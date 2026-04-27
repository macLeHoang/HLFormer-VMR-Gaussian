"""
Dataset and dataloader for Video Moment Retrieval (VMR).
Adapted from QD-DETR (https://github.com/wjun0830/QD-DETR) and
HLFormer data_provider.py.

Expected annotation format (JSONL, one record per line):
{
  "qid": 7803,
  "query": "Man in gray top walks from outside to inside.",
  "duration": 150.0,
  "vid": "RoripwjYFp8_360.0_510.0",
  "relevant_clip_ids": [13, 14, 15, 16, 17],
  "relevant_windows": [[26, 36]],
  "saliency_scores": [[2, 2, 3], [2, 2, 2], ...]   # optional, for QVHighlights
}

Feature files on disk:
  video : <v_feat_dir>/<vid>.npz  with key "features"  -> (L_vid, D_vid)
  query : <q_feat_dir>/qid<qid>.npz  with key "last_hidden_state" -> (L_txt, D_txt)
"""

import json
import random
import logging
from os.path import join, exists
from statistics import mean

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader

from VMR.Models.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def resample_feat(feat, target_len):
    """Resample feature array to target length using linear interpolation."""
    if len(feat) == target_len:
        return feat
    factor = target_len / len(feat)
    return scipy.ndimage.zoom(feat, (factor, 1), order=1)


def l2_normalize_np_array(arr, eps=1e-5):
    """L2-normalize the last dimension of a numpy array."""
    return arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + eps)


def pad_sequences_1d(sequences, dtype=torch.float32, fixed_length=None):
    """Pad a list of variable-length tensors or numpy arrays to the same length.

    Args:
        sequences: list of 1-D or 2-D arrays/tensors
        dtype: target dtype for the padded tensor
        fixed_length: if given, pad/truncate to this length; otherwise use max

    Returns:
        padded: (B, max_len) or (B, max_len, D) tensor
        mask:   (B, max_len) float32 tensor, 1=valid, 0=pad
    """
    if isinstance(sequences[0], np.ndarray):
        sequences = [torch.from_numpy(s) for s in sequences]

    lengths = [s.shape[0] for s in sequences]
    max_len = fixed_length if fixed_length is not None else max(lengths)
    B = len(sequences)

    extra_dims = sequences[0].shape[1:]
    padded = torch.zeros(B, max_len, *extra_dims, dtype=dtype)
    mask   = torch.zeros(B, max_len, dtype=torch.float32)

    for i, seq in enumerate(sequences):
        length = min(seq.shape[0], max_len)
        padded[i, :length] = seq[:length].to(dtype)
        mask[i, :length]   = 1.0

    return padded, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VMRDataset(Dataset):
    """PyTorch Dataset for Video Moment Retrieval.

    Each sample is one (video, query, temporal annotations) tuple.
    Supports QVHighlights, Charades-STA, ActivityNet Captions, TACoS.

    Args:
        dset_name:       dataset identifier, e.g. "qvhighlights", "charades_sta"
        data_path:       path to annotation JSONL file
        v_feat_dirs:     list of directories containing video .npz feature files
        q_feat_dir:      directory (or list of dirs) containing query .npz feature files
        q_feat_type:     key inside query .npz, e.g. "last_hidden_state"
        max_q_l:         max query token length (will truncate)
        max_v_l:         max video clip length (will truncate)
        clip_len:        seconds per clip (used to convert time -> clip index)
        max_windows:     max number of ground-truth windows per sample
        normalize_v:     L2-normalize video features
        normalize_t:     L2-normalize text features
        load_labels:     whether to load span/saliency labels (False for test)
        txt_drop_ratio:  probability of randomly zeroing out a query token row
        data_ratio:      use only this fraction of data (useful for debugging)
        v_feat_len_mode: how to align multi-source video features before concat.
                         "max" – resample shorter sources up to the longest
                                 (preserves all temporal information).
                         "min" – truncate longer sources down to the shortest
                                 (faster, no interpolation artefacts).
        q_feat_len_mode: same as v_feat_len_mode but for query features.
        temporal_crop_ratio: if > 0, randomly crop this fraction of video during
                             training and rescale GT windows accordingly.
        feat_mask_ratio: if > 0, randomly zero out this fraction of video clips.
        gt_jitter_frames: if > 0, jitter GT boundaries by up to this many frames.
        use_tef:          append Temporal Endpoint Features (start/end in [0,1]).
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, clip_len=2.0,
                 max_windows=5, normalize_v=True, normalize_t=True,
                 load_labels=True, txt_drop_ratio=0.0, data_ratio=1.0,
                 v_feat_len_mode="max", q_feat_len_mode="min",
                 temporal_crop_ratio=0.0, feat_mask_ratio=0.0,
                 gt_jitter_frames=0, use_tef=False):

        self.dset_name      = dset_name
        self.data_path      = data_path
        self.v_feat_dirs    = v_feat_dirs if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dirs    = q_feat_dir  if isinstance(q_feat_dir,  list) else [q_feat_dir]
        self.q_feat_type    = q_feat_type
        self.max_q_l        = max_q_l
        self.max_v_l        = max_v_l
        self.clip_len       = clip_len
        self.max_windows    = max_windows
        self.normalize_v      = normalize_v
        self.normalize_t      = normalize_t
        self.load_labels      = load_labels
        self.txt_drop_ratio   = txt_drop_ratio
        assert v_feat_len_mode in ("max", "min"), \
            f"v_feat_len_mode must be 'max' or 'min', got '{v_feat_len_mode}'"
        self.v_feat_len_mode  = v_feat_len_mode
        assert q_feat_len_mode in ("max", "min"), \
            f"q_feat_len_mode must be 'max' or 'min', got '{q_feat_len_mode}'"
        self.q_feat_len_mode  = q_feat_len_mode
        self.temporal_crop_ratio = temporal_crop_ratio
        self.feat_mask_ratio     = feat_mask_ratio
        self.gt_jitter_frames    = gt_jitter_frames
        self.use_tef             = use_tef

        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0.0, \
                "txt_drop_ratio must be 0 for val/test"

        self.data = self._load_data(data_ratio)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, data_ratio):
        data = load_jsonl(self.data_path)
        if data_ratio < 1.0:
            n = int(len(data) * data_ratio)
            data = data[:n]
            logger.info("Using %.0f%% of data: %d samples", data_ratio * 100, n)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = {}
        model_inputs["query_feat"] = self._get_query_feat(meta["qid"])   # (L_q, D_q)
        model_inputs["video_feat"] = self._get_video_feat(meta["vid"])   # (L_v, D_v)
        orig_ctx_l = len(model_inputs["video_feat"])                     # #clips before crop
        ctx_l = orig_ctx_l
        orig_duration = float(meta["duration"])
        orig_clip_len = orig_duration / max(orig_ctx_l, 1)

        # --- Temporal crop augmentation (training only) ---
        # Randomly crop a contiguous sub-sequence containing the GT window,
        # then rescale GT boundaries to the cropped range.
        crop_offset = 0  # frames removed from the start
        if (self.load_labels and self.temporal_crop_ratio > 0
                and random.random() < 0.5):
            model_inputs["video_feat"], crop_offset, ctx_l = \
                self._temporal_crop(model_inputs["video_feat"],
                                    meta["relevant_windows"], orig_duration,
                                    ctx_l)

        # --- Random feature masking (training only) ---
        if self.load_labels and self.feat_mask_ratio > 0:
            model_inputs["video_feat"] = self._random_mask_clips(
                model_inputs["video_feat"])
            
        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / max(ctx_l, 1)
            tef_ed = tef_st + 1.0 / max(ctx_l, 1)
            tef = torch.stack([tef_st, tef_ed], dim=1)
            model_inputs["video_feat"] = torch.cat([model_inputs["video_feat"], tef], dim=1)

        if self.load_labels:
            # Adjust GT windows for any temporal crop offset
            adjusted_windows = meta["relevant_windows"]
            adjusted_duration = orig_duration
            if crop_offset > 0 or ctx_l != orig_ctx_l:
                adjusted_duration = max(orig_clip_len * ctx_l, 1e-6)
                crop_shift = crop_offset * orig_clip_len
                adjusted_windows = [
                    [max(0.0, w[0] - crop_shift),
                     min(adjusted_duration, w[1] - crop_shift)]
                    for w in meta["relevant_windows"]
                ]
                adjusted_windows = [[max(0.0, s), max(0.0, e)]
                                    for s, e in adjusted_windows]

            # GT jitter: randomly perturb boundaries by a few frames
            if self.gt_jitter_frames > 0:
                adjusted_windows = self._jitter_windows(
                    adjusted_windows, adjusted_duration, ctx_l)

            model_inputs["span_labels"] = self._get_span_labels(
                adjusted_windows, ctx_l, adjusted_duration)

            if self.dset_name in ["charades_sta", "tacos", "activitynet"]:
                pos, neg, score_arr = self._get_saliency_sub_as_query(
                    adjusted_windows[0], adjusted_duration, ctx_l)
            else:
                pos, neg, score_arr = self._get_saliency_all(
                    meta["relevant_clip_ids"],
                    meta.get("saliency_scores", []),
                    ctx_l)

            model_inputs["saliency_pos_labels"] = pos
            model_inputs["saliency_neg_labels"] = neg
            model_inputs["saliency_all_labels"] = score_arr

        return {"meta": meta, "model_inputs": model_inputs}

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _get_query_feat(self, qid):
        feat_list = []
        for feat_dir in self.q_feat_dirs:
            if self.dset_name in ["tacos", "nlq", "youtube_uni"]:
                path = join(feat_dir, f"{qid}.npz")
            else:
                path = join(feat_dir, f"qid{qid}.npz")

            feat = np.load(path)[self.q_feat_type].astype(np.float32)

            if self.q_feat_type == "last_hidden_state":
                feat = feat[:self.max_q_l]
            if self.normalize_t:
                feat = l2_normalize_np_array(feat)
                
            feat_list.append(feat)

        if self.q_feat_len_mode == "max":
            target_len = max(len(f) for f in feat_list)
            feat_list = [resample_feat(f, target_len) for f in feat_list]
        else:  # "min"
            target_len = min(len(f) for f in feat_list)
            feat_list = [f[:target_len] for f in feat_list]

        feat = np.concatenate(feat_list, axis=1)

        if self.txt_drop_ratio > 0:
            feat = self._random_drop_rows(feat)

        return torch.from_numpy(feat)   # (L_q, D_q)

    def _get_video_feat(self, vid):
        feat_list = []
        raw_lengths = []

        for feat_dir in self.v_feat_dirs:
            path = join(feat_dir, f"{vid}.npz")
            feat = np.load(path)["features"].astype(np.float32)
            raw_lengths.append(len(feat))
            feat_list.append(feat)

        target_len = self._aligned_video_length_from_lengths(raw_lengths)
        target_len = min(target_len, self.max_v_l)

        feat_list = [resample_feat(f, target_len) for f in feat_list]

        if self.normalize_v:
            feat_list = [l2_normalize_np_array(f) for f in feat_list]

        feat = np.concatenate(feat_list, axis=1)
        return torch.from_numpy(feat)   # (target_len, D_v)

    def _aligned_video_length_from_lengths(self, lengths):
        if not lengths:
            return 0
        return max(lengths) if self.v_feat_len_mode == "max" else min(lengths)

    def summarize_temporal_alignment(self, max_samples=None):
        samples = self.data if max_samples is None else self.data[:max_samples]
        if not samples:
            return {"samples": 0}

        raw_lengths_per_stream = [[] for _ in self.v_feat_dirs]
        final_ctx_lengths = []
        durations = []
        aligned_clip_lens = []
        stream_length_mismatch_count = 0
        cfg_clip_len_mismatch_count = 0
        clamp_count = 0
        load_failures = 0

        for meta in samples:
            try:
                stream_lengths = []
                for stream_idx, feat_dir in enumerate(self.v_feat_dirs):
                    path = join(feat_dir, f"{meta['vid']}.npz")
                    feat = np.load(path)["features"]
                    raw_len = int(min(len(feat), self.max_v_l))
                    raw_lengths_per_stream[stream_idx].append(raw_len)
                    stream_lengths.append(raw_len)

                ctx_l = self._aligned_video_length_from_lengths(stream_lengths)
                duration = float(meta.get("duration", 0.0))
                aligned_clip_len = duration / max(ctx_l, 1)

                final_ctx_lengths.append(ctx_l)
                durations.append(duration)
                aligned_clip_lens.append(aligned_clip_len)

                if len(set(stream_lengths)) > 1:
                    stream_length_mismatch_count += 1

                if self.clip_len > 0:
                    rel_err = abs(aligned_clip_len - self.clip_len) / max(self.clip_len, 1e-6)
                    if rel_err > 0.05:
                        cfg_clip_len_mismatch_count += 1

                if self.load_labels:
                    total_len = max(duration, 1e-6)
                    windows = torch.tensor(meta.get("relevant_windows", []), dtype=torch.float32)
                    if windows.numel() > 0:
                        windows_norm_raw = windows / total_len
                        windows_norm = windows_norm_raw.clamp(0.0, 1.0)
                        if not torch.equal(windows_norm, windows_norm_raw):
                            clamp_count += 1
            except Exception:
                load_failures += 1

        summary = {
            "samples": len(samples),
            "load_failures": load_failures,
            "stream_length_mismatch_count": stream_length_mismatch_count,
            "stream_length_mismatch_ratio": stream_length_mismatch_count / max(len(samples), 1),
            "cfg_clip_len_mismatch_count": cfg_clip_len_mismatch_count,
            "cfg_clip_len_mismatch_ratio": cfg_clip_len_mismatch_count / max(len(samples), 1),
            "span_clamp_count": clamp_count,
            "span_clamp_ratio": clamp_count / max(len(samples), 1),
            "ctx_l_mean": mean(final_ctx_lengths) if final_ctx_lengths else 0.0,
            "duration_mean": mean(durations) if durations else 0.0,
            "aligned_clip_len_mean": mean(aligned_clip_lens) if aligned_clip_lens else 0.0,
        }
        for i, vals in enumerate(raw_lengths_per_stream):
            summary[f"stream{i}_raw_len_mean"] = mean(vals) if vals else 0.0
        return summary

    def _random_drop_rows(self, feat):
        n_drop = round(len(feat) * self.txt_drop_ratio)
        if n_drop > 0:
            idx = np.random.choice(len(feat), size=n_drop, replace=False)
            feat[idx] = 0.0
        return feat

    def _temporal_crop(self, video_feat, relevant_windows, duration, ctx_l):
        """Randomly crop a contiguous sub-sequence that contains the GT window.

        Returns:
            cropped_feat: (new_L, D) tensor
            crop_offset:  number of frames removed from start
            new_ctx_l:    new number of clips
        """
        clip_len = duration / max(ctx_l, 1)
        # Find the frame range that must be included (GT window)
        gt_st_frame = int(relevant_windows[0][0] / clip_len)
        gt_ed_frame = min(int(relevant_windows[0][1] / clip_len), ctx_l - 1)
        gt_st_frame = min(gt_st_frame, gt_ed_frame)

        # How many frames to keep: at least gt_span + some context
        gt_len = gt_ed_frame - gt_st_frame + 1
        min_keep = max(gt_len + 4, int(ctx_l * (1.0 - self.temporal_crop_ratio)))
        min_keep = min(min_keep, ctx_l)

        keep_len = random.randint(min_keep, ctx_l)

        # Random start such that GT is fully contained
        max_start = max(0, gt_st_frame)
        min_end = gt_ed_frame + 1
        # start must satisfy: start <= gt_st_frame and start + keep_len >= gt_ed_frame + 1
        start_lo = max(0, min_end - keep_len)
        start_hi = min(max_start, ctx_l - keep_len)
        if start_lo > start_hi:
            start_lo = start_hi = max(0, ctx_l - keep_len)
        crop_start = random.randint(start_lo, max(start_lo, start_hi))

        cropped = video_feat[crop_start:crop_start + keep_len]
        return cropped, crop_start, len(cropped)

    def _random_mask_clips(self, video_feat):
        """Randomly zero out a fraction of video clip features."""
        n_clips = len(video_feat)
        n_mask = max(1, round(n_clips * self.feat_mask_ratio))
        idx = np.random.choice(n_clips, size=n_mask, replace=False)
        video_feat = video_feat.clone()
        video_feat[idx] = 0.0
        return video_feat

    def _jitter_windows(self, windows, duration, ctx_l):
        """Randomly jitter GT window boundaries by up to gt_jitter_frames."""
        clip_len = duration / max(ctx_l, 1)
        jitter_sec = self.gt_jitter_frames * clip_len
        jittered = []
        for st, ed in windows:
            st_new = st + random.uniform(-jitter_sec, jitter_sec)
            ed_new = ed + random.uniform(-jitter_sec, jitter_sec)
            st_new = max(0, st_new)
            ed_new = min(duration, ed_new)
            if ed_new <= st_new:
                ed_new = st_new + clip_len  # ensure non-zero span
            jittered.append([st_new, ed_new])
        return jittered

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def _get_span_labels(self, windows, ctx_l, duration):
        """Convert list of [start_sec, end_sec] windows to normalized (cx, w) format.

        Args:
            windows: list of [st, ed] in seconds
            ctx_l:   number of clips in the video
            duration: total video duration in seconds

        Returns:
            Tensor (n_windows, 2) of normalized (center, width) spans
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]

        # total_len = ctx_l * self.clip_len
        total_len = max(float(duration), 1e-6)
        windows_t = torch.tensor(windows, dtype=torch.float32)
        windows_norm = windows_t / total_len        # normalize to [0, 1]
        windows_norm = windows_norm.clamp(0.0, 1.0)
        return span_xx_to_cxw(windows_norm)         # (n_windows, 2)

    def _get_saliency_all(self, rel_clip_ids, scores, ctx_l, max_n=1):
        """Saliency labels for QVHighlights (multiple annotator scores).

        Positive clips: highest aggregated scores among relevant clips.
        Negative clips: lowest aggregated scores among relevant clips +
                        easy negatives outside relevant clips.
        """
        if len(scores) == 0:
            # Fall back to window-based saliency if no scores provided
            return self._get_saliency_sub_as_query([[0, 0]], ctx_l * self.clip_len, ctx_l)

        scores_arr = np.array(scores)           # (#rel_clips, #annotators)
        agg = scores_arr.sum(axis=1)            # (#rel_clips,)
        sort_idx = np.argsort(agg)              # ascending

        score_array = np.zeros(ctx_l, dtype=np.float32)
        for i, cid in enumerate(rel_clip_ids):
            if cid < ctx_l:
                score_array[cid] = agg[i]

        hard_pos = [min(rel_clip_ids[i], ctx_l - 1) for i in sort_idx[-max_n:]]
        hard_neg = [min(rel_clip_ids[i], ctx_l - 1) for i in sort_idx[:max_n]]

        easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
        if len(easy_neg_pool) >= max_n:
            easy_pos = random.sample(rel_clip_ids, k=max_n)
            easy_neg = random.sample(easy_neg_pool, k=max_n)
        else:
            easy_pos = hard_pos
            easy_neg = hard_neg

        return hard_pos + easy_pos, hard_neg + easy_neg, score_array

    def _get_saliency_sub_as_query(self, gt_window, duration, ctx_l, max_n=2):
        """Saliency labels derived from a single GT window (Charades-STA, etc.).

        Positive clips: sampled inside the GT window.
        Negative clips: sampled outside the GT window.
        """
        clip_len = duration / ctx_l
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clips = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clips = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))
        try:
            neg_clips = random.sample(neg_pool, k=max_n)
        except ValueError:
            neg_clips = pos_clips   # edge case: entire video is relevant

        score_array = np.zeros(ctx_l, dtype=np.float32)
        score_array[gt_st:gt_ed + 1] = 1.0

        return pos_clips, neg_clips, score_array


# ---------------------------------------------------------------------------
# Collate and prepare
# ---------------------------------------------------------------------------

def make_collate_fn(max_v_l):
    """Return a collate function that pads each batch to its own max length.

    Video features and saliency_all_labels are padded dynamically so their
    temporal length always matches within a batch.
    """
    def vmr_collate_fn(batch):
        batch_meta = [e["meta"] for e in batch]
        keys = batch[0]["model_inputs"].keys()
        batched_data = {}

        for k in keys:
            if k == "span_labels":
                batched_data[k] = [{"spans": e["model_inputs"][k]} for e in batch]

            elif k in ("saliency_pos_labels", "saliency_neg_labels"):
                batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])

            elif k == "saliency_all_labels":
                padded, _ = pad_sequences_1d(
                    [e["model_inputs"][k] for e in batch],
                    dtype=torch.float32, fixed_length=None)
                batched_data[k] = padded

            elif k == "video_feat":
                padded, mask = pad_sequences_1d(
                    [e["model_inputs"][k] for e in batch],
                    dtype=torch.float32, fixed_length=None)
                batched_data[k] = (padded, mask)

            elif k == "query_feat":
                padded, mask = pad_sequences_1d(
                    [e["model_inputs"][k] for e in batch], dtype=torch.float32)
                batched_data[k] = (padded, mask)

            else:
                batched_data[k] = torch.stack([e["model_inputs"][k] for e in batch])

        return batch_meta, batched_data

    return vmr_collate_fn


def prepare_batch_inputs(batched_data, device, non_blocking=False):
    """Move batched data from collate_fn to the correct device and format.

    Returns:
        model_inputs: dict with src_vid, src_vid_mask, src_txt, src_txt_mask
        targets:      dict with span_labels, saliency_*_labels (or None for test)
    """
    to = lambda t: t.to(device, non_blocking=non_blocking)

    model_inputs = dict(
        src_txt      = to(batched_data["query_feat"][0]),
        src_txt_mask = to(batched_data["query_feat"][1]),
        src_vid      = to(batched_data["video_feat"][0]),
        src_vid_mask = to(batched_data["video_feat"][1]),
    )

    targets = {}
    if "span_labels" in batched_data:
        targets["span_labels"] = [
            {"spans": to(d["spans"])} for d in batched_data["span_labels"]
        ]
    if "saliency_pos_labels" in batched_data:
        targets["saliency_pos_labels"] = to(batched_data["saliency_pos_labels"])
        targets["saliency_neg_labels"] = to(batched_data["saliency_neg_labels"])
    if "saliency_all_labels" in batched_data:
        targets["saliency_all_labels"] = to(batched_data["saliency_all_labels"])

    return model_inputs, targets if targets else None


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_vmr_dataloaders(cfg):
    """Build train/val/test DataLoaders from config.

    Expected config keys:
        train_path, val_path, test_path (optional)
        v_feat_dirs, q_feat_dir, q_feat_type
        max_q_l, max_v_l, clip_len
        batchsize, num_workers, pin_memory
        normalize_v, normalize_t
        txt_drop_ratio, data_ratio
        dset_name

    Returns:
        train_loader, val_loader, test_loader (test_loader may be None)
    """
    common = dict(
        dset_name      = cfg["dset_name"],
        v_feat_dirs    = cfg["v_feat_dirs"],
        q_feat_dir     = cfg["q_feat_dir"],
        q_feat_type    = cfg.get("q_feat_type", "last_hidden_state"),
        max_q_l        = cfg["max_q_l"],
        max_v_l        = cfg["max_v_l"],
        clip_len       = cfg.get("clip_len", 2.0),
        max_windows    = cfg.get("max_windows", 5),
        normalize_v      = cfg.get("normalize_v", True),
        normalize_t      = cfg.get("normalize_t", True),
        v_feat_len_mode  = cfg.get("v_feat_len_mode", "max"),
        q_feat_len_mode  = cfg.get("q_feat_len_mode", "min"),
        use_tef          = cfg.get("use_tef", False),
    )

    train_dset = VMRDataset(
        data_path     = cfg["train_path"],
        load_labels   = True,
        txt_drop_ratio= cfg.get("txt_drop_ratio", 0.0),
        data_ratio    = cfg.get("data_ratio", 1.0),
        temporal_crop_ratio = cfg.get("temporal_crop_ratio", 0.0),
        feat_mask_ratio     = cfg.get("feat_mask_ratio", 0.0),
        gt_jitter_frames    = cfg.get("gt_jitter_frames", 0),
        **common,
    )
    val_dset = VMRDataset(
        data_path     = cfg["val_path"],
        load_labels   = True,
        txt_drop_ratio= 0.0,
        data_ratio    = 1.0,
        **common,
    )

    loader_kw = dict(
        batch_size  = cfg["batchsize"],
        num_workers = cfg.get("num_workers", 4),
        pin_memory  = cfg.get("pin_memory", True),
        collate_fn  = make_collate_fn(cfg["max_v_l"]),
    )

    train_loader = DataLoader(train_dset, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_dset,   shuffle=False, **loader_kw)

    test_loader = None
    if cfg.get("test_path") and exists(cfg["test_path"]):
        test_dset = VMRDataset(
            data_path    = cfg["test_path"],
            load_labels  = False,
            txt_drop_ratio=0.0,
            data_ratio   = 1.0,
            **common,
        )
        test_loader = DataLoader(test_dset, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
