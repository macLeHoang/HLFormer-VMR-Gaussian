"""
Evaluation metrics for Video Moment Retrieval (VMR).

Metrics:
  - R1@IoU{0.5, 0.7}  : Recall@1 at IoU thresholds 0.5 and 0.7
  - mAP@IoU{0.5, 0.7} : mean Average Precision at IoU thresholds
  - Highlight mAP     : average precision for saliency scores (QVHighlights only)
  - HIT@1             : saliency score > 0 at the top-1 retrieved clip
"""

import numpy as np
import torch
from VMR.Models.span_utils import span_cxw_to_xx, temporal_iou


SPAN_SOURCE_KEYS = {
    "coarse":   "pred_spans",
    "refined":  "pred_spans_refined",
    "final":    "pred_spans_final",
}


# ---------------------------------------------------------------------------
# Post-processing: model output -> list of prediction dicts
# ---------------------------------------------------------------------------

def _select_span_tensor(outputs, use_refined_spans=True, span_key=None):
    """Select a span tensor from model outputs."""
    if span_key is not None:
        return outputs.get(span_key)

    if use_refined_spans:
        for key in ("pred_spans_final", "pred_spans_refined", "pred_spans"):
            span_tensor = outputs.get(key)
            if span_tensor is not None:
                return span_tensor
        return outputs["pred_spans"]

    return outputs["pred_spans"]


def post_process_predictions(outputs, metas, duration_key="duration",
                             top_k=5, nms_thresh=0.4,
                             use_refined_spans=True, span_key=None):
    """Convert raw model outputs to per-sample prediction lists.

    Args:
        outputs:      dict from model.forward() – includes 'pred_logits' and span outputs
        metas:        list of sample metadata dicts (length B)
        duration_key: key in meta that stores video duration in seconds
        top_k:        keep at most this many spans per sample (before NMS)
        nms_thresh:   IoU threshold for non-maximum suppression

    Returns:
        predictions: list of B dicts, each with
            "pred_spans":  (K, 2) float numpy, [start_sec, end_sec]
            "pred_scores": (K,)   float numpy, quality/ranking scores
            "vid":         str
            "qid":         int
    """
    pred_logits = outputs["pred_logits"].detach().cpu()   # (B, Q)
    # Inference span selection is independent from matcher/label supervision.
    # When use_refined_spans=True, prefer final spans, then refined spans, then
    # coarse spans as a safe fallback for older checkpoints or disabled refinement.
    _raw = _select_span_tensor(outputs, use_refined_spans=use_refined_spans, span_key=span_key)
    if _raw is None:
        return None
    pred_spans  = _raw.detach().cpu()                     # (B, Q, 2) cxw [0,1]

    # Quality/ranking score: pred_logits is a single logit per query slot.
    fg_probs = pred_logits.sigmoid()                       # (B, Q)

    predictions = []
    for i, meta in enumerate(metas):
        duration = meta.get(duration_key, 1.0)

        probs     = fg_probs[i]                           # (Q,)
        spans_cxw = pred_spans[i]                         # (Q, 2)

        # Convert (center, width) normalized -> (start, end) in seconds
        spans_xx  = span_cxw_to_xx(spans_cxw)            # (Q, 2) in [0,1]
        spans_sec = spans_xx * duration                   # (Q, 2) in seconds
        spans_sec = spans_sec.clamp(min=0.0, max=duration)
        spans_sec[:, 1] = torch.maximum(
            spans_sec[:, 1], spans_sec[:, 0] + 0.01)     # ensure start < end

        # Sort by quality/ranking score descending, keep top-k
        order = probs.argsort(descending=True)[:top_k]
        probs     = probs[order]
        spans_sec = spans_sec[order]

        # Temporal NMS
        kept_spans, kept_probs = temporal_nms(spans_sec, probs, nms_thresh)

        predictions.append({
            "pred_spans":  kept_spans.numpy(),    # (K, 2)
            "pred_scores": kept_probs.numpy(),    # (K,)
            "vid":         meta.get("vid", ""),
            "qid":         meta.get("qid", -1),
        })

    return predictions


def _spans_to_seconds(span_tensor, metas, duration_key="duration"):
    """Convert normalized cxw spans to seconds for each sample."""
    spans_xx = span_cxw_to_xx(span_tensor.detach().cpu())
    spans_sec = spans_xx.clone()
    for i, meta in enumerate(metas):
        duration = float(meta.get(duration_key, 1.0))
        spans_sec[i] = spans_sec[i] * duration
        spans_sec[i, :, 0] = spans_sec[i, :, 0].clamp(min=0.0, max=duration)
        spans_sec[i, :, 1] = spans_sec[i, :, 1].clamp(min=0.0, max=duration)
        spans_sec[i, :, 1] = torch.maximum(
            spans_sec[i, :, 1], spans_sec[i, :, 0] + 0.01)
    return spans_sec


def _append_span_delta(metrics_acc, src_name, dst_name, src_tensor, dst_tensor, metas):
    """Accumulate mean absolute start/end deltas between two span sources."""
    if src_tensor is None or dst_tensor is None:
        return

    src_sec = _spans_to_seconds(src_tensor, metas)
    dst_sec = _spans_to_seconds(dst_tensor, metas)
    deltas = (dst_sec - src_sec).abs()
    pair_key = f"{src_name}_to_{dst_name}"
    metrics_acc.setdefault(pair_key, {"start": [], "end": []})
    metrics_acc[pair_key]["start"].append(deltas[:, :, 0].reshape(-1).numpy())
    metrics_acc[pair_key]["end"].append(deltas[:, :, 1].reshape(-1).numpy())


def temporal_nms(spans, scores, iou_thresh=0.4):
    """Greedy temporal NMS.

    Args:
        spans:  (N, 2) torch tensor [start, end]
        scores: (N,)   torch tensor, descending order assumed
        iou_thresh: suppress spans with IoU > threshold

    Returns:
        kept_spans:  (K, 2)
        kept_scores: (K,)
    """
    if len(spans) == 0:
        return spans, scores

    keep = []
    suppressed = torch.zeros(len(spans), dtype=torch.bool)

    for i in range(len(spans)):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 == len(spans):
            break
        iou, _ = temporal_iou(spans[i].unsqueeze(0), spans[i + 1:])
        suppressed[i + 1:] |= (iou.squeeze(0) > iou_thresh)

    keep = torch.tensor(keep)
    return spans[keep], scores[keep]


# ---------------------------------------------------------------------------
# Ground-truth extraction helpers
# ---------------------------------------------------------------------------

def extract_gt_windows(metas, duration_key="duration"):
    """Extract list of GT windows (seconds) per sample.

    Returns:
        gt_list: list of B lists, each containing [start_sec, end_sec] windows
    """
    gt_list = []
    for meta in metas:
        duration = meta.get(duration_key, 1.0)
        windows  = meta.get("relevant_windows", [])
        gt_list.append([[float(s), float(e)] for s, e in windows])
    return gt_list


# ---------------------------------------------------------------------------
# Moment Retrieval metrics
# ---------------------------------------------------------------------------

def compute_iou_with_gt(pred_span, gt_windows):
    """Compute max IoU between a single predicted span and all GT windows.

    Args:
        pred_span:  [start, end]  (list or array)
        gt_windows: list of [start, end]

    Returns:
        max_iou: float
    """
    if not gt_windows:
        return 0.0
    pred = torch.tensor(np.array([pred_span]), dtype=torch.float32)
    gts  = torch.tensor(np.array(gt_windows), dtype=torch.float32)
    iou, _ = temporal_iou(pred, gts)          # (1, n_gt)
    return iou.max().item()


def compute_r1(predictions, gt_list, iou_thresh=0.5):
    """Recall@1: fraction of samples where the top-1 prediction exceeds IoU threshold.

    Args:
        predictions: list of B prediction dicts (from post_process_predictions)
        gt_list:     list of B GT windows lists
        iou_thresh:  IoU threshold

    Returns:
        r1: float in [0, 100]
    """
    hits = 0
    for pred, gts in zip(predictions, gt_list):
        if len(pred["pred_spans"]) == 0:
            continue
        top1_span = pred["pred_spans"][0]
        max_iou   = compute_iou_with_gt(top1_span, gts)
        if max_iou >= iou_thresh:
            hits += 1
    return 100.0 * hits / max(len(predictions), 1)


def compute_map(predictions, gt_list, iou_thresh=0.5):
    """Compute mAP for moment retrieval (simplified, per-sample average).

    For each sample we compute AP across the ranked predicted spans.

    Args:
        predictions: list of B prediction dicts
        gt_list:     list of B GT window lists
        iou_thresh:  IoU threshold to count a prediction as correct

    Returns:
        map_val: float in [0, 100]
    """
    aps = []
    for pred, gts in zip(predictions, gt_list):
        spans  = pred["pred_spans"]
        scores = pred["pred_scores"]
        if len(spans) == 0 or not gts:
            aps.append(0.0)
            continue

        matched_gt = set()
        tp_list    = []
        for span in spans:
            max_iou = -1.0
            best_j  = -1
            for j, gt in enumerate(gts):
                iou = compute_iou_with_gt(span, [gt])
                if iou > max_iou:
                    max_iou = iou
                    best_j  = j
            if max_iou >= iou_thresh and best_j not in matched_gt:
                tp_list.append(1)
                matched_gt.add(best_j)
            else:
                tp_list.append(0)

        tp_cumsum = np.cumsum(tp_list)
        precisions = tp_cumsum / (np.arange(len(tp_list)) + 1)
        recalls    = tp_cumsum / max(len(gts), 1)

        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precisions, recalls):
            if r > prev_recall:
                ap += p * (r - prev_recall)
                prev_recall = r
        aps.append(ap)

    return 100.0 * float(np.mean(aps))


# ---------------------------------------------------------------------------
# Highlight Detection metrics (QVHighlights)
# ---------------------------------------------------------------------------

def compute_highlight_map(saliency_scores, saliency_labels, vid_mask):
    """Compute highlight mAP over all samples.

    Args:
        saliency_scores: (B, L) torch tensor
        saliency_labels: (B, L) numpy array with raw annotation scores
        vid_mask:        (B, L) torch tensor, 1=valid

    Returns:
        hl_map: float in [0, 100]
    """
    aps = []
    B = saliency_scores.shape[0]
    scores_np = saliency_scores.detach().cpu().numpy()
    mask_np   = vid_mask.detach().cpu().numpy()

    for i in range(B):
        valid = mask_np[i].astype(bool)
        s     = scores_np[i][valid]
        lbl   = saliency_labels[i][:valid.sum()]

        if lbl.max() == 0:
            aps.append(0.0)
            continue

        # Binary: clip is "positive" if label > median
        binary_lbl = (lbl >= lbl.mean()).astype(int)
        order      = np.argsort(-s)
        s_sorted   = s[order]
        l_sorted   = binary_lbl[order]

        tp_cumsum  = np.cumsum(l_sorted)
        precisions = tp_cumsum / (np.arange(len(l_sorted)) + 1)
        recalls    = tp_cumsum / max(l_sorted.sum(), 1)

        ap = 0.0
        prev_r = 0.0
        for p, r in zip(precisions, recalls):
            if r > prev_r:
                ap += p * (r - prev_r)
                prev_r = r
        aps.append(ap)

    return 100.0 * float(np.mean(aps))


def compute_hit1(saliency_scores, saliency_labels, vid_mask):
    """HIT@1: top-1 clip has a positive saliency label.

    Args:
        saliency_scores: (B, L) tensor
        saliency_labels: (B, L) numpy
        vid_mask:        (B, L) tensor

    Returns:
        hit1: float in [0, 100]
    """
    scores = saliency_scores.detach().cpu()
    mask   = vid_mask.detach().cpu()

    # Mask out padding
    scores = scores * mask + (1.0 - mask) * -1e9

    top1_idx = scores.argmax(dim=1).numpy()   # (B,)
    hits = 0
    for i, idx in enumerate(top1_idx):
        if saliency_labels[i][idx] > 0:
            hits += 1

    return 100.0 * hits / max(len(top1_idx), 1)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_vmr(model, dataloader, device, cfg):
    """Run full evaluation on a dataloader and return metrics.

    Args:
        model:      GaussianFormer_VMR (eval mode)
        dataloader: DataLoader returning (batch_meta, batched_data)
        device:     torch.device
        cfg:        config dict with iou_thresholds, nms_thresh, top_k

    Returns:
        metrics: dict with R1@0.5, R1@0.7, mAP@0.5, mAP@0.7,
                 hl_mAP (if saliency labels present), HIT@1
    """
    from VMR.Datasets.vmr_data_provider import prepare_batch_inputs

    model.eval()
    iou_thresholds  = cfg.get("iou_thresholds", [0.5, 0.7])
    top_k           = cfg.get("top_k", 5)
    nms_thresh      = cfg.get("nms_thresh", 0.4)
    use_refined     = cfg.get("use_refined_spans", True)
    eval_span_source_metrics = cfg.get("eval_span_source_metrics", False)
    eval_refine_diagnostics  = cfg.get("eval_refine_diagnostics",  False)

    all_preds          = []
    all_gt             = []
    all_saliency_scores = []
    all_saliency_labels = []
    all_vid_masks       = []
    all_gate_means      = []
    all_gate_values     = []
    span_source_preds   = {name: [] for name in SPAN_SOURCE_KEYS}
    span_source_counts  = {name: 0 for name in SPAN_SOURCE_KEYS}
    span_delta_acc      = {}

    for batch_meta, batched_data in dataloader:
        model_inputs, targets = prepare_batch_inputs(batched_data, device)
        outputs = model(**model_inputs)

        preds = post_process_predictions(outputs, batch_meta, top_k=top_k,
                                         nms_thresh=nms_thresh,
                                         use_refined_spans=use_refined)
        gts   = extract_gt_windows(batch_meta)

        all_preds.extend(preds)
        all_gt.extend(gts)

        if eval_span_source_metrics:
            for source_name, span_key in SPAN_SOURCE_KEYS.items():
                source_preds = post_process_predictions(
                    outputs,
                    batch_meta,
                    top_k=top_k,
                    nms_thresh=nms_thresh,
                    use_refined_spans=use_refined,
                    span_key=span_key,
                )
                if source_preds is None:
                    continue
                span_source_preds[source_name].extend(source_preds)
                span_source_counts[source_name] += len(source_preds)

        gate = outputs.get("pred_refine_gate")
        if gate is not None:
            all_gate_means.append(gate.mean().item())
            if eval_refine_diagnostics:
                all_gate_values.append(gate.detach().cpu().reshape(-1).float().numpy())

        if eval_refine_diagnostics:
            coarse_spans = outputs.get(SPAN_SOURCE_KEYS["coarse"])
            refined_spans = outputs.get(SPAN_SOURCE_KEYS["refined"])
            final_spans = outputs.get(SPAN_SOURCE_KEYS["final"])
            _append_span_delta(span_delta_acc, "coarse", "refined",
                               coarse_spans, refined_spans, batch_meta)
            _append_span_delta(span_delta_acc, "coarse", "final",
                               coarse_spans, final_spans, batch_meta)

        if "saliency_all_labels" in batched_data:
            all_saliency_scores.append(outputs["saliency_scores"].cpu())
            all_saliency_labels.append(batched_data["saliency_all_labels"].numpy())
            all_vid_masks.append(outputs["video_mask"].cpu())

    metrics = {}
    for thr in iou_thresholds:
        metrics[f"R1@{thr}"]  = compute_r1(all_preds, all_gt, iou_thresh=thr)
        metrics[f"mAP@{thr}"] = compute_map(all_preds, all_gt, iou_thresh=thr)

    if eval_span_source_metrics:
        total_samples = len(all_gt)
        for source_name, source_preds in span_source_preds.items():
            if span_source_counts[source_name] != total_samples:
                continue
            for thr in iou_thresholds:
                metrics[f"{source_name}_R1@{thr}"] = compute_r1(
                    source_preds, all_gt, iou_thresh=thr)
                metrics[f"{source_name}_mAP@{thr}"] = compute_map(
                    source_preds, all_gt, iou_thresh=thr)

    if all_saliency_scores:
        max_l = max(s.shape[1] for s in all_saliency_scores)

        def _pad_t(t, pad_value=0.0):
            if t.shape[1] == max_l:
                return t
            pad = t.new_full((t.shape[0], max_l - t.shape[1]), pad_value)
            return torch.cat([t, pad], dim=1)

        def _pad_np(a, pad_value=0.0):
            if a.shape[1] == max_l:
                return a
            pad = np.full((a.shape[0], max_l - a.shape[1]), pad_value, dtype=a.dtype)
            return np.concatenate([a, pad], axis=1)

        sal_scores = torch.cat([_pad_t(s, pad_value=0.0) for s in all_saliency_scores], dim=0)
        sal_labels = np.concatenate([_pad_np(a, pad_value=0.0) for a in all_saliency_labels], axis=0)
        sal_masks  = torch.cat([_pad_t(m, pad_value=0.0) for m in all_vid_masks], dim=0)
        metrics["hl_mAP"] = compute_highlight_map(sal_scores, sal_labels, sal_masks)
        metrics["HIT@1"]  = compute_hit1(sal_scores, sal_labels, sal_masks)

    # Convenience scalar: primary metric for early stopping.
    # Average of R1@0.5 and R1@0.7 so checkpointing improves on both thresholds,
    # not just the easier one.
    r1_05  = metrics.get("R1@0.5",  metrics.get(f"R1@{iou_thresholds[0]}",  0.0))
    r1_07  = metrics.get("R1@0.7",  metrics.get(f"R1@{iou_thresholds[-1]}", 0.0))
    metrics["primary"] = 0.5 * (r1_05 + r1_07)

    if all_gate_means:
        metrics["refine_gate_mean"] = sum(all_gate_means) / len(all_gate_means)
        if eval_refine_diagnostics and all_gate_values:
            gate_values = np.concatenate(all_gate_values, axis=0)
            metrics["refine_gate_p25"] = float(np.percentile(gate_values, 25))
            metrics["refine_gate_p50"] = float(np.percentile(gate_values, 50))
            metrics["refine_gate_p75"] = float(np.percentile(gate_values, 75))
            metrics["refine_gate_p90"] = float(np.percentile(gate_values, 90))

    if eval_refine_diagnostics:
        for pair_key, deltas in span_delta_acc.items():
            if deltas["start"]:
                metrics[f"{pair_key}_start_abs_delta_mean_sec"] = float(
                    np.concatenate(deltas["start"], axis=0).mean())
            if deltas["end"]:
                metrics[f"{pair_key}_end_abs_delta_mean_sec"] = float(
                    np.concatenate(deltas["end"], axis=0).mean())

    raw_model = model.module if hasattr(model, "module") else model
    vid_proj = getattr(raw_model, "input_vid_proj", None)
    if hasattr(vid_proj, "stream_logits"):
        alpha = torch.softmax(vid_proj.stream_logits, dim=0).tolist()
        metrics["stream_weights"] = [round(a, 3) for a in alpha]
        last_alpha = getattr(vid_proj, "last_stream_weights", None)
        if last_alpha is not None:
            metrics["last_stream_weights"] = [
                round(float(a), 3) for a in last_alpha.detach().cpu().tolist()
            ]
        last_blend = getattr(vid_proj, "last_hybrid_blend_mean", None)
        if last_blend is not None:
            metrics["hybrid_blend_mean"] = round(float(last_blend.detach().cpu().item()), 4)

    return metrics
