"""
Configuration for HLFormer-VMR on Charades-STA dataset.

Charades-STA features:
  Video: SlowFast (2304) + CLIP (512) + BLIP (768)  -> <vid>.npz  (key: "features")
  Text:  CLIP (512) + BLIP (768)                    -> <qid>.npz  (key: "last_hidden_state")
"""

import os
import yaml
# Import the raw dict without calling get_cfg_defaults() to avoid side effects
from VMR.Configs import qvhighlights as _qvh

cfg = dict(_qvh.cfg)   # inherit defaults, then override

cfg["seed"]          = 2026

cfg["model_name"]    = "GaussianFormer_VMR_v1"
cfg["dataset_name"]  = "charades_sta"
cfg["dset_name"]     = "charades_sta"

cfg["data_root"]   = ""
cfg["v_feat_dirs"] = [
    "/content/charades/slowfast_features",
    "/content/charades/clip_features",
    "/content/charades/blip_video_features",
]
cfg["q_feat_dir"]  = [
    "/content/charades/clip_text_features",
    "/content/charades/blip_text_features",
]

cfg["train_path"]  = "/content/drive/MyDrive/Master/Thesis/QD-DETR/data/charades-sta/train.jsonl"
cfg["val_path"]    = "/content/drive/MyDrive/Master/Thesis/QD-DETR/data/charades-sta/test.jsonl"
cfg["test_path"]   = None

cfg["model_root"]  = os.path.join(cfg["root"], cfg["dataset_name"], cfg["model_name"])

# Per-stream dims drive MultiStreamVidProjection; v_feat_dim/t_feat_dim are
# kept as totals for logging and any legacy code that reads them.
cfg["v_feat_dims"] = [2304, 512, 768]            # SlowFast | BLIP | CLIP  (order must match v_feat_dirs)
cfg["v_feat_dim"]  = sum(cfg["v_feat_dims"])     # 3584
cfg["t_feat_dims"] = [512, 768]                  # BLIP text | CLIP text   (order must match q_feat_dir)
cfg["t_feat_dim"]  = sum(cfg["t_feat_dims"])     # 1280

cfg["max_v_l"]     = 75
cfg["clip_len"]    = 1.0
cfg["use_tef"]     = True
cfg["v_feat_len_mode"] = "max" 

# ---- Model architecture --------------------------------------------------
cfg["hidden_size"]    = 384   # v30: restored to 384; v28=320 hit capacity ceiling
cfg["n_heads"]        = 6     # 384/8=48 per head
cfg["num_queries"]    = 10    # v37: increase proposal capacity for better candidate coverage
cfg["dec_layers"]     = 3
cfg["input_drop"]     = 0.25
cfg["drop"]           = 0.15
cfg["txt_drop_ratio"] = 0.1
cfg["t2v_layers"]     = 3
cfg["attention_num"]  = 6
cfg["pos_enc_type"]   = "trainable"
cfg["sft_factor"]     = 0.1   # v11=0.06; stronger query-conditioned feature shift
cfg["weight_token_mode"] = "hybrid"       # global | mean | hybrid
cfg["weight_token_hybrid_init"] = 0.7      # gate on global token at init

# v11=False; enabling text-in-memory makes encoder query-conditioned (not just decoder)
cfg["use_txt_in_memory"]     = True
cfg["use_global_in_encoder"] = True

cfg["use_refined_spans"] = True

# ---- v37 alignment flags ---------------------------------------------------
# Match and quality-target supervision should follow the same span space used
# at evaluation time (refined when available), with safe fallback to coarse.
cfg["match_span_source"]   = "refined"   # coarse | refined | dual
cfg["refined_cost_weight"] = 0.5         # used only when match_span_source="dual"
cfg["label_span_source"]   = "matched"   # coarse | refined | matched

# ---- Contrastive alignment -----------------------------------------------
# temperature=0.07: MoCo-style sharp negatives; v11=0.3 was too smooth → loss never converged
# contrastive_align_loss_coef=0.01: reduce contrastive dominance over span/boundary losses
cfg["contrastive_hdim"]            = 64
cfg["contrastive_align_loss_coef"] = 0.15  # v33: 0.05→0.15; cross-modal alignment is critical for boundary quality
# temperature for cross-sample NCE (v26 fix):
# The previous 0.1 was calibrated for the broken fg-only logsumexp (which was constant anyway).
# With real cross-sample NCE (B=32 negatives), 0.1 is too sharp — loss collapses to 0.26 by
# epoch 13, meaning the model memorises batch-level text identity rather than learning
# span-text alignment.  0.3 gives softer gradients; loss should stabilise around 2.0–2.5
# (log(32)≈3.47 at random, ~1.5–2.0 at good discrimination without memorisation).
cfg["temperature"]                 = 0.15  # v31: was 0.3; sharper contrastive discrimination

# ---- Saliency ------------------------------------------------------------
cfg["lw_saliency"]     = 0.2   # v34: 0.2→0.05; saliency doesn't contribute to R1, redirect budget to span losses.
cfg["saliency_margin"] = 1.0

# ---- Loss weights --------------------------------------------------------
cfg["span_loss_coef"]     = 10.0  # v17→v18: boost L1 span signal (dominant early training)
cfg["giou_loss_coef"]     = 6.0   # v33: 4.0→6.0; boundary precision is the primary bottleneck for R1@0.7
cfg["boundary_loss_coef"] = 0.0   # disabled: smooth-L1 on (start,end) is redundant with
                                   # DIoU which supervises the same coordinates; the old
                                   # value of 16.0 was swamping the DIoU gradient 16:1.
                                   # Boundary supervision is kept in the refinement head.
cfg["label_loss_coef"]    = 2.0   # v33: 4.0→2.0; quality head converges fast, reduce dominance over span losses
# cfg["label_smoothing"]    = 0.1  # v32: 0.1→0.2; softer targets reduce overconfident predictions

# ---- Boundary refinement losses (BoundaryRefinementHead, v15) -----------
# Applied to pred_spans_refined (final layer only, same Hungarian indices).
# v31: enabled from epoch 0 — the BoundaryRefineHead is zero-initialized so it
# starts as identity and gradually activates.  With giou_coef=4.0 the model gets
# direct boundary supervision from the beginning.
cfg["boundary_refine_coef"]           = 1.0   # phase-0 default; schedule ramps refinement gradually.
cfg["boundary_refine_giou_coef"]      = 1.0
cfg["boundary_refine_window"]         = 16     # v34: at max_v_l=77, sigma=16/154≈0.10 (10% of video); tight enough for precise boundary pooling
cfg["boundary_refine_learnable_sigma"] = True
cfg["boundary_refine_max_delta"]      = 0.2
cfg["alpha_iou_alpha"]                = 2.0   # v34: 2.5→3.0; stronger gradient amplification in 0.5-0.7 IoU zone

# aux_loss_scale starts low: early decoder layers produce near-zero IoU targets,
# creating a strong "everything is background" pull if aux losses are weighted heavily.
# Ramped via loss_schedule once coarse localization stabilises.
cfg["aux_loss_scale"]     = 0.2   # v31: was 0.1; stronger deep supervision from start

# ---- Hungarian matcher ---------------------------------------------------
# v31: set_cost_class and set_cost_giou raised from 1.0 to 2.0 from start
# to stabilize matching early (quality predictions are good enough by ep3-5).
cfg["set_cost_class"]  = 2.0   # v31: was 1.0; stabilize query-target assignment
cfg["set_cost_span"]   = 10.0  # v17→v18: match span_loss_coef ratio
cfg["set_cost_giou"]   = 2.0   # v31: was 1.0; IoU-aware matching from start

# ---- Post-processing -----------------------------------------------------
cfg["top_k"]      = 10  # keep all query candidates before NMS (matches num_queries)
cfg["nms_thresh"] = 0.6  # looser NMS to preserve near-duplicate high-IoU candidates for top-1 selection

# ---- Optimizer -----------------------------------------------------------
cfg["lr"]            = 1.5e-4  # v34: 1e-4→2e-4; stronger gradient signal, more budget before cosine trough
cfg["lr_vid_enc"]    = 0.75e-4   # v34: 5e-5→1e-4; proportional to lr increase
cfg["lr_drop"]       = 30     # kept for backward compat; unused when cosine_T0 > 0
cfg["lr_gamma"]      = 0.5    # kept for backward compat
cfg["warmup_epochs"] = 5
cfg["cosine_T0"]     = 35
cfg["cosine_Tmult"]  = 2      # v32: NEW; second cycle = 60 epochs (total ~90 before 2nd restart)
cfg["cosine_eta_min_ratio"] = 0.01  # v32: NEW; min LR = 1% of base at cycle trough
cfg["wd"]            = 5e-5   # v32: 5e-6→1e-4; 20x increase to regularize weights (5e-6 is effectively zero)
cfg["grad_clip"]     = 0.3

# ---- EMA ----------------------------------------------------------------
cfg["use_ema"]       = True   # v32: NEW; exponential moving average smooths val fluctuations (+1-2 R1)
cfg["ema_decay"]     = 0.999
cfg["n_epoch"]       = 100
cfg["max_es_cnt"]    = 20     # v17→v18: give more room after LR drop
cfg["batchsize"]     = 32

# ---- DataLoader ----------------------------------------------------------
cfg["num_workers"] = 2

# ---- Data augmentation (training only) -----------------------------------
cfg["temporal_crop_ratio"] = 0.2   # v32: randomly crop 20% of video, forces context-invariant localization
cfg["feat_mask_ratio"]     = 0.15  # v32: randomly mask 15% of clips, prevents reliance on specific frames
cfg["gt_jitter_frames"]    = 2  # v32: jitter GT boundaries ±2 frames, smooths supervision signal
cfg["feat_noise_std"]      = 0.0  # v32: Gaussian noise σ added to projected features during training

# Epoch-wise augmentation schedule (train only).
# Each entry updates train_dset augmentation strengths from `from_epoch` onward.
cfg["aug_schedule"] = [
    (0, {
        "temporal_crop_ratio": 0.2,
        "feat_mask_ratio":     0.15,
        "gt_jitter_frames":    2,
    }),
    (45, {
        "temporal_crop_ratio": 0.05,
        "feat_mask_ratio":     0.05,
        "gt_jitter_frames":    0,
    }),
]

cfg["iou_thresholds"] = [0.3, 0.5, 0.7]

# ---- Loss schedule -------------------------------------------------------
# Each entry: (from_epoch, {cfg_key: new_value, ...})
# apply_loss_schedule() in main_vmr.py reads this and patches criterion.weight_dict
# and matcher costs at phase boundaries.
#
# v34 schedule: boundary refinement starts at 4.0 from ep0 (zero-init guarantees safety),
#
# Phase 1  (ep  0-14): Coarse localization — boundary refine at 1.0 (head warms up),
#                       zero-init head activates gradually.
# Phase 2  (ep 15-19): Boundary refine jumps to 4.0 — refine head actively correcting.
# Phase 3  (ep 20-39): Boundary refine at 5.0 — refine head learns width+text conditioning.
# Phase 4  (ep 40-59): IoU dominant — span L1 reduces, giou+boundary amplified.  Cap at 6.0.
# Phase 5  (ep 60+  ): Cap at 6.0 — refinement must not dominate over decoder supervision.
#
# Static cfg values above must match Phase 1 so build_criterion() starts correctly.
cfg["loss_schedule"] = [
    (0, {
        "span_loss_coef":             10.0,
        "giou_loss_coef":              6.0,
        "boundary_refine_coef":        1.0,
        "boundary_refine_giou_coef":   1.0,
        "contrastive_align_loss_coef": 0.15,
        "aux_loss_scale":              0.2,
        "set_cost_class":              2.0,
        "set_cost_span":              10.0,
        "set_cost_giou":               2.0,
    }),
    (15, {
        "boundary_refine_coef":        3.0,
        "boundary_refine_giou_coef":   3.0,
    }),
    (20, {
        "boundary_refine_coef":        4.5,
        "boundary_refine_giou_coef":   4.5,
    }),
    (40, {
        "span_loss_coef":              9.5,
        "giou_loss_coef":              6.2,
        "boundary_refine_coef":        5.0,
        "boundary_refine_giou_coef":   5.0,
        "set_cost_class":              2.2,
        "set_cost_giou":               2.2,
    }),
    (50, {
        "span_loss_coef":              9.0,
        "giou_loss_coef":              6.5,
        "boundary_refine_coef":        5.4,
        "boundary_refine_giou_coef":   5.4,
        "contrastive_align_loss_coef": 0.1,
        "set_cost_class":              2.5,
        "set_cost_giou":               2.5,
    }),
    (55, {
        "boundary_refine_coef":        6.0,
        "boundary_refine_giou_coef":   6.0,
        "contrastive_align_loss_coef": 0.05,
        "set_cost_class":              3.0,
        "set_cost_giou":               3.0,
    }),
    (60, {
        "aux_loss_scale":              0.1,
    }),
    (65, {
        "span_loss_coef":              6.0,
        "giou_loss_coef":              8.0,
        "boundary_refine_coef":        6.0,
        "boundary_refine_giou_coef":   6.0,
        "set_cost_class":              3.5,
        "aux_loss_scale":              0.05,
    }),
]


def get_cfg_defaults():
    os.makedirs(cfg["model_root"], exist_ok=True)
    with open(os.path.join(cfg["model_root"], "hyperparams.yaml"), "w") as f:
        yaml.dump(cfg, f)
    return cfg