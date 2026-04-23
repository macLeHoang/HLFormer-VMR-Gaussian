"""
Configuration for GaussianFormer-VMR on Charades-STA dataset.

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

cfg["model_name"]    = "GaussianFormer_VMR_v3"
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
cfg["v_feat_len_mode"] = "min"

# ---- Model architecture --------------------------------------------------
cfg["hidden_size"]    = 256   # v30: restored to 384; v28=320 hit capacity ceiling
cfg["n_heads"]        = 4     # 384/8=48 per head
cfg["num_queries"]    = 8     # remediation: 5→8; more queries to prevent gradient starvation in aux decoders
cfg["dec_layers"]     = 3
cfg["input_drop"]     = 0.25
cfg["drop"]           = 0.25
cfg["txt_drop_ratio"] = 0.1
cfg["t2v_layers"]     = 3  # deepened 3->4 to unblock cross-modal fusion
cfg["attention_num"]  = 5
cfg["pos_enc_type"]   = "trainable"
cfg["sft_factor"]     = 0.3   # v11=0.06; stronger query-conditioned feature shift
cfg["gauss_bias_mode"] = "add_log"        # additive log-space Gaussian bias
cfg["weight_token_mode"] = "hybrid"       # global | mean | hybrid
cfg["weight_token_hybrid_init"] = 0.5      # gate on global token at init
cfg["txt_enc_layers"]  = 1                 # scaled from 2 → 1 for fresh run

# v11=False; enabling text-in-memory makes encoder query-conditioned (not just decoder)
cfg["use_txt_in_memory"]     = True
cfg["use_global_in_encoder"] = True

cfg["use_refined_spans"] = False

# ---- v37 alignment flags ---------------------------------------------------
# Match and quality-target supervision should follow the same span space used
# at evaluation time (refined when available), with safe fallback to coarse.
cfg["match_span_source"]   = "coarse"   # remediation: "refined"→"coarse"; prevents matcher churn when refiner is cold early
cfg["refined_cost_weight"] = 0.0         # ignored under "coarse"; set to 0 for clarity
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
cfg["temperature"]                 = 0.07  # sharpened from 0.15; faster NCE convergence with real cross-sample negatives (B=32)

# ---- Saliency ------------------------------------------------------------
cfg["lw_saliency"]     = 0.05  # saliency doesn't contribute to R1, redirect budget to span losses.
cfg["saliency_margin"] = 1.0
cfg["sal_prior_scale"] = 0.3   # remediation: 1.0→0.3; reduce saliency prior dominance early in training

# ---- Loss weights --------------------------------------------------------
cfg["span_loss_coef"]     = 10.0  # v17→v18: boost L1 span signal (dominant early training)
cfg["giou_loss_coef"]     = 6.0   # v33: 4.0→6.0; boundary precision is the primary bottleneck for R1@0.7
cfg["boundary_loss_coef"] = 0.0   # disabled: smooth-L1 on (start,end) is redundant with
                                   # DIoU which supervises the same coordinates; the old
                                   # value of 16.0 was swamping the DIoU gradient 16:1.
                                   # Boundary supervision is kept in the refinement head.
cfg["label_loss_coef"]    = 2.5   # remediation: 3.5→2.5; reduce quality-head dominance to let span losses breathe
# cfg["label_smoothing"]    = 0.1  # v32: 0.1→0.2; softer targets reduce overconfident predictions
cfg["final_loss_coef_span"] = 0.0  # refiner disabled
cfg["final_loss_coef_giou"] = 0.0  # refiner disabled

# ---- Boundary refinement losses (BoundaryRefinementHead, v15) -----------
# Applied to pred_spans_refined (final layer only, same Hungarian indices).
# Refiner is auxiliary: constant low coef keeps it contributing without
# overwhelming the coarse decoder.  Head gradient unblocked by removing
# zero-init on joint_mlp final layer (vmr_model.py) and raising lr_refiner.
cfg["boundary_refine_coef"]           = 0.0   # refiner disabled for fresh run
cfg["boundary_refine_giou_coef"]      = 0.0   # refiner disabled for fresh run
cfg["boundary_refine_window"]         = 12     # v34: at max_v_l=77, sigma=16/154≈0.10 (10% of video); tight enough for precise boundary pooling
cfg["boundary_refine_learnable_sigma"] = True
cfg["boundary_refine_max_delta"]      = 0.15  # unused when refiner disabled
cfg["refine_num_passes"]              = 1      # single pass; no iterative update
cfg["alpha_iou_alpha"]                = 2.0   # static; no ramp

# aux_loss_scale starts low: early decoder layers produce near-zero IoU targets,
# creating a strong "everything is background" pull if aux losses are weighted heavily.
# Ramped via loss_schedule once coarse localization stabilises.
cfg["aux_loss_scale"]     = 0.2   # v31: was 0.1; stronger deep supervision from start

# ---- Hungarian matcher ---------------------------------------------------
# v31: set_cost_class and set_cost_giou raised from 1.0 to 2.0 from start
# to stabilize matching early (quality predictions are good enough by ep3-5).
cfg["set_cost_class"]  = 3.0   # v31: was 1.0; stabilize query-target assignment
cfg["set_cost_span"]   = 5.0   # v17→v18: match span_loss_coef ratio
cfg["set_cost_giou"]   = 4.0   # v31: was 1.0; IoU-aware matching from start

# ---- Post-processing -----------------------------------------------------
cfg["top_k"]      = 8  # remediation: 10→8; aligned to num_queries=8 to avoid empty candidate slots
cfg["nms_thresh"] = 0.45  # tighter NMS for Charades short moments

# ---- Optimizer -----------------------------------------------------------
cfg["lr"]            = 1.5e-4  # stronger gradient signal, more budget before cosine trough
cfg["lr_vid_enc"]    = 0.75e-4    # proportional to lr
cfg["lr_refiner"]    = 1.5e-4    # 3x base lr; pushes boundary_refine head out of dead basin
cfg["lr_txt_enc"]    = 0.75e-4    # learning rate for text encoder
cfg["lr_drop"]       = 30     # kept for backward compat; unused when cosine_T0 > 0
cfg["lr_gamma"]      = 0.5    # kept for backward compat
cfg["warmup_epochs"] = 5
cfg["cosine_T0"]     = 35
cfg["cosine_Tmult"]  = 1      # no second cycle in 50-epoch budget
cfg["cosine_eta_min_ratio"] = 0.01  # v32: NEW; min LR = 1% of base at cycle trough
cfg["wd"]            = 1e-4   # v32: 5e-6→1e-4; 20x increase to regularize weights (5e-6 is effectively zero)
cfg["grad_clip"]     = 0.3

# ---- EMA ----------------------------------------------------------------
cfg["use_ema"]       = True   # v32: NEW; exponential moving average smooths val fluctuations (+1-2 R1)
cfg["ema_decay"]     = 0.995
cfg["n_epoch"]       = 70
cfg["max_es_cnt"]    = 15     # tighter: prior run wasted 23+ no-improve epochs
cfg["batchsize"]     = 32

# ---- DataLoader ----------------------------------------------------------
cfg["num_workers"] = 2

# ---- Data augmentation (training only) -----------------------------------
cfg["temporal_crop_ratio"] = 0.25   # v32: randomly crop 20% of video, forces context-invariant localization
cfg["feat_mask_ratio"]     = 0.25  # v32: randomly mask 15% of clips, prevents reliance on specific frames
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
    (35, {
        "temporal_crop_ratio": 0.05,
        "feat_mask_ratio":     0.05,
        "gt_jitter_frames":    1,
    }),
]

cfg["iou_thresholds"] = [0.3, 0.5, 0.7]

# ---- Validation schedule -------------------------------------------------
cfg["val_freq"]       = 3     # validate every 3 epochs before val_full_epoch
cfg["val_full_epoch"] = 12   # dense validation from near prior best (ep15)

# ---- Loss schedule -------------------------------------------------------
# Each entry: (from_epoch, {cfg_key: new_value, ...})
# apply_loss_schedule() in main_vmr.py reads this and patches criterion.weight_dict
# and matcher costs at phase boundaries.
#
# Two phases only — no mid-run ramps; previous ramps caused consistent regressions.
# Phase 1 (ep  0-34): Coarse localization with low constant refine coef.
# Phase 2 (ep 35+  ): IoU-budget rebalance: giou up, span down, contrastive down.
#
# Static cfg values above must match Phase 1 so build_criterion() starts correctly.
cfg["loss_schedule"] = [
    (0,  {'span_loss_coef': 10.0, 'giou_loss_coef': 6.0,
          'boundary_refine_coef': 0.0, 'boundary_refine_giou_coef': 0.0,
          'contrastive_align_loss_coef': 0.08, 'aux_loss_scale': 0.2,
          'set_cost_class': 3.0, 'set_cost_span': 5.0, 'set_cost_giou': 4.0,
          'alpha_iou_alpha': 2.0}),
    (30, {'contrastive_align_loss_coef': 0.03, 'aux_loss_scale': 0.1}),
]


def get_cfg_defaults():
    os.makedirs(cfg["model_root"], exist_ok=True)
    with open(os.path.join(cfg["model_root"], "hyperparams.yaml"), "w") as f:
        yaml.dump(cfg, f)
    return cfg