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

cfg["seed"]          = 36

cfg["model_name"]    = "GaussianFormer_VMR_v9"
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
cfg["v_feat_len_mode"] = "max"  # remediation: "min"→"max"; use longest stream to avoid clipping valid frames

# ---- Model architecture --------------------------------------------------
cfg["hidden_size"]    = 384   # v30: restored to 384; v28=320 hit capacity ceiling
cfg["n_heads"]        = 8     # 320/8=40 per head; fixes odd-head warning
cfg["num_queries"]    = 8     # remediation: 5→8; more queries to prevent gradient starvation in aux decoders
cfg["dec_layers"]     = 3
cfg["input_drop"]     = 0.3
cfg["drop"]           = 0.2
cfg["txt_drop_ratio"] = 0.1
cfg["t2v_layers"]     = 3  
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

cfg["use_refined_spans"] = True  # boundary refiner now active; refined spans used at eval time

# ---- v37 alignment flags ---------------------------------------------------
# Match and quality-target supervision should follow the same span space used
# at evaluation time (refined when available), with safe fallback to coarse.
cfg["match_span_source"]   = "coarse"   # remediation: "refined"→"coarse"; prevents matcher churn when refiner is cold early
cfg["refined_cost_weight"] = 0.0         # ignored under "coarse"; set to 0 for clarity
cfg["label_span_source"]   = "final"   # coarse | refined | matched | final

# ---- Contrastive alignment -----------------------------------------------
# temperature=0.07: MoCo-style sharp negatives; v11=0.3 was too smooth → loss never converged
# contrastive_align_loss_coef=0.01: reduce contrastive dominance over span/boundary losses
cfg["contrastive_hdim"]            = 64
cfg["contrastive_align_loss_coef"] = 0.03  # remediation: 0.15→0.03; span head was starved by dominant contrastive early
# temperature for cross-sample NCE (v26 fix):
# The previous 0.1 was calibrated for the broken fg-only logsumexp (which was constant anyway).
# With real cross-sample NCE (B=32 negatives), 0.1 is too sharp — loss collapses to 0.26 by
# epoch 13, meaning the model memorises batch-level text identity rather than learning
# span-text alignment.  0.3 gives softer gradients; loss should stabilise around 2.0–2.5
# (log(32)≈3.47 at random, ~1.5–2.0 at good discrimination without memorisation).
cfg["temperature"]                 = 0.1   # remediation: 0.07→0.1; soften NCE gradients to prevent contrastive memorisation

# ---- Saliency ------------------------------------------------------------
cfg["lw_saliency"]     = 0.1  # saliency doesn't contribute to R1, redirect budget to span losses.
cfg["saliency_margin"] = 1.0
cfg["sal_prior_scale"] = 0.3   # remediation: 1.0→0.3; reduce saliency prior dominance early in training

# ---- Loss weights --------------------------------------------------------
cfg["span_loss_coef"]     = 10.0  # v17→v18: boost L1 span signal (dominant early training)
cfg["giou_loss_coef"]     = 6.0   # v33: 4.0→6.0; boundary precision is the primary bottleneck for R1@0.7
cfg["boundary_loss_coef"] = 0.0   # disabled: smooth-L1 on (start,end) is redundant with
                                   # DIoU which supervises the same coordinates; the old
                                   # value of 16.0 was swamping the DIoU gradient 16:1.
                                   # Boundary supervision is kept in the refinement head.
cfg["label_loss_coef"]    = 1.5   # remediation: 2.5→1.5; further reduce quality-head dominance to let span losses breathe
# cfg["label_smoothing"]    = 0.1  # v32: 0.1→0.2; softer targets reduce overconfident predictions
cfg["final_loss_coef_span"] = 1.0  # v7: rollback 2.0→1.0
cfg["final_loss_coef_giou"] = 0.75  # v7: rollback 1.5→0.75

# ---- Boundary refinement losses (BoundaryRefinementHead, v15) -----------
# Applied to pred_spans_refined (final layer only, same Hungarian indices).
# Refiner is auxiliary: constant low coef keeps it contributing without
# overwhelming the coarse decoder.  Head gradient unblocked by removing
# zero-init on joint_mlp final layer (vmr_model.py) and raising lr_refiner.
cfg["boundary_refine_coef"]           = 1.0   # v7: rollback 1.5→1.0
cfg["boundary_refine_giou_coef"]      = 0.5   # v7: rollback 1.0→0.5
cfg["boundary_refine_window"]         = 12     # v7: rollback 10→12
cfg["boundary_refine_learnable_sigma"] = True
cfg["boundary_refine_max_delta"]      = 0.07  # v7: rollback 0.10→0.07
cfg["iou_floor"] = 0.2   # v6: 0.0→0.2; matched slots always have target ≥ 0.2, fixes label-head cold start
cfg["refine_num_passes"]              = 2      # v7: rollback 3→2
cfg["alpha_iou_alpha"]                = 2.0   # static; no ramp

# aux_loss_scale starts low: early decoder layers produce near-zero IoU targets,
# creating a strong "everything is background" pull if aux losses are weighted heavily.
# Ramped via loss_schedule once coarse localization stabilises.
cfg["aux_loss_scale"]     = 0.2   # v6: matches Phase 0 of loss_schedule

# ---- Hungarian matcher ---------------------------------------------------
# v6: set_cost_* frozen at Phase 0 static values; no three-way ramp.
# Phase 1 (ep5) raises costs to (2,4,3); no Phase 2 jump (removes matcher churn).
cfg["set_cost_class"]  = 1.0   # v6: Phase 0 static; matches loss_schedule Phase 0
cfg["set_cost_span"]   = 3.0   # v6: Phase 0 static; matches loss_schedule Phase 0
cfg["set_cost_giou"]   = 2.0   # v6: Phase 0 static; matches loss_schedule Phase 0

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
cfg["warmup_epochs"] = 3      # v6: 5→3
cfg["cosine_T0"]     = 50     # v6: 35→50; single full cycle over the 50-epoch budget
cfg["cosine_Tmult"]  = 1      # no second cycle in 50-epoch budget
cfg["cosine_eta_min_ratio"] = 0.005  # v6: 0.01→0.005; deeper trough for final fine-tune
cfg["wd"]            = 5e-5   # remediation: 1e-4→5e-5; slight relaxation to avoid over-regularising refiner head
cfg["grad_clip"]     = 0.3

# ---- EMA ----------------------------------------------------------------
cfg["use_ema"]       = True   # v32: NEW; exponential moving average smooths val fluctuations (+1-2 R1)
cfg["ema_decay"]     = 0.999
cfg["n_epoch"]       = 50     # v6: 70→50
cfg["max_es_cnt"]    = 12     # v6: 25→12; stop 12 epochs after peak
cfg["batchsize"]     = 32

# ---- DataLoader ----------------------------------------------------------
cfg["num_workers"] = 2

# ---- Data augmentation (training only) -----------------------------------
cfg["temporal_crop_ratio"] = 0.20   # v6: kept strong (was 0.25); matches aug_schedule Phase 0
cfg["feat_mask_ratio"]     = 0.15   # v6: kept strong (was 0.25); matches aug_schedule Phase 0
cfg["gt_jitter_frames"]    = 2      # v6: unchanged; matches aug_schedule Phase 0
cfg["feat_noise_std"]      = 0.01   # v6: 0.0→0.01; gentle projection-layer regularisation

# Epoch-wise augmentation schedule (train only).
# Each entry updates train_dset augmentation strengths from `from_epoch` onward.
# v6: three phases — stronger augmentation kept until ep35 to prevent ep35 overfitting cliff.
cfg["aug_schedule"] = [
    (0,  {"temporal_crop_ratio": 0.20, "feat_mask_ratio": 0.15, "gt_jitter_frames": 2}),
    (20, {"temporal_crop_ratio": 0.15, "feat_mask_ratio": 0.15, "gt_jitter_frames": 2}),
]

cfg["iou_thresholds"] = [0.3, 0.5, 0.7]

# ---- Validation schedule -------------------------------------------------
cfg["val_freq"]       = 3     # validate every 3 epochs before val_full_epoch
cfg["val_full_epoch"] = 10   # v6: 12→10; dense validation earlier in a 50-epoch budget

# ---- Loss schedule -------------------------------------------------------
# Each entry: (from_epoch, {cfg_key: new_value, ...})
# apply_loss_schedule() in main_vmr.py reads this and patches criterion.weight_dict
# and matcher costs at phase boundaries.
#
# Three phases: cold start → matcher stabilises → late IoU fine-tune.
# ep15 matcher-cost jump REMOVED (was (3,5,4)) — no matcher churn.
# ep35 contrastive cooldown REMOVED — contrastive stays active throughout.
# Static cfg values above must match Phase 0 so build_criterion() starts correctly.
cfg["loss_schedule"] = [
    # Phase 0 (ep 0-4): cold start — refiner hot from start
    (0,  {'span_loss_coef': 10.0, 'giou_loss_coef': 6.0,
          'boundary_refine_coef': 1.0, 'boundary_refine_giou_coef': 0.5,
          'final_loss_coef_span': 1.0, 'final_loss_coef_giou': 0.75,
          'contrastive_align_loss_coef': 0.03, 'aux_loss_scale': 0.2,
          'set_cost_class': 1.0, 'set_cost_span': 3.0, 'set_cost_giou': 2.0,
          'alpha_iou_alpha': 2.0, 'iou_floor': 0.2}),
    # Phase 1 (ep 5-11): matcher stabilises
    (5,  {'set_cost_class': 2.0, 'set_cost_span': 4.0, 'set_cost_giou': 3.0,
          'aux_loss_scale': 0.4}),
    # Phase 2 (ep 12+): reduce quality-target floor once coarse localisation is stable
    (12, {'iou_floor': 0.05}),
]


def get_cfg_defaults():
    os.makedirs(cfg["model_root"], exist_ok=True)
    with open(os.path.join(cfg["model_root"], "hyperparams.yaml"), "w") as f:
        yaml.dump(cfg, f)
    return cfg