"""Configuration for GaussianFormer-VMR on Charades-STA.

Feature layout:
  Video: SlowFast (2304) + CLIP (512) + BLIP (768) -> <vid>.npz["features"]
  Text:  CLIP (512) + BLIP (768)                   -> <qid>.npz["last_hidden_state"]
"""

import os
import yaml

from VMR.Configs import qvhighlights as _qvh

cfg = dict(_qvh.cfg)

cfg["seed"]         = 36

cfg["model_name"]   = "GaussianFormer_VMR_v22"
cfg["dataset_name"] = "charades_sta"
cfg["dset_name"]    = "charades_sta"

cfg["data_root"]   = ""
cfg["v_feat_dirs"] = [
    "/content/charades/slowfast_features",
    "/content/charades/clip_features",
    "/content/charades/blip_video_features",
]
cfg["q_feat_dir"] = [
    "/content/charades/clip_text_features",
    "/content/charades/blip_text_features",
]

cfg["train_path"] = "/content/drive/MyDrive/Master/Thesis/QD-DETR/data/charades-sta/train.jsonl"
cfg["val_path"]   = "/content/drive/MyDrive/Master/Thesis/QD-DETR/data/charades-sta/test.jsonl"
cfg["test_path"]  = None

cfg["model_root"] = os.path.join(cfg["root"], cfg["dataset_name"], cfg["model_name"])

# Per-stream dimensions drive the multi-stream projection layers.
# The summed totals are kept for logging and legacy code paths.
cfg["v_feat_dims"]                     = [2304, 512, 768]  # SlowFast | CLIP | BLIP; order must match v_feat_dirs.
cfg["v_feat_dim"]                      = sum(cfg["v_feat_dims"])
cfg["use_multistream_projection"]      = True
cfg["t_feat_dims"]                     = [512, 768]  # CLIP | BLIP; order must match q_feat_dir.
cfg["t_feat_dim"]                      = sum(cfg["t_feat_dims"])
cfg["use_multistream_text_projection"] = True
cfg["n_input_proj"]                    = 2

cfg["max_v_l"]                         = 75
cfg["clip_len"]                        = 1.0
cfg["use_tef"]                         = True
cfg["v_feat_len_mode"]                 = "max"  # Options: max | min. Keep the longest stream so valid frames are not clipped.

# ---- Model architecture --------------------------------------------------
cfg["hidden_size"]               = 384
cfg["n_heads"]                   = 8
cfg["num_queries"]               = 6            # Stage-3 ablation: try 6 for lower ranking noise on single-moment data.
cfg["dec_layers"]                = 3
cfg["input_drop"]                = 0.25
cfg["drop"]                      = 0.15
cfg["txt_drop_ratio"]            = 0.0
cfg["t2v_layers"]                = 3
cfg["attention_num"]             = 6
cfg["pos_enc_type"]              = "trainable"  # Options: trainable | sinusoidal.
cfg["sft_factor"]                = 0.3
cfg["gauss_bias_mode"]           = "add_log"    # Options: add_log | other fallback. Apply Gaussian bias additively in log space.
cfg["weight_token_mode"]         = "hybrid"     # Options: global | mean | hybrid. Mix global-token and mean-token weighting.
cfg["weight_token_hybrid_init"]  = 0.5          # Start the hybrid gate at an even split.
cfg["txt_enc_layers"]            = 2

# Make the encoder text-conditioned instead of relying on the decoder alone.
cfg["use_txt_in_memory"]         = True
cfg["use_mem_kv_for_txt_memory"] = True
cfg["use_global_in_encoder"]     = True
cfg["use_boundary_refinement"]   = True
cfg["refine_gate_max"]           = 0.85

cfg["use_refined_spans"]         = True  # Prefer refined spans at evaluation when available.

# Matcher supervision, label supervision, and inference span selection are related but separate:
# - match_span_source selects the spans used in matching cost.
# - label_span_source selects the spans used for soft IoU targets.
# - use_refined_spans controls validation-time span preference.
cfg["match_span_source"]         = "coarse"  # Options: coarse | refined | dual. Match on coarse spans to keep early training stable.
cfg["refined_cost_weight"]       = 0.0       # Weight for refined-span cost when match_span_source="dual".
cfg["label_span_source"]         = "refined"   # Options: coarse | refined | matched | final.

# ---- Contrastive alignment -----------------------------------------------
cfg["contrastive_hdim"]            = 64
cfg["contrastive_align_loss_coef"] = 0.03  # Keep contrastive loss from overwhelming span learning.
cfg["temperature"]                 = 0.1   # Softer NCE temperature avoids over-sharp batch discrimination.

# ---- Saliency ------------------------------------------------------------
cfg["lw_saliency"]     = 0.1  # Keep saliency lightweight; retrieval quality depends more on span accuracy.
cfg["saliency_margin"] = 1.0
cfg["sal_prior_scale"] = 0.1  # Stage-3 ablation: set 0.0 to remove saliency-biased decoder attention.

# ---- Loss weights --------------------------------------------------------
cfg["span_loss_coef"]       = 10.0
cfg["giou_loss_coef"]       = 6.0
cfg["boundary_loss_coef"]   = 1.0   # Keep boundary loss light; the refiner already supervises boundaries directly.
cfg["label_loss_coef"]      = 1.5
cfg["final_loss_coef_span"] = 2.0
cfg["final_loss_coef_giou"] = 2.0

# Boundary refinement is auxiliary and should help without overpowering the coarse decoder.
cfg["boundary_refine_coef"]            = 1.0
cfg["boundary_refine_giou_coef"]       = 0.5
cfg["boundary_refine_window"]          = 8
cfg["boundary_refine_learnable_sigma"] = True
cfg["boundary_refine_max_delta"]       = 0.06
cfg["iou_floor"]                       = 0.2   # Prevent matched slots from starting with near-zero quality targets.
cfg["refine_num_passes"]               = 1
cfg["alpha_iou_alpha"]                 = 2.5

# Start auxiliary loss low, then increase it once coarse localization is stable.
cfg["aux_loss_scale"]                  = 0.2

# ---- Hungarian matcher ---------------------------------------------------
# These are the phase-0 matcher costs; later phases are applied through loss_schedule.
cfg["set_cost_class"] = 1.0
cfg["set_cost_span"]  = 3.0
cfg["set_cost_giou"]  = 2.0

# ---- Post-processing -----------------------------------------------------
cfg["top_k"]      = 6     # Keep one candidate per query.
cfg["nms_thresh"] = 0.45  # Slightly tighter NMS suits short Charades moments.

# ---- Optimizer -----------------------------------------------------------
cfg["lr"]                   = 1.25e-4
cfg["lr_vid_enc"]           = 0.625e-4
cfg["lr_refiner"]           = 1.25e-4      # Give the refinement head enough learning signal.
cfg["lr_txt_enc"]           = 0.625e-4
cfg["lr_drop"]              = 25        # Step decay after warmup at epoch 30; aligns with the observed validation peak.
cfg["lr_gamma"]             = 0.5       # Legacy field kept for compatibility.
cfg["warmup_epochs"]        = 5
cfg["cosine_T0"]            = 0
cfg["cosine_Tmult"]         = 2
cfg["cosine_eta_min_ratio"] = 0.001
cfg["wd"]                   = 5e-5
cfg["grad_clip"]            = 0.3

# ---- EMA ----------------------------------------------------------------
cfg["use_ema"]    = True   # EMA smooths validation noise and usually gives steadier checkpoints.
cfg["ema_decay"]  = 0.999
cfg["n_epoch"]    = 100
cfg["max_es_cnt"] = 10     # Give the fresh diagnostic run more room after the earlier LR drop.
cfg["batchsize"]  = 32

# ---- DataLoader ----------------------------------------------------------
cfg["num_workers"] = 2

# ---- Data augmentation (training only) -----------------------------------
cfg["temporal_crop_ratio"] = 0.20
cfg["feat_mask_ratio"]     = 0.15
cfg["gt_jitter_frames"]    = 2
cfg["feat_noise_std"]      = 0.00

# Update training augmentation strength at the listed epochs.
cfg["aug_schedule"] = [
    (0,  {"temporal_crop_ratio": 0.2, "feat_mask_ratio": 0.15, "gt_jitter_frames": 2}),
    (20, {"temporal_crop_ratio": 0.2, "feat_mask_ratio": 0.10, "gt_jitter_frames": 1}),
    (30, {"temporal_crop_ratio": 0.1, "feat_mask_ratio": 0.05, "gt_jitter_frames": 1}),
]

cfg["iou_thresholds"] = [0.3, 0.5, 0.7]

# ---- Validation schedule -------------------------------------------------
cfg["val_freq"]                 = 3
cfg["val_full_epoch"]           = 20         # Run full validation from this epoch onward.
cfg["eval_span_source_metrics"] = True       # Report coarse/refined/final validation metrics without changing primary.
cfg["eval_refine_diagnostics"]  = True      # Report gate percentiles and boundary-delta diagnostics.

# ---- Loss schedule -------------------------------------------------------
# Each entry is (from_epoch, {cfg_key: new_value, ...}).
# main_vmr.py applies these updates to criterion weights and matcher costs.
cfg["loss_schedule"] = [
    # Phase 0 (ep 0-4): cold start — moderate refinement while coarse spans stabilise
    (0,  {
        'span_loss_coef':               10.0,
        'giou_loss_coef':               6.0,
        'boundary_refine_coef':         0.5,
        'boundary_refine_giou_coef':    0.25,
        'final_loss_coef_span':         1.0,
        'final_loss_coef_giou':         1.0,
        'contrastive_align_loss_coef':  0.03,
        'aux_loss_scale':               0.2,
        'set_cost_class':               1.0,
        'set_cost_span':                3.0,
        'set_cost_giou':                2.0,
        'alpha_iou_alpha':              2.0,
        'iou_floor':                    0.2,
    }),
    # Phase 1 (ep 5-11): matcher stabilises
    (5,  {
        'boundary_refine_coef':         1.0,
        'boundary_refine_giou_coef':    0.5,
        'final_loss_coef_span':         2.0,
        'final_loss_coef_giou':         2.0,
        'set_cost_class':               2.0,
        'set_cost_span':                4.0,
        'set_cost_giou':                3.0,
        'alpha_iou_alpha':              2.5,
        "aux_loss_scale":               0.3,
    }),
    # Phase 2 (ep 12-23): reduce quality-target floor once coarse localisation is stable
    (12, {
        'iou_floor':                    0.05,
    }),
]


def get_cfg_defaults():
    os.makedirs(cfg["model_root"], exist_ok=True)
    with open(os.path.join(cfg["model_root"], "hyperparams.yaml"), "w") as f:
        yaml.dump(cfg, f)
    return cfg
