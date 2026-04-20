"""
Configuration for HLFormer-VMR on QVHighlights dataset.

QVHighlights data directory should contain:
  highlight_train_release.jsonl
  highlight_val_release.jsonl
  highlight_test_release.jsonl (optional, no labels)
  features/
    clip_features/  -> <vid>.npz  (key: "features", shape: (L, 512))
    clip_text_features/ -> qid<qid>.npz  (key: "last_hidden_state", shape: (L_q, 512))

Feature dimensions shown here correspond to CLIP (ViT-B/32).
Adjust v_feat_dim / t_feat_dim if using different backbone features.
"""

import os
import yaml

cfg = {}

# ---- Experiment identity ------------------------------------------------
cfg["model_name"]   = "HLFormer_VMR"
cfg["dataset_name"] = "qvhighlights"
cfg["dset_name"]    = "qvhighlights"
cfg["seed"]         = 2024

# ---- Paths  (EDIT THESE to match your local setup) ---------------------
cfg["root"]       = "./experiments"                     # output root
cfg["data_root"]  = "./data/qvhighlights"              # annotation root
cfg["v_feat_dirs"] = ["./data/qvhighlights/features/clip_features"]
cfg["q_feat_dir"]  = "./data/qvhighlights/features/clip_text_features"
cfg["train_path"]  = os.path.join(cfg["data_root"], "highlight_train_release.jsonl")
cfg["val_path"]    = os.path.join(cfg["data_root"], "highlight_val_release.jsonl")
cfg["test_path"]   = os.path.join(cfg["data_root"], "highlight_test_release.jsonl")

cfg["model_root"]  = os.path.join(cfg["root"], cfg["dataset_name"], cfg["model_name"])

# ---- Feature dimensions -------------------------------------------------
cfg["v_feat_dim"]   = 512    # CLIP video features
cfg["t_feat_dim"]   = 512    # CLIP text features
cfg["q_feat_type"]  = "last_hidden_state"

# ---- Data settings -------------------------------------------------------
cfg["max_v_l"]      = 75     # max video clip length (QVHighlights: 2s clips)
cfg["max_q_l"]      = 32     # max query token length
cfg["clip_len"]     = 2.0    # seconds per clip
cfg["max_windows"]  = 5      # max GT windows per sample
cfg["normalize_v"]  = True
cfg["normalize_t"]  = True
cfg["txt_drop_ratio"] = 0.0
cfg["data_ratio"]   = 1.0
cfg["v_feat_len_mode"] = "max"   # "max": resample to longest | "min": truncate to shortest
cfg["q_feat_len_mode"] = "min"   # "min": truncate (safe default for discrete token embeds)
cfg["use_tef"]        = False

# ---- Model architecture --------------------------------------------------
cfg["hidden_size"]    = 256
cfg["n_heads"]        = 8
cfg["num_queries"]    = 10   # number of detection query slots
cfg["dec_layers"]     = 2    # transformer decoder layers
cfg["input_drop"]     = 0.5
cfg["drop"]           = 0.1
cfg["n_input_proj"]   = 2    # stacked input projection layers (QD-DETR style)
cfg["initializer_range"] = 0.02
cfg["pos_enc_type"]   = "trainable"   # "trainable": learned | "sinusoidal": fixed sin/cos

# HLFormerBlock settings (from HLFormer)
cfg["sft_factor"]     = 0.09
cfg["attention_num"]  = 8    # total attention branches (all Euclidean Gaussian)

# Early Query Fusion (T2V encoder, QD-DETR style)
cfg["t2v_layers"]     = 6    # number of stacked T2VEncoderLayer blocks

# Auxiliary decoder losses (losses from each intermediate decoder layer)
cfg["aux_loss"]              = True
cfg["use_txt_in_memory"]     = True   # True=[video|text] in decoder, False=video only (QD-DETR style)
cfg["use_global_in_encoder"] = False  # True=global token passes through HLFormerBlock (QD-DETR style)

# ---- Loss weights --------------------------------------------------------
cfg["span_loss_coef"]   = 10.0
cfg["giou_loss_coef"]   = 1.0
cfg["label_loss_coef"]  = 4.0
cfg["lw_saliency"]      = 1.0
cfg["eos_coef"]         = 0.1    # class weight for background class
cfg["saliency_margin"]  = 1.0

# Contrastive alignment loss (aligns matched query slots with text tokens)
cfg["use_contrastive"]             = True
cfg["contrastive_hdim"]            = 64
cfg["contrastive_align_loss_coef"] = 0.1
cfg["temperature"]                 = 0.07

# Hungarian matcher costs
cfg["set_cost_class"]   = 4.0
cfg["set_cost_span"]    = 10.0
cfg["set_cost_giou"]    = 1.0

# ---- Optimizer -----------------------------------------------------------
cfg["lr"]               = 1e-4
cfg["lr_vid_enc"]       = 1e-4   # learning rate for video encoder (HLFormerBlock)
cfg["lr_drop"]          = 40     # epochs after which LR is multiplied by lr_gamma
cfg["lr_gamma"]         = 0.1
cfg["wd"]               = 1e-4   # weight decay
cfg["grad_clip"]        = 0.1    # gradient clipping norm
cfg["n_epoch"]          = 200
cfg["max_es_cnt"]       = 50     # early stopping patience (epochs)

# ---- DataLoader ----------------------------------------------------------
cfg["batchsize"]    = 32
cfg["num_workers"]  = 4
cfg["pin_memory"]   = True

# ---- Evaluation ----------------------------------------------------------
cfg["iou_thresholds"] = [0.5, 0.7]   # for R1@IoU metric
cfg["nms_thresh"]     = 0.5          # NMS IoU threshold for post-processing
cfg["top_k"]          = 5            # keep top-K spans before NMS


def get_cfg_defaults():
    os.makedirs(cfg["model_root"], exist_ok=True)
    with open(os.path.join(cfg["model_root"], "hyperparams.yaml"), "w") as f:
        yaml.dump(cfg, f)
    return cfg
