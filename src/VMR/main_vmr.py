"""
Training entry point for GaussianFormer-VMR.

Usage:
    cd ICCV25-HLFormer/src
    python VMR/main_vmr.py -d qvhighlights --gpu 0
    python VMR/main_vmr.py -d qvhighlights --gpu 0 --eval --resume path/to/best.ckpt
    python VMR/main_vmr.py -d charades    --gpu 0

Supported datasets (-d):
    qvhighlights   QVHighlights (moment retrieval + highlight detection)
    charades       Charades-STA (moment retrieval)
"""

import os
import sys
import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --------------------------------------------------------------------------
# Make sure 'src/' is on sys.path so HLFormer components can be imported
# --------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Now import VMR modules (relative to src/)
from VMR.Configs.qvhighlights import get_cfg_defaults as qvh_cfg
from VMR.Configs.charades     import get_cfg_defaults as cha_cfg
from VMR.Datasets.vmr_data_provider import build_vmr_dataloaders, prepare_batch_inputs
from VMR.Models.vmr_model          import build_model
from VMR.Losses.vmr_loss           import build_criterion
from VMR.Validations.vmr_validations import evaluate_vmr

from Utils.basic_utils import AverageMeter
from Utils.utils import set_seed, set_log, save_ckpt, load_ckpt


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) model
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains exponential moving average of model parameters.

    The EMA parameters are updated after each training step:
        ema_param = decay * ema_param + (1 - decay) * model_param

    Use ema_model.module for evaluation (call .eval() on it).

    Args:
        model:  the training model
        decay:  EMA decay factor (0.999–0.9999 typical)
        device: if set, EMA params are stored on this device
    """

    def __init__(self, model, decay=0.9995, device=None):
        import copy
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay  = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_p = model_p.to(device=self.device)
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="GaussianFormer-VMR Training")
parser.add_argument("-d", "--dataset_name", default="qvhighlights", type=str,
                    choices=["qvhighlights", "charades"],
                    help="Dataset to train on")
parser.add_argument("--gpu",    default="0", type=str,  help="CUDA device index")
parser.add_argument("--eval",   action="store_true",    help="Run evaluation only")
parser.add_argument("--resume", default="",  type=str,  help="Path to checkpoint")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def get_cfg(dataset_name):
    cfg_map = {
        "qvhighlights": qvh_cfg,
        "charades":     cha_cfg,
    }
    return cfg_map[dataset_name]()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    """AdamW with separate LRs for video encoder, boundary refinement head, and text encoder."""
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    vid_enc_params  = list(raw_model.video_encoder.parameters())
    vid_enc_ids     = set(id(p) for p in vid_enc_params)

    refiner_module  = getattr(raw_model, "boundary_refine", None)
    refiner_params  = list(refiner_module.parameters()) if refiner_module is not None else []
    refiner_ids     = set(id(p) for p in refiner_params)

    txt_enc         = getattr(raw_model, "txt_encoder", None)
    txt_enc_params  = []
    txt_enc_ids     = set()
    if txt_enc is not None:
        txt_enc_params = list(txt_enc.parameters())
        txt_enc_ids    = {id(p) for p in txt_enc_params}

    other_params    = [p for p in model.parameters()
                       if id(p) not in vid_enc_ids
                       and id(p) not in refiner_ids
                       and id(p) not in txt_enc_ids
                       and p.requires_grad]
    vid_params      = [p for p in vid_enc_params if p.requires_grad]
    refiner_params  = [p for p in refiner_params if p.requires_grad]
    txt_enc_params  = [p for p in txt_enc_params if p.requires_grad]

    param_groups = [
        {"params": other_params,   "lr": cfg["lr"]},
        {"params": vid_params,     "lr": cfg.get("lr_vid_enc", cfg["lr"])},
        {"params": refiner_params, "lr": cfg.get("lr_refiner",  cfg["lr"])},
    ]
    if txt_enc_params:
        param_groups.append({
            "params": txt_enc_params,
            "lr":     cfg.get("lr_txt_enc", cfg["lr"]),
        })
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["wd"])
    logger_ref = globals().get("logger")
    if logger_ref is not None:
        n_groups = len(param_groups)
        logger_ref.info(
            f"Optimizer: {n_groups} param groups — "
            f"others lr={cfg['lr']:.2e}, "
            f"vid_enc lr={cfg.get('lr_vid_enc', cfg['lr']):.2e}, "
            f"refiner lr={cfg.get('lr_refiner', cfg['lr']):.2e}"
            + (f", txt_enc lr={cfg.get('lr_txt_enc', cfg['lr']):.2e}"
               if txt_enc_params else "")
        )
    return optimizer


def build_scheduler(optimizer, cfg):
    """LR scheduler: cosine annealing with linear warmup.

    - Epochs [0, warmup_epochs): LR scales linearly from 0 → 1.
    - Epochs [warmup_epochs, ...): Cosine annealing with warm restarts.
      T_0 = cosine_T0 epochs per cycle, T_mult = cosine_Tmult.
    - Falls back to step decay if cosine_T0 is not set in config.
    """
    warmup_epochs = cfg.get("warmup_epochs", 0)
    cosine_T0     = cfg.get("cosine_T0", 0)

    if cosine_T0 > 0:
        # Cosine annealing with warm restarts + linear warmup
        cosine_Tmult = cfg.get("cosine_Tmult", 2)
        eta_min_ratio = cfg.get("cosine_eta_min_ratio", 0.01)  # min LR = 1% of base

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            # After warmup, use cosine schedule
            t = epoch - warmup_epochs
            # Find which cycle we're in
            T_cur = cosine_T0
            cycle_start = 0
            while t >= cycle_start + T_cur:
                cycle_start += T_cur
                T_cur = int(T_cur * cosine_Tmult)
            progress = (t - cycle_start) / max(T_cur, 1)
            return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Original step decay fallback
        lr_drop  = cfg.get("lr_drop", 40)
        lr_gamma = cfg.get("lr_gamma", 0.1)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            steps = (epoch - warmup_epochs) // lr_drop
            return lr_gamma ** steps

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss schedule
# ---------------------------------------------------------------------------

def apply_loss_schedule(epoch, criterion, cfg, logger=None):
    """Patch criterion.weight_dict and matcher costs at phase boundaries.

    Reads cfg["loss_schedule"] — a list of (from_epoch, weight_dict) pairs.
    The last entry whose from_epoch <= epoch is the active phase.
    Only applies changes when the phase actually changes (epoch boundary).
    """
    schedule = cfg.get("loss_schedule")
    if not schedule:
        return

    # Determine active phase for this epoch and for the previous epoch
    def _active_phase(ep):
        active = None
        active_start = -1
        for (from_epoch, weights) in schedule:
            if ep >= from_epoch >= active_start:
                active = weights
                active_start = from_epoch
        return active, active_start

    current_phase, current_start = _active_phase(epoch)
    prev_phase,    prev_start    = _active_phase(epoch - 1)

    # No change — skip to avoid redundant work every epoch
    if current_start == prev_start and epoch > 0:
        return

    if current_phase is None:
        return

    if logger:
        logger.info(f"  [LossSchedule] Epoch {epoch}: phase from_epoch={current_start}  "
                    + "  ".join(f"{k}={v}" for k, v in current_phase.items()))

    aux_scale = cfg.get("aux_loss_scale", 1.0)
    n_aux     = cfg.get("dec_layers", 2) - 1

    # cfg key → weight_dict key
    key_map = {
        "span_loss_coef":             "loss_span",
        "giou_loss_coef":             "loss_giou",
        "boundary_loss_coef":         "loss_boundary",
        "boundary_refine_coef":       "loss_boundary_refined",
        "boundary_refine_giou_coef":  "loss_giou_refined",
        "final_loss_coef_span":       "loss_span_final",
        "final_loss_coef_giou":       "loss_giou_final",
        "contrastive_align_loss_coef":"loss_contrastive_align",
        "lw_saliency":                "loss_saliency",
        "label_loss_coef":            "loss_label",
    }

    refinement_schedule_keys = {
        "boundary_refine_coef",
        "boundary_refine_giou_coef",
        "final_loss_coef_span",
        "final_loss_coef_giou",
    }
    use_boundary_refinement = cfg.get("use_boundary_refinement", True)

    for cfg_key, new_val in current_phase.items():
        if cfg_key in ("set_cost_span", "set_cost_giou", "set_cost_class"):
            continue  # handled below
        if not use_boundary_refinement and cfg_key in refinement_schedule_keys:
            continue

        wk = key_map.get(cfg_key)
        if wk is None:
            continue

        # Update primary weight
        if wk in criterion.weight_dict:
            criterion.weight_dict[wk] = new_val

        # Update auxiliary decoder layer weights (loss_span_0, loss_giou_0, …)
        for i in range(n_aux):
            aux_key = f"{wk}_{i}"
            if aux_key in criterion.weight_dict:
                criterion.weight_dict[aux_key] = new_val * aux_scale

    # Matcher costs — update directly on the matcher object
    if "set_cost_span" in current_phase:
        criterion.matcher.cost_span = current_phase["set_cost_span"]
    if "set_cost_giou" in current_phase:
        criterion.matcher.cost_giou = current_phase["set_cost_giou"]
    if "set_cost_class" in current_phase:
        criterion.matcher.cost_class = current_phase["set_cost_class"]

    # alpha_iou_alpha — ramp the α-IoU exponent on the criterion directly
    if "alpha_iou_alpha" in current_phase:
        new_alpha = float(current_phase["alpha_iou_alpha"])
        setattr(criterion, "alpha_iou_alpha", new_alpha)
        if logger:
            logger.info(f"  [LossSchedule] alpha_iou_alpha -> {new_alpha}")

    if "iou_floor" in current_phase:
        new_iou_floor = float(current_phase["iou_floor"])
        setattr(criterion, "iou_floor", new_iou_floor)
        cfg["iou_floor"] = new_iou_floor
        if logger:
            logger.info(f"  [LossSchedule] iou_floor -> {new_iou_floor}")

    # aux_loss_scale — re-scale all existing aux weight_dict entries to the new scale.
    # cfg is updated first so subsequent schedule phases inherit the correct value.
    if "aux_loss_scale" in current_phase:
        new_aux_scale = current_phase["aux_loss_scale"]
        old_aux_scale = cfg.get("aux_loss_scale", 1.0)
        cfg["aux_loss_scale"] = new_aux_scale
        if old_aux_scale > 0 and new_aux_scale != old_aux_scale:
            ratio = new_aux_scale / old_aux_scale
            for k in list(criterion.weight_dict.keys()):
                # Aux keys have the pattern "<loss_name>_<layer_idx>" where
                # layer_idx is a non-negative integer suffix.
                parts = k.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    criterion.weight_dict[k] = criterion.weight_dict[k] * ratio


# ---------------------------------------------------------------------------
# Augmentation schedule
# ---------------------------------------------------------------------------

def apply_augmentation_schedule(epoch, train_loader, cfg, logger=None):
    """Patch training dataset augmentation strengths at phase boundaries."""
    schedule = cfg.get("aug_schedule")
    if not schedule:
        return

    def _active_phase(ep):
        active = None
        active_start = -1
        for (from_epoch, params) in schedule:
            if ep >= from_epoch >= active_start:
                active = params
                active_start = from_epoch
        return active, active_start

    current_phase, current_start = _active_phase(epoch)
    prev_phase, prev_start = _active_phase(epoch - 1)

    if current_phase is None:
        return
    if current_start == prev_start and epoch > 0:
        return

    train_dset = getattr(train_loader, "dataset", None)
    if train_dset is None:
        return

    applied = []
    for key, val in current_phase.items():
        if hasattr(train_dset, key):
            setattr(train_dset, key, val)
            cfg[key] = val
            applied.append(f"{key}={val}")

    if logger and applied:
        logger.info(f"  [AugSchedule] Epoch {epoch}: phase from_epoch={current_start}  " + "  ".join(applied))


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(epoch, train_loader, model, criterion, optimizer, cfg, device, logger,
                    ema=None):
    model.train()
    loss_meter = AverageMeter()
    loss_components = {}

    bar = tqdm(train_loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)
    for batch_meta, batched_data in bar:
        model_inputs, targets = prepare_batch_inputs(batched_data, device)

        optimizer.zero_grad()
        outputs = model(**model_inputs)
        loss_dict, total_loss = criterion(outputs, targets)

        total_loss.backward()
        if cfg.get("grad_clip", 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        # Update EMA after each optimization step
        if ema is not None:
            ema.update(model.module if isinstance(model, nn.DataParallel) else model)

        loss_meter.update(total_loss.item())
        # Track individual loss components for logging
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

        bar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            **{k: f"{v.item():.4f}" for k, v in loss_dict.items()
               if isinstance(v, torch.Tensor)}
        )

    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    return loss_meter.avg, avg_components


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_one_epoch(epoch, val_loader, model, cfg, device, logger):
    model.eval()
    metrics = evaluate_vmr(model, val_loader, device, cfg)
    return metrics


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_metrics(logger, epoch, train_loss, metrics, best_metrics, avg_components=None):
    logger.info("=" * 80)
    logger.info(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}")
    if avg_components:
        # Log only main-loss keys (skip auxiliary decoder losses like loss_span_0)
        main = {k: v for k, v in sorted(avg_components.items())
                if not any(k.endswith(f"_{i}") for i in range(10))}
        logger.info("  Loss components (epoch avg):")
        for k, v in main.items():
            logger.info(f"    {k}: {v:.4f}")
    if metrics:
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.2f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        logger.info("  Best so far:")
        for k, v in best_metrics.items():
            logger.info(f"    {k}: {v:.2f}" if isinstance(v, (int, float)) else f"    {k}: {v}")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = get_cfg(args.dataset_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    logger = set_log(cfg["model_root"], "log_vmr.txt")
    logger.info(f"GaussianFormer-VMR  dataset={cfg['dataset_name']}")
    logger.info(str(cfg))

    # Reproducibility
    set_all_seeds(cfg["seed"])
    logger.info(f"Seed: {cfg['seed']}")

    # ---------- Data ----------------------------------------------------------
    logger.info("Building dataloaders ...")
    train_loader, val_loader, test_loader = build_vmr_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)} samples  "
                f"Val: {len(val_loader.dataset)} samples")

    for split_name, loader in (("train", train_loader), ("val", val_loader)):
        dset = getattr(loader, "dataset", None)
        if dset is not None and hasattr(dset, "summarize_temporal_alignment"):
            summary = dset.summarize_temporal_alignment()
            logger.info(f"{split_name.title()} temporal alignment: {summary}")

    # ---------- Model ---------------------------------------------------------
    logger.info("Building model ...")
    model = build_model(cfg).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Parameters: {total_params:.2f}M total, {trainable:.2f}M trainable")

    # ---------- Loss ----------------------------------------------------------
    criterion = build_criterion(cfg).to(device)
    logger.info(
        "Span semantics: matcher=%s label=%s eval=%s",
        cfg.get("match_span_source", "coarse"),
        cfg.get("label_span_source", "coarse"),
        "final->refined->coarse" if cfg.get("use_refined_spans", True) else "coarse-only",
    )

    # ---------- Checkpoint resume ---------------------------------------------
    current_epoch      = -1
    best_metrics       = {"primary": 0.0}
    metrics            = {}   # retains last validated epoch's metrics for last.pt on skipped epochs
    optimizer          = build_optimizer(model, cfg)
    scheduler          = build_scheduler(optimizer, cfg)

    _resume_ema_state = None   # applied after EMA is constructed below
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        saved_cfg, state_dict, opt_state, sched_state, current_epoch, saved_metrics = load_ckpt(args.resume)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        _info = raw_model.load_state_dict(state_dict, strict=False)
        if _info.missing_keys:
            logger.info(f"  Missing keys (will use init): {_info.missing_keys}")
        if _info.unexpected_keys:
            logger.info(f"  Unexpected keys (ignored): {_info.unexpected_keys}")
        try:
            optimizer.load_state_dict(opt_state)
        except ValueError as exc:
            logger.info(f"  Optimizer state restore skipped: {exc}")
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
            except ValueError as exc:
                logger.info(f"  Scheduler state restore skipped: {exc}")
        best_metrics = saved_metrics if isinstance(saved_metrics, dict) \
                       else {"primary": saved_metrics}
        # Stash EMA weights if present (last.pt stores them under "ema_state_dict")
        _raw_ckpt = torch.load(args.resume, map_location="cpu")
        _resume_ema_state = _raw_ckpt.get("ema_state_dict", None)
        if _resume_ema_state is not None:
            logger.info("  EMA state found in checkpoint — will restore after EMA init")

    # ---------- Eval-only mode ------------------------------------------------
    if args.eval:
        if not args.resume:
            logger.warning("--eval specified but no --resume checkpoint provided!")
        loader = test_loader if test_loader is not None else val_loader
        metrics = val_one_epoch(-1, loader, model, cfg, device, logger)
        logger.info("Evaluation results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.2f}")
        return

    # ---------- Training loop -------------------------------------------------
    es_cnt = 0
    # Create EMA model for smoother evaluation (reduces epoch-to-epoch fluctuation)
    ema_decay = cfg.get("ema_decay", 0.9995)
    use_ema   = cfg.get("use_ema", True)
    ema = None
    if use_ema:
        raw_for_ema = model.module if isinstance(model, nn.DataParallel) else model
        ema = ModelEMA(raw_for_ema, decay=ema_decay, device=device)
        if _resume_ema_state is not None:
            ema.load_state_dict(_resume_ema_state)
            logger.info("  EMA weights restored from checkpoint")
        logger.info(f"EMA enabled with decay={ema_decay}")

    for epoch in range(current_epoch + 1, cfg["n_epoch"]):

        # --- Loss schedule (phase-boundary updates to criterion/matcher) ---
        apply_loss_schedule(epoch, criterion, cfg, logger)

        # --- Augmentation schedule (phase-boundary updates to train dataset) ---
        apply_augmentation_schedule(epoch, train_loader, cfg, logger)

        # --- Train ---
        train_loss, loss_components = train_one_epoch(
            epoch, train_loader, model, criterion, optimizer, cfg, device, logger,
            ema=ema)
        scheduler.step()

        # --- Validate (use EMA model if available) ---
        val_freq       = cfg.get("val_freq", 1)
        val_full_epoch = cfg.get("val_full_epoch", 0)
        do_validate = (
            epoch >= val_full_epoch
            or epoch % val_freq == 0
            or epoch == cfg["n_epoch"] - 1
        )

        if do_validate:
            eval_model = ema.module if ema is not None else model
            with torch.no_grad():
                metrics = val_one_epoch(epoch, val_loader, eval_model, cfg, device, logger)

        log_metrics(logger, epoch, train_loss, metrics if do_validate else None,
                    best_metrics, avg_components=loss_components)

        # --- Checkpoint & early stopping ---
        raw_model = model.module if isinstance(model, nn.DataParallel) else model

        if do_validate:
            primary = metrics.get("primary", 0.0)
            if primary > best_metrics.get("primary", 0.0):
                best_metrics = metrics
                es_cnt = 0
                ckpt_path = os.path.join(cfg["model_root"], "best.ckpt")
                # Save EMA model as the checkpoint when available (better generalization)
                save_model = ema.module if ema is not None else raw_model
                save_ckpt(save_model, optimizer, scheduler, cfg, ckpt_path, epoch, best_metrics)
                logger.info(f"  New best checkpoint saved -> {ckpt_path}"
                            + (" (EMA)" if ema is not None else ""))
            else:
                es_cnt += 1
                logger.info(f"  No improvement ({es_cnt}/{cfg['max_es_cnt']})")
                if cfg["max_es_cnt"] != -1 and es_cnt > cfg["max_es_cnt"]:
                    logger.info("Early stopping triggered.")
                    break

        # --- Separate saliency checkpoint (hl_mAP) ---
        # hl_map = metrics.get("hl_mAP", 0.0)
        # if hl_map > best_sal_metrics.get("hl_mAP", 0.0):
        #     best_sal_metrics = metrics
        #     sal_path = os.path.join(cfg["model_root"], "best_saliency.ckpt")
        #     save_model = ema.module if ema is not None else raw_model
        #     save_ckpt(save_model, optimizer, scheduler, cfg, sal_path, epoch, best_sal_metrics)
        #     logger.info(f"  New best saliency checkpoint saved -> {sal_path}  "
        #                 f"(hl_mAP={hl_map:.2f})")

        # --- Latest checkpoint: raw model + EMA (always overwritten) ---
        last_path = os.path.join(cfg["model_root"], "last.pt")
        last_ckpt = {
            "config":     cfg,
            "epoch":      epoch,
            "model_val":  metrics,
            "state_dict": raw_model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
        }
        if ema is not None:
            last_ckpt["ema_state_dict"] = ema.state_dict()
        torch.save(last_ckpt, last_path)
        logger.info(f"  Latest checkpoint saved -> {last_path}"
                    + (" (raw + EMA)" if ema is not None else " (raw)"))

    # ---------- Final test evaluation ----------------------------------------
    if test_loader is not None:
        logger.info("Running final evaluation on test set ...")
        best_ckpt = os.path.join(cfg["model_root"], "best.ckpt")
        if os.path.exists(best_ckpt):
            _, state_dict, _, _, _, _ = load_ckpt(best_ckpt)
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            _info = raw_model.load_state_dict(state_dict, strict=False)
            if _info.missing_keys:
                logger.info(f"  Missing keys (will use init): {_info.missing_keys}")
            if _info.unexpected_keys:
                logger.info(f"  Unexpected keys (ignored): {_info.unexpected_keys}")

        test_metrics = val_one_epoch(-1, test_loader, model, cfg, device, logger)
        logger.info("Test results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
