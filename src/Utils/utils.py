import os
import torch
import torch.nn as nn
import random
import numpy as np
import logging
import math

import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

import ipdb


def set_seed(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:                   # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_log(file_path, file_name):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(os.path.join(file_path, file_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_ckpt(model, optimizer, scheduler, config, ckpt_file, epoch, model_val):
    torch.save({
        'config': config,
        'epoch': epoch,
        'model_val': model_val,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, ckpt_file)


def load_ckpt(ckpt_file):
    ckpt = torch.load(ckpt_file, map_location="cpu")
    config = ckpt['config']
    model = ckpt['state_dict']
    optimizer = ckpt['optimizer']
    scheduler = ckpt.get('scheduler', None)   # None for checkpoints saved before this fix
    current_epoch = ckpt['epoch']
    model_val = ckpt['model_val']
    return config, model, optimizer, scheduler, current_epoch, model_val


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


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


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def _format_value(value, digits=2):
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v, digits=3) if _is_number(v) else str(v)
                               for v in value) + "]"
    return str(value)


def _render_table(headers, rows):
    if not rows:
        return []

    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(row):
        return "  " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "  " + "-+-".join("-" * width for width in widths)
    lines = [_fmt_row(headers), separator]
    lines.extend(_fmt_row(row) for row in str_rows)
    return lines


def _log_section(logger, title, lines):
    if not lines:
        return
    logger.info(f"[{title}]")
    for line in lines:
        logger.info(line)


def _metric_or_dash(metrics, key, digits=2):
    value = metrics.get(key)
    return _format_value(value, digits=digits) if value is not None else "-"


def _format_loss_section(avg_components):
    if not avg_components:
        return []
    main = {k: v for k, v in sorted(avg_components.items())
            if not any(k.endswith(f"_{i}") for i in range(10))}
    rows = [(name, _format_value(value, digits=4)) for name, value in main.items()]
    return _render_table(["Loss", "Avg"], rows)


def _format_main_metrics_section(metrics):
    if not metrics:
        return []
    preferred = [
        "primary",
        "R1@0.3", "R1@0.5", "R1@0.7",
        "mAP@0.3", "mAP@0.5", "mAP@0.7",
        "hl_mAP", "HIT@1",
    ]
    present = [key for key in preferred if key in metrics]
    if not present:
        return []
    return _render_table(present, [[_metric_or_dash(metrics, key) for key in present]])


def _format_span_source_section(metrics):
    rows = []
    headers = ["Source", "R1@0.3", "R1@0.5", "R1@0.7", "mAP@0.3", "mAP@0.5", "mAP@0.7"]
    for source in ("coarse", "refined", "final"):
        if any(f"{source}_{metric}" in metrics for metric in headers[1:]):
            rows.append([
                source,
                _metric_or_dash(metrics, f"{source}_R1@0.3"),
                _metric_or_dash(metrics, f"{source}_R1@0.5"),
                _metric_or_dash(metrics, f"{source}_R1@0.7"),
                _metric_or_dash(metrics, f"{source}_mAP@0.3"),
                _metric_or_dash(metrics, f"{source}_mAP@0.5"),
                _metric_or_dash(metrics, f"{source}_mAP@0.7"),
            ])
    return _render_table(headers, rows)


def _format_refinement_gain_section(metrics):
    pairs = [
        ("refined - coarse", "refined", "coarse"),
        ("final - coarse", "final", "coarse"),
        ("final - refined", "final", "refined"),
    ]
    rows = []
    for label, lhs, rhs in pairs:
        r1_lhs = metrics.get(f"{lhs}_R1@0.7")
        r1_rhs = metrics.get(f"{rhs}_R1@0.7")
        map_lhs = metrics.get(f"{lhs}_mAP@0.7")
        map_rhs = metrics.get(f"{rhs}_mAP@0.7")
        if None in (r1_lhs, r1_rhs, map_lhs, map_rhs):
            continue
        rows.append([
            label,
            f"{(r1_lhs - r1_rhs):+.2f}",
            f"{(map_lhs - map_rhs):+.2f}",
        ])
    return _render_table(["Delta", "R1@0.7", "mAP@0.7"], rows)


def _format_stream_weight_section(metrics):
    rows = []
    for key, label in (
        ("stream_weights", "stream_weights"),
        ("last_stream_weights", "last_stream_weights"),
        ("hybrid_blend_mean", "hybrid_blend_mean"),
    ):
        values = metrics.get(key)
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            value_text = "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"
        elif isinstance(values, (float, int)):
            value_text = f"{float(values):.4f}"
        else:
            value_text = str(values)
        rows.append((label, value_text))
    return _render_table(["Stream", "Value"], rows)


def _format_refinement_diagnostics_section(metrics):
    lines = []
    gate_rows = []
    for key in ("refine_gate_mean", "refine_gate_p25", "refine_gate_p50",
                "refine_gate_p75", "refine_gate_p90"):
        if key in metrics:
            gate_rows.append((key, _metric_or_dash(metrics, key, digits=4)))
    if gate_rows:
        lines.extend(_render_table(["Gate", "Value"], gate_rows))

    delta_rows = []
    for pair in ("coarse_to_refined", "coarse_to_final"):
        start_key = f"{pair}_start_abs_delta_mean_sec"
        end_key = f"{pair}_end_abs_delta_mean_sec"
        if start_key in metrics or end_key in metrics:
            delta_rows.append([
                pair,
                _metric_or_dash(metrics, start_key, digits=4),
                _metric_or_dash(metrics, end_key, digits=4),
            ])
    if delta_rows:
        if lines:
            lines.append("  ")
        lines.extend(_render_table(["Transition", "Start d_sec", "End d_sec"], delta_rows))
    stream_rows = _format_stream_weight_section(metrics)
    if stream_rows:
        if lines:
            lines.append("  ")
        lines.extend(stream_rows)
    return lines


def _build_validation_warnings(metrics, cfg):
    warnings = []

    gate_cap = cfg.get("refine_gate_max")
    gate_p50 = metrics.get("refine_gate_p50")
    gate_p90 = metrics.get("refine_gate_p90")
    if gate_cap is not None and gate_p50 is not None and gate_p50 >= 0.90 * float(gate_cap):
        warnings.append(
            f"refine_gate median is near the configured cap "
            f"({gate_p50:.4f} / {float(gate_cap):.4f})"
        )
    if gate_cap is not None and gate_p90 is not None and gate_p90 >= 0.98 * float(gate_cap):
        warnings.append(
            f"refine_gate high-percentile values look saturated "
            f"(p90={gate_p90:.4f}, cap={float(gate_cap):.4f})"
        )

    coarse_map_07 = metrics.get("coarse_mAP@0.7")
    refined_map_07 = metrics.get("refined_mAP@0.7")
    if coarse_map_07 is not None and refined_map_07 is not None and refined_map_07 < coarse_map_07:
        warnings.append(
            f"refined mAP@0.7 is below coarse "
            f"({refined_map_07:.2f} vs {coarse_map_07:.2f})"
        )

    final_r1_07 = metrics.get("final_R1@0.7")
    refined_r1_07 = metrics.get("refined_R1@0.7")
    if final_r1_07 is not None and refined_r1_07 is not None and final_r1_07 < refined_r1_07:
        warnings.append(
            f"final R1@0.7 is below refined "
            f"({final_r1_07:.2f} vs {refined_r1_07:.2f})"
        )

    final_map_07 = metrics.get("final_mAP@0.7")
    if final_map_07 is not None and refined_map_07 is not None and final_map_07 < refined_map_07:
        warnings.append(
            f"final mAP@0.7 is below refined "
            f"({final_map_07:.2f} vs {refined_map_07:.2f})"
        )

    return warnings


def _format_best_section(best_metrics, state=None):
    if not best_metrics:
        return []
    row = [[
        _metric_or_dash(best_metrics, "primary"),
        _metric_or_dash(best_metrics, "R1@0.5"),
        _metric_or_dash(best_metrics, "R1@0.7"),
        _metric_or_dash(best_metrics, "mAP@0.5"),
        _metric_or_dash(best_metrics, "mAP@0.7"),
        _metric_or_dash(best_metrics, "hl_mAP"),
    ]]
    lines = _render_table(["primary", "R1@0.5", "R1@0.7", "mAP@0.5", "mAP@0.7", "hl_mAP"], row)
    source_lines = _format_span_source_section(best_metrics)
    if source_lines:
        lines.append("  ")
        lines.append("  Span source metrics:")
        lines.extend(source_lines)
    stream_lines = _format_stream_weight_section(best_metrics)
    if stream_lines:
        lines.append("  ")
        lines.append("  Stream fusion:")
        lines.extend(stream_lines)
    if state:
        status = "updated" if state.get("best_updated") else "unchanged"
        lines.append(
            f"  best.ckpt: {status}"
            + (f" | {state['best_ckpt_path']}" if state.get("best_ckpt_path") else "")
        )
    return lines


def _format_training_state_section(epoch, train_loss, state):
    rows = [
        ("epoch", epoch),
        ("train_loss", _format_value(train_loss, digits=4)),
        ("validated", "yes" if state.get("validated") else "no"),
        ("ema_eval", "yes" if state.get("ema_active") else "no"),
    ]
    if state.get("validated"):
        rows.append(("best_updated", "yes" if state.get("best_updated") else "no"))
    if state.get("early_stop_counter") is not None:
        max_es = state.get("max_es_cnt")
        rows.append(("early_stop", f"{state['early_stop_counter']}/{max_es}"))
    if state.get("learning_rates"):
        lr_text = ", ".join(f"{lr:.2e}" for lr in state["learning_rates"])
        rows.append(("lr", lr_text))
    if state.get("latest_ckpt_path"):
        rows.append(("last_ckpt", state["latest_ckpt_path"]))
    if state.get("early_stop_triggered"):
        rows.append(("stop", "early_stop"))
    return _render_table(["State", "Value"], rows)


def log_validation_summary(logger, epoch, train_loss, metrics, best_metrics, cfg,
                           avg_components=None, state=None, title_prefix="Epoch"):
    state = state or {}
    logger.info("=" * 80)
    logger.info(f"{title_prefix} {epoch:3d} validation summary")

    _log_section(logger, "Losses", _format_loss_section(avg_components))

    if metrics:
        _log_section(logger, "Main Metrics", _format_main_metrics_section(metrics))
        _log_section(logger, "Span Source Metrics", _format_span_source_section(metrics))
        _log_section(logger, "Refinement Diagnostics", _format_refinement_diagnostics_section(metrics))
        _log_section(logger, "Refinement Gain", _format_refinement_gain_section(metrics))

        for warning in _build_validation_warnings(metrics, cfg):
            logger.warning(f"[Validation Warning] {warning}")

    _log_section(logger, "Best So Far", _format_best_section(best_metrics, state=state))
    _log_section(logger, "Training State", _format_training_state_section(epoch, train_loss, state))

    if metrics and cfg.get("log_raw_val_metrics", False):
        raw_rows = [(key, _format_value(value, digits=4)) for key, value in sorted(metrics.items())]
        _log_section(logger, "Raw Metric Dump", _render_table(["Metric", "Value"], raw_rows))

    logger.info("=" * 80)
