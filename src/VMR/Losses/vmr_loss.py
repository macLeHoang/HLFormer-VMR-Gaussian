"""
Loss criterion for Video Moment Retrieval.

Contains four loss components (all from QD-DETR):
  1. loss_span   – L1 + GIoU regression on matched span pairs
  2. loss_label  – foreground / background cross-entropy
  3. loss_saliency – ranking contrastive loss + negative-pair loss + margin loss

Adapted from QD-DETR (https://github.com/wjun0830/QD-DETR).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VMR.Models.span_utils import span_cxw_to_xx, temporal_iou


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy for the given output/target pair."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ---------------------------------------------------------------------------
# IoU-based loss helpers
# ---------------------------------------------------------------------------

def diou_temporal_loss(src_xx, tgt_xx):
    """DIoU loss for matched 1D temporal spans.

    DIoU = IoU - ρ²/c²
      ρ = center distance between predicted and GT span
      c = length of the minimum convex hull enclosing both spans

    Strictly better than GIoU: provides direct center-alignment gradient
    even when spans do not overlap, where GIoU only shrinks the convex hull.

    Args:
        src_xx: (N, 2) predicted spans in (start, end) format, normalized [0,1]
        tgt_xx: (N, 2) GT spans in (start, end) format, normalized [0,1]
    Returns:
        scalar loss
    """
    iou_mat, _ = temporal_iou(src_xx, tgt_xx)          # (N, N)
    iou = torch.diag(iou_mat).clamp(min=0.0)            # (N,)

    c_pred = (src_xx[:, 0] + src_xx[:, 1]) * 0.5       # (N,) predicted center
    c_gt   = (tgt_xx[:, 0] + tgt_xx[:, 1]) * 0.5       # (N,) GT center
    rho2   = (c_pred - c_gt).pow(2)                     # (N,) squared center dist

    hull_left  = torch.min(src_xx[:, 0], tgt_xx[:, 0])
    hull_right = torch.max(src_xx[:, 1], tgt_xx[:, 1])
    c2 = (hull_right - hull_left).clamp(min=1e-7).pow(2)   # (N,) squared hull len

    return (1.0 - (iou - rho2 / c2)).mean()


def alpha_iou_temporal_loss(src_xx, tgt_xx, alpha=2.0):
    """Alpha-IoU loss for matched 1D temporal spans.

    L = 1 - IoU^alpha

    With alpha > 1, the gradient is amplified for medium-IoU predictions
    (0.5–0.7 range), directly targeting R1@0.7 conversions.
    alpha=1 recovers standard IoU loss; alpha=2 or 3 recommended for R1@0.7.

    Args:
        src_xx: (N, 2) predicted spans in (start, end) format, normalized [0,1]
        tgt_xx: (N, 2) GT spans in (start, end) format, normalized [0,1]
        alpha:  exponent > 1 (default 2.0)
    Returns:
        scalar loss
    """
    iou_mat, _ = temporal_iou(src_xx, tgt_xx)          # (N, N)
    iou = torch.diag(iou_mat).clamp(min=0.0)            # (N,)
    return (1.0 - iou.pow(alpha)).mean()


# ---------------------------------------------------------------------------
# VMR SetCriterion
# ---------------------------------------------------------------------------

class VMRSetCriterion(nn.Module):
    """Compute all losses for the GaussianFormer-VMR model.

    Process:
        1. Run the Hungarian matcher to assign predictions to GT spans.
        2. Compute span L1 + GIoU losses on matched pairs.
        3. Compute foreground/background classification loss.
        4. Compute saliency loss (ranking contrastive + margin + neg-pair).

    Args:
        matcher:          HungarianMatcher instance
        weight_dict:      {loss_name: weight} for final weighted sum
        eos_coef:         class weight for the background (non-object) class
        losses:           list of active loss names, e.g. ["spans","labels","saliency"]
        saliency_margin:  margin for the margin-based saliency ranking loss
        use_matcher:      if False, skip matching and only compute saliency loss
                          (used for highlight-detection only datasets like TVSum)
    """

    FOREGROUND = 0
    BACKGROUND = 1

    def __init__(self, matcher, weight_dict, eos_coef=0.1,
                 losses=("spans", "labels", "saliency"),
                 saliency_margin=1.0, use_matcher=True,
                 temperature=0.07, label_smoothing=0.0,
                 alpha_iou_alpha=2.0,
                 label_span_source="coarse"):
        super().__init__()
        self.matcher         = matcher
        self.weight_dict     = weight_dict
        self.losses          = list(losses)
        self.saliency_margin = saliency_margin
        self.use_matcher     = use_matcher
        self.temperature     = temperature
        self.label_smoothing = label_smoothing
        self.alpha_iou_alpha = alpha_iou_alpha   # exponent for Alpha-IoU in refiner head
        self.label_span_source = label_span_source
        assert self.label_span_source in {"coarse", "refined", "matched"}, \
            "label_span_source must be 'coarse', 'refined', or 'matched'"

        # Class weights: foreground=1.0, background=eos_coef
        empty_weight = torch.ones(2)
        empty_weight[self.BACKGROUND] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # ------------------------------------------------------------------
    # Individual losses
    # ------------------------------------------------------------------

    def loss_spans(self, outputs, targets, indices):
        """L1 regression loss + DIoU loss on matched (prediction, GT) pairs.

        Both are in normalized (center, width) format in [0, 1].
        DIoU replaces GIoU: provides direct center-alignment gradient even when
        spans do not overlap, which is common in early training epochs.
        """
        assert "pred_spans" in outputs
        idx      = self._src_permutation_idx(indices)
        src_cxw  = outputs["pred_spans"][idx]                                 # (#matched, 2)
        tgt_cxw  = torch.cat(
            [t["spans"][j] for t, (_, j) in zip(targets["span_labels"], indices)],
            dim=0)                                                             # (#matched, 2)

        # Convert to (start, end) once — shared by DIoU and boundary losses
        src_xx = span_cxw_to_xx(src_cxw)   # (#matched, 2)
        tgt_xx = span_cxw_to_xx(tgt_cxw)

        loss_l1       = F.l1_loss(src_cxw, tgt_cxw, reduction="none").mean()
        loss_diou     = diou_temporal_loss(src_xx, tgt_xx)
        loss_alpha    = alpha_iou_temporal_loss(src_xx, tgt_xx, alpha=self.alpha_iou_alpha)

        # Boundary smooth-L1: direct (start, end) supervision with beta=0.05.
        # beta=0.05 ≈ 6 frames in a 128-frame video (normalized coords).
        loss_boundary = F.smooth_l1_loss(src_xx, tgt_xx, reduction="mean", beta=0.05)

        # Combined coarse loss: DIoU handles center alignment (strong gradient when
        # spans do not overlap); Alpha-IoU amplifies gradient in the IoU=[0.5, 0.7)
        # zone that separates R1@0.5 hits from R1@0.7 hits.
        # v34: 0.5/0.5 → 0.3/0.7 — put 70% of gradient budget on Alpha-IoU to
        # maximise signal in the 0.5→0.7 IoU transition region (R1@0.7 bottleneck).
        loss_giou_combined = 0.5 * loss_diou + 0.5 * loss_alpha

        return {"loss_span": loss_l1, "loss_giou": loss_giou_combined, "loss_boundary": loss_boundary}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """QD-DETR style contrastive alignment loss.

        Computes query-token similarities, sums over text tokens to get
        per-query logits, then applies positive-map NCE using matched queries.
        """
        if "proj_queries" not in outputs:
            return {"loss_contrastive_align": torch.tensor(0.0,
                    device=outputs["pred_spans"].device)}
        if indices is None:
            return {"loss_contrastive_align": torch.tensor(0.0,
                    device=outputs["pred_spans"].device)}

        normalized_text_embed = outputs["proj_txt_mem"]   # (B, L_t, d)
        normalized_img_embed  = outputs["proj_queries"]   # (B, Q, d)

        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed
        )
        logits = logits.sum(2) / self.temperature          # (B, Q)

        idx = self._src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        if idx[0].numel() > 0:
            positive_map[idx] = True

        num_pos = positive_map.sum(1)                      # (B,)
        valid = num_pos > 0
        if not valid.any():
            return {"loss_contrastive_align": torch.tensor(0.0,
                    device=outputs["pred_spans"].device)}

        positive_logits = logits.masked_fill(~positive_map, 0.0)
        pos_term = positive_logits.sum(1)
        neg_term = logits.logsumexp(1)

        loss_nce = -pos_term[valid] / num_pos[valid].float() + neg_term[valid]
        return {"loss_contrastive_align": loss_nce.mean()}

    def loss_labels(self, outputs, targets, indices, log=True):
        """Quality score loss: train fg logit to predict IoU-with-GT.

        Replaces binary foreground/background cross-entropy with a softer
        formulation where matched slots are trained toward their actual
        IoU with the matched GT span (range [0.1, 1.0]) and unmatched slots
        are trained toward 0.0.

        At inference: sigmoid(pred_logits) ≈ expected localization
        quality — high only when the span is both foreground AND precise.
        """
        assert "pred_logits" in outputs
        idx = self._src_permutation_idx(indices)

        # pred_logits is now (B, Q) — single quality logit per query slot
        fg_logits = outputs["pred_logits"]   # (B, Q)

        # Compute actual IoU for each matched (pred, gt) pair — no gradient
        with torch.no_grad():
            use_refined_for_labels = False
            if self.label_span_source == "refined":
                use_refined_for_labels = True
            elif self.label_span_source == "matched":
                matcher_source = getattr(self.matcher, "match_span_source", "coarse")
                use_refined_for_labels = matcher_source in {"refined", "dual"}

            span_key = "pred_spans_refined" if (
                use_refined_for_labels and outputs.get("pred_spans_refined") is not None
            ) else "pred_spans"

            src_spans = span_cxw_to_xx(outputs[span_key][idx])            # (#matched, 2)
            tgt_spans = span_cxw_to_xx(
                torch.cat(
                    [t["spans"][j] for t, (_, j) in zip(targets["span_labels"], indices)],
                    dim=0))                                                 # (#matched, 2)
            iou_mat, _ = temporal_iou(src_spans, tgt_spans)
            raw_iou   = torch.diag(iou_mat).clamp(0.0, 1.0)               # (#matched,)
            # Use raw IoU directly with a floor of 0.1.
            # The old formula (0.5 + 0.5 * raw_iou) set a floor of 0.5, which
            # compressed the matched vs unmatched target gap to 0.5 vs 0.0 and
            # weakened the quality-score gradient for the first ~15 epochs.
            # With floor=0.1 the separation is 0.1 vs 0.0 early on and naturally
            # grows toward 1.0 as IoU improves, giving a cleaner signal.
            iou_scores = raw_iou.clamp(min=0.1)                            # (#matched,) in [0.1, 1.0]

        # Soft targets: matched -> IoU score, unmatched -> 0.0
        soft_targets = torch.zeros_like(fg_logits)   # (B, Q)
        soft_targets[idx] = iou_scores

        # pos_weight rebalances foreground vs background gradient.
        # With num_queries=Q and ~n_gt foreground slots per sample,
        # background slots outnumber foreground by (Q - n_gt) / n_gt.
        # pos_weight = (Q - n_gt) / n_gt  makes gradient magnitudes equal.
        # We estimate n_gt from soft_targets (any slot with target > 0).
        n_fg = (soft_targets > 0).float().sum().clamp(min=1)
        n_total = soft_targets.numel()
        # Cap at 25 to match the realistic Q/n_gt ratio for Charades-STA
        # (num_queries=24, ~1 GT per sample → true ratio ≈ 23).
        # The old cap of 15 suppressed foreground gradient by ~35%,
        # causing the quality head to under-weight matched spans.
        pos_weight = torch.tensor(
            min((n_total - n_fg.item()) / n_fg.item(), 25.0),
            device=fg_logits.device
        )
        loss_bce = F.binary_cross_entropy_with_logits(
            fg_logits, soft_targets, pos_weight=pos_weight, reduction="mean")

        losses = {"loss_label": loss_bce}
        if log:
            with torch.no_grad():
                pred_quality = fg_logits[idx].sigmoid()
                # class_error: % of matched predictions with predicted quality < 0.5
                # (i.e., model thinks its own matched span is low quality)
                losses["class_error"] = (
                    100.0 * (pred_quality < 0.5).float().mean())
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """Three-component saliency loss (from QD-DETR):

        1. Negative-pair loss  – suppress saliency scores for negative (video, query) pairs
        2. Ranking contrastive – rank clips by their ground-truth saliency level
        3. Margin-based loss   – pos_clip_score > neg_clip_score + margin
        """
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": torch.tensor(0.0, device=outputs["pred_spans"].device)}

        vid_mask = outputs["video_mask"]                        # (B, L_vid)

        # ---- 1. Negative-pair loss ----
        scores_neg = outputs["saliency_scores_neg"].clone()    # (B, L_vid)
        loss_neg_pair = (
            -torch.log(1.0 - torch.sigmoid(scores_neg) + 1e-6) * vid_mask
        ).sum(dim=1).mean()

        # ---- 2. Ranking contrastive loss ----
        scores     = outputs["saliency_scores"].clone()        # (B, L_vid)
        contrast_labels = targets["saliency_all_labels"]       # (B, L_vid)

        # Concatenate pos+neg for contrastive computation
        scores_cat  = torch.cat([scores, scores_neg], dim=1)
        labels_cat  = torch.cat([contrast_labels,
                                  torch.zeros_like(contrast_labels)], dim=1)
        mask_cat    = vid_mask.repeat(1, 2)
        scores_cat  = mask_cat * scores_cat + (1.0 - mask_cat) * -1e3

        tau = 0.5
        loss_rank_contrastive = torch.tensor(0.0, device=scores.device)
        n_active = 0
        for rank_thr in range(1, 12):
            pos_mask  = labels_cat >= rank_thr
            if pos_mask.sum() == 0:
                continue
            n_active += 1
            batch_has_pos = pos_mask.sum(dim=1) > 0            # (B,)

            cur_scores = scores_cat / tau
            cur_scores = cur_scores - cur_scores.max(dim=1, keepdim=True)[0]

            exp_s  = torch.exp(cur_scores)
            log_p  = cur_scores - torch.log(exp_s.sum(1, keepdim=True) + 1e-6)
            mean_log_prob_pos = (
                pos_mask * log_p * mask_cat
            ).sum(1) / (pos_mask.sum(1) + 1e-6)
            loss = -mean_log_prob_pos * batch_has_pos

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        # Divide by number of thresholds that were actually active, not hardcoded 11.
        # On Charades-STA labels are binary (0/1), so only rank_thr=1 fires — the
        # old /11 was silently suppressing the loss to 1/11 of its intended magnitude.
        loss_rank_contrastive = loss_rank_contrastive / max(n_active, 1)

        # ---- 3. Margin-based ranking loss ----
        pos_idx  = targets["saliency_pos_labels"]   # (B, n_pairs)
        neg_idx  = targets["saliency_neg_labels"]   # (B, n_pairs)
        n_pairs  = pos_idx.shape[1]
        bidx     = torch.arange(len(scores), device=scores.device)
        pos_sc   = torch.stack(
            [scores[bidx, pos_idx[:, c]] for c in range(n_pairs)], dim=1)
        neg_sc   = torch.stack(
            [scores[bidx, neg_idx[:, c]] for c in range(n_pairs)], dim=1)
        loss_margin = torch.clamp(
            self.saliency_margin + neg_sc - pos_sc, min=0.0
        ).sum() / (len(scores) * n_pairs) * 2.0

        total = loss_margin + loss_rank_contrastive + loss_neg_pair
        return {"loss_saliency": total}

    def loss_spans_refined(self, outputs, targets, indices):
        """Boundary + Alpha-IoU loss on BoundaryRefinementHead output.

        Reuses the same Hungarian `indices` from the coarse decoder —
        no second matching needed.  Only smooth L1 boundary and Alpha-IoU are
        applied (not center/width L1) because precision at the boundary
        is the bottleneck for R1@0.7.

        Alpha-IoU (L = 1 - IoU^alpha) amplifies gradient for medium-IoU
        predictions (0.5–0.7 range), directly targeting R1@0.7 conversions.
        """
        if "pred_spans_refined" not in outputs or outputs["pred_spans_refined"] is None:
            return {}

        idx     = self._src_permutation_idx(indices)
        src_cxw = outputs["pred_spans_refined"][idx]          # (#matched, 2)
        tgt_cxw = torch.cat(
            [t["spans"][j] for t, (_, j) in zip(targets["span_labels"], indices)],
            dim=0)                                             # (#matched, 2)

        src_xx = span_cxw_to_xx(src_cxw)
        tgt_xx = span_cxw_to_xx(tgt_cxw)

        # beta=0.02 ≈ 1.5 frames at max_v_l=75: keeps quadratic gradient in the 1–3 frame
        # zone that separates R1@0.5 hits from R1@0.7 hits (was 0.05 ≈ 3.75 frames).
        loss_bdy = F.smooth_l1_loss(src_xx, tgt_xx, reduction="mean", beta=0.02)
        loss_iou = alpha_iou_temporal_loss(src_xx, tgt_xx, alpha=self.alpha_iou_alpha)

        return {"loss_boundary_refined": loss_bdy, "loss_giou_refined": loss_iou}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, outputs, targets):
        """Compute the total weighted loss.

        Args:
            outputs: dict from HLFormer_VMR.forward()
            targets: dict from prepare_batch_inputs() - has span_labels, saliency_*

        Returns:
            losses: dict of individual loss tensors (used for logging)
            total:  scalar weighted sum
        """
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        if self.use_matcher:
            indices = self.matcher(outputs_no_aux, targets)
        else:
            indices = None

        losses = {}

        loss_names = self.losses if self.use_matcher else ["saliency"]
        for loss_name in loss_names:
            losses.update(self._get_loss(loss_name, outputs_no_aux, targets, indices))

        # Boundary refinement loss (final layer only — uses same indices as coarse decoder)
        if outputs_no_aux.get("pred_spans_refined") is not None and indices is not None:
            losses.update(self.loss_spans_refined(outputs_no_aux, targets, indices))

        # Auxiliary decoder layer losses (all except saliency)
        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                if self.use_matcher:
                    aux_indices = self.matcher(aux_out, targets)
                else:
                    aux_indices = None

                for loss_name in (self.losses if self.use_matcher else []):
                    if loss_name == "saliency":
                        continue
                    l_dict = self._get_loss(loss_name, aux_out, targets, aux_indices)
                    losses.update({f"{k}_{i}": v for k, v in l_dict.items()})

        # Weighted sum
        total = sum(
            self.weight_dict[k] * v
            for k, v in losses.items()
            if k in self.weight_dict
        )
        return losses, total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_loss(self, name, outputs, targets, indices):
        loss_map = {
            "spans":       self.loss_spans,
            "labels":      self.loss_labels,
            "saliency":    self.loss_saliency,
            "contrastive": self.loss_contrastive_align,
        }
        assert name in loss_map, f"Unknown loss: {name}"
        return loss_map[name](outputs, targets, indices)

    def _src_permutation_idx(self, indices):
        """Return (batch_idx, src_idx) tensors for all matched predictions."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_criterion(cfg):
    """Build the VMRSetCriterion from a config dict.

    Required config keys:
        set_cost_class, set_cost_span, set_cost_giou
        span_loss_coef, giou_loss_coef, label_loss_coef, lw_saliency
        eos_coef, saliency_margin, max_v_l
        aux_loss       (optional, default False) – include aux decoder losses
        dec_layers     (optional) – number of decoder layers
    """
    from VMR.Models.matcher import build_matcher

    matcher = build_matcher(cfg)

    weight_dict = {
        "loss_span":     cfg["span_loss_coef"],
        "loss_giou":     cfg["giou_loss_coef"],
        "loss_boundary": cfg.get("boundary_loss_coef", 0.0),  # 0.0 = disabled unless set
        "loss_label":    cfg["label_loss_coef"],
        "loss_saliency": cfg["lw_saliency"],
    }
    # Boundary refinement losses (final decoder layer only — not propagated to aux layers)
    weight_dict["loss_boundary_refined"] = cfg.get("boundary_refine_coef",      0.0)
    weight_dict["loss_giou_refined"]     = cfg.get("boundary_refine_giou_coef", 0.0)

    losses = ["spans", "labels", "saliency"]
    if cfg.get("use_contrastive", False):
        weight_dict["loss_contrastive_align"] = cfg.get("contrastive_align_loss_coef", 0.1)
        losses.append("contrastive")

    # Aux decoder layers: same loss keys as final layer (except saliency), scaled by
    # aux_loss_scale to prevent early layers from over-fitting to independent solutions.
    # Default scale=1.0 preserves original behaviour.
    _final_only = {"loss_saliency", "loss_boundary_refined", "loss_giou_refined"}
    if cfg.get("aux_loss", False):
        n_aux      = cfg.get("dec_layers", 2) - 1
        _aux_scale = cfg.get("aux_loss_scale", 1.0)
        aux_weights = {
            f"{k}_{i}": v * _aux_scale
            for i in range(n_aux)
            for k, v in weight_dict.items()
            if k not in _final_only
        }
        weight_dict.update(aux_weights)

    criterion = VMRSetCriterion(
        matcher        = matcher,
        weight_dict    = weight_dict,
        eos_coef       = cfg.get("eos_coef", 0.1),
        losses         = losses,
        saliency_margin= cfg.get("saliency_margin", 1.0),
        use_matcher    = True,
        temperature    = cfg.get("temperature", 0.07),
        label_smoothing= cfg.get("label_smoothing", 0.0),
        alpha_iou_alpha= cfg.get("alpha_iou_alpha", 2.0),
        label_span_source=cfg.get("label_span_source", "coarse"),
    )
    return criterion
