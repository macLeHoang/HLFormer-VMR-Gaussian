"""
Hungarian matcher for bipartite assignment between predicted and target spans.
Adapted from QD-DETR (https://github.com/wjun0830/QD-DETR).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from VMR.Models.span_utils import generalized_temporal_iou, span_cxw_to_xx


class HungarianMatcher(nn.Module):
    """Solves the bipartite matching between model predictions and ground-truth spans.

    In general there are more predictions (num_queries) than targets.
    We do a 1-to-1 optimal matching; unmatched predictions are treated as background.

    Args:
        cost_class: weight for the classification cost term
        cost_span:  weight for the L1 span distance cost term
        cost_giou:  weight for the GIoU cost term
        max_v_l:    maximum number of video clips (only needed for CE span loss type)
    """

    def __init__(self, cost_class: float = 1.0, cost_span: float = 1.0,
                 cost_giou: float = 1.0, max_v_l: int = 75,
                 match_span_source: str = "coarse",
                 refined_cost_weight: float = 0.5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span  = cost_span
        self.cost_giou  = cost_giou
        self.max_v_l    = max_v_l
        self.match_span_source = match_span_source
        self.refined_cost_weight = refined_cost_weight
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, \
            "At least one cost term must be non-zero"
        assert self.match_span_source in {"coarse", "refined", "dual"}, \
            "match_span_source must be 'coarse', 'refined', or 'dual'"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Compute the optimal assignment.

        Args:
            outputs: dict with keys
                "pred_spans":  (B, num_queries, 2)  – (center, width) normalized
                "pred_logits": (B, num_queries)    – quality/ranking logit (single value per query)
            targets: dict with key
                "span_labels": list of B dicts, each {"spans": (num_gt, 2)}

        Returns:
            list of B tuples (pred_indices, tgt_indices), both LongTensors
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        tgt_list = targets["span_labels"]

        # Use sigmoid of the per-query quality/ranking logit. This score is used
        # for relative matching preference; weighted BCE in the criterion does not
        # guarantee that it is a calibrated probability or unbiased IoU estimate.
        out_quality = outputs["pred_logits"].flatten(0, 1).sigmoid()            # (B*Q,)
        coarse_spans = outputs["pred_spans"].flatten(0, 1)                      # (B*Q, 2)
        refined_spans = outputs.get("pred_spans_refined", None)
        if refined_spans is not None:
            refined_spans = refined_spans.flatten(0, 1)

        tgt_spans = torch.cat([t["spans"] for t in tgt_list], dim=0)          # (total_gt, 2)

        # Classification cost: -quality_score (higher quality = lower cost = preferred match)
        cost_class = -out_quality.unsqueeze(1).expand(-1, len(tgt_spans))     # (B*Q, total_gt)

        # Span source selection for matching cost
        if self.match_span_source == "refined" and refined_spans is not None:
            match_spans = refined_spans
            cost_span = torch.cdist(match_spans, tgt_spans, p=1)
            cost_giou = -generalized_temporal_iou(
                span_cxw_to_xx(match_spans),
                span_cxw_to_xx(tgt_spans)
            )
        elif self.match_span_source == "dual" and refined_spans is not None:
            cost_span_coarse = torch.cdist(coarse_spans, tgt_spans, p=1)
            cost_giou_coarse = -generalized_temporal_iou(
                span_cxw_to_xx(coarse_spans),
                span_cxw_to_xx(tgt_spans)
            )
            cost_span_refined = torch.cdist(refined_spans, tgt_spans, p=1)
            cost_giou_refined = -generalized_temporal_iou(
                span_cxw_to_xx(refined_spans),
                span_cxw_to_xx(tgt_spans)
            )
            w = float(self.refined_cost_weight)
            cost_span = (1.0 - w) * cost_span_coarse + w * cost_span_refined
            cost_giou = (1.0 - w) * cost_giou_coarse + w * cost_giou_refined
        else:
            # coarse mode, or fallback when refined spans are unavailable (e.g., aux outputs)
            match_spans = coarse_spans
            cost_span = torch.cdist(match_spans, tgt_spans, p=1)
            cost_giou = -generalized_temporal_iou(
                span_cxw_to_xx(match_spans),
                span_cxw_to_xx(tgt_spans)
            )

        # Combined cost matrix
        C = (self.cost_class * cost_class
             + self.cost_span  * cost_span
             + self.cost_giou  * cost_giou)      # (B*Q, total_gt)
        C = C.view(bs, num_queries, -1).cpu()    # (B, Q, total_gt)

        sizes = [len(t["spans"]) for t in tgt_list]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, dim=-1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


def build_matcher(cfg):
    """Factory function.

    Args:
        cfg: dict-like with keys set_cost_class, set_cost_span, set_cost_giou, max_v_l
    """
    return HungarianMatcher(
        cost_class=cfg["set_cost_class"],
        cost_span=cfg["set_cost_span"],
        cost_giou=cfg["set_cost_giou"],
        max_v_l=cfg["max_v_l"],
        match_span_source=cfg.get("match_span_source", "coarse"),
        refined_cost_weight=cfg.get("refined_cost_weight", 0.5),
    )
