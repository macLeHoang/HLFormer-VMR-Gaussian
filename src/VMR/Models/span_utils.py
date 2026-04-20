"""
Temporal span utility functions for Video Moment Retrieval.
Ported from QD-DETR (https://github.com/wjun0830/QD-DETR).
"""
import torch


def span_xx_to_cxw(xx_spans):
    """Convert spans from [start, end] to [center, width] format.

    Args:
        xx_spans: tensor (..., 2), each row is (start, end) normalized in [0,1]

    Returns:
        cxw_spans: tensor (..., 2), each row is (center, width)

    >>> spans = torch.Tensor([[0, 1], [0.2, 0.4]])
    >>> span_xx_to_cxw(spans)
    tensor([[0.5000, 1.0000],
        [0.3000, 0.2000]])
    """
    center = xx_spans.sum(-1) * 0.5
    width = xx_spans[..., 1] - xx_spans[..., 0]
    return torch.stack([center, width], dim=-1)


def span_cxw_to_xx(cxw_spans):
    """Convert spans from [center, width] to [start, end] format.

    Args:
        cxw_spans: tensor (..., 2), each row is (center, width)

    Returns:
        xx_spans: tensor (..., 2), each row is (start, end)

    >>> spans = torch.Tensor([[0.5, 1.0], [0.3, 0.2]])
    >>> span_cxw_to_xx(spans)
    tensor([[0.0000, 1.0000],
        [0.2000, 0.4000]])
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)


def temporal_iou(spans1, spans2):
    """Compute pairwise temporal IoU.

    Args:
        spans1: (N, 2) tensor, each row is [start, end]
        spans2: (M, 2) tensor

    Returns:
        iou:   (N, M) tensor
        union: (N, M) tensor
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N,)
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M,)

    left  = torch.max(spans1[:, None, 0], spans2[:, 0])   # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])   # (N, M)

    inter = (right - left).clamp(min=0)                    # (N, M)
    union = areas1[:, None] + areas2 - inter               # (N, M)

    iou = inter / union.clamp(min=1e-6)
    return iou, union


def temporal_intersection_over_pred(gt_spans, pred_spans):
    """Intersection divided by the prediction span length.

    Args:
        gt_spans:   (N, 2)
        pred_spans: (M, 2)

    Returns:
        inter_over_pred: (N, M)
    """
    left  = torch.max(gt_spans[:, None, 0], pred_spans[:, 0])
    right = torch.min(gt_spans[:, None, 1], pred_spans[:, 1])
    inter = (right - left).clamp(min=0)
    pred_len = (pred_spans[:, 1] - pred_spans[:, 0]).clamp(min=1e-6)
    return inter / pred_len


def generalized_temporal_iou(spans1, spans2):
    """Generalized temporal IoU (GIoU), analogous to spatial GIoU in DETR.

    Args:
        spans1: (N, 2) tensor in [start, end] format
        spans2: (M, 2) tensor in [start, end] format

    Returns:
        giou: (N, M) tensor in [-1, 1]
    """
    spans1 = spans1.float().nan_to_num(0.0)
    spans2 = spans2.float().nan_to_num(0.0)
    # Ensure start <= end (NaN or negative-width spans are clamped to valid range)
    spans1 = torch.stack([spans1[:, 0], torch.maximum(spans1[:, 0], spans1[:, 1])], dim=1)
    spans2 = torch.stack([spans2[:, 0], torch.maximum(spans2[:, 0], spans2[:, 1])], dim=1)

    iou, union = temporal_iou(spans1, spans2)

    left           = torch.min(spans1[:, None, 0], spans2[:, 0])   # (N, M)
    right          = torch.max(spans1[:, None, 1], spans2[:, 1])   # (N, M)
    enclosing_area = (right - left).clamp(min=1e-6)                 # (N, M)

    return iou - (enclosing_area - union) / enclosing_area
