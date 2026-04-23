"""
GaussianFormer-VMR: GaussianFormer extended for Video Moment Retrieval.

Architecture:
  1. Feature projection: vid/txt -> hidden_size  (LinearLayer + TrainablePosEnc)
  2. Early Query Fusion (T2V Encoder, QD-DETR style):
       [global_token | video] as query, text as key/value.
       Global token is a learned parameter prepended to the video sequence;
       it accumulates cross-modal context through all T2V layers.
       After T2V: global token extracted for saliency, video passed to GaussianBlock.
  3. Video Encoder:   GaussianBlock — multi-scale Euclidean Gaussian attention with
                      per-position learned branch fusion.
  4. Decoder: VMRDecoder — iterative reference-point refinement (QD-DETR style).
       - VMRDecoderLayer with sine-conditioned cross-attention queries.
       - bbox_embed MLP refines reference points at each layer.
       - query_scale MLP conditions query positional signal on content.
       - Content-adaptive query init (CAQ): each slot attends over vid_feat
         conditioned on txt_mean before entering the decoder (v29).
       - Auxiliary outputs collected for intermediate supervision.
  5. Prediction heads: class head (fg/bg) + spans from bbox_embed.
  6. Saliency head: dot-product scoring with global_rep from T2V.
  7. Contrastive alignment (optional): NCE loss aligning query slots with text.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

# --- Reuse HLFormer components ---
from Models.HLFormer.model_components import (
    LinearLayer,
    TrainablePositionalEncoding,
    GaussianBlock,
)


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Drop-in replacement for TrainablePositionalEncoding:
    same constructor signature and forward interface.
    Embeddings are fixed buffers — no learned parameters.
    """

    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout   = nn.Dropout(dropout)

        pe       = torch.zeros(max_position_embeddings, hidden_size)
        position = torch.arange(0, max_position_embeddings,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)              # (max_len, H), not a parameter

    def forward(self, input_feat):
        seq_len    = input_feat.shape[1]
        embeddings = self.LayerNorm(input_feat + self.pe[:seq_len])
        return self.dropout(embeddings)


def _build_pos_enc(enc_type, max_len, hidden_size, dropout):
    """Return a positional encoding module by name.

    Args:
        enc_type:   "trainable" | "sinusoidal"
        max_len:    maximum sequence length
        hidden_size: embedding dimension
        dropout:    dropout probability
    """
    if enc_type == "sinusoidal":
        return SinusoidalPositionalEncoding(max_len, hidden_size, dropout)
    if enc_type == "trainable":
        return TrainablePositionalEncoding(max_len, hidden_size, dropout)
    raise ValueError(f"pos_enc_type must be 'trainable' or 'sinusoidal', got '{enc_type}'")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def inverse_sigmoid(x, eps=1e-3):
    x  = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor, H=256):
    """Sine positional embeddings from normalized (center, width) reference points.

    Args:
        pos_tensor: (B, Q, 2) — normalized center and width in [0, 1]
        H:          output embedding dimension (must be even)

    Returns:
        (B, Q, H) — first H/2 dims encode center, last H/2 dims encode width
    """
    half_H = H // 2
    dim_t  = torch.arange(half_H, dtype=torch.float32, device=pos_tensor.device)
    dim_t  = 10000 ** (2 * (dim_t // 2) / half_H)

    scale  = 2 * math.pi
    center = pos_tensor[:, :, 0:1] * scale / dim_t   # (B, Q, half_H)
    width  = pos_tensor[:, :, 1:2] * scale / dim_t   # (B, Q, half_H)

    center = torch.stack([center[..., 0::2].sin(),
                          center[..., 1::2].cos()], dim=-1).flatten(-2)
    width  = torch.stack([width[..., 0::2].sin(),
                          width[..., 1::2].cos()],  dim=-1).flatten(-2)

    return torch.cat([center, width], dim=-1)         # (B, Q, H)


# ---------------------------------------------------------------------------
# Auxiliary modules
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Feed-forward MLP with ReLU activations on all but the last layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# T2V Encoder (Text-to-Video, QD-DETR style)
# ---------------------------------------------------------------------------

class T2VEncoderLayer(nn.Module):
    """One layer of the Text-to-Video encoder.

    [global | video] tokens are the query; text tokens are key and value.
    Post-LN Transformer block: cross-attn -> residual+LN -> FFN -> residual+LN.
    Uses a 2-D padding-aware attention mask.
    """

    def __init__(self, hidden_size, n_heads, dropout=0.1, dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_size
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)
        self.norm1   = nn.LayerNorm(hidden_size)
        self.norm2   = nn.LayerNorm(hidden_size)
        self.drop    = nn.Dropout(dropout)
        self.drop1   = nn.Dropout(dropout)
        self.drop2   = nn.Dropout(dropout)
        self.act     = nn.PReLU()

    def forward(self, vid_feat, txt_feat,
                txt_key_padding_mask=None, attn_mask=None):
        vid2, _ = self.cross_attn(
            query=vid_feat, key=txt_feat, value=txt_feat,
            key_padding_mask=txt_key_padding_mask,
            attn_mask=attn_mask)
        vid_feat = self.norm1(vid_feat + self.drop1(vid2))
        vid2     = self.linear2(self.drop(self.act(self.linear1(vid_feat))))
        vid_feat = self.norm2(vid_feat + self.drop2(vid2))
        return vid_feat


# ---------------------------------------------------------------------------
# VMR Decoder (iterative reference-point refinement, QD-DETR style)
# ---------------------------------------------------------------------------

class VMRDecoderLayer(nn.Module):
    """Custom decoder layer with sine-conditioned cross-attention.

    Self-attention:
        Q = sa_qcontent(tgt) + sa_qpos(query_pos)   [H]
        K = sa_kcontent(tgt) + sa_kpos(query_pos)   [H]
        V = sa_v(tgt)                                [H]

    Cross-attention (sine-conditioned Q):
        Q = ca_qcontent(tgt) + ca_qpos_sine(sine_embed)   [H]
        K = ca_kcontent(memory)                            [H]
        V = ca_v(memory)                                   [H]

    FFN: Linear(H, 4H) -> ReLU -> Linear(4H, H)
    """

    def __init__(self, H, n_heads, dropout=0.1):
        super().__init__()
        # Self-attention projections
        self.sa_qcontent = nn.Linear(H, H)
        self.sa_qpos     = nn.Linear(H, H)
        self.sa_kcontent = nn.Linear(H, H)
        self.sa_kpos     = nn.Linear(H, H)
        self.sa_v        = nn.Linear(H, H)
        self.self_attn   = nn.MultiheadAttention(
            H, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention projections (sine-conditioned query)
        self.ca_qcontent   = nn.Linear(H, H)
        self.ca_qpos_sine  = nn.Linear(H, H)
        self.ca_kcontent   = nn.Linear(H, H)
        self.ca_v          = nn.Linear(H, H)
        self.cross_attn    = nn.MultiheadAttention(
            H, n_heads, dropout=dropout, batch_first=True)

        # FFN
        self.ffn_linear1 = nn.Linear(H, 4 * H)
        self.ffn_linear2 = nn.Linear(4 * H, H)
        self.ffn_act     = nn.ReLU()
        self.ffn_drop    = nn.Dropout(dropout)

        # Norms + drops
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)
        self.norm3 = nn.LayerNorm(H)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos, sine_embed,
                memory_key_padding_mask=None, cross_attn_bias=None):
        """
        Args:
            tgt:                    (B, Q, H)
            memory:                 (B, L, H) — encoder memory [video | text]
            query_pos:              (B, Q, H) — positional signal from ref_point_head
            sine_embed:             (B, Q, H) — sine embeddings from reference points
            memory_key_padding_mask:(B, L) bool, True = padded
            cross_attn_bias:        (B, L_mem) optional additive bias over memory keys
        """
        # Self-attention
        q  = self.sa_qcontent(tgt) + self.sa_qpos(query_pos)
        k  = self.sa_kcontent(tgt) + self.sa_kpos(query_pos)
        v  = self.sa_v(tgt)
        tgt2, _ = self.self_attn(q, k, v)
        tgt = self.norm1(tgt + self.drop1(tgt2))

        # Cross-attention (sine-conditioned)
        q2  = self.ca_qcontent(tgt) + self.ca_qpos_sine(sine_embed)
        k2  = self.ca_kcontent(memory)
        v2  = self.ca_v(memory)
        # Build additive attn_mask for cross-attention if saliency bias provided
        ca_attn_mask = None
        if cross_attn_bias is not None:
            B, L_mem = cross_attn_bias.shape
            nh = self.cross_attn.num_heads
            Lq = tgt.shape[1]  # batch_first=True, so dim 1 is query length
            bias = cross_attn_bias.view(B, 1, 1, L_mem).expand(B, nh, Lq, L_mem)
            ca_attn_mask = bias.reshape(B * nh, Lq, L_mem)
        tgt2, _ = self.cross_attn(q2, k2, v2,
                                   attn_mask=ca_attn_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.drop2(tgt2))

        # FFN
        tgt2 = self.ffn_linear2(
            self.ffn_drop(self.ffn_act(self.ffn_linear1(tgt))))
        tgt  = self.norm3(tgt + self.drop3(tgt2))
        return tgt


class VMRDecoder(nn.Module):
    """Iterative decoder: refines reference points (span predictions) at each layer."""

    def __init__(self, decoder_layer, num_layers, H):
        super().__init__()
        self.layers          = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm            = nn.LayerNorm(H)
        self.ref_point_head  = nn.ModuleList([MLP(H, H, H, 2) for _ in range(num_layers)])   # sine_embed -> query pos signal
        # Per-layer query_scale: each decoder layer conditions its own positional
        # scaling on content, matching the per-layer bbox_embed design.
        self.query_scale     = nn.ModuleList([MLP(H, H, H, 2) for _ in range(num_layers)])
        # Per-layer bbox_embed: each decoder layer predicts its own Δ(center, width).
        # Sharing a single MLP conflates coarse (layer 0) and fine (layer N-1) gradients.
        self.bbox_embed      = nn.ModuleList([MLP(H, H, 2, 3) for _ in range(num_layers)])
        self._H              = H

        # Zero-init final layer of every bbox_embed for stable training start
        for embed in self.bbox_embed:
            nn.init.zeros_(embed.layers[-1].weight)
            nn.init.zeros_(embed.layers[-1].bias)

    def forward(self, tgt, memory, memory_key_padding_mask, reference_points,
                cross_attn_bias=None):
        """
        Args:
            tgt:                    (B, Q, H)  — initial query content
            memory:                 (B, L, H)  — encoder memory
            memory_key_padding_mask:(B, L) bool — True = padded
            reference_points:       (B, Q, 2)  — initial refs in [0, 1]
            cross_attn_bias:        (B, L_mem) optional additive bias over memory keys

        Returns:
            final_out:       (B, Q, H)       — normed last-layer output
            aux_list:        list[(B, Q, H)] — intermediate outputs (unnormed)
            spans_per_layer: list[(B, Q, 2)] — span prediction per layer
        """
        aux_list        = []
        spans_per_layer = []

        for i, layer in enumerate(self.layers):
            # Positional signal from current reference points
            sine_embed = gen_sineembed_for_position(
                reference_points, self._H)          # (B, Q, H)
            query_pos  = self.ref_point_head[i](sine_embed)    # (B, Q, H)
            scale      = self.query_scale[i](tgt)            # (B, Q, H)
            query_pos  = query_pos * scale

            tgt = layer(tgt, memory, query_pos, sine_embed,
                        memory_key_padding_mask=memory_key_padding_mask,
                        cross_attn_bias=cross_attn_bias)

            # Refine reference points — use this layer's own bbox_embed
            delta          = self.bbox_embed[i](tgt)                         # (B, Q, 2)
            new_refs       = (inverse_sigmoid(reference_points) + delta).sigmoid()
            spans_per_layer.append(new_refs)

            if i < len(self.layers) - 1:
                aux_list.append(tgt)                 # save unnormed for aux loss
            reference_points = new_refs   # gradient flows through all layers for high-IoU precision

        return self.norm(tgt), aux_list, spans_per_layer


# ---------------------------------------------------------------------------
# Multi-stream video feature projection
# ---------------------------------------------------------------------------

class MultiStreamVidProjection(nn.Module):
    """Project heterogeneous video feature streams independently, then fuse.

    Each stream gets its own LinearLayer (stream_dim → H), then all projected
    streams are concatenated and fused through a single Linear+LN+GELU layer.

    Handles any number of streams dynamically:
      - 1 stream  → single LinearLayer, no fusion (Identity)
      - N streams → N LinearLayers + Linear(N*H → H) fusion

    Args:
        stream_dims : list[int]  per-stream feature dimensions, must match
                                 the concatenation order in the input tensor
                                 e.g. [2304, 512, 768] for SlowFast+CLIP+BLIP
                                 e.g. [512]            for CLIP-only
        H           : int        target hidden size
        dropout     : float      dropout probability for each LinearLayer

    Forward:
        x : (B, L, sum(stream_dims))
        returns (B, L, H)
    """

    def __init__(self, stream_dims: list, H: int, dropout: float = 0.1):
        super().__init__()
        assert len(stream_dims) >= 1, "stream_dims must have at least one element"
        self.stream_dims = stream_dims

        self.stream_projs = nn.ModuleList([
            LinearLayer(dim, H, layer_norm=True, dropout=dropout, relu=True)
            for dim in stream_dims
        ])

        n = len(stream_dims)
        if n > 1:
            # Learned per-stream importance logits; zeros → uniform (= current behaviour) at init.
            # softmax(stream_logits) is inspectable after training to see which stream dominates.
            self.stream_logits = nn.Parameter(torch.zeros(n))
            self.fusion = nn.Sequential(
                nn.Linear(n * H, H, bias=False),
                nn.LayerNorm(H),
                nn.GELU(),
            )
        else:
            self.fusion = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        streams = x.split(self.stream_dims, dim=-1)           # tuple of (B, L, dim_i)
        projected = [proj(s) for proj, s in zip(self.stream_projs, streams)]
        if len(projected) > 1:
            alpha = F.softmax(self.stream_logits, dim=0)      # (n,) — learned stream weights
            projected = [a * p for a, p in zip(alpha, projected)]
            return self.fusion(torch.cat(projected, dim=-1))  # (B, L, H)
        return projected[0]                                   # (B, L, H)


# ---------------------------------------------------------------------------
# Boundary Refinement Head
# ---------------------------------------------------------------------------

class BoundaryRefinementHead(nn.Module):
    """Post-decoder local boundary refinement.

    Pools video features around predicted start/end boundaries using
    differentiable Gaussian soft-attention, then predicts a clamped delta
    offset via a small MLP.  The head operates in (start, end) space and
    converts back to (center, width) on return.

    Sigma is conditioned on both span width (geometric) and a cross-modal
    text representation (semantic).  Wider spans get a wider pooling window;
    queries with semantically diffuse descriptions also widen their window.
    All conditioning terms are zero-initialized so training starts from
    scalar-sigma behaviour and learns the conditioning gradually.

    Args:
        H:             Model hidden size.
        window_frames: Number of frames defining the base Gaussian sigma
                       (sigma_base = window_frames / (2 * max_v_l)).  Default 8.
        max_delta:     Maximum absolute shift of each boundary in normalized
                       [0, 1] coords.  tanh gates keep deltas in
                       (−max_delta, +max_delta).  Default 0.1 (≈12 frames at L=128).
    """

    def __init__(self, H: int, window_frames: int = 8, max_delta: float = 0.1,
                 max_v_l: int = 128, learnable_sigma: bool = True,
                 num_passes: int = 1):
        super().__init__()
        self.max_delta        = max_delta
        self.learnable_sigma  = learnable_sigma
        self.num_passes       = max(int(num_passes), 1)
        self._H               = H
        self._sigma_ref_len   = float(max_v_l)
        # Base Gaussian sigma per boundary in normalized [0,1] space.
        # Initialised so sigma ≈ window_frames / (2 * max_v_l).
        init_log_sigma = math.log(max(window_frames / (2.0 * max_v_l), 1e-6))
        if learnable_sigma:
            self.log_sigma_start = nn.Parameter(torch.tensor(init_log_sigma))
            self.log_sigma_end   = nn.Parameter(torch.tensor(init_log_sigma))
            # Width-scale: multiplies predicted span width in log-sigma space.
            # Zero-init → starts as pure base sigma, grows with training.
            self.sigma_width_scale_start = nn.Parameter(torch.tensor(0.0))
            self.sigma_width_scale_end   = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_sigma_start", torch.tensor(init_log_sigma))
            self.register_buffer("log_sigma_end",   torch.tensor(init_log_sigma))
            self.register_buffer("sigma_width_scale_start", torch.tensor(0.0))
            self.register_buffer("sigma_width_scale_end",   torch.tensor(0.0))
        # Text-conditioned sigma offset: maps global cross-modal rep → 2 scalars
        # (one per boundary).  Zero-init → text starts contributing nothing.
        self.sigma_txt_proj = nn.Linear(H, 2, bias=True)
        # Joint MLP: takes concatenated (start_feat, end_feat, txt_rep) → (delta_s, delta_e).
        # Text context lets the MLP decide *which direction* to shift based on query semantics.
        # txt_rep defaults to zeros when not provided (backward-compat with old call sites).
        self.joint_mlp = MLP(3 * H, H, 2, 2)

    def forward(
        self,
        pred_spans: torch.Tensor,          # (B, Q, 2)  center/width in [0, 1]
        vid_feat:   torch.Tensor,          # (B, L, H)
        vid_mask:   torch.Tensor,          # (B, L)     1=valid
        txt_rep:    torch.Tensor = None,   # (B, H)     cross-modal global rep (optional)
    ):                                     # -> (Tensor(B,Q,2), list[Tensor(B,Q,2)])
        B, Q, _ = pred_spans.shape
        L, H    = vid_feat.shape[1], vid_feat.shape[2]
        dev     = vid_feat.device

        start = (pred_spans[..., 0] - pred_spans[..., 1] / 2).clamp(0., 1.)  # (B, Q)
        end   = (pred_spans[..., 0] + pred_spans[..., 1] / 2).clamp(0., 1.)  # (B, Q)

        # Pre-compute fixed quantities shared across passes.
        t_pos   = torch.linspace(0., 1., L, device=dev)         # (L,)
        min_sig = 1.0 / (4.0 * L)                               # floor: 0.25 frames

        if txt_rep is not None:
            # Bound text offset to ±0.5 * log_range so sigma_txt_proj can't explode.
            _log_range = math.log(max(float(L) * min_sig * 4.0, 1.0 + 1e-6))
            txt_off = torch.tanh(self.sigma_txt_proj(txt_rep)) * (0.5 * _log_range)  # (B, 2)
            txt_off_s = txt_off[:, 0:1]                          # (B, 1)
            txt_off_e = txt_off[:, 1:2]                          # (B, 1)
            txt_expand = txt_rep.unsqueeze(1).expand(-1, Q, -1)  # (B, Q, H)
        else:
            txt_off_s = txt_off_e = torch.zeros(B, 1, device=dev, dtype=vid_feat.dtype)
            txt_expand = torch.zeros(B, Q, H, device=dev, dtype=vid_feat.dtype)

        def _pool(anchor, sigma):
            # anchor: (B, Q), sigma: (B, Q), returns (B, Q, H)
            dist2 = ((t_pos[None, None, :] - anchor[:, :, None]) / sigma[:, :, None]) ** 2
            w = torch.exp(-0.5 * dist2)                       # (B, Q, L)
            w = w * vid_mask[:, None, :].float()
            w = w / (w.sum(-1, keepdim=True).clamp(min=1e-8))
            return torch.einsum("bql,blh->bqh", w, vid_feat)  # (B, Q, H)

        all_passes = []  # list of (B, Q, 2) tensors, one per pass

        for _pass_idx in range(self.num_passes):
            # Per-pass sigma: recomputed from current (start, end) so wider spans
            # get a wider pooling window each iteration.
            width = (end - start).clamp(min=1e-6)              # (B, Q)

            log_sigma_s = (self.log_sigma_start
                           + self.sigma_width_scale_start * width
                           + txt_off_s)                        # (B, Q)
            log_sigma_e = (self.log_sigma_end
                           + self.sigma_width_scale_end   * width
                           + txt_off_e)                        # (B, Q)
            sigma_s = log_sigma_s.exp().clamp(min=min_sig)     # (B, Q)
            sigma_e = log_sigma_e.exp().clamp(min=min_sig)     # (B, Q)

            start_feat = _pool(start, sigma_s)                 # (B, Q, H)
            end_feat   = _pool(end,   sigma_e)                 # (B, Q, H)

            # Joint prediction: start and end deltas conditioned on both
            # boundaries + text.  Shared weights across passes.
            joint_feat = torch.cat([start_feat, end_feat, txt_expand], dim=-1)  # (B, Q, 3H)
            deltas     = torch.tanh(self.joint_mlp(joint_feat)) * self.max_delta  # (B, Q, 2)
            delta_s    = deltas[..., 0]                        # (B, Q)
            delta_e    = deltas[..., 1]                        # (B, Q)

            start_r  = (start + delta_s).clamp(0., 1.)
            end_r    = (end   + delta_e).clamp(0., 1.)
            center_r = (start_r + end_r) / 2.0
            width_r  = (end_r - start_r).clamp(min=1e-6)
            pass_span = torch.stack([center_r, width_r], dim=-1)  # (B, Q, 2)
            all_passes.append(pass_span)

            # Update boundaries for next pass.
            start = start_r
            end   = end_r

        final_span_cxw = all_passes[-1]                        # (B, Q, 2)
        return final_span_cxw, all_passes


# ---------------------------------------------------------------------------
# GaussianFormer-VMR
# ---------------------------------------------------------------------------

class GaussianFormer_VMR(nn.Module):
    """Video Moment Retrieval model built on GaussianBlock encoder backbone.

    Args:
        config: edict / dict-like with all hyperparameters
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        H = config.hidden_size

        # ---- Video feature projection -----------------------------------
        #  Two modes controlled by config:
        #
        #  Multi-stream (recommended):
        #    config.v_feat_dims = [dim1, dim2, ...]  e.g. [2304, 512, 768]
        #    → MultiStreamVidProjection: one LinearLayer per stream + fusion.
        #    → Works for any subset: CLIP-only [512], SF+CLIP [2304,512], etc.
        #    → config.v_feat_dim is auto-set to sum(v_feat_dims) if absent.
        #
        #  Single-stream (legacy / backward-compat):
        #    config.v_feat_dims absent or single-element list
        #    → stacked n_input_proj LinearLayers: feat_dim→H→H (QD-DETR style).
        _stream_dims = getattr(config, "v_feat_dims", None)
        if _stream_dims and len(_stream_dims) > 1:
            self.input_vid_proj = MultiStreamVidProjection(
                _stream_dims, H, dropout=config.input_drop)
        else:
            # Legacy single-stream: n_input_proj stacked layers
            _n_proj    = getattr(config, "n_input_proj", 2)
            _relu_args = [True] * _n_proj
            if _n_proj >= 3:
                _relu_args[-1] = False
            _in_dim = _stream_dims[0] if _stream_dims else config.v_feat_dim
            self.input_vid_proj = nn.Sequential(*[
                LinearLayer(_in_dim if i == 0 else H, H,
                            layer_norm=True, dropout=config.input_drop,
                            relu=_relu_args[i])
                for i in range(_n_proj)
            ])

        # ---- Text feature projection ------------------------------------
        #  Same multi-stream logic as video: set config.t_feat_dims as a list
        #  e.g. [512, 768] for CLIP+BLIP text, [512] for CLIP-only.
        #  Falls back to stacked n_input_proj layers when t_feat_dims is absent.
        _txt_stream_dims = getattr(config, "t_feat_dims", None)
        if _txt_stream_dims and len(_txt_stream_dims) > 1:
            self.input_txt_proj = MultiStreamVidProjection(
                _txt_stream_dims, H, dropout=config.input_drop)
        else:
            _n_proj_t    = getattr(config, "n_input_proj", 2)
            _relu_args_t = [True] * _n_proj_t
            if _n_proj_t >= 3:
                _relu_args_t[-1] = False
            _in_dim_t = _txt_stream_dims[0] if _txt_stream_dims else config.t_feat_dim
            self.input_txt_proj = nn.Sequential(*[
                LinearLayer(_in_dim_t if i == 0 else H, H,
                            layer_norm=True, dropout=config.input_drop,
                            relu=_relu_args_t[i])
                for i in range(_n_proj_t)
            ])

        # ---- Positional encodings ---------------------------------------
        _pos_enc = getattr(config, "pos_enc_type", "trainable")
        self.vid_pos_embed = _build_pos_enc(_pos_enc, config.max_v_l, H, config.input_drop)
        self.txt_pos_embed = _build_pos_enc(_pos_enc, config.max_q_l, H, config.input_drop)

        # ---- Optional text transformer encoder --------------------------
        _txt_enc_layers = int(getattr(config, "txt_enc_layers", 0))
        if _txt_enc_layers > 0:
            txt_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.n_heads,
                dim_feedforward=4 * config.hidden_size,
                dropout=getattr(config, "drop", 0.15),
                batch_first=True,
                norm_first=False,
            )
            self.txt_encoder = nn.TransformerEncoder(txt_layer, num_layers=_txt_enc_layers)
        else:
            self.txt_encoder = None

        # ---- Global token (prepended to [global | video] for T2V) ------
        #  Accumulates cross-modal context through all T2V layers.
        #  Extracted after T2V as the saliency global representation.
        self.global_token = nn.Parameter(torch.randn(H))
        self.global_pos   = nn.Parameter(torch.randn(H))

        # ---- T2V encoder (QD-DETR style, 6 layers) ---------------------
        _t2v_layers = getattr(config, "t2v_layers", 6)
        self.t2v_encoder = nn.ModuleList([
            T2VEncoderLayer(H, config.n_heads, dropout=config.drop)
            for _ in range(_t2v_layers)])

        # ---- Video encoder (GaussianBlock) ---------------------------------
        #  frame_len controls GaussianBlock's internal Gaussian window size.
        #  When use_global_in_encoder=True the global token is prepended to
        #  the video sequence before GaussianBlock, so frame_len = max_v_l+1.
        self._use_global_in_encoder = getattr(config, "use_global_in_encoder", False)
        _frame_len = config.max_v_l + (1 if self._use_global_in_encoder else 0)
        vid_enc_cfg = edict(
            hidden_size=H, intermediate_size=H,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop,
            frame_len=_frame_len,
            sft_factor=config.sft_factor,
            drop=config.drop,
            attention_num=config.attention_num,
            weight_token_mode=getattr(config, "weight_token_mode", "global"),
            weight_token_hybrid_init=getattr(config, "weight_token_hybrid_init", 0.7),
            gauss_bias_mode=getattr(config, "gauss_bias_mode", "add_log"))
        self.video_encoder = GaussianBlock(vid_enc_cfg)

        # ---- Decoder (iterative reference-point refinement) ------------
        decoder_layer = VMRDecoderLayer(H, config.n_heads, dropout=config.drop)
        self.decoder  = VMRDecoder(decoder_layer, config.dec_layers, H)

        # Query slots: content embedding + reference point initialisation
        self.query_content_embed = nn.Embedding(config.num_queries, H)
        self.query_embed         = nn.Embedding(config.num_queries, 2)

        # ---- Content-adaptive query initialisation (v29) -------------------
        # Instead of projecting a single txt_mean to all Q slots identically,
        # each query slot attends over the *video* encoder output conditioned on
        # text via cross-attention.  This gives each slot a unique, content-driven
        # starting point so the decoder iterates from a good initialisation rather
        # than a static learned embedding.
        #
        # Architecture:
        #   caq_q_proj  : (Q, H) learned query seeds for the cross-attention
        #   caq_txt_gate: text-mean projected to H; gated into each query seed
        #                 so different queries are seeded differently per text
        #   caq_attn    : Q-slot cross-attention over video encoder memory,
        #                 with text-gated query seeds as queries and vid_feat
        #                 as keys/values
        #   caq_norm    : LayerNorm on attended output before adding to base embed
        #   caq_ref_head: 2-output MLP on attended features → reference-point
        #                 offset (added to the learned uniform initialisation)
        #
        # Zero-initialised final layers ensure identity start (same as v28 ep0).
        Q = config.num_queries
        self.caq_q_proj   = nn.Embedding(Q, H)          # learned per-slot seeds
        self.caq_txt_gate = nn.Linear(H, Q * H)         # text-mean → Q per-slot gates
        self.caq_attn     = nn.MultiheadAttention(
            embed_dim=H, num_heads=config.n_heads,
            dropout=config.drop, batch_first=True)
        self.caq_norm     = nn.LayerNorm(H)
        # 2-layer MLP: H → H → 2 (reference-point offset in inv-sigmoid space).
        # Receives caq_out (video-attended content) — kept for backward compat.
        self.caq_ref_head = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, 2))
        # Separate text-only reference initializer: txt_mean → (Δcx, Δw).
        # Decouples *what* the query attends to (video content via caq_attn)
        # from *where* it points initially (text prior via caq_ref_init).
        # Zero-init final layer: identity start, offset grows as training progresses.
        self.caq_ref_init = MLP(H, H, 2, 2)

        # ---- Prediction heads ------------------------------------------
        self.class_head  = nn.Linear(H, 1)   # single quality logit; [..0] indexing removed
        # Gate that learns per-query when to trust boundary refinement vs coarse spans.
        # Bias=-1.0 so initial gate≈0.27 — refinement contributes but doesn't dominate at ep0.
        self.refine_gate = nn.Linear(H, 1)

        # ---- Saliency projections --------------------------------------
        self.saliency_proj1 = nn.Linear(H, H)
        self.saliency_proj2 = nn.Linear(H, H)

        # ---- Boundary refinement head (post-decoder local precision) ---
        _win      = getattr(config, "boundary_refine_window",          16)
        _maxd     = getattr(config, "boundary_refine_max_delta",       0.1)
        _learn_sg = getattr(config, "boundary_refine_learnable_sigma", True)
        _npasses  = getattr(config, "refine_num_passes",               1)
        self.boundary_refine = BoundaryRefinementHead(H, _win, _maxd,
                                                      max_v_l=config.max_v_l,
                                                      learnable_sigma=_learn_sg,
                                                      num_passes=_npasses)

        # ---- Contrastive alignment (optional) --------------------------
        self._use_contrastive = getattr(config, "use_contrastive", False)
        if self._use_contrastive:
            hdim = getattr(config, "contrastive_hdim", 64)
            self.ca_proj_query = nn.Linear(H, hdim)
            self.ca_proj_txt   = nn.Linear(H, hdim)
            self.ca_proj_vid   = nn.Linear(H, hdim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        # Re-zero final layer of each per-layer bbox_embed after general init
        for embed in self.decoder.bbox_embed:
            nn.init.zeros_(embed.layers[-1].weight)
            nn.init.zeros_(embed.layers[-1].bias)
        # Zero-init for joint_mlp final layer intentionally removed:
        # zero-init combined with tanh*max_delta clamp starved the head's gradient
        # (deltas=0 at ep0 → tanh saturation near zero → near-zero gradient back through
        # the head for many epochs).  Default Kaiming init lets deltas start nonzero so
        # gradient reaches the head from step 1.  Head contribution is bounded by
        # boundary_refine_coef=0.5 in the loss schedule.
        # Zero-init text+width conditioning terms so ep0 behaviour = scalar base sigma.
        nn.init.zeros_(self.boundary_refine.sigma_txt_proj.weight)
        nn.init.zeros_(self.boundary_refine.sigma_txt_proj.bias)
        if self.boundary_refine.learnable_sigma:
            nn.init.zeros_(self.boundary_refine.sigma_width_scale_start)
            nn.init.zeros_(self.boundary_refine.sigma_width_scale_end)
        # Small-normal init for text portion of joint_mlp first layer (columns 2H:3H).
        # Was zero-init: text input was stuck behind a zero floor and required large
        # gradients to activate. std=0.01 lets text inform refinement from step 1
        # while staying small enough that it doesn't corrupt the ep0 boundary estimate.
        with torch.no_grad():
            _H = self.boundary_refine._H
            nn.init.normal_(self.boundary_refine.joint_mlp.layers[0].weight[:, 2 * _H:],
                            mean=0.0, std=0.01)
        # Zero-init caq_ref_head final layer so initial reference-point offsets=0.
        # This means at ep0 CAQ produces the same reference points as the static
        # uniform linspace init, and the offset grows as training progresses.
        nn.init.zeros_(self.caq_ref_head[-1].weight)
        nn.init.zeros_(self.caq_ref_head[-1].bias)
        # Zero-init caq_ref_init final layer (text-only ref offset MLP) for same reason.
        nn.init.zeros_(self.caq_ref_init.layers[-1].weight)
        nn.init.zeros_(self.caq_ref_init.layers[-1].bias)
        # Zero-init caq_attn output projection so caq_out=0 at ep0.
        # Without this, MHA produces random-magnitude output that corrupts tgt
        # at initialisation (observed: class_error=87% vs v28's 45%).
        # Standard residual-branch identity trick: new branch adds zero, then
        # activates as training progresses.
        nn.init.zeros_(self.caq_attn.out_proj.weight)
        nn.init.zeros_(self.caq_attn.out_proj.bias)
        # Init refine_gate bias to -1.0 so sigmoid(0 * x + (-1)) ≈ 0.27 at ep0.
        # Refinement contributes but doesn't dominate; gate grows as training progresses.
        nn.init.constant_(self.refine_gate.bias, -1.0)
        # Spread decoder query reference points uniformly: cx in [0.1, 0.9], w = 0.3
        # sigmoid(query_embed.weight) is used as the initial reference point in _decode,
        # so we store inv_sigmoid of the desired initial values.
        Q  = self.query_embed.weight.shape[0]
        cx = torch.linspace(0.1, 0.9, Q)
        w  = torch.full((Q,), 0.3)
        self.query_embed.weight.data[:, 0] = cx.logit()   # inv_sigmoid of center
        self.query_embed.weight.data[:, 1] = w.logit()    # inv_sigmoid of width

    # ------------------------------------------------------------------
    def _encode_query(self, src_txt, src_txt_mask):
        """Project + pos-encode + (optionally) transformer-encode text features."""
        txt_feat = self.input_txt_proj(src_txt)
        txt_feat = self.txt_pos_embed(txt_feat)
        if self.txt_encoder is not None:
            kpm = (src_txt_mask == 0)              # True = padded
            txt_feat = self.txt_encoder(txt_feat, src_key_padding_mask=kpm)
        return txt_feat                                      # (B, L_t, H)

    def _early_query_fusion(self, vid_feat, txt_feat, vid_mask, txt_mask):
        """T2V encoder with global token prepended to video sequence.

        [global | video] = Q,  text = K/V.
        Global token accumulates cross-modal context through all T2V layers.

        Args:
            vid_feat: (B, L_v, H) — projected + pos-encoded video features
            txt_feat: (B, L_t, H) — query-encoded text features
            vid_mask: (B, L_v)    — 1=valid, 0=padded
            txt_mask: (B, L_t)    — 1=valid, 0=padded

        Returns:
            vid_feat:   (B, L_v, H) — text-conditioned video features
            global_rep: (B, H)      — cross-modal global representation
        """
        B, L_v, H = vid_feat.shape

        # Prepend global token: [global | video]
        g = (self.global_token + self.global_pos).reshape(1, 1, H).expand(B, 1, H)
        vid_with_g = torch.cat([g, vid_feat], dim=1)            # (B, 1+L_v, H)

        # txt_kpm: True where text is padded — passed as key_padding_mask so
        # PyTorch MHA suppresses padded text keys in cross-attention.
        # The previous 2-D attn_mask (vid_pad & txt_pad) was a no-op: it only
        # fired when BOTH the video query AND the text key were padding, which
        # almost never happens and duplicated what txt_kpm already handles.
        txt_kpm = (txt_mask == 0)                                # (B, L_t) True=pad

        for layer in self.t2v_encoder:
            vid_with_g = layer(vid_with_g, txt_feat,
                               txt_key_padding_mask=txt_kpm)

        global_rep = vid_with_g[:, 0, :]    # (B, H)
        vid_feat   = vid_with_g[:, 1:, :]   # (B, L_v, H)
        return vid_feat, global_rep

    def _decode(self, memory, memory_mask, txt_feat=None, txt_mask=None,
                vid_feat=None, vid_mask=None, cross_attn_bias=None):
        """Run the iterative VMRDecoder with content-adaptive query initialisation.

        Args:
            memory:       (B, L, H)   — encoder output (video [+ text])
            memory_mask:  (B, L)      — 1=valid, 0=padded
            txt_feat:     (B, L_t, H) — text features for query conditioning
            txt_mask:     (B, L_t)    — 1=valid, 0=padded
            vid_feat:     (B, L_v, H) — video-only encoder output (for CAQ attn)
            vid_mask:     (B, L_v)    — 1=valid, 0=padded
            cross_attn_bias: (B, L_mem) optional additive saliency bias over memory keys
        """
        B, Q = memory.shape[0], self.query_content_embed.weight.shape[0]
        mem_pad = (memory_mask == 0)                             # (B, L) True=pad

        # ---- Base content embedding (learned, same as before) ---------------
        tgt = self.query_content_embed.weight \
                  .unsqueeze(0).expand(B, -1, -1).clone()       # (B, Q, H)

        # ---- Content-adaptive query initialisation --------------------------
        # Attend each query slot over the video encoder output, conditioned on
        # the text query.  vid_feat is the video-only encoder output (before
        # cat with text), so cross-attention focuses on temporal positions only.
        if txt_feat is not None and vid_feat is not None:
            # 1. Text-gated query seeds: start from learned per-slot embeddings
            #    and add a text-mean gate so different queries diverge based on
            #    the current text query.
            txt_valid = txt_mask.unsqueeze(-1).float()           # (B, L_t, 1)
            txt_sum   = (txt_feat * txt_valid).sum(1)            # (B, H)
            txt_count = txt_valid.sum(1).clamp(min=1.0)          # (B, 1)
            txt_mean  = txt_sum / txt_count                      # (B, H)

            # caq_txt_gate projects txt_mean to (B, Q*H) → unique gate per slot
            txt_gate = self.caq_txt_gate(txt_mean).view(B, Q, -1)   # (B, Q, H)
            q_seeds  = self.caq_q_proj.weight.unsqueeze(0) \
                           .expand(B, -1, -1) + txt_gate              # (B, Q, H)

            # 2. Cross-attend over vid_feat: each query seed retrieves the most
            #    text-relevant temporal positions from the video encoder output.
            vid_key_pad = (vid_mask == 0) if vid_mask is not None else None
            caq_out, _ = self.caq_attn(
                q_seeds, vid_feat, vid_feat,
                key_padding_mask=vid_key_pad)                    # (B, Q, H)
            caq_out = self.caq_norm(caq_out)                     # (B, Q, H)

            # 3. Add attended features to base content embedding.
            tgt = tgt + caq_out

            # 4. Content-adaptive reference points: combine per-query offset from
            #    caq_out with a text-global offset from caq_ref_init(txt_mean).
            #    Zero-init of both final layers means offset=0 at ep0.
            ref_offset_q = self.caq_ref_head(caq_out)                                 # (B, Q, 2) per-query
            ref_offset_t = self.caq_ref_init(txt_mean).unsqueeze(1).expand(B, Q, 2)   # (B, Q, 2) text-global
            ref_offset   = ref_offset_q + ref_offset_t
            base_ref     = self.query_embed.weight.sigmoid() \
                               .unsqueeze(0).expand(B, -1, -1)    # (B, Q, 2)
            # Add offset in inv-sigmoid space then re-sigmoid to stay in [0,1]
            ref = (inverse_sigmoid(base_ref) + ref_offset).sigmoid()  # (B, Q, 2)
        else:
            ref = self.query_embed.weight.sigmoid() \
                      .unsqueeze(0).expand(B, -1, -1)           # (B, Q, 2)

        return self.decoder(tgt, memory, mem_pad, ref,
                            cross_attn_bias=cross_attn_bias)

    # ------------------------------------------------------------------
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        """
        Args:
            src_vid:      (B, L_v, D_v) — padded video clip features
            src_vid_mask: (B, L_v)       — 1=valid, 0=padded
            src_txt:      (B, L_t, D_t) — padded query token features
            src_txt_mask: (B, L_t)       — 1=valid, 0=padded
        """
        H        = self.config.hidden_size
        aux_loss = getattr(self.config, "aux_loss", False)

        expected_vid_dim = sum(getattr(self.config, "v_feat_dims", [self.config.v_feat_dim]))
        if src_vid.shape[-1] != expected_vid_dim:
            raise ValueError(
                f"src_vid feature dim mismatch: got {src_vid.shape[-1]}, expected {expected_vid_dim}. "
                f"Check v_feat_dims/v_feat_dim and use_tef config alignment."
            )

        # ---- 1. Encode text ------------------------------------------
        txt_feat = self._encode_query(src_txt, src_txt_mask)    # (B, L_t, H)

        # ---- 2. Project video (reused for negative pair) -------------
        vid_feat_proj = self.vid_pos_embed(
            self.input_vid_proj(src_vid))                        # (B, L_v, H)

        # ---- 2b. Feature noise augmentation (training only) ----------
        # Adds small Gaussian noise to projected features to regularize
        # and prevent the model from memorizing exact feature patterns.
        _feat_noise_std = getattr(self.config, "feat_noise_std", 0.0)
        if self.training and _feat_noise_std > 0:
            vid_feat_proj = vid_feat_proj + torch.randn_like(vid_feat_proj) * _feat_noise_std
            txt_feat      = txt_feat      + torch.randn_like(txt_feat)      * _feat_noise_std

        # ---- 3. T2V fusion + GaussianBlock ---------------------------
        vid_feat, global_rep = self._early_query_fusion(
            vid_feat_proj, txt_feat, src_vid_mask, src_txt_mask)

        if self._use_global_in_encoder:
            # Prepend global token: [global | video] → GaussianBlock → split
            g_in    = global_rep.unsqueeze(1)                        # (B, 1, H)
            g_mask  = torch.ones(vid_feat.shape[0], 1,
                                 dtype=src_vid_mask.dtype,
                                 device=src_vid_mask.device)
            enc_in  = torch.cat([g_in, vid_feat], dim=1)             # (B, 1+L_v, H)
            enc_mask = torch.cat([g_mask, src_vid_mask], dim=1)      # (B, 1+L_v)
            enc_out = self.video_encoder(enc_in, enc_mask.unsqueeze(1),
                                         weight_token=global_rep.unsqueeze(1))
            global_rep = enc_out[:, 0, :]                            # (B, H) updated
            vid_feat   = enc_out[:, 1:, :]                           # (B, L_v, H)
        else:
            vid_feat = self.video_encoder(
                vid_feat, src_vid_mask.unsqueeze(1),
                weight_token=global_rep.unsqueeze(1))                # (B, L_v, H)

        vid_feat = vid_feat * src_vid_mask.unsqueeze(-1)

        # ---- 4. Decoder memory (video only, or [video | text]) ----------
        if getattr(self.config, "use_txt_in_memory", True):
            memory      = torch.cat([vid_feat, txt_feat], dim=1)
            memory_mask = torch.cat([src_vid_mask, src_txt_mask], dim=1)
        else:
            memory      = vid_feat
            memory_mask = src_vid_mask

        # ---- 4a. Saliency scores (positive pair) — computed before decode ----
        # Both vid_feat and global_rep are available here; saliency is used both
        # as a supervision target (step 6) and to bias decoder cross-attention.
        saliency_scores = (
            self.saliency_proj1(vid_feat)
            * self.saliency_proj2(global_rep).unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(H)                             # (B, L_v)

        # ---- 4b. Build cross-attention saliency bias for decoder ----------
        # cross_attn_bias: (B, L_mem) additive log-sigmoid bias injected into
        # decoder cross-attention over video positions; text positions get 0.
        # Memory layout is [video | text] (confirmed above).
        sal_scale = float(getattr(self.config, "sal_prior_scale", 0.0))
        if sal_scale > 0.0:
            sal_bias_vid = (
                torch.log(torch.sigmoid(saliency_scores).clamp_min(1e-6)) * sal_scale
            )  # (B, L_v)
            if getattr(self.config, "use_txt_in_memory", True):
                B_s, L_v = sal_bias_vid.shape
                L_t = txt_feat.shape[1]
                # Memory is [vid; txt]: zeros for text positions
                cross_attn_bias = torch.cat(
                    [sal_bias_vid,
                     torch.zeros(B_s, L_t,
                                 device=sal_bias_vid.device,
                                 dtype=sal_bias_vid.dtype)],
                    dim=1,
                )  # (B, L_v + L_t)
            else:
                cross_attn_bias = sal_bias_vid  # (B, L_v)
        else:
            cross_attn_bias = None

        dec_out, aux_list, spans_per_layer = self._decode(memory, memory_mask,
                                                          txt_feat=txt_feat,
                                                          txt_mask=src_txt_mask,
                                                          vid_feat=vid_feat,
                                                          vid_mask=src_vid_mask,
                                                          cross_attn_bias=cross_attn_bias)

        # ---- 5. Prediction heads -------------------------------------
        pred_logits = self.class_head(dec_out).squeeze(-1)          # (B, Q)
        pred_spans  = spans_per_layer[-1]                        # (B, Q, 2)

        # ---- 5b. Boundary refinement (local Gaussian pooling + MLP) --
        pred_spans_refined, refined_passes = self.boundary_refine(
            pred_spans, vid_feat, src_vid_mask,
            txt_rep=global_rep)                               # (B, Q, 2)

        # ---- 5c. Learned gate: blend coarse and refined spans --------
        # gate ≈ 0.27 at ep0 (refine_gate.bias=-1.0); grows as training progresses.
        # This removes unconditional refinement application, preventing the
        # boundary_refine loss-coef ramp from starving the coarse decoder gradient.
        gate = torch.sigmoid(self.refine_gate(dec_out))         # (B, Q, 1)
        pred_spans_final = gate * pred_spans_refined + (1.0 - gate) * pred_spans  # (B, Q, 2)

        # ---- 6. Saliency (negative pair — cyclic text shift) ---------
        #  v31: simplified from full encoder re-run to shifted dot product.
        #  The negative pair only needs to show "mismatched text gives low saliency."
        #  Re-running the full encoder was 40% of forward cost and introduced
        #  conflicting gradients between positive and negative text encoder updates.
        #  vid_feat is detached so the negative-pair gradient only updates
        #  the text encoder and saliency projections.
        if txt_feat.shape[0] > 1:
            txt_neg_feat = torch.cat([txt_feat[1:], txt_feat[:1]], dim=0)       # (B, L_t, H)
            txt_neg_mask = torch.cat([src_txt_mask[1:], src_txt_mask[:1]], dim=0)
            txt_neg_valid = txt_neg_mask.unsqueeze(-1).float()                  # (B, L_t, 1)
            global_rep_neg = (txt_neg_feat * txt_neg_valid).sum(1) / \
                             txt_neg_valid.sum(1).clamp(min=1.0)                # (B, H)
        else:
            global_rep_neg = global_rep.detach()

        saliency_scores_neg = (
            self.saliency_proj1(vid_feat.detach())
            * self.saliency_proj2(global_rep_neg).unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(H)                             # (B, L_v)

        # ---- 9. Collect output ---------------------------------------
        out = {
            "pred_logits":         pred_logits,
            "pred_spans":          pred_spans,              # COARSE decoder output (for loss_spans)
            "pred_spans_refined":        pred_spans_refined,      # (B, Q, 2) – raw refined (for loss_spans_refined)
            "pred_spans_refined_passes": refined_passes,           # list[Tensor(B,Q,2)] – per-pass spans (deep-supervision)
            "pred_spans_final":    pred_spans_final,        # gated coarse+refined blend — used at inference
            "pred_refine_gate":    gate,                    # (B, Q, 1) – in [0,1], diagnostic
            "saliency_scores":     saliency_scores,
            "saliency_scores_neg": saliency_scores_neg,
            "video_mask":          src_vid_mask,
            "text_mask":           src_txt_mask,            # (B, L_t) – for contrastive-align mask
        }

        # ---- 10. Contrastive alignment (optional) --------------------
        if self._use_contrastive:
            out["proj_queries"] = F.normalize(
                self.ca_proj_query(dec_out),  p=2, dim=-1)      # (B, Q, hdim)
            out["proj_txt_mem"] = F.normalize(
                self.ca_proj_txt(txt_feat),   p=2, dim=-1)      # (B, L_t, hdim)
            out["proj_vid_mem"] = F.normalize(
                self.ca_proj_vid(vid_feat),   p=2, dim=-1)      # (B, L_v, hdim)

        # ---- 11. Auxiliary outputs (per intermediate decoder layer) --
        if aux_loss and aux_list:
            aux_outputs = []
            for i, h in enumerate(aux_list):
                h_norm = self.decoder.norm(h)
                aux_dict = {
                    "pred_logits": self.class_head(h_norm).squeeze(-1),   # (B, Q)
                    "pred_spans":  spans_per_layer[i],
                }
                if self._use_contrastive:
                    aux_dict["proj_queries"] = F.normalize(
                        self.ca_proj_query(h_norm), p=2, dim=-1)
                    aux_dict["proj_txt_mem"] = out["proj_txt_mem"]  # shared across layers
                aux_outputs.append(aux_dict)
            out["aux_outputs"] = aux_outputs

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg):
    """Build GaussianFormer_VMR from a config dict."""
    cfg = dict(cfg)
    if cfg.get("use_tef", False):
        stream_dims = cfg.get("v_feat_dims", None)
        if stream_dims is not None and len(stream_dims) == len(cfg.get("v_feat_dirs", [])):
            cfg["v_feat_dims"] = list(stream_dims) + [2]
            cfg["v_feat_dim"] = sum(cfg["v_feat_dims"])
        else:
            cfg["v_feat_dim"] = cfg.get("v_feat_dim", 0) + 2
    return GaussianFormer_VMR(edict(cfg))
