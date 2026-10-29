"""
Integration of HolonomicAttention into Bonsai's ternary-weight model.

Bonsai uses extremely low-bit ternary weights ``{-1, 0, 1}`` for fast edge
inference. This module provides a replacement attention block that keeps the
projections in ternary form while preserving the full-precision gauge and
temperature parameters needed for phase calibration.

Since the Bonsai codebase is not available in this environment, this module
defines a local ``TernaryLinear`` stub and the ``BonsaiHolonomicAttention``
class. In a real deployment, replace ``TernaryLinear`` with the upstream Bonsai
implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryLinear(nn.Module):
    """Stub ternary fully-connected layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool, optional
        Whether to include a bias term. Default ``True``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("ternary_weight", torch.zeros_like(self.weight))
        self._quantize_weight()

    def _quantize_weight(self) -> None:
        """Quantize the real-valued weight to ``{-1, 0, +1}``."""
        with torch.no_grad():
            self.ternary_weight.copy_(torch.sign(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the ternary weight approximation."""
        self._quantize_weight()
        out = F.linear(x, self.ternary_weight, self.bias)
        return out


class BonsaiHolonomicAttention(nn.Module):
    """Holonomic attention block adapted for Bonsai ternary inference.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability. Default 0.0.
    use_gauges : bool, optional
        Whether to use gauge rotation. Default ``True``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_gauges: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = self.d_k
        self.use_gauges = use_gauges

        # Ternary linear projections for Q, K, V, and output.
        self.W_q = TernaryLinear(d_model, n_heads * self.d_k * 2)
        self.W_k = TernaryLinear(d_model, n_heads * self.d_k * 2)
        self.W_v = TernaryLinear(d_model, n_heads * self.d_v * 2)
        self.W_o = TernaryLinear(n_heads * self.d_v * 2, d_model)

        # Gauges and temperature remain full precision for phase accuracy.
        self.gauges = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.dropout_p = dropout
        self._init_gauges()

    def _init_gauges(self) -> None:
        """Initialize gauges as identity matrices."""
        with torch.no_grad():
            identity = torch.eye(self.d_k).unsqueeze(0).repeat(self.n_heads, 1, 1)
            self.gauges.copy_(identity)

    def _quantize_gauge_to_ternary(self) -> torch.Tensor:
        """Return a ternary approximation of the gauge matrices.

        The phase rotation can be aggressively quantized without breaking
        unitarity in the limit of small rotations. We use ``sign(gauge)``
        and then project back to the closest orthogonal matrix.
        """
        ternary = torch.sign(self.gauges)
        q, _ = torch.linalg.qr(ternary)
        return q

    def _split_heads(self, x: torch.Tensor, d_head: int) -> torch.Tensor:
        """Split projection output into real/imag and multi-head shape."""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, d_head, 2)

    def _phase_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute phase-sensitive real inner-product scores."""
        q_r, q_i = q[..., 0], q[..., 1]
        k_r, k_i = k[..., 0], k[..., 1]
        re_scores = torch.einsum("bthd,bshd->bhts", q_r, k_r) + torch.einsum(
            "bthd,bshd->bhts", q_i, k_i
        )
        return re_scores

    def _rotate_keys(self, k: torch.Tensor, gauges: torch.Tensor) -> torch.Tensor:
        """Apply gauge rotation ``k' = U @ k``."""
        k_r, k_i = k[..., 0], k[..., 1]
        kr_rot = torch.einsum("hdg,bshg->bshd", gauges, k_r)
        ki_rot = torch.einsum("hdg,bshg->bshd", gauges, k_i)
        return torch.stack([kr_rot, ki_rot], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run Bonsai ternary holonomic attention.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``.
        key_padding_mask : torch.Tensor, optional
            ``(batch, seq_len)`` boolean mask.
        attn_mask : torch.Tensor, optional
            ``(tgt_len, src_len)`` additive mask.

        Returns
        -------
        torch.Tensor
            ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.W_q(x), self.d_k)
        k = self._split_heads(self.W_k(x), self.d_k)
        v = self._split_heads(self.W_v(x), self.d_v)

        if self.use_gauges:
            # Use full-precision gauges; optionally quantize during inference.
            gauges = self._quantize_gauge_to_ternary() if not self.training else self.gauges
            # Orthogonalize to keep unitarity.
            gauges = torch.linalg.qr(gauges)[0]
            k = self._rotate_keys(k, gauges)

        re_scores = self._phase_scores(q, k)
        temp = torch.clamp(self.temperature, min=1e-6)
        scores = (re_scores ** 2) / temp

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)

        out_r = torch.einsum("bhts,bshd->bthd", attn, v[..., 0])
        out_i = torch.einsum("bhts,bshd->bthd", attn, v[..., 1])
        out = torch.cat([out_r, out_i], dim=-1).view(batch, seq_len, -1)

        return self.W_o(out), None


class BonsaiHolonomicTransformerLayer(nn.Module):
    """Single transformer layer for a Bonsai ternary model.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of heads.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float, optional
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = BonsaiHolonomicAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            TernaryLinear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            TernaryLinear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one Bonsai holonomic transformer layer."""
        attn_out, _ = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x
