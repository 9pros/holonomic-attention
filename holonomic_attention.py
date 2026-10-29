"""
Holonomic Attention - PyTorch core implementation.

A phase-sensitive attention mechanism that treats queries and keys as complex
numbers (real + imaginary parts) and applies a learnable unitary gauge rotation
to the key phase.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HolonomicAttention(nn.Module):
    """Phase-sensitive multi-head attention with learnable unitary gauges.

    This module is intended as a drop-in replacement for
    ``nn.MultiheadAttention`` in settings where an explicit phase/gauge degree
    of freedom is useful.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    n_heads : int
        Number of attention heads.
    d_k : int, optional
        Per-head key/query dimension. Defaults to ``d_model // n_heads``.
    d_v : int, optional
        Per-head value dimension. Defaults to ``d_model // n_heads``.
    dropout : float, optional
        Dropout probability applied to attention weights. Default 0.0.
    use_gauges : bool, optional
        Whether to apply the learnable unitary gauge rotation to keys.
        Default ``True``.

    Attributes
    ----------
    W_q, W_k, W_v, W_o : nn.Linear
        Linear projections. The Q/K/V projections output ``n_heads * d_k * 2``
        features (real and imaginary parts concatenated).
    gauges : nn.Parameter
        Unitary rotation matrices of shape ``(n_heads, d_k, d_k)``.
    temperature : nn.Parameter
        Learnable positive temperature used to scale attention scores.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
        use_gauges: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads
        self.dropout_p = dropout
        self.use_gauges = use_gauges

        # Linear projections. Output is twice the per-head dim because we split
        # the representation into real and imaginary parts.
        self.W_q = nn.Linear(d_model, n_heads * self.d_k * 2)
        self.W_k = nn.Linear(d_model, n_heads * self.d_k * 2)
        self.W_v = nn.Linear(d_model, n_heads * self.d_v * 2)
        self.W_o = nn.Linear(n_heads * self.d_v * 2, d_model)

        # Learnable unitary gauge matrices, one per head.
        # We store them as raw matrices and project to the Stiefel manifold
        # via a Cayley / QR retraction in an auxiliary method.
        self.gauges = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k))

        # Positive learnable temperature.
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize projections with Xavier-like gain and orthogonal gauges."""
        for proj in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Initialize gauges as (approximate) identity matrices for stability.
        with torch.no_grad():
            identity = torch.eye(self.d_k).unsqueeze(0).repeat(self.n_heads, 1, 1)
            self.gauges.copy_(identity)

    def _unitarize(self) -> torch.Tensor:
        """Return a unitary/orthogonal approximation of the gauge matrices.

        Uses QR decomposition to map each head's gauge to the closest
        orthogonal matrix, preserving the direction of the learned rotation.
        """
        # gauges: (n_heads, d_k, d_k)
        q, _ = torch.linalg.qr(self.gauges)
        return q

    def _split_heads(self, x: torch.Tensor, d_head: int) -> torch.Tensor:
        """Split projected tensor into real/imag parts and multiple heads.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, seq_len, n_heads * d_head * 2)``.
        d_head : int
            Per-head dimension (d_k or d_v).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, seq_len, n_heads, d_head, 2)`` where the
            last dimension indexes real (0) and imaginary (1) parts.
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, d_head, 2)
        return x

    def _phase_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute phase-sensitive real inner-product scores.

        Parameters
        ----------
        q, k : torch.Tensor
            Complex tensors of shape ``(batch, seq_len, n_heads, d_k, 2)``.

        Returns
        -------
        torch.Tensor
            Real scores of shape ``(batch, n_heads, tgt_len, src_len)``.
        """
        q_r, q_i = q[..., 0], q[..., 1]
        k_r, k_i = k[..., 0], k[..., 1]
        # einsum: (b, t, h, d) x (b, s, h, d) -> (b, h, t, s)
        re_scores = torch.einsum("bthd,bshd->bhts", q_r, k_r) + torch.einsum(
            "bthd,bshd->bhts", q_i, k_i
        )
        return re_scores

    def _rotate_keys(self, k: torch.Tensor, gauges: torch.Tensor) -> torch.Tensor:
        """Apply unitary gauge rotation to keys: ``k' = U @ k``.

        Parameters
        ----------
        k : torch.Tensor
            Complex keys of shape ``(batch, src_len, n_heads, d_k, 2)``.
        gauges : torch.Tensor
            Unitary matrices of shape ``(n_heads, d_k, d_k)``.

        Returns
        -------
        torch.Tensor
            Rotated keys with the same shape as ``k``.
        """
        k_r, k_i = k[..., 0], k[..., 1]
        # Apply the same real rotation to both real and imaginary parts.
        # U is real orthogonal, so U(a + ib) = U a + i U b.
        # einsum: (h, d_out, d_in) @ (b, s, h, d_in) -> (b, s, h, d_out)
        kr_rot = torch.einsum("hdg,bshg->bshd", gauges, k_r)
        ki_rot = torch.einsum("hdg,bshg->bshd", gauges, k_i)
        return torch.stack([kr_rot, ki_rot], dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run holonomic attention.

        Parameters
        ----------
        query : torch.Tensor
            ``(batch, tgt_len, d_model)``.
        key : torch.Tensor
            ``(batch, src_len, d_model)``.
        value : torch.Tensor
            ``(batch, src_len, d_model)``.
        key_padding_mask : torch.Tensor, optional
            ``(batch, src_len)`` boolean mask; ``True`` means masked out.
        need_weights : bool, optional
            If ``True``, also return the averaged attention weights.
        attn_mask : torch.Tensor, optional
            ``(tgt_len, src_len)`` additive mask.

        Returns
        -------
        output : torch.Tensor
            ``(batch, tgt_len, d_model)``.
        attn_weights : torch.Tensor or None
            Averaged attention weights if ``need_weights`` is ``True``.
        """
        batch, tgt_len, _ = query.shape
        src_len = key.size(1)

        # 1. Project Q, K, V and split into complex components.
        q = self._split_heads(self.W_q(query), self.d_k)
        k = self._split_heads(self.W_k(key), self.d_k)
        v = self._split_heads(self.W_v(value), self.d_v)

        # 2. Optional unitary gauge rotation of keys.
        if self.use_gauges:
            gauges = self._unitarize()
            k = self._rotate_keys(k, gauges)

        # 3. Phase-sensitive scores and temperature scaling.
        re_scores = self._phase_scores(q, k)
        # Clamp temperature for numerical stability.
        temp = torch.clamp(self.temperature, min=1e-6)
        scores = (re_scores ** 2) / temp

        # 4. Apply masks.
        if attn_mask is not None:
            # Expand for broadcast: (t, s) -> (1, 1, t, s)
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            # True means masked; expand to (b, 1, 1, s).
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        # 5. Softmax and weighted sum of values.
        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)

        # attn: (b, h, t, s); v: (b, s, h, d_v, 2)
        out_r = torch.einsum("bhts,bshd->bthd", attn, v[..., 0])
        out_i = torch.einsum("bhts,bshd->bthd", attn, v[..., 1])
        out = torch.cat([out_r, out_i], dim=-1)
        # out now: (b, t, h, d_v*2)
        out = out.view(batch, tgt_len, self.n_heads * self.d_v * 2)

        # 6. Final linear projection back to d_model.
        output = self.W_o(out)

        if need_weights:
            # Average over heads.
            attn_weights = attn.mean(dim=1)
            return output, attn_weights
        return output, None

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_k={self.d_k}, d_v={self.d_v}, "
            f"use_gauges={self.use_gauges}, dropout={self.dropout_p}"
        )
