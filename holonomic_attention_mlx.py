"""
Holonomic Attention - MLX port.

Implements the same phase-sensitive attention layer using ``mlx.core`` arrays
and ``mlx.nn`` modules.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class HolonomicAttentionMLX(nn.Module):
    """MLX implementation of HolonomicAttention.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_k : int, optional
        Per-head key/query dimension. Defaults to ``d_model // n_heads``.
    d_v : int, optional
        Per-head value dimension. Defaults to ``d_model // n_heads``.
    dropout : float, optional
        Dropout probability (kept for API parity; currently unused by MLX).
    use_gauges : bool, optional
        Whether to apply the learnable unitary gauge rotation. Default ``True``.

    Notes
    -----
    MLX stores complex numbers as ``mx.complex64`` arrays; however, to keep the
    implementation aligned with the PyTorch version, we still explicitly split
    real and imaginary parts and store them as a final axis of size 2.
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
        self.dropout = dropout
        self.use_gauges = use_gauges

        self.W_q = nn.Linear(d_model, n_heads * self.d_k * 2)
        self.W_k = nn.Linear(d_model, n_heads * self.d_k * 2)
        self.W_v = nn.Linear(d_model, n_heads * self.d_v * 2)
        self.W_o = nn.Linear(n_heads * self.d_v * 2, d_model)

        # Gauge matrices: initialize as identity for stability.
        identity = mx.eye(self.d_k)[None, ...]  # (1, d_k, d_k)
        self.gauges = mx.array(mx.repeat(identity, n_heads, axis=0))

        # Learnable parameters must be nn.Parameter instances.
        self.gauges = nn.Parameter(self.gauges)
        self.temperature = nn.Parameter(mx.array(1.0))

    def _split_heads(self, x: mx.array, d_head: int) -> mx.array:
        """Split projected tensor into real/imag parts and multiple heads."""
        batch, seq_len, _ = x.shape
        # (batch, seq, n_heads, d_head, 2)
        x = x.reshape(batch, seq_len, self.n_heads, d_head, 2)
        return x

    def _unitarize(self) -> mx.array:
        """Return an orthogonal approximation of the gauge matrices."""
        # MLX currently does not expose a direct QR primitive; fallback to
        # a Cayley transform: U = (I - A)(I + A)^{-1} for skew-symmetric A.
        # We approximate A = (G - G^T) / 2, which is skew-symmetric.
        g = self.gauges
        a = (g - g.transpose(0, 2, 1)) / 2.0
        d = g.shape[-1]
        eye = mx.eye(d)
        i_minus_a = eye[None, ...] - a
        i_plus_a = eye[None, ...] + a
        # Solve (I + A) X = (I - A) for X.
        inv, _ = mx.linalg.solve(i_plus_a, i_minus_a)
        return inv

    def _phase_scores(self, q: mx.array, k: mx.array) -> mx.array:
        """Compute phase-sensitive real inner-product scores."""
        q_r, q_i = q[..., 0], q[..., 1]
        k_r, k_i = k[..., 0], k[..., 1]
        re_scores = mx.einsum("bthd,bshd->bhts", q_r, k_r) + mx.einsum(
            "bthd,bshd->bhts", q_i, k_i
        )
        return re_scores

    def _rotate_keys(self, k: mx.array, gauges: mx.array) -> mx.array:
        """Apply gauge rotation ``k' = U @ k`` to keys."""
        k_r, k_i = k[..., 0], k[..., 1]
        kr_rot = mx.einsum("hdg,bshg->bshd", gauges, k_r)
        ki_rot = mx.einsum("hdg,bshg->bshd", gauges, k_i)
        return mx.stack([kr_rot, ki_rot], axis=-1)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Run MLX holonomic attention.

        Parameters
        ----------
        query, key, value : mx.array
            ``(batch, seq_len, d_model)``.
        mask : mx.array, optional
            ``(batch, 1, seq_len)`` or ``(batch, tgt_len, src_len)`` boolean
            mask.

        Returns
        -------
        mx.array
            ``(batch, tgt_len, d_model)``.
        """
        batch, tgt_len, _ = query.shape

        q = self._split_heads(self.W_q(query), self.d_k)
        k = self._split_heads(self.W_k(key), self.d_k)
        v = self._split_heads(self.W_v(value), self.d_v)

        if self.use_gauges:
            gauges = self._unitarize()
            k = self._rotate_keys(k, gauges)

        re_scores = self._phase_scores(q, k)
        temp = mx.clip(self.temperature, 1e-6, None)
        scores = (re_scores ** 2) / temp

        if mask is not None:
            # mask True -> masked out
            scores = mx.where(mask, float("-inf"), scores)

        attn = mx.softmax(scores, axis=-1)
        # (b, h, t, s) x (b, s, h, d, 2)
        out_r = mx.einsum("bhts,bshd->bthd", attn, v[..., 0])
        out_i = mx.einsum("bhts,bshd->bthd", attn, v[..., 1])
        out = mx.concatenate([out_r, out_i], axis=-1)
        out = out.reshape(batch, tgt_len, self.n_heads * self.d_v * 2)
        return self.W_o(out), None

    def __repr__(self) -> str:
        return (
            f"HolonomicAttentionMLX("
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_k={self.d_k}, d_v={self.d_v}, use_gauges={self.use_gauges})"
        )
