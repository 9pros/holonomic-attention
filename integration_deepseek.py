"""
Integration of HolonomicAttention into DeepSeek's MoE architecture.

DeepSeek-style models route tokens through a sparse Mixture of Experts (MoE).
This module replaces the standard attention with HolonomicAttention and adds
holonomic expert routing via gauge-derived scores. The MoE feed-forward block
is modelled as a simple stub; in production, swap it for the upstream DeepSeek
implementation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .holonomic_attention import HolonomicAttention


class MoEFeedForward(nn.Module):
    """Stub MoE feed-forward block.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_experts : int
        Number of experts.
    top_k : int
        Number of experts to activate per token.
    d_ff : int, optional
        Hidden dimension of each expert.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        d_ff: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_ff = d_ff
        # Each expert is a small two-layer FFN.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor, router_scores: torch.Tensor) -> torch.Tensor:
        """Route tokens to top-k experts using provided router scores.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``.
        router_scores : torch.Tensor
            ``(batch, seq_len, n_experts)`` router logits.

        Returns
        -------
        torch.Tensor
            Updated hidden states of shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape
        top_k_values, top_k_indices = torch.topk(router_scores, self.top_k, dim=-1)
        weights = F.softmax(top_k_values, dim=-1)

        out = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]
            expert_weight = weights[..., i]
            # Gather inputs for each expert is tricky in batched form; use a
            # simple loop over the batch as a reference implementation.
            for b in range(batch):
                for s in range(seq_len):
                    e_idx = int(expert_idx[b, s].item())
                    out[b, s] += expert_weight[b, s] * self.experts[e_idx](x[b, s])
        return out


class DeepSeekHolonomicAttention(nn.Module):
    """Holonomic attention block for DeepSeek-style models.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    n_experts : int
        Number of MoE experts (used only when this module is paired with an
        MoE FFN for routing).
    top_k : int
        Top-k experts to activate.
    dropout : float, optional
        Dropout probability.
    use_gauges : bool, optional
        Whether to use gauge rotation. Default ``True``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        top_k: int,
        dropout: float = 0.0,
        use_gauges: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.top_k = top_k

        self.attention = HolonomicAttention(
            d_model, n_heads, dropout=dropout, use_gauges=use_gauges
        )

        # Holonomic expert gates used to derive router scores from hidden state.
        # Shape: (n_experts, d_model).
        self.expert_gates = nn.Parameter(torch.randn(n_experts, d_model) / d_model)
        self.dropout = nn.Dropout(p=dropout)

    def compute_router_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute DeepSeek MoE router logits from holonomic gates.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            ``(batch, seq_len, n_experts)`` router logits.
        """
        # Standard linear routing.
        scores = torch.einsum("ed,btd->bte", self.expert_gates, x)
        return scores

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run DeepSeek holonomic attention and return router scores.

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
            Attention output, same shape as ``x``.
        torch.Tensor
            Router logits ``(batch, seq_len, n_experts)``.
        """
        attn_out, _ = self.attention(x, x, x, key_padding_mask, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        router_scores = self.compute_router_scores(x)
        return x, router_scores


class DeepSeekHolonomicLayer(nn.Module):
    """Full DeepSeek-style layer with holonomic attention + MoE FFN.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    n_experts : int
        Number of MoE experts.
    top_k : int
        Number of experts per token.
    d_ff : int, optional
        Expert hidden dimension.
    dropout : float, optional
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        top_k: int,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = DeepSeekHolonomicAttention(
            d_model, n_heads, n_experts, top_k, dropout=dropout
        )
        self.moe_ffn = MoEFeedForward(d_model, n_experts, top_k, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one DeepSeek-style holonomic layer.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        attn_out, router_scores = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.moe_ffn(self.norm2(x), router_scores)
        x = x + self.dropout(ffn_out)
        return x
