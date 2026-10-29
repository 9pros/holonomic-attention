"""
Holonomic Transformer - full transformer language model using HolonomicAttention.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .holonomic_attention import HolonomicAttention


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    max_len : int
        Maximum sequence length for which to precompute encodings.
    dropout : float, optional
        Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor ``x``.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            ``x`` with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_ff : int
        Hidden dimension of the feed-forward layer.
    dropout : float, optional
        Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class HolonomicTransformerLayer(nn.Module):
    """A single transformer layer with holonomic attention.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Hidden dimension of the feed-forward layer.
    dropout : float, optional
        Dropout probability.
    use_gauges : bool, optional
        Passed to :class:`HolonomicAttention`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_gauges: bool = True,
    ) -> None:
        super().__init__()
        self.attention = HolonomicAttention(
            d_model, n_heads, dropout=dropout, use_gauges=use_gauges
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one transformer layer with pre-norm residual connections.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        # Pre-norm: attention block.
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        # Pre-norm: feed-forward block.
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class HolonomicTransformer(nn.Module):
    """Full autoregressive transformer language model using holonomic attention.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    d_ff : int
        Hidden dimension of the feed-forward layer.
    dropout : float, optional
        Dropout probability.
    max_len : int, optional
        Maximum sequence length.
    use_gauges : bool, optional
        Passed to every :class:`HolonomicTransformerLayer`.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        use_gauges: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                HolonomicTransformerLayer(d_model, n_heads, d_ff, dropout, use_gauges)
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply standard initialization to embeddings and the output head."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full transformer language model.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len)`` integer token indices.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
