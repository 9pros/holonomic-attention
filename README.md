# Holonomic Attention

A self-contained reference implementation of **Holonomic Attention**, a phase-sensitive multi-head attention mechanism with learnable unitary gauge rotations.

## Overview

Standard scaled dot-product attention treats queries and keys as real vectors. Holonomic Attention instead treats each as a **complex vector** (real + imaginary parts) and computes the phase-sensitive real inner product:

```
Re(⟨q, k⟩)² = (q_r · k_r + q_i · k_i)²
```

A learnable unitary gauge matrix `U` is applied to the key phase before scoring:

```
k' = U · k
```

The attention scores are then temperature-scaled:

```
scores = softmax( Re(⟨q, k⟩)² / τ )
```

This repository provides:

- `holonomic_attention.py` — core PyTorch `HolonomicAttention` layer.
- `holonomic_transformer.py` — full transformer language model built on top of it.
- `meta_ga_evolution.py` — multi-objective genetic algorithm (MetaGA) for evolving gauge matrices and temperature.
- `holonomic_attention_mlx.py` — MLX port of the core layer.
- `integration_bonsai.py` — integration with Bonsai-style ternary-weight inference.
- `integration_deepseek.py` — integration with DeepSeek-style MoE routing.

## Installation

```bash
pip install -r requirements.txt
```

For the MLX port on Apple Silicon:

```bash
pip install mlx>=0.0.9
```

## Usage

### Core attention

```python
import torch
from holonomic_attention import HolonomicAttention

model = HolonomicAttention(d_model=512, n_heads=8)
q = torch.randn(2, 16, 512)
k = torch.randn(2, 20, 512)
v = torch.randn(2, 20, 512)
out, weights = model(q, k, v)
```

### Full transformer

```python
from holonomic_transformer import HolonomicTransformer

transformer = HolonomicTransformer(
    vocab_size=10000, d_model=512, n_heads=8, num_layers=6
)
tokens = torch.randint(0, 10000, (2, 32))
logits = transformer(tokens)
```

### MetaGA evolution

```python
from meta_ga_evolution import EvolvedHolonomicAttention

layer = EvolvedHolonomicAttention(d_model=512, n_heads=8, population_size=32)
best_genome = layer.evolve()
print(best_genome.fitness)
```

### MLX port

```python
import mlx.core as mx
from holonomic_attention_mlx import HolonomicAttentionMLX

model = HolonomicAttentionMLX(d_model=512, n_heads=8)
q = mx.random.normal((2, 16, 512))
k = mx.random.normal((2, 20, 512))
v = mx.random.normal((2, 20, 512))
out, _ = model(q, k, v)
```

### Bonsai ternary integration

```python
from integration_bonsai import BonsaiHolonomicAttention

layer = BonsaiHolonomicAttention(d_model=512, n_heads=8)
x = torch.randn(2, 16, 512)
out, _ = layer(x)
```

### DeepSeek MoE integration

```python
from integration_deepseek import DeepSeekHolonomicLayer

layer = DeepSeekHolonomicLayer(
    d_model=512, n_heads=8, n_experts=8, top_k=2
)
x = torch.randn(2, 16, 512)
out = layer(x)
```

## Performance Benchmarks

Benchmarks are placeholders pending full evaluation. Expected characteristics:

| Component                | Notes                                           |
|--------------------------|-------------------------------------------------|
| HolonomicAttention       | ~1-2× runtime of standard MHA due to complex split |
| HolonomicTransformer     | Standard transformer with custom attention      |
| MetaGA                   | Population-size dependent; easily parallelized  |
| MLX port                 | Similar runtime on Apple Silicon                |
| Bonsai integration       | Reduced memory footprint; ternary projections   |
| DeepSeek integration     | Adds MoE routing overhead                       |

## Citation

```bibtex
@software{holonomic_attention,
  title = {Holonomic Attention: Phase-Sensitive Multi-Head Attention with Unitary Gauges},
  year = {2025},
  url = {https://github.com/example/holonomic-attention}
}
```

## License

MIT License.
