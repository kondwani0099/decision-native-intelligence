"""
SwiGLU Gated Feed-Forward Network in JAX/Flax.

Architecture:
    gate = silu(x @ W_gate)
    up   = x @ W_up
    out  = (gate * up) @ W_down
"""

import jax
import jax.numpy as jnp
from flax import linen as nn


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network module.

    Args:
        dim: Input and output representation dimension (hidden_dim)
        ffn_dim: Expanded intermediate dimension (e.g. 2560 for 200M)
        dropout: Dropout rate
        dtype: Computation dtype
        param_dtype: Parameter storage dtype
    """

    dim: int
    ffn_dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        w_gate = nn.Dense(self.ffn_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="w_gate")
        w_up = nn.Dense(self.ffn_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="w_up")
        w_down = nn.Dense(self.dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="w_down")

        gate = nn.silu(w_gate(x))
        up = w_up(x)
        h = gate * up
        out = w_down(h)

        if not deterministic and self.dropout > 0.0:
            out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)

        return out
