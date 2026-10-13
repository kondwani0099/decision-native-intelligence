"""
Grouped Query Attention (GQA) and Rotary Position Embeddings (RoPE) in JAX/Flax.

Features:
  - GQA (16 query heads : 4 KV heads for UDM-200M)
  - Rotary Position Embeddings (RoPE) applied to queries and keys
  - Bidirectional attention for non-autoregressive decision representations
  - Optimized for XLA compilation on NVIDIA GPUs and Google TPUs
"""

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jax.Array, jax.Array]:
    """
    Precompute cosine and sine frequency matrices for RoPE.

    Returns:
        cos: [max_seq_len, head_dim // 2]
        sin: [max_seq_len, head_dim // 2]
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(dtype)
    sin = jnp.sin(freqs).astype(dtype)
    return cos, sin


def apply_rotary_embedding(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """
    Apply rotary embeddings to queries or keys.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim // 2]
        sin: [seq_len, head_dim // 2]
    """
    seq_len = x.shape[1]
    cos_sliced = cos[:seq_len][None, :, None, :]  # [1, seq_len, 1, head_dim // 2]
    sin_sliced = sin[:seq_len][None, :, None, :]  # [1, seq_len, 1, head_dim // 2]

    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = jnp.concatenate(
        [
            x1 * cos_sliced - x2 * sin_sliced,
            x1 * sin_sliced + x2 * cos_sliced,
        ],
        axis=-1,
    )
    return rotated


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) module in Flax.

    Args:
        hidden_dim: Model hidden dimension (e.g. 1024)
        num_heads: Number of query attention heads (e.g. 16)
        num_kv_heads: Number of key/value heads (e.g. 4)
        head_dim: Dimension per head (e.g. 64)
        dropout: Attention dropout probability
        dtype: Computation data type (jnp.float32 or jnp.bfloat16)
        param_dtype: Parameter storage data type
    """

    hidden_dim: int = 1024
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        rope_cos: jax.Array,
        rope_sin: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            rope_cos: [max_seq_len, head_dim // 2]
            rope_sin: [max_seq_len, head_dim // 2]
            attention_mask: [batch, seq_len] 1 for valid tokens, 0 for padding
            deterministic: If True, disable dropout
        """
        batch_size, seq_len, _ = x.shape

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        q = nn.Dense(q_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="q_proj")(x)
        k = nn.Dense(kv_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="k_proj")(x)
        v = nn.Dense(kv_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="v_proj")(x)

        q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Apply RoPE
        q = apply_rotary_embedding(q, rope_cos, rope_sin)
        k = apply_rotary_embedding(k, rope_cos, rope_sin)

        # Expand KV heads for GQA (group ratio = num_heads // num_kv_heads)
        kv_groups = self.num_heads // self.num_kv_heads
        if kv_groups > 1:
            k = jnp.repeat(k, kv_groups, axis=2)
            v = jnp.repeat(v, kv_groups, axis=2)

        scale = 1.0 / math.sqrt(self.head_dim)
        # Scaled dot-product attention: [batch, heads, q_len, k_len]
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale

        if attention_mask is not None:
            # Mask pad tokens (shape: [batch, 1, 1, seq_len])
            mask_4d = attention_mask[:, None, None, :]
            # Large negative value for masking (safe for bfloat16 and float32)
            neg_inf = -1e4 if self.dtype == jnp.bfloat16 else -1e9
            logits = jnp.where(mask_4d > 0, logits, neg_inf)

        attn_weights = jax.nn.softmax(logits, axis=-1)

        if not deterministic and self.dropout > 0.0:
            attn_weights = nn.Dropout(rate=self.dropout)(attn_weights, deterministic=deterministic)

        # Context output: [batch, seq_len, num_heads, head_dim]
        out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        out = out.reshape((batch_size, seq_len, self.num_heads * self.head_dim))

        # Output projection
        out = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="o_proj")(out)

        if not deterministic and self.dropout > 0.0:
            out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)

        return out
