"""
Transformer Backbone for UDM in JAX/Flax.

Architecture:
  - RMSNorm (pre-normalization)
  - TransformerBlock = RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual
  - TransformerBackbone = Token Embedding + N × TransformerBlock + Final RMSNorm + Latent Pooling
  - Non-autoregressive bidirectional contextual representations
"""

from typing import Optional, List
import jax
import jax.numpy as jnp
from flax import linen as nn

from .attention import GroupedQueryAttention, precompute_rope_frequencies
from .mlp import SwiGLU
from ..config import UDMJaxConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    norm(x) = x * rsqrt(mean(x^2, axis=-1) + eps) * scale
    """

    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        # Compute variance in float32 for numerical stability
        orig_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        normed = normed.astype(orig_dtype) * scale.astype(orig_dtype)
        return normed


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    Flow:
      x → RMSNorm → GQA → + residual → RMSNorm → SwiGLU → + residual
    """

    config: UDMJaxConfig
    layer_idx: int = 0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        rope_cos: jax.Array,
        rope_sin: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        # Pre-norm before Attention
        h = RMSNorm(
            dim=self.config.hidden_dim,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            name="attn_norm",
        )(x)

        h = GroupedQueryAttention(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            dropout=self.config.dropout,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            name="attention",
        )(
            h,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )
        x = x + h

        # Pre-norm before FFN
        h = RMSNorm(
            dim=self.config.hidden_dim,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            name="ffn_norm",
        )(x)

        h = SwiGLU(
            dim=self.config.hidden_dim,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            name="ffn",
        )(h, deterministic=deterministic)
        x = x + h

        return x


class TransformerBackbone(nn.Module):
    """
    Full transformer backbone with token embeddings and stacked blocks.

    Converts token IDs into contextual hidden representations and pools:
      token_ids → embedding → N × TransformerBlock → final RMSNorm → pooled vector
    """

    config: UDMJaxConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """
        Args:
            input_ids: [batch, seq_len] integer token IDs
            attention_mask: [batch, seq_len] 1 for valid tokens, 0 for pad
            deterministic: disable dropout if True

        Returns:
            hidden_states: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding
        embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            param_dtype=self.config.param_dtype,
            name="token_embed",
        )(input_ids).astype(self.config.dtype)

        # Precompute RoPE tables
        rope_cos, rope_sin = precompute_rope_frequencies(
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_seq_len,
            theta=self.config.rope_theta,
            dtype=jnp.float32,
        )

        x = embed
        for i in range(self.config.num_layers):
            x = TransformerBlock(
                config=self.config,
                layer_idx=i,
                name=f"layer_{i}",
            )(
                x,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attention_mask=attention_mask,
                deterministic=deterministic,
            )

        # Final RMSNorm
        x = RMSNorm(
            dim=self.config.hidden_dim,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            name="final_norm",
        )(x)

        return x


def pool_hidden_states(
    hidden_states: jax.Array,
    attention_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """
    Mean pooling over valid sequence tokens.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]

    Returns:
        pooled: [batch, hidden_dim]
    """
    if attention_mask is None:
        return jnp.mean(hidden_states, axis=1)

    # Cast mask to hidden_states dtype
    mask_expanded = attention_mask[:, :, None].astype(hidden_states.dtype)
    sum_hidden = jnp.sum(hidden_states * mask_expanded, axis=1)
    counts = jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)
    return sum_hidden / counts
