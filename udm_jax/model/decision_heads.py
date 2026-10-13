"""
Multi-Task Decision Heads for UDM in JAX/Flax.

The shared transformer backbone produces a pooled hidden vector [batch, hidden_dim].
Task-specific heads project this vector directly into the structured decision space:

1. ClassificationHead  →  logits over N options  →  softmax  →  probabilities
2. RegressionHead      →  scalar prediction  →  clamped to [min, max] range
3. RankingHead         →  per-candidate score
4. AbstentionGate      →  epistemic abstention probability (escalate to human)
"""

from dataclasses import dataclass
from typing import Optional, Any, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass
class JaxDecisionOutput:
    """Typed decision container returned from UDM forward pass."""
    decision_type: str
    logits: jax.Array
    probabilities: Optional[jax.Array] = None
    predicted_label: Optional[jax.Array] = None
    confidence: Optional[jax.Array] = None
    abstain_prob: Optional[jax.Array] = None
    should_abstain: Optional[jax.Array] = None
    calibrated_probs: Optional[jax.Array] = None
    pooled_hidden: Optional[jax.Array] = None


class ClassificationHead(nn.Module):
    """
    Classification head projecting pooled representation to option logits.
    """

    hidden_dim: int = 1024
    max_options: int = 8
    proj_dim: int = 256
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pooled: jax.Array,
        num_options: int = 2,
        deterministic: bool = True,
    ) -> jax.Array:
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm")(pooled)
        x = nn.Dense(self.proj_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="dense1")(x)
        x = nn.gelu(x)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        logits = nn.Dense(self.max_options, dtype=self.dtype, param_dtype=self.param_dtype, name="dense2")(x)
        return logits[:, :num_options]


class RegressionHead(nn.Module):
    """
    Regression head projecting pooled representation to continuous scalar values.
    """

    hidden_dim: int = 1024
    proj_dim: int = 256
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pooled: jax.Array,
        output_range: Optional[Tuple[float, float]] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm")(pooled)
        x = nn.Dense(self.proj_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="dense1")(x)
        x = nn.gelu(x)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        pred = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype, name="dense2")(x)

        if output_range is not None:
            min_val, max_val = output_range
            pred = min_val + nn.sigmoid(pred) * (max_val - min_val)

        return pred


class RankingHead(nn.Module):
    """
    Ranking head projecting pooled representation to relative preference scores.
    """

    hidden_dim: int = 1024
    proj_dim: int = 256
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pooled: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm")(pooled)
        x = nn.Dense(self.proj_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="dense1")(x)
        x = nn.gelu(x)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        score = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype, name="dense2")(x)
        return score


class AbstentionGate(nn.Module):
    """
    Epistemic abstention gate estimating model confidence / representation entropy.
    Outputs probability p in [0, 1] of abstaining (escalating to human review).
    """

    hidden_dim: int = 1024
    gate_dim: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pooled: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm")(pooled)
        x = nn.Dense(self.gate_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="gate_dense")(x)
        x = nn.gelu(x)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        gate_logit = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype, name="gate_out")(x)
        abstain_prob = nn.sigmoid(gate_logit)
        return abstain_prob
