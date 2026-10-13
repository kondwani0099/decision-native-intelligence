"""
Main UDM / JEV Model Module in JAX/Flax.

Wires together:
  1. TransformerBackbone (16 layers, GQA, RoPE, SwiGLU, RMSNorm for UDM-200M)
  2. Latent Pooling (bidirectional token aggregation)
  3. Multi-task Decision Heads (Classification, Regression, Ranking)
  4. Epistemic Abstention Gate (human-in-the-loop escalation trigger)
"""

from typing import Optional, Dict, Any, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

from ..config import UDMJaxConfig
from .transformer import TransformerBackbone, pool_hidden_states
from .decision_heads import (
    ClassificationHead,
    RegressionHead,
    RankingHead,
    AbstentionGate,
    JaxDecisionOutput,
)


class UDMJaxModel(nn.Module):
    """
    Uniplexity Decision Model in Flax / JAX.

    A decision-native transformer designed for ultra-low latency,
    calibrated inference on GPUs, TPUs, and CPUs.
    """

    config: UDMJaxConfig

    def setup(self):
        # 16-layer Backbone (for 200M configuration)
        self.backbone = TransformerBackbone(self.config)

        # Multi-task heads
        self.cls_head = ClassificationHead(
            hidden_dim=self.config.hidden_dim,
            max_options=self.config.max_options,
            proj_dim=256,
            dropout=self.config.dropout,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.reg_head = RegressionHead(
            hidden_dim=self.config.hidden_dim,
            proj_dim=256,
            dropout=self.config.dropout,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.rank_head = RankingHead(
            hidden_dim=self.config.hidden_dim,
            proj_dim=256,
            dropout=self.config.dropout,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.abstention_gate = AbstentionGate(
            hidden_dim=self.config.hidden_dim,
            gate_dim=64,
            dropout=self.config.dropout,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        decision_type: str = "classification",
        num_options: int = 2,
        deterministic: bool = True,
    ) -> Dict[str, jax.Array]:
        """
        Forward pass producing structured decision outputs.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            decision_type: "classification", "regression", or "ranking"
            num_options: Number of active options for classification
            deterministic: If True, disables dropout

        Returns:
            Dictionary containing logits, probabilities, predicted_label,
            confidence, abstain_prob, should_abstain, pooled_hidden.
        """
        # 1. Forward through transformer backbone
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )

        # 2. Pool representations
        pooled = pool_hidden_states(hidden_states, attention_mask=attention_mask)

        # 3. Route to requested head
        # We compute heads conditionally or all for JIT branch symmetry
        cls_logits = self.cls_head(pooled, num_options=num_options, deterministic=deterministic)
        reg_output = self.reg_head(pooled, deterministic=deterministic)
        rank_scores = self.rank_head(pooled, deterministic=deterministic)

        # 4. Abstention Gate
        abstain_prob = self.abstention_gate(pooled, deterministic=deterministic).squeeze(-1)
        should_abstain = abstain_prob > self.config.abstention_threshold

        # Format outputs depending on decision_type
        if decision_type == "classification":
            probs = jax.nn.softmax(cls_logits, axis=-1)
            pred_label = jnp.argmax(probs, axis=-1)
            confidence = jnp.max(probs, axis=-1)
            head_logits = cls_logits
        elif decision_type == "regression":
            probs = None
            pred_label = reg_output.squeeze(-1)
            confidence = 1.0 - abstain_prob
            head_logits = reg_output
        elif decision_type == "ranking":
            probs = None
            pred_label = rank_scores.squeeze(-1)
            confidence = 1.0 - abstain_prob
            head_logits = rank_scores
        else:
            probs = jax.nn.softmax(cls_logits, axis=-1)
            pred_label = jnp.argmax(probs, axis=-1)
            confidence = jnp.max(probs, axis=-1)
            head_logits = cls_logits

        return {
            "logits": head_logits,
            "probabilities": probs,
            "predicted_label": pred_label,
            "confidence": confidence,
            "abstain_prob": abstain_prob,
            "should_abstain": should_abstain,
            "pooled_hidden": pooled,
        }
