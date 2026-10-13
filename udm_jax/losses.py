"""
Loss Functions for UDM in JAX.

Implements:
  - Classification loss with label smoothing (Cross-Entropy)
  - Regression loss (Huber / Smooth-L1)
  - Ranking loss (Pairwise margin)
  - Abstention gate loss (Binary Cross-Entropy for confidence escalation)
  - Combined multi-task decision loss
"""

from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
from .config import UDMJaxConfig


def classification_loss(
    logits: jax.Array,
    targets: jax.Array,
    label_smoothing: float = 0.1,
) -> jax.Array:
    """
    Cross-entropy loss with label smoothing.

    Args:
        logits: [batch, num_classes] raw class logits
        targets: [batch] integer label indices
        label_smoothing: epsilon for smoothing

    Returns:
        scalar loss
    """
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(targets, num_classes)
    if label_smoothing > 0.0:
        smooth_targets = one_hot * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    else:
        smooth_targets = one_hot

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(smooth_targets * log_probs, axis=-1)
    return jnp.mean(loss)


def huber_regression_loss(
    predictions: jax.Array,
    targets: jax.Array,
    delta: float = 1.0,
) -> jax.Array:
    """
    Huber loss (Smooth L1) for regression predictions.
    """
    diff = jnp.abs(predictions - targets)
    loss = jnp.where(diff < delta, 0.5 * (diff ** 2), delta * (diff - 0.5 * delta))
    return jnp.mean(loss)


def ranking_loss(
    scores: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """
    Pairwise ranking loss with margin:
      L = -log(sigmoid(s_correct - s_incorrect))
    """
    s_correct = scores[:, 0:1]
    s_incorrect = scores[:, 1:]
    diff = s_correct - s_incorrect
    loss = -jnp.log(jax.nn.sigmoid(diff) + 1e-7)
    return jnp.mean(loss)


def abstention_loss(
    abstain_probs: jax.Array,
    targets: jax.Array,
    eps: float = 1e-7,
) -> jax.Array:
    """
    Binary Cross-Entropy loss for abstention gate.

    Args:
        abstain_probs: [batch] probability in [0, 1]
        targets: [batch] 1 if example should be escalated, 0 otherwise
    """
    targets = targets.astype(jnp.float32)
    p = jnp.clip(abstain_probs, eps, 1.0 - eps)
    bce = -(targets * jnp.log(p) + (1.0 - targets) * jnp.log(1.0 - p))
    return jnp.mean(bce)


def compute_combined_loss(
    outputs: Dict[str, jax.Array],
    batch: Dict[str, jax.Array],
    config: UDMJaxConfig,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Computes total weighted loss and individual metrics.

    Args:
        outputs: Dict returned from UDMJaxModel forward pass
        batch: Dict with 'labels', 'domain_ids', 'should_abstain' (optional)
        config: UDMJaxConfig with loss weights

    Returns:
        total_loss, metrics_dict
    """
    labels = batch["labels"]
    logits = outputs["logits"]
    abstain_probs = outputs["abstain_prob"]

    # 1. Classification loss
    cls_loss = classification_loss(logits, labels, label_smoothing=config.label_smoothing)

    # 2. Abstention loss
    # If the prediction is incorrect, target should_abstain = 1, else 0
    preds = outputs["predicted_label"]
    is_incorrect = (preds != labels).astype(jnp.float32)
    abstain_tgt = batch.get("should_abstain", is_incorrect)
    abs_loss = abstention_loss(abstain_probs, abstain_tgt)

    # 3. Total Loss
    total_loss = config.lambda_cls * cls_loss + config.lambda_abstain * abs_loss

    # Accuracy metric
    acc = jnp.mean((preds == labels).astype(jnp.float32))

    metrics = {
        "loss": total_loss,
        "cls_loss": cls_loss,
        "abstain_loss": abs_loss,
        "accuracy": acc,
    }

    return total_loss, metrics
