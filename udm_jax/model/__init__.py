"""
UDM JAX Model Package.
"""

from .calibration import TemperatureScaler, compute_ece, compute_brier_score

try:
    from .attention import GroupedQueryAttention, apply_rotary_embedding, precompute_rope_frequencies
    from .mlp import SwiGLU
    from .transformer import RMSNorm, TransformerBlock, TransformerBackbone, pool_hidden_states
    from .decision_heads import (
        ClassificationHead,
        RegressionHead,
        RankingHead,
        AbstentionGate,
        JaxDecisionOutput,
    )
    from .udm import UDMJaxModel
except ImportError:
    GroupedQueryAttention = None
    apply_rotary_embedding = None
    precompute_rope_frequencies = None
    SwiGLU = None
    RMSNorm = None
    TransformerBlock = None
    TransformerBackbone = None
    pool_hidden_states = None
    ClassificationHead = None
    RegressionHead = None
    RankingHead = None
    AbstentionGate = None
    JaxDecisionOutput = None
    UDMJaxModel = None

__all__ = [
    "GroupedQueryAttention",
    "apply_rotary_embedding",
    "precompute_rope_frequencies",
    "SwiGLU",
    "RMSNorm",
    "TransformerBlock",
    "TransformerBackbone",
    "pool_hidden_states",
    "ClassificationHead",
    "RegressionHead",
    "RankingHead",
    "AbstentionGate",
    "JaxDecisionOutput",
    "TemperatureScaler",
    "compute_ece",
    "compute_brier_score",
    "UDMJaxModel",
]
