"""
UDM / JEV Model Configuration for JAX.

Defines model architecture parameters, hardware sharding, and configuration presets
including the target UDM-200M (201.4M parameter) model.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import math


@dataclass
class UDMJaxConfig:
    """Configuration for the JAX Uniplexity / Decision Model."""

    # --- Model Architecture ---
    vocab_size: int = 32000
    hidden_dim: int = 1024          # Default 200M: 1024
    num_layers: int = 16          # Default 200M: 16 layers
    num_heads: int = 16           # Query heads: 16
    num_kv_heads: int = 4         # GQA: 16 query heads, 4 KV heads (4:1 ratio)
    head_dim: int = 64            # hidden_dim // num_heads = 64
    ffn_dim: int = 2560           # SwiGLU: 2.5x hidden_dim rounded to multiple of 256
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0          # Dropout rate (0.0 for inference, >0 for training)

    # Dtype / Compute Precision
    dtype: str = "float32"        # "float32", "bfloat16", "float16"
    param_dtype: str = "float32"

    # --- Decision Head ---
    max_options: int = 8          # Maximum classification options per decision
    num_domains: int = 16         # Maximum supported domains
    abstention_threshold: float = 0.60  # Confidence below this -> abstain

    # --- Task Type Tokens ---
    task_token_cls: int = 6
    task_token_reg: int = 7
    task_token_rank: int = 8

    # --- Training ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_epochs: int = 50
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 1

    # --- Distributed Sharding (GPU / TPU) ---
    mesh_axes: Tuple[str, ...] = ("data", "model")
    data_axis_name: str = "data"
    model_axis_name: str = "model"

    # --- Loss Weights ---
    lambda_cls: float = 1.0
    lambda_reg: float = 1.0
    lambda_rank: float = 1.0
    lambda_abstain: float = 0.1
    lambda_distill: float = 0.5

    # --- Data ---
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation_factor: int = 10

    @property
    def kv_dim(self) -> int:
        """Total dimension for key/value projections."""
        return self.num_kv_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        """Total dimension for query projections."""
        return self.num_heads * self.head_dim

    @property
    def kv_groups(self) -> int:
        """Number of query heads per KV head."""
        return self.num_heads // self.num_kv_heads

    def estimate_parameters(self) -> Dict[str, Any]:
        """Calculate exact parameter counts for all model components."""
        # Embeddings
        embed = self.vocab_size * self.hidden_dim

        # Per-layer attention: Q + K + V + O projections
        q_proj = self.hidden_dim * self.q_dim
        k_proj = self.hidden_dim * self.kv_dim
        v_proj = self.hidden_dim * self.kv_dim
        o_proj = self.q_dim * self.hidden_dim
        attn_per_layer = q_proj + k_proj + v_proj + o_proj

        # Per-layer FFN (SwiGLU has 3 weight matrices: gate, up, down)
        ffn_per_layer = 3 * self.hidden_dim * self.ffn_dim

        # Per-layer RMSNorms (attn_norm, ffn_norm)
        norm_per_layer = 2 * self.hidden_dim

        layer_total = attn_per_layer + ffn_per_layer + norm_per_layer
        backbone = self.num_layers * layer_total
        final_norm = self.hidden_dim

        # Decision heads (projection dense + biases + output dense + biases)
        cls_head = (self.hidden_dim) + (self.hidden_dim * 256 + 256) + (256 * self.max_options + self.max_options)
        reg_head = (self.hidden_dim) + (self.hidden_dim * 256 + 256) + (256 * 1 + 1)
        rank_head = (self.hidden_dim) + (self.hidden_dim * 256 + 256) + (256 * 1 + 1)
        abstain_head = (self.hidden_dim) + (self.hidden_dim * 64 + 64) + (64 * 1 + 1)
        heads_total = cls_head + reg_head + rank_head + abstain_head

        total = embed + backbone + final_norm + heads_total

        return {
            "embedding": embed,
            "attention_per_layer": attn_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "norm_per_layer": norm_per_layer,
            "layer_total": layer_total,
            "backbone": backbone,
            "final_norm": final_norm,
            "decision_heads": heads_total,
            "total": total,
            "total_millions": round(total / 1e6, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)


# ==============================================================================
# Preset Architectures
# ==============================================================================

# UDM-200M (Target Specification: ~201.4M parameters)
UDM_200M = UDMJaxConfig(
    hidden_dim=1024,
    num_layers=16,
    num_heads=16,
    num_kv_heads=4,
    head_dim=64,
    ffn_dim=2560,
    max_seq_len=2048,
)

# UDM-125M (~100.7M parameters)
UDM_125M = UDMJaxConfig(
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=4,
    head_dim=64,
    ffn_dim=2048,
    max_seq_len=2048,
)

# UDM-350M (~350M parameters)
UDM_350M = UDMJaxConfig(
    hidden_dim=1024,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=64,
    ffn_dim=2816,
    max_seq_len=4096,
)

# UDM-1B (~1.05B parameters)
UDM_1B = UDMJaxConfig(
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=128,
    ffn_dim=5504,
    max_seq_len=8192,
)
