"""
UDM Model Configuration.

Defines the UDM-125M architecture parameters as a Python dataclass.
Matches the target spec:
  - 12 layers, 768 hidden dim, 12 attention heads, 4 KV heads (GQA)
  - SwiGLU activation, RMSNorm, RoPE positions
  - ~125M parameters
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class UDMConfig:
    """Configuration for the Uniplexity Decision Model."""

    # --- Model Architecture ---
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4          # GQA: 12 query heads, 4 KV heads (3:1 ratio)
    head_dim: int = 64             # hidden_dim // num_heads
    ffn_dim: int = 2048            # SwiGLU: int(2/3 * 4 * hidden_dim) rounded to multiple of 256
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0          # Dropout rate (0.0 for inference, set >0 for training regularization)

    # --- Decision Head ---
    max_options: int = 8           # Maximum classification options per decision
    num_domains: int = 16          # Maximum supported domains
    abstention_threshold: float = 0.60  # Confidence below this -> abstain

    # --- Task Type Tokens ---
    # These are special token IDs injected at position 0 to signal the task type
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
    augmentation_factor: int = 10  # Multiply dataset by this factor via augmentation

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

    def estimate_parameters(self) -> dict:
        """Estimate parameter count by component."""
        embed = self.vocab_size * self.hidden_dim

        # Per-layer attention: Q + K + V + O projections
        attn_per_layer = (
            self.hidden_dim * self.q_dim +       # Q
            self.hidden_dim * self.kv_dim +       # K
            self.hidden_dim * self.kv_dim +       # V
            self.q_dim * self.hidden_dim          # O
        )
        # Per-layer FFN (SwiGLU has 3 weight matrices)
        ffn_per_layer = 3 * self.hidden_dim * self.ffn_dim
        # Per-layer norms
        norm_per_layer = 2 * self.hidden_dim

        layer_total = attn_per_layer + ffn_per_layer + norm_per_layer
        backbone = self.num_layers * layer_total
        final_norm = self.hidden_dim

        # Decision heads (approximate)
        cls_head = self.hidden_dim * 256 + 256 * self.max_options
        reg_head = self.hidden_dim * 256 + 256
        rank_head = self.hidden_dim * 256 + 256
        abstain_head = self.hidden_dim * 64 + 64
        heads_total = cls_head + reg_head + rank_head + abstain_head

        total = embed + backbone + final_norm + heads_total

        return {
            "embedding": embed,
            "attention_per_layer": attn_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "norm_per_layer": norm_per_layer,
            "layer_total": layer_total,
            "backbone": backbone,
            "decision_heads": heads_total,
            "total": total,
            "total_millions": total / 1e6,
        }


# --- Preset Configurations ---

UDM_125M = UDMConfig()  # Default is 125M

UDM_350M = UDMConfig(
    hidden_dim=1024,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=64,
    ffn_dim=2816,
    max_seq_len=4096,
)

UDM_1B = UDMConfig(
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=128,
    ffn_dim=5504,  # int(2/3 * 4 * 2048) rounded to 256
    max_seq_len=8192,
)
