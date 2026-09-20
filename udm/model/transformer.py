"""
Transformer Backbone for UDM.

Architecture:
  - RMSNorm (pre-normalization)
  - TransformerBlock = RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual
  - TransformerBackbone = Token Embedding + N × TransformerBlock + Final RMSNorm

Key difference from standard LLM:
  - Bidirectional attention (no causal mask)
  - Output is pooled hidden state, not next-token logits
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .attention import GroupedQueryAttention, precompute_rope_frequencies
from .mlp import SwiGLU


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm — no mean subtraction or bias.
    norm(x) = x * rsqrt(mean(x^2) + eps) * weight

    Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability
        input_dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm).to(input_dtype) * self.weight


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    Flow:
      x → RMSNorm → GQA → + residual → RMSNorm → SwiGLU → + residual
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm before attention
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)

        # Pre-norm before FFN
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_dim, config.ffn_dim, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            rope_cos, rope_sin: Precomputed RoPE frequencies
            attention_mask: [batch, seq_len] boolean pad mask

        Returns:
            [batch, seq_len, hidden_dim]
        """
        # Attention with residual
        h = self.attn_norm(x)
        h = self.attention(h, rope_cos, rope_sin, attention_mask)
        x = x + h

        # FFN with residual
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x


class TransformerBackbone(nn.Module):
    """
    Full transformer backbone for UDM.

    Converts token IDs into contextual hidden representations:
      token_ids → embedding → N × TransformerBlock → final_norm → hidden_states

    The output hidden_states are then consumed by task-specific decision heads.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding (no separate position embedding — RoPE handles positions)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Precompute RoPE frequencies (cached, not a parameter)
        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] boolean mask (1 = valid, 0 = pad)

        Returns:
            hidden_states: [batch, seq_len, hidden_dim]
        """
        # Embed tokens
        x = self.token_embedding(input_ids)

        # Move RoPE buffers to same device as input
        rope_cos = self.rope_cos.to(x.device)
        rope_sin = self.rope_sin.to(x.device)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, attention_mask)

        # Final normalization
        x = self.final_norm(x)

        return x

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mean pool hidden states over non-pad tokens.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] boolean mask

        Returns:
            pooled: [batch, hidden_dim]
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # Expand mask for broadcasting: [batch, seq_len, 1]
        mask = attention_mask.unsqueeze(-1).float()
        # Sum of valid hidden states
        summed = (hidden_states * mask).sum(dim=1)
        # Count of valid tokens (avoid division by zero)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
