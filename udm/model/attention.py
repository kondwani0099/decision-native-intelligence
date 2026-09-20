"""
Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE).

Architecture:
  - 12 query heads, 4 KV heads (3:1 ratio) for UDM-125M
  - RoPE applied to Q and K before attention computation
  - Bidirectional attention (no causal mask) — UDM is an encoder, not a generator
  - Uses torch.nn.functional.scaled_dot_product_attention for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cos and sin frequencies for RoPE.

    Returns:
        cos_cached: [max_seq_len, head_dim//2] cosine values
        sin_cached: [max_seq_len, head_dim//2] sine values
    """
    # Frequency bands: theta^(-2i/d) for i in [0, d/2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # Position indices
    t = torch.arange(max_seq_len, device=device).float()
    # Outer product: [seq_len, head_dim//2]
    freqs = torch.outer(t, inv_freq)
    cos_cached = freqs.cos()
    sin_cached = freqs.sin()
    return cos_cached, sin_cached


def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim//2]
        sin: [seq_len, head_dim//2]

    Returns:
        Tensor with RoPE applied, same shape as x
    """
    seq_len = x.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]

    # Split x into two halves along head_dim
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)

    return rotated


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE.

    Key design choice: Q heads are grouped into G groups, each sharing one K/V head.
    For UDM-125M: 12 query heads, 4 KV heads → 3 query heads per KV group.

    This reduces memory and compute for KV projections while maintaining
    representation capacity in the query space.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_groups = config.kv_groups  # num_heads // num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

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
            rope_cos: [max_seq_len, head_dim//2]
            rope_sin: [max_seq_len, head_dim//2]
            attention_mask: [batch, seq_len] boolean mask (True = attend, False = pad)

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rotary_embedding(q, rope_cos, rope_sin)
        k = apply_rotary_embedding(k, rope_cos, rope_sin)

        # Expand KV heads to match query heads (GQA)
        # Each KV head is repeated kv_groups times
        if self.kv_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.kv_groups, -1)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.kv_groups, -1)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, heads, seq, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build attention mask for SDPA
        # SDPA expects attn_mask shape [batch, 1, seq, seq] or None
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] with True for valid tokens
            # Convert to [batch, 1, 1, seq_len] float mask for broadcasting
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        # Scaled dot-product attention (bidirectional — no causal mask)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,  # Bidirectional for decision encoding
        )

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        output = self.o_proj(attn_output)
        output = self.dropout(output)

        return output
