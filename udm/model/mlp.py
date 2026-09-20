"""
SwiGLU Feed-Forward Network.

Architecture:
  output = (Swish(x @ W_gate) * (x @ W_up)) @ W_down

SwiGLU uses 3 weight matrices instead of 2, with gated activation.
The hidden dimension is set to 2/3 of the standard 4x expansion to
compensate for the extra parameters from the gate projection,
keeping total FFN parameter count comparable to a standard MLP.

Reference: Shazeer, "GLU Variants Improve Transformer" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    For UDM-125M:
      input_dim = 768
      hidden_dim = 2048 (= int(2/3 * 4 * 768) rounded to multiple of 256)

    Parameters per layer: 3 * 768 * 2048 = 4,718,592
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            [batch, seq_len, dim]
        """
        # SwiGLU: Swish(x @ W_gate) * (x @ W_up) @ W_down
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))
