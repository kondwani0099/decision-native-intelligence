"""
Multi-Task Decision Heads for UDM.

Architecture:
  The shared transformer backbone produces a pooled hidden vector [batch, hidden_dim].
  Task-specific heads project this into the appropriate decision space:

  1. ClassificationHead  →  logits over N options  →  softmax  →  probabilities
  2. RegressionHead      →  scalar prediction  →  clamped to [min, max] range
  3. RankingHead         →  per-candidate scores  →  pairwise ranking
  4. AbstentionGate      →  abstention probability  →  binary abstain/decide

The key insight: the model does NOT generate JSON text.
The neural network outputs tensors:
  logits → softmax → typed Python object → JSON

This is faster and more reliable than autoregressive generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DecisionOutput:
    """Typed output from a UDM decision head."""
    decision_type: str                    # "classification", "regression", "ranking"
    label: Any                            # Selected option, scalar, or ranked list
    logits: torch.Tensor                  # Raw output from head
    probabilities: Optional[torch.Tensor] = None  # Softmax probs (classification)
    confidence: float = 0.0               # Max probability or 1-uncertainty
    abstain: bool = False                 # Whether model chose to abstain
    abstain_probability: float = 0.0      # Abstention gate output
    reason_code: str = ""                 # Reason for abstention if applicable


class ClassificationHead(nn.Module):
    """
    Projects pooled hidden state to classification logits.

    Architecture:
      hidden → LayerNorm → Linear(hidden, proj_dim) → GELU → Dropout → Linear(proj_dim, max_options)
    """

    def __init__(self, hidden_dim: int, max_options: int = 8, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, max_options),
        )

    def forward(self, pooled: torch.Tensor, num_options: int = 2) -> torch.Tensor:
        """
        Args:
            pooled: [batch, hidden_dim] - pooled transformer output
            num_options: number of valid options for this decision (slices logits)

        Returns:
            logits: [batch, num_options]
        """
        x = self.norm(pooled)
        logits = self.projector(x)
        # Slice to actual number of options (mask unused positions)
        return logits[:, :num_options]


class RegressionHead(nn.Module):
    """
    Projects pooled hidden state to a scalar prediction, optionally clamped to a range.

    Architecture:
      hidden → LayerNorm → Linear(hidden, proj_dim) → GELU → Linear(proj_dim, 1) → sigmoid → scale to range
    """

    def __init__(self, hidden_dim: int, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(
        self,
        pooled: torch.Tensor,
        output_range: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Args:
            pooled: [batch, hidden_dim]
            output_range: (min_val, max_val) to scale output into

        Returns:
            score: [batch, 1]
        """
        x = self.norm(pooled)
        raw = self.projector(x)  # [batch, 1]

        if output_range is not None:
            lo, hi = output_range
            # Sigmoid to [0, 1] then scale to [lo, hi]
            raw = torch.sigmoid(raw) * (hi - lo) + lo

        return raw


class RankingHead(nn.Module):
    """
    Scores each candidate independently, producing a ranking.

    Architecture:
      For each candidate, concatenate pooled_context + candidate_embedding,
      then project to a scalar score.
      hidden → LayerNorm → Linear(hidden, proj_dim) → GELU → Linear(proj_dim, 1)

    For now (Milestone 1), this uses the same pooled hidden state
    and produces per-option scores that can be sorted into a ranking.
    """

    def __init__(self, hidden_dim: int, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, pooled: torch.Tensor, num_candidates: int = 3) -> torch.Tensor:
        """
        For Milestone 1, uses the classification-style approach:
        produces num_candidates scores from the same pooled representation.

        Args:
            pooled: [batch, hidden_dim]
            num_candidates: number of candidates to rank

        Returns:
            scores: [batch, num_candidates]
        """
        x = self.norm(pooled)
        # Expand and project to get per-candidate scores
        # For now, use a single scorer applied to the pooled state
        # In future: per-candidate embeddings would be concatenated
        score = self.scorer(x)  # [batch, 1]
        # Replicate and add learned noise for initial ranking differentiation
        # This is a placeholder — real ranking needs candidate-specific inputs
        return score.expand(-1, num_candidates)


class AbstentionGate(nn.Module):
    """
    Binary gate that determines whether the model should abstain from deciding.

    Architecture:
      hidden → Linear(hidden, gate_dim) → GELU → Linear(gate_dim, 1) → sigmoid

    Output is a probability in [0, 1]:
      - High value (>threshold) → ABSTAIN (insufficient confidence)
      - Low value (<threshold)  → DECIDE (confident enough)

    This is explicitly safer than forcing a low-confidence model to guess.
    The abstention gate is trained separately from the decision heads,
    using examples where the model should abstain (out-of-domain, ambiguous, missing data).
    """

    def __init__(self, hidden_dim: int, gate_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: [batch, hidden_dim]

        Returns:
            abstain_prob: [batch, 1] probability of abstention
        """
        return self.gate(pooled)
