"""
Main UDM (Uniplexity Decision Model) Module.

This is the top-level model that wires together:
  1. TransformerBackbone  → contextual hidden representations
  2. Mean Pooling         → single vector per input
  3. Decision Heads       → task-specific outputs (classification, regression, ranking)
  4. AbstentionGate       → confidence-based abstention
  5. TemperatureScaler    → post-hoc calibration

Flow:
  input_ids → backbone → pool → route to head → abstention check → DecisionOutput

Key difference from standard LLM:
  LLM:  hidden → vocabulary logits → next token
  UDM:  hidden → decision representation → decision logits → typed object
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .transformer import TransformerBackbone
from .decision_heads import (
    ClassificationHead,
    RegressionHead,
    RankingHead,
    AbstentionGate,
    DecisionOutput,
)
from .calibration import TemperatureScaler


@dataclass
class UDMOutput:
    """Complete output from a UDM forward pass."""
    # Core decision
    logits: torch.Tensor                       # Raw head output
    probabilities: Optional[torch.Tensor]       # Softmax (classification)
    predicted_label: Optional[torch.Tensor]     # Argmax prediction
    confidence: torch.Tensor                    # Max probability or 1 - uncertainty

    # Abstention
    abstain_prob: torch.Tensor                  # Abstention gate output
    should_abstain: torch.Tensor                # Boolean: abstain_prob > threshold

    # Calibration (populated after calibration step)
    calibrated_probs: Optional[torch.Tensor] = None

    # Metadata
    decision_type: str = "classification"
    pooled_hidden: Optional[torch.Tensor] = None  # For downstream analysis


class UDMModel(nn.Module):
    """
    Uniplexity Decision Model — a decision-native transformer.

    Architecture (UDM-125M):
      - 12-layer transformer backbone with GQA + RoPE + SwiGLU + RMSNorm
      - Multi-task decision heads: classification, regression, ranking
      - Abstention gate for confidence-based routing
      - Temperature scaling for post-hoc calibration

    The model is NOT generative. It processes a canonically-serialized
    decision problem and outputs a typed decision object via tensor operations:
      logits → softmax → typed Python object → JSON
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Shared transformer backbone
        self.backbone = TransformerBackbone(config)

        # Task-specific decision heads
        self.cls_head = ClassificationHead(
            hidden_dim=config.hidden_dim,
            max_options=config.max_options,
            proj_dim=256,
            dropout=config.dropout,
        )
        self.reg_head = RegressionHead(
            hidden_dim=config.hidden_dim,
            proj_dim=256,
            dropout=config.dropout,
        )
        self.rank_head = RankingHead(
            hidden_dim=config.hidden_dim,
            proj_dim=256,
            dropout=config.dropout,
        )

        # Abstention gate
        self.abstention_gate = AbstentionGate(
            hidden_dim=config.hidden_dim,
            gate_dim=64,
        )

        # Post-hoc calibration (applied after training)
        self.temperature_scaler = TemperatureScaler()

        # Abstention threshold
        self.abstention_threshold = config.abstention_threshold

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decision_type: str = "classification",
        num_options: int = 2,
        output_range: Optional[tuple] = None,
        num_candidates: int = 3,
    ) -> UDMOutput:
        """
        Full forward pass: input → backbone → pool → decision head → output.

        Args:
            input_ids: [batch, seq_len] tokenized canonical decision format
            attention_mask: [batch, seq_len] pad mask (1=valid, 0=pad)
            decision_type: "classification" | "regression" | "ranking"
            num_options: number of classification options (used if decision_type="classification")
            output_range: (min, max) for regression clamping
            num_candidates: number of candidates for ranking

        Returns:
            UDMOutput with all decision fields populated
        """
        # 1. Backbone: tokens → contextualized hidden states
        hidden_states = self.backbone(input_ids, attention_mask)

        # 2. Pool: [batch, seq, hidden] → [batch, hidden]
        pooled = self.backbone.pool_hidden_states(hidden_states, attention_mask)

        # 3. Abstention gate
        abstain_prob = self.abstention_gate(pooled)  # [batch, 1]
        should_abstain = abstain_prob > self.abstention_threshold

        # 4. Route to appropriate decision head
        if decision_type == "classification":
            logits = self.cls_head(pooled, num_options=num_options)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)
            return UDMOutput(
                logits=logits,
                probabilities=probs,
                predicted_label=predicted,
                confidence=confidence,
                abstain_prob=abstain_prob.squeeze(-1),
                should_abstain=should_abstain.squeeze(-1),
                decision_type="classification",
                pooled_hidden=pooled,
            )
        elif decision_type == "regression":
            score = self.reg_head(pooled, output_range=output_range)
            return UDMOutput(
                logits=score,
                probabilities=None,
                predicted_label=score.squeeze(-1),
                confidence=1.0 - abstain_prob.squeeze(-1),
                abstain_prob=abstain_prob.squeeze(-1),
                should_abstain=should_abstain.squeeze(-1),
                decision_type="regression",
                pooled_hidden=pooled,
            )
        elif decision_type == "ranking":
            scores = self.rank_head(pooled, num_candidates=num_candidates)
            rankings = scores.argsort(dim=-1, descending=True)
            return UDMOutput(
                logits=scores,
                probabilities=F.softmax(scores, dim=-1),
                predicted_label=rankings,
                confidence=1.0 - abstain_prob.squeeze(-1),
                abstain_prob=abstain_prob.squeeze(-1),
                should_abstain=should_abstain.squeeze(-1),
                decision_type="ranking",
                pooled_hidden=pooled,
            )
        else:
            raise ValueError(f"Unknown decision_type: {decision_type}. Expected: classification, regression, ranking")

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["backbone_embedding"] = sum(
            p.numel() for p in self.backbone.token_embedding.parameters()
        )
        counts["backbone_layers"] = sum(
            p.numel() for p in self.backbone.layers.parameters()
        )
        counts["backbone_norm"] = sum(
            p.numel() for p in self.backbone.final_norm.parameters()
        )
        counts["cls_head"] = sum(p.numel() for p in self.cls_head.parameters())
        counts["reg_head"] = sum(p.numel() for p in self.reg_head.parameters())
        counts["rank_head"] = sum(p.numel() for p in self.rank_head.parameters())
        counts["abstention_gate"] = sum(p.numel() for p in self.abstention_gate.parameters())
        counts["temperature_scaler"] = sum(p.numel() for p in self.temperature_scaler.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
