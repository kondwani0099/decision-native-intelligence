"""
Loss Functions for UDM Multi-Task Training.

Combined objective:
  L = L_cls + lambda_1 * L_rank + lambda_2 * L_reg + lambda_3 * L_abstain + lambda_4 * L_KD

Only the applicable terms are active for each batch example:
  - Classification examples: L_cls + L_abstain
  - Regression examples:     L_reg + L_abstain
  - Ranking examples:        L_rank + L_abstain
  - Distillation examples:   L_KD (from teacher probability distribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ClassificationLoss(nn.Module):
    """
    Cross-entropy with label smoothing.

    Label smoothing prevents overconfident predictions by distributing
    a small probability mass uniformly across all classes:
      target = (1 - epsilon) * one_hot + epsilon / num_classes

    This acts as a regularizer and improves calibration.
    """

    def __init__(self, label_smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes] raw classification logits
            targets: [batch] integer class indices

        Returns:
            Scalar loss
        """
        return self.loss_fn(logits, targets)


class RegressionLoss(nn.Module):
    """
    Huber loss (smooth L1) for regression.

    Combines the best of MSE (smooth gradients near zero) and MAE (robust to outliers).
    For |error| < delta: 0.5 * error^2
    For |error| >= delta: delta * (|error| - 0.5 * delta)
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, 1] predicted scalar values
            targets: [batch, 1] ground truth values

        Returns:
            Scalar loss
        """
        return self.loss_fn(predictions, targets)


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss using log-sigmoid margin.

    For each pair (correct, incorrect):
      L = -log(sigmoid(s_correct - s_incorrect))

    This encourages the model to score the correct item higher than incorrect items.
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        rankings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scores: [batch, num_candidates] predicted scores per candidate
            rankings: [batch, num_candidates] ground truth ranking (0 = best)

        Returns:
            Scalar pairwise ranking loss
        """
        batch_size, num_candidates = scores.shape
        total_loss = torch.tensor(0.0, device=scores.device)
        num_pairs = 0

        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                # Get pairs where ranking[i] < ranking[j] (i is ranked higher)
                mask = rankings[:, i] < rankings[:, j]
                if mask.sum() == 0:
                    continue

                s_better = scores[:, i][mask]
                s_worse = scores[:, j][mask]
                pair_loss = -F.logsigmoid(s_better - s_worse - self.margin).mean()
                total_loss = total_loss + pair_loss
                num_pairs += 1

                # Reverse pairs
                mask_rev = rankings[:, j] < rankings[:, i]
                if mask_rev.sum() > 0:
                    s_better_rev = scores[:, j][mask_rev]
                    s_worse_rev = scores[:, i][mask_rev]
                    pair_loss_rev = -F.logsigmoid(s_better_rev - s_worse_rev - self.margin).mean()
                    total_loss = total_loss + pair_loss_rev
                    num_pairs += 1

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss


class AbstentionLoss(nn.Module):
    """
    Binary cross-entropy loss for the abstention gate.

    Trained on:
      - should_abstain=1 for: out-of-domain examples, missing data, ambiguous cases
      - should_abstain=0 for: clear, in-domain decision examples

    This teaches the model when to say "I don't know" — critical for enterprise safety.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction=reduction)

    def forward(
        self,
        abstain_prob: torch.Tensor,
        should_abstain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            abstain_prob: [batch] predicted abstention probability (0-1)
            should_abstain: [batch] ground truth (1 = should abstain, 0 = should decide)

        Returns:
            Scalar BCE loss
        """
        return self.loss_fn(abstain_prob, should_abstain.float())


class DistillationLoss(nn.Module):
    """
    KL divergence loss for distilling a teacher model's probability distribution.

    L_KD = KL(P_teacher || P_student) = sum P_teacher * log(P_teacher / P_student)

    Temperature parameter controls the softness of teacher/student distributions:
      Higher temperature → softer distributions → transfers more "dark knowledge"
    """

    def __init__(self, temperature: float = 4.0, reduction: str = "batchmean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [batch, num_classes] student model raw logits
            teacher_logits: [batch, num_classes] teacher model raw logits

        Returns:
            Scalar KL divergence loss
        """
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction=self.reduction)
        # Scale by T^2 to maintain gradient magnitude
        return kl_loss * (T * T)


class CombinedDecisionLoss(nn.Module):
    """
    Combined multi-task loss for UDM training.

    L = lambda_cls * L_cls + lambda_rank * L_rank + lambda_reg * L_reg
        + lambda_abstain * L_abstain + lambda_distill * L_KD

    Only applicable terms are computed per batch (determined by decision_type).
    """

    def __init__(self, config):
        super().__init__()
        self.cls_loss = ClassificationLoss(label_smoothing=config.label_smoothing)
        self.reg_loss = RegressionLoss()
        self.rank_loss = RankingLoss()
        self.abstain_loss = AbstentionLoss()
        self.distill_loss = DistillationLoss()

        self.lambda_cls = config.lambda_cls
        self.lambda_reg = config.lambda_reg
        self.lambda_rank = config.lambda_rank
        self.lambda_abstain = config.lambda_abstain
        self.lambda_distill = config.lambda_distill

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        decision_type: str = "classification",
        abstain_prob: Optional[torch.Tensor] = None,
        should_abstain: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Returns:
            Dict with 'total', 'cls', 'abstain', etc. losses
        """
        losses = {}
        total = torch.tensor(0.0, device=logits.device)

        if decision_type == "classification":
            cls = self.cls_loss(logits, targets)
            losses["cls"] = cls
            total = total + self.lambda_cls * cls

        elif decision_type == "regression":
            reg = self.reg_loss(logits, targets.unsqueeze(-1).float())
            losses["reg"] = reg
            total = total + self.lambda_reg * reg

        # Abstention loss (always active if abstain signals provided)
        if abstain_prob is not None and should_abstain is not None:
            abstain = self.abstain_loss(abstain_prob, should_abstain)
            losses["abstain"] = abstain
            total = total + self.lambda_abstain * abstain

        # Distillation loss (if teacher provided)
        if teacher_logits is not None:
            distill = self.distill_loss(logits, teacher_logits)
            losses["distill"] = distill
            total = total + self.lambda_distill * distill

        losses["total"] = total
        return losses
