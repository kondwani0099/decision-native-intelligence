"""
Post-hoc Calibration for UDM.

Implements:
  1. Temperature Scaling - learns a single temperature parameter on validation set
  2. Expected Calibration Error (ECE) - measures calibration quality
  3. Reliability Diagram - visual calibration assessment

A calibrated model means: if it says 80% confidence, it should be correct ~80% of the time.
This is critical for enterprise decision systems where confidence drives routing:
  - High confidence → auto-execute
  - Low confidence  → escalate to human / larger model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for calibrating softmax probabilities.

    After training the base model, freeze all weights and optimize
    a single scalar temperature T on the validation set:
      calibrated_probs = softmax(logits / T)

    Temperature > 1 → softer (less confident) distribution
    Temperature < 1 → sharper (more confident) distribution
    Temperature = 1 → original distribution
    """

    def __init__(self):
        super().__init__()
        # Initialize temperature to 1.0 (identity)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: [batch, num_classes] raw model outputs

        Returns:
            calibrated_probs: [batch, num_classes] calibrated probabilities
        """
        scaled_logits = logits / self.temperature.clamp(min=0.01)
        return F.softmax(scaled_logits, dim=-1)

    def calibrate(
        self,
        logits_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        lr: float = 0.01,
        max_iters: int = 100,
    ) -> float:
        """
        Optimize temperature on validation set using NLL loss.

        Args:
            logits_list: list of [batch, num_classes] logits from validation set
            labels_list: list of [batch] ground truth labels
            lr: learning rate for temperature optimization
            max_iters: maximum optimization steps

        Returns:
            Optimal temperature value
        """
        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        def eval_fn():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature.clamp(min=0.01)
            loss = F.cross_entropy(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        return self.temperature.item()


def compute_ece(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Tuple[float, List[dict]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum over bins: (|bin| / N) * |accuracy(bin) - confidence(bin)|

    Lower ECE = better calibrated model.
    A perfectly calibrated model has ECE = 0.

    Args:
        probabilities: [N, num_classes] predicted probabilities
        labels: [N] ground truth class indices
        n_bins: number of confidence bins

    Returns:
        ece: scalar ECE value
        bin_data: list of dicts with per-bin stats for reliability diagram
    """
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean().item()

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = accuracies[in_bin].mean().item()
            bin_ece = abs(avg_accuracy - avg_confidence) * prop_in_bin
            ece += bin_ece
            bin_data.append({
                "bin_lo": lo,
                "bin_hi": hi,
                "count": in_bin.sum().item(),
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "gap": abs(avg_accuracy - avg_confidence),
            })
        else:
            bin_data.append({
                "bin_lo": lo,
                "bin_hi": hi,
                "count": 0,
                "avg_confidence": (lo + hi) / 2,
                "avg_accuracy": 0.0,
                "gap": 0.0,
            })

    return ece, bin_data


def compute_brier_score(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Brier score (lower is better).

    Brier = mean( sum_k (p_k - y_k)^2 )

    For binary: Brier = mean( (p - y)^2 )

    Args:
        probabilities: [N, num_classes]
        labels: [N] ground truth indices

    Returns:
        Brier score
    """
    num_classes = probabilities.shape[1]
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    brier = ((probabilities - one_hot) ** 2).sum(dim=1).mean().item()
    return brier
