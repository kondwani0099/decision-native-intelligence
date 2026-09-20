"""
Evaluation Metrics for UDM-Bench.

Computes:
  Accuracy metrics: accuracy, macro-F1, precision, recall, balanced accuracy
  Calibration metrics: ECE, Brier score, reliability curves
  Abstention metrics: coverage, selective accuracy, risk-coverage curve
  Performance metrics: latency (p50/p95/p99), throughput
"""

import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        predictions: [N] predicted class indices
        labels: [N] ground truth class indices
        num_classes: total number of classes

    Returns:
        Dict with accuracy, macro_f1, per-class precision/recall
    """
    predictions = predictions.cpu()
    labels = labels.cpu()
    N = len(labels)

    # Overall accuracy
    accuracy = (predictions == labels).float().mean().item()

    # Per-class precision, recall, F1
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = ((predictions == c) & (labels == c)).sum().item()
        fp = ((predictions == c) & (labels != c)).sum().item()
        fn = ((predictions != c) & (labels == c)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0

    # Balanced accuracy (average per-class recall)
    balanced_accuracy = macro_recall

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": balanced_accuracy,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
    }


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Returns:
        [num_classes, num_classes] numpy array where [i,j] = count of true=i, pred=j
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        cm[int(true), int(pred)] += 1
    return cm


def compute_abstention_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    abstain_flags: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute abstention-aware metrics.

    Args:
        predictions: [N] predicted labels
        labels: [N] ground truth labels
        confidences: [N] model confidence scores
        abstain_flags: [N] boolean abstention flags

    Returns:
        coverage: fraction of examples where model decides (doesn't abstain)
        selective_accuracy: accuracy on non-abstained examples
        abstention_accuracy: fraction of abstained examples that were actually wrong
    """
    decided = ~abstain_flags.bool()
    abstained = abstain_flags.bool()

    coverage = decided.float().mean().item()

    # Selective accuracy: accuracy only on decided examples
    if decided.sum() > 0:
        selective_accuracy = (predictions[decided] == labels[decided]).float().mean().item()
    else:
        selective_accuracy = 0.0

    # Abstention accuracy: what fraction of abstained would have been wrong?
    if abstained.sum() > 0:
        would_be_wrong = (predictions[abstained] != labels[abstained]).float().mean().item()
    else:
        would_be_wrong = 0.0

    return {
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "abstention_rate": 1.0 - coverage,
        "abstention_correctness": would_be_wrong,  # How often abstention saved us from errors
    }


def compute_risk_coverage_curve(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    num_points: int = 20,
) -> List[Dict[str, float]]:
    """
    Compute risk-coverage curve by sweeping confidence thresholds.

    At each threshold, compute:
      - coverage (fraction of examples above threshold)
      - risk (error rate on covered examples)
      - selective accuracy

    Returns:
        List of dicts with {threshold, coverage, risk, selective_accuracy}
    """
    curve = []
    thresholds = torch.linspace(0.0, 1.0, num_points)

    for threshold in thresholds:
        covered = confidences >= threshold.item()
        coverage = covered.float().mean().item()

        if covered.sum() > 0:
            correct = (predictions[covered] == labels[covered]).float()
            sel_acc = correct.mean().item()
            risk = 1.0 - sel_acc
        else:
            sel_acc = 0.0
            risk = 0.0

        curve.append({
            "threshold": threshold.item(),
            "coverage": coverage,
            "risk": risk,
            "selective_accuracy": sel_acc,
        })

    return curve


def measure_inference_latency(
    model: torch.nn.Module,
    sample_input: Tuple[torch.Tensor, torch.Tensor],
    num_warmup: int = 10,
    num_runs: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Measure inference latency (p50, p95, p99).

    Args:
        model: UDM model
        sample_input: (input_ids, attention_mask) tuple
        num_warmup: warmup iterations (not measured)
        num_runs: measured iterations
        device: inference device

    Returns:
        Dict with p50_ms, p95_ms, p99_ms, mean_ms, throughput_per_sec
    """
    model.eval()
    input_ids, attention_mask = sample_input
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_ids, attention_mask, decision_type="classification", num_options=2)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids, attention_mask, decision_type="classification", num_options=2)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies_np = np.array(latencies)

    return {
        "mean_ms": float(np.mean(latencies_np)),
        "p50_ms": float(np.percentile(latencies_np, 50)),
        "p95_ms": float(np.percentile(latencies_np, 95)),
        "p99_ms": float(np.percentile(latencies_np, 99)),
        "throughput_per_sec": 1000.0 / float(np.mean(latencies_np)) * input_ids.shape[0],
    }


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format metrics as a readable ASCII table."""
    lines = [f"\n{'='*50}", f"  {title}", f"{'='*50}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:30s} {v:.4f}")
        elif isinstance(v, (list, np.ndarray)):
            lines.append(f"  {k:30s} {v}")
        else:
            lines.append(f"  {k:30s} {v}")
    lines.append(f"{'='*50}\n")
    return "\n".join(lines)
