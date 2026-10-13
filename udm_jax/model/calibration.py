"""
Post-Hoc Probability Calibration and Reliability Metrics in JAX.

Implements:
  1. Temperature Scaling optimization using negative log-likelihood (NLL)
  2. Expected Calibration Error (ECE)
  3. Brier Score calculation
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import minimize


class TemperatureScaler:
    """
    Post-hoc temperature scaling for calibrated softmax probabilities.

    Given uncalibrated logits z, calculates:
      p_calibrated = softmax(z / T)
    where T > 0 is optimized on the validation set.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature to logits."""
        temp = max(self.temperature, 1e-4)
        scaled = logits / temp
        # Stable softmax
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        init_temperature: float = 1.5,
    ) -> float:
        """
        Optimize temperature T on validation logits and true labels using L-BFGS-B.

        Args:
            logits: [N, num_classes] validation logits
            labels: [N] true integer labels

        Returns:
            optimal temperature T
        """
        N, C = logits.shape

        def nll_obj(params: np.ndarray) -> float:
            T = max(float(params[0]), 1e-3)
            scaled = logits / T
            max_logits = np.max(scaled, axis=-1, keepdims=True)
            log_sum_exp = max_logits + np.log(np.sum(np.exp(scaled - max_logits), axis=-1, keepdims=True))
            # Pick true class log-probabilities
            true_logits = scaled[np.arange(N), labels]
            nll = -np.mean(true_logits - log_sum_exp.squeeze(-1))
            return float(nll)

        res = minimize(nll_obj, x0=[init_temperature], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        if res.success:
            self.temperature = float(res.x[0])
        else:
            self.temperature = float(init_temperature)

        return self.temperature


def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            diff = abs(bin_acc - bin_conf)
            ece += diff * (np.sum(in_bin) / n_samples)
            bin_details.append({
                "bin_idx": i,
                "range": (float(bin_lower), float(bin_upper)),
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "count": int(np.sum(in_bin)),
            })

    return float(ece), bin_details


def compute_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute multi-class Brier score: mean squared difference from one-hot target.
    Lower is better (0.0 = perfect).
    """
    N, C = probabilities.shape
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(N), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
