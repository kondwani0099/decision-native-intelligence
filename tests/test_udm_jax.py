"""
Unit and Integration Tests for UDM-200M JAX Architecture.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from udm_jax.config import UDM_200M, UDM_125M, UDMJaxConfig
from udm_jax.model.calibration import TemperatureScaler, compute_ece, compute_brier_score


class TestUDMJaxArchitecture(unittest.TestCase):
    """Test architectural specifications and parameter counts."""

    def test_200m_parameter_specification(self):
        config = UDM_200M
        stats = config.estimate_parameters()

        # Check total parameter count is ~200M (between 195M and 205M)
        total_params = stats["total"]
        self.assertGreater(total_params, 195_000_000, "UDM-200M should have at least 195M parameters")
        self.assertLess(total_params, 205_000_000, "UDM-200M should have at most 205M parameters")
        self.assertAlmostEqual(stats["total_millions"], 201.43, delta=0.5)

        # Architectural specifics
        self.assertEqual(config.num_layers, 16)
        self.assertEqual(config.hidden_dim, 1024)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.num_kv_heads, 4)
        self.assertEqual(config.kv_groups, 4)
        self.assertEqual(config.ffn_dim, 2560)
        self.assertEqual(config.vocab_size, 32000)

    def test_125m_parameter_specification(self):
        config = UDM_125M
        stats = config.estimate_parameters()
        self.assertGreater(stats["total"], 95_000_000)
        self.assertLess(stats["total"], 110_000_000)

    def test_temperature_scaler_calibration(self):
        # Synthetic uncalibrated overconfident logits
        rng = np.random.default_rng(42)
        N = 200
        labels = rng.integers(0, 2, size=N)
        logits = rng.normal(scale=1.0, size=(N, 2))
        logits[np.arange(N), labels] += 1.2

        scaler = TemperatureScaler(temperature=1.0)
        opt_t = scaler.calibrate(logits, labels)
        self.assertGreater(opt_t, 0.05)
        self.assertLess(opt_t, 10.0)

        scaled_probs = scaler.scale_logits(logits)
        self.assertEqual(scaled_probs.shape, (N, 2))
        np.testing.assert_allclose(np.sum(scaled_probs, axis=-1), np.ones(N), atol=1e-5)

    def test_ece_computation(self):
        # Perfect calibration test
        N = 1000
        rng = np.random.default_rng(42)
        probs = np.full((N, 2), 0.5)
        labels = rng.binomial(1, 0.5, size=N)

        ece, bins = compute_ece(probs, labels, n_bins=10)
        self.assertIsInstance(ece, float)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_brier_score_computation(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        brier = compute_brier_score(probs, labels)
        self.assertAlmostEqual(brier, 0.0, places=4)

    def test_model_forward_pass(self):
        import jax
        import jax.numpy as jnp
        from udm_jax.model.udm import UDMJaxModel

        # Small test configuration
        test_cfg = UDMJaxConfig(
            vocab_size=100,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            ffn_dim=256,
            max_seq_len=64,
        )
        model = UDMJaxModel(config=test_cfg)
        rng = jax.random.PRNGKey(0)

        batch_size = 2
        seq_len = 16
        input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        variables = model.init(
            rng,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decision_type="classification",
            num_options=2,
            deterministic=True,
        )

        outputs = model.apply(
            variables,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decision_type="classification",
            num_options=2,
            deterministic=True,
        )

        self.assertIn("logits", outputs)
        self.assertIn("probabilities", outputs)
        self.assertIn("predicted_label", outputs)
        self.assertIn("confidence", outputs)
        self.assertIn("abstain_prob", outputs)
        self.assertEqual(outputs["logits"].shape, (batch_size, 2))
        self.assertEqual(outputs["probabilities"].shape, (batch_size, 2))
        self.assertEqual(outputs["predicted_label"].shape, (batch_size,))
        self.assertEqual(outputs["abstain_prob"].shape, (batch_size,))

    def test_hardware_detection(self):
        from udm_jax.distributed import detect_hardware
        hw = detect_hardware()
        self.assertIn("backend", hw)
        self.assertIn("target_name", hw)
        self.assertIn("num_devices", hw)
        self.assertGreaterEqual(hw["num_devices"], 1)


if __name__ == "__main__":
    unittest.main()

