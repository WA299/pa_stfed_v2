from __future__ import annotations

import unittest

import numpy as np

from code.models.centralized_gru import masked_scaled_mae
from code.models.electrical_attention import (
    TORCH_AVAILABLE,
    ElectricalAttentionBaseline,
    normalize_impedance_abs_distance,
)


class ElectricalAttentionTest(unittest.TestCase):
    def test_distance_normalization_scale_diagonal_and_finite_values(self) -> None:
        distance = np.asarray([[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]])
        normalized, scale = normalize_impedance_abs_distance(distance)
        self.assertAlmostEqual(scale, 4.0)
        self.assertTrue(np.allclose(np.diag(normalized), 0.0))
        self.assertTrue(np.all(np.isfinite(normalized)))
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertAlmostEqual(float(normalized[0, 1]), np.log1p(0.5))

    def test_closer_distance_has_higher_score_at_equal_dynamic_score(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed")
        import torch

        distance = np.asarray([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
        model = ElectricalAttentionBaseline(distance)
        bias = model.lambda_value.detach() * model.normalized_distance
        scores = torch.zeros_like(bias) - bias
        self.assertGreater(float(scores[0, 1]), float(scores[0, 2]))

    def test_attention_is_global_normalized_and_lambda_has_gradient(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed")
        import torch

        distance = np.asarray([[0.0, 1.0, 3.0, 5.0], [1.0, 0.0, 2.0, 4.0], [3.0, 2.0, 0.0, 1.0], [5.0, 4.0, 1.0, 0.0]])
        model = ElectricalAttentionBaseline(distance)
        prediction, weights = model(torch.randn(2, 168, 4, 6), return_attention=True)
        self.assertEqual(tuple(weights.shape), (2, 4, 4))
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6))
        self.assertTrue(torch.all(weights[:, 0, 3] > 0))
        self.assertFalse(hasattr(model, "attention_mask"))
        self.assertFalse(hasattr(model, "edge_index"))
        loss = masked_scaled_mae(prediction, torch.randn(2, 4), torch.tensor([True, False, True, False]))
        loss.backward()
        self.assertIsNotNone(model.raw_lambda.grad)
        self.assertGreaterEqual(float(model.lambda_value.detach()), 0.0)

    def test_different_node_counts_output_and_backward(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed")
        import torch

        for node_count in (3, 5):
            distance = np.abs(np.subtract.outer(np.arange(node_count), np.arange(node_count))).astype(float)
            model = ElectricalAttentionBaseline(distance)
            prediction = model(torch.randn(2, 168, node_count, 6))
            self.assertEqual(tuple(prediction.shape), (2, node_count))
            loss = masked_scaled_mae(prediction, torch.randn(2, node_count), torch.ones(node_count, dtype=torch.bool))
            loss.backward()
            self.assertIsNotNone(model.raw_lambda.grad)


if __name__ == "__main__":
    unittest.main()
