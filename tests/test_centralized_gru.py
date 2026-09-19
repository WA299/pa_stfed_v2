from __future__ import annotations

import unittest

import numpy as np

from code.models.centralized_gru import TORCH_AVAILABLE, evaluate_validation, persistence_1h_predictions


class CentralizedGRUSmokeTest(unittest.TestCase):
    def test_validation_metric_shapes(self) -> None:
        actual = np.array([[1.0, 0.0], [2.0, 0.0]])
        prediction = np.array([[1.5, 99.0], [1.0, 99.0]])
        result = evaluate_validation(actual, prediction, np.array([True, False]))
        self.assertEqual(result["node_count"], 1)
        self.assertEqual(result["sample_count"], 2)
        self.assertAlmostEqual(result["node_macro"]["mae"], 0.75)
        self.assertAlmostEqual(result["grid_aggregate"]["mae"], 0.75)

    def test_persistence_uses_previous_timestamp(self) -> None:
        p = np.arange(12, dtype=float).reshape(4, 3)
        result = persistence_1h_predictions(p, np.array([1, 3]))
        np.testing.assert_array_equal(result, p[[0, 2]])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_single_batch_forward_backward(self) -> None:
        import torch
        from code.models.centralized_gru import SharedNodeGRU, masked_scaled_mae
        model = SharedNodeGRU(input_size=6, hidden_size=32, num_layers=1)
        x = torch.randn(2, 168, 39, 6)
        y = torch.randn(2, 39)
        mask = torch.ones(39, dtype=torch.bool)
        loss = masked_scaled_mae(model(x), y, mask)
        loss.backward()
        self.assertEqual(tuple(model(x).shape), (2, 39))


if __name__ == "__main__":
    unittest.main()
