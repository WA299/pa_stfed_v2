from __future__ import annotations

import unittest

import numpy as np

from code.models.centralized_gru import (
    TORCH_AVAILABLE,
    best_epoch_summary,
    compare_gru_persistence,
    evaluate_validation,
    persistence_1h_predictions,
)


class CentralizedGRUSmokeTest(unittest.TestCase):
    def test_validation_metric_shapes(self) -> None:
        actual = np.array([[1.0, 0.0], [2.0, 0.0]])
        prediction = np.array([[1.5, 99.0], [1.0, 99.0]])
        result = evaluate_validation(actual, prediction, np.array([True, False]))
        self.assertEqual(result["node_count"], 1)
        self.assertEqual(result["sample_count"], 2)
        self.assertAlmostEqual(result["node_macro"]["mae"], 0.75)
        self.assertAlmostEqual(result["grid_aggregate"]["mae"], 0.75)

    def test_percentage_metric_units(self) -> None:
        result = evaluate_validation(
            np.array([[1.0], [1.0]]),
            np.array([[1.1], [1.1]]),
            np.array([True]),
        )
        self.assertAlmostEqual(result["node_macro"]["wape_pct"], 10.0)
        self.assertAlmostEqual(result["node_macro"]["smape_pct"], 9.5238095238)

    def test_best_epoch_summary(self) -> None:
        history = [
            {"epoch": 1, "train_scaled_mae": 0.8, "validation": {"node_macro": {"mae": 2.0}}},
            {"epoch": 2, "train_scaled_mae": 0.6, "validation": {"node_macro": {"mae": 1.0}}},
        ]
        self.assertEqual(
            best_epoch_summary(history),
            {
                "best_epoch": 2,
                "best_validation_node_macro_mae": 1.0,
                "best_train_scaled_mae": 0.6,
                "epochs_run": 2,
            },
        )

    def test_comparison_helper_is_metric_and_scope_specific(self) -> None:
        gru = {
            "node_macro": {"mae": 1.0, "rmse": 2.0, "wape_pct": 3.0, "smape_pct": 4.0},
            "grid_aggregate": {"mae": 5.0, "rmse": 6.0, "wape_pct": 7.0, "smape_pct": 8.0},
        }
        persistence = {
            "node_macro": {"mae": 1.0, "rmse": 1.0, "wape_pct": 4.0, "smape_pct": 5.0},
            "grid_aggregate": {"mae": 4.0, "rmse": 7.0, "wape_pct": 7.0, "smape_pct": 9.0},
        }
        comparison = compare_gru_persistence(gru, persistence)
        self.assertEqual(comparison["node_macro"], {
            "mae": "equal", "rmse": "gru_higher", "wape_pct": "gru_lower", "smape_pct": "gru_lower",
        })
        self.assertEqual(comparison["grid_aggregate"], {
            "mae": "gru_higher", "rmse": "gru_lower", "wape_pct": "equal", "smape_pct": "gru_lower",
        })

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
