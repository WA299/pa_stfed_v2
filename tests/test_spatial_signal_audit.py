from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from code.audits.audit_spatial_signal import (
    _hour_of_week_residual,
    _pair_records,
    build_report,
    hop_group,
)


class _Split:
    start_index = 0
    end_index = 8
    sample_count = 8


class _Grid:
    grid_name = "synthetic"
    num_nodes = 4
    node_ids = np.asarray(["a", "b", "c", "d"], dtype=object)
    load_bus_mask = np.asarray([True, True, False, True])
    p = np.asarray([
        [1.0, 2.0, 0.0, 4.0], [1.0, 4.0, 0.0, 8.0],
        [1.0, 6.0, 0.0, 12.0], [1.0, 8.0, 0.0, 16.0],
        [1.0, 10.0, 0.0, 20.0], [1.0, 12.0, 0.0, 24.0],
        [1.0, 14.0, 0.0, 28.0], [1.0, 16.0, 0.0, 32.0],
    ])
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h")
    distance_matrices = {
        "hop_distance": np.asarray([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float),
        "impedance_abs_distance": np.asarray([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float),
    }
    splits = {"train": _Split()}


class SpatialSignalAuditTest(unittest.TestCase):
    def test_hop_groups(self) -> None:
        self.assertEqual(hop_group(1), "hop=1")
        self.assertEqual(hop_group(2), "hop=2")
        self.assertEqual(hop_group(3), "hop=3-4")
        self.assertEqual(hop_group(4), "hop=3-4")
        self.assertEqual(hop_group(5), "hop>=5")

    def test_hour_of_week_residual_uses_train_values(self) -> None:
        timestamps = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-08 00:00"])
        values = np.asarray([[1.0], [10.0], [1000.0]])
        residual = _hour_of_week_residual(values, timestamps)
        self.assertAlmostEqual(float(residual[0, 0]), -499.5)
        self.assertAlmostEqual(float(residual[2, 0]), 499.5)
        self.assertAlmostEqual(float(residual[1, 0]), 0.0)

    def test_pair_count_load_only_no_self_and_distance_alignment(self) -> None:
        records, selection = _pair_records(_Grid())
        self.assertEqual(len(selection["load_bus_indices"]), 3)
        self.assertEqual(len(records), 3 * 2 // 2)
        self.assertTrue(all(item["node_i_index"] != item["node_j_index"] for item in records))
        pair_ab = next(item for item in records if item["node_i_id"] == "a" and item["node_j_id"] == "b")
        self.assertEqual(pair_ab["hop_distance"], 1)
        self.assertEqual(pair_ab["impedance_abs_distance"], 1.0)

    def test_zero_variance_pair_is_explicitly_excluded(self) -> None:
        records, _ = _pair_records(_Grid())
        pair = next(item for item in records if item["node_i_id"] == "a" and item["node_j_id"] == "b")
        self.assertIsNone(pair["correlations"]["raw_standardized"])
        self.assertEqual(pair["correlation_exclusion_reasons"]["raw_standardized"], "zero_variance_pair")


if __name__ == "__main__":
    unittest.main()
