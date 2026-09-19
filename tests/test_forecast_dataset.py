from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from code.data.forecast_dataset import (
    FEATURE_MODES,
    ForecastWindowDataset,
    build_forecast_datasets,
)
from code.data.lv_grid_loader import LVGridLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("LV_GRID_DATA_ROOT", str(REPO_ROOT.parent / "pa_stfed_data_v2" / "raw")))
MAPPING_PATH = REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json"
EXPECTED_NODES = {
    "39_bus_semi_urban_reference_grid": 39,
    "50_bus_rural_reference_grid": 50,
    "56_bus_semi_urban_reference_grid": 56,
    "80_bus_rural_reference_grid": 80,
}


@unittest.skipUnless(
    (DATA_ROOT / "norway_4_lv_grids").is_dir() and MAPPING_PATH.is_file(),
    "external LV data and verified mapping report are required",
)
class ForecastDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loader = LVGridLoader(DATA_ROOT, MAPPING_PATH)

    def test_history_and_target_shapes_for_both_modes(self) -> None:
        for name, grid in self.loader.load_all().items():
            for mode, dataset in build_forecast_datasets(grid).items():
                sample = dataset[0]
                expected_features = 6 if mode == "p_calendar" else 7
                self.assertEqual(sample["x"].shape, (168, EXPECTED_NODES[name], expected_features))
                self.assertEqual(sample["y"].shape, (EXPECTED_NODES[name],))
                self.assertEqual(dataset.x_shape, (8592, 168, EXPECTED_NODES[name], expected_features))
                self.assertEqual(dataset.y_shape, (8592, EXPECTED_NODES[name]))

    def test_target_timestamp_ranges_are_exact_and_split_safe(self) -> None:
        for grid in self.loader.load_all().values():
            for dataset in build_forecast_datasets(grid).values():
                self.assertEqual(dataset.first_target_timestamp, "2021-01-08 00:00:00")
                self.assertEqual(dataset.last_target_timestamp, "2021-12-31 23:00:00")
                self.assertEqual(len(dataset.split("train")), 5964)
                self.assertEqual(len(dataset.split("validation")), 1314)
                self.assertEqual(len(dataset.split("test")), 1314)
                for split_name in ("train", "validation", "test"):
                    subset = dataset.split(split_name)
                    split = grid.splits[split_name]
                    self.assertTrue(np.all(subset.target_indices >= split.start_index))
                    self.assertTrue(np.all(subset.target_indices < split.end_index))

    def test_validation_history_may_cross_boundary_but_never_future(self) -> None:
        for grid in self.loader.load_all().values():
            dataset = ForecastWindowDataset.from_grid(grid, "p_calendar")
            validation = dataset.split("validation")
            first = validation[0]
            self.assertEqual(first["target_timestamp"].isoformat(sep=" "), "2021-09-13 12:00:00")
            self.assertEqual(first["history_start_index"], 5964)
            self.assertEqual(first["history_end_index"], 6132)
            self.assertEqual(first["target_index"], 6132)
            for index in (0, len(validation) - 1):
                sample = validation[index]
                self.assertLessEqual(sample["history_end_index"], sample["target_index"])
                self.assertEqual(sample["history_end_index"] - sample["history_start_index"], 168)

    def test_scaler_is_train_only(self) -> None:
        for grid in self.loader.load_all().values():
            train = grid.splits["train"]
            load_mask = grid.load_bus_mask
            expected_count = train.sample_count * int(load_mask.sum())
            for mode in FEATURE_MODES:
                dataset = ForecastWindowDataset.from_grid(grid, mode)
                scaler = dataset.fit_scaler()
                self.assertEqual(scaler.fit_split, "train")
                self.assertEqual(scaler.fit_start_index, train.start_index)
                self.assertEqual(scaler.fit_end_index, train.end_index)
                self.assertEqual(scaler.fit_value_count, expected_count)
                self.assertEqual(scaler.fit_load_bus_count, int(load_mask.sum()))
                self.assertEqual(scaler.fit_value_count_by_feature["p"], expected_count)
                train_p = grid.p[train.start_index : train.end_index][:, load_mask]
                self.assertAlmostEqual(scaler.p_mean_, float(np.mean(train_p)), places=8)
                self.assertAlmostEqual(scaler.p_scale_, float(np.std(train_p)), places=8)
                if mode == "pq_calendar":
                    self.assertEqual(scaler.fit_value_count_by_feature["q"], expected_count)
                    train_q = grid.q[train.start_index : train.end_index][:, load_mask]
                    self.assertAlmostEqual(scaler.q_mean_, float(np.mean(train_q)), places=8)
                    self.assertAlmostEqual(scaler.q_scale_, float(np.std(train_q)), places=8)
                else:
                    self.assertIsNone(scaler.q_mean_)
                    self.assertIsNone(scaler.q_scale_)
                with self.assertRaises(ValueError):
                    scaler.fit(dataset, split="validation")

                sample = dataset[0]["x"]
                transformed = scaler.transform(sample)
                restored_features = scaler.inverse_transform(transformed)
                self.assertEqual(transformed.shape, sample.shape)
                calendar_names = {"hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "weekend"}
                for position, name in enumerate(dataset.feature_names):
                    if name in calendar_names:
                        self.assertTrue(np.array_equal(transformed[..., position], sample[..., position]))
                        self.assertTrue(np.array_equal(restored_features[..., position], sample[..., position]))
                    if name in {"p", "q"}:
                        self.assertTrue(np.all(transformed[..., ~load_mask, position] == 0.0))
                        self.assertTrue(np.all(restored_features[..., ~load_mask, position] == 0.0))

                target = dataset[0]["y"]
                target_scaled = scaler.transform_target(target)
                target_restored = scaler.inverse_transform_target(target_scaled)
                self.assertTrue(np.allclose(target_restored, target))
                self.assertTrue(np.all(target_scaled[~load_mask] == 0.0))
                self.assertTrue(np.all(target_restored[~load_mask] == 0.0))


if __name__ == "__main__":
    unittest.main()
