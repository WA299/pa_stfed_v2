from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from code.data.lv_grid_loader import LVGridLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "pa_stfed_data_v2" / "raw"
DATA_ROOT = Path(os.environ.get("LV_GRID_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
MAPPING_PATH = REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json"
EXPECTED_NODES = {
    "39_bus_semi_urban_reference_grid": 39,
    "50_bus_rural_reference_grid": 50,
    "56_bus_semi_urban_reference_grid": 56,
    "80_bus_rural_reference_grid": 80,
}
EXPECTED_EDGES = {name: count - 1 for name, count in EXPECTED_NODES.items()}
EXPECTED_LOAD_BUSES = {
    "39_bus_semi_urban_reference_grid": 28,
    "50_bus_rural_reference_grid": 21,
    "56_bus_semi_urban_reference_grid": 44,
    "80_bus_rural_reference_grid": 32,
}


@unittest.skipUnless(
    (DATA_ROOT / "norway_4_lv_grids").is_dir() and MAPPING_PATH.is_file(),
    "external LV data and verified mapping report are required",
)
class LVGridLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loader = LVGridLoader(DATA_ROOT, MAPPING_PATH)
        cls.grids = cls.loader.load_all()

    def test_all_physical_buses_and_edges_are_retained(self) -> None:
        self.assertEqual(set(self.grids), set(EXPECTED_NODES))
        for name, grid in self.grids.items():
            self.assertEqual(grid.num_nodes, EXPECTED_NODES[name])
            self.assertEqual(grid.num_edges, EXPECTED_EDGES[name])
            self.assertEqual(grid.edge_index.shape, (2, EXPECTED_EDGES[name]))

    def test_timestamps_and_pq_alignment(self) -> None:
        for grid in self.grids.values():
            self.assertEqual(grid.num_timesteps, 8760)
            self.assertEqual(grid.p.shape, (8760, grid.num_nodes))
            self.assertEqual(grid.q.shape, grid.p.shape)
            self.assertTrue(grid.metadata["p_q_timestamp_aligned"])
            self.assertTrue(grid.metadata["p_q_profile_shape_aligned"])
            self.assertTrue(grid.metadata["time_order_preserved"])
            self.assertEqual(grid.dynamic_features.shape, (8760, grid.num_nodes, 7))

    def test_load_bus_mask_counts_and_non_load_zero(self) -> None:
        for name, grid in self.grids.items():
            self.assertEqual(int(grid.load_bus_mask.sum()), EXPECTED_LOAD_BUSES[name])
            non_load = ~grid.load_bus_mask
            self.assertTrue(np.all(grid.p[:, non_load] == 0))
            self.assertTrue(np.all(grid.q[:, non_load] == 0))

    def test_distance_matrices_are_symmetric_with_zero_diagonal(self) -> None:
        for grid in self.grids.values():
            for matrix in grid.distance_matrices.values():
                self.assertEqual(matrix.shape, (grid.num_nodes, grid.num_nodes))
                self.assertTrue(np.allclose(matrix, matrix.T, equal_nan=True))
                self.assertTrue(np.allclose(np.diag(matrix), 0.0, equal_nan=True))

    def test_static_and_edge_features_have_canonical_shapes(self) -> None:
        for grid in self.grids.values():
            self.assertEqual(grid.static_features.shape, (grid.num_nodes, 4))
            self.assertEqual(grid.edge_features.shape, (grid.num_edges, 4))
            self.assertEqual(len(grid.branch_type_values), grid.num_edges)
            self.assertEqual(set(grid.branch_type_mapping), {value for value in grid.branch_type_values if value is not None})

    def test_chronological_splits_and_train_only_scaler_interface(self) -> None:
        for grid in self.grids.values():
            self.assertEqual(
                {name: split.sample_count for name, split in grid.splits.items()},
                {"train": 6132, "validation": 1314, "test": 1314},
            )
            scaler = grid.fit_scaler()
            self.assertEqual(scaler.train_sample_count_, 6132)
            transformed = scaler.transform(grid.dynamic_features[:2])
            self.assertEqual(transformed.shape, grid.dynamic_features[:2].shape)


if __name__ == "__main__":
    unittest.main()
