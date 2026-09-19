from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.run_global_attention import (
    build_parser,
    build_physical_hop2_mask,
    build_electrical_load_knn_mask,
    output_paths,
    validate_attention_options,
)


class GlobalAttentionRunnerTest(unittest.TestCase):
    def test_parser_default_is_all_nodes(self) -> None:
        self.assertEqual(build_parser().parse_args([]).attention_source, "all_nodes")

    def test_all_nodes_output_path_is_unchanged(self) -> None:
        json_path, md_path = output_paths(
            "80_bus_rural_reference_grid", Path("results/centralized"), "all_nodes"
        )
        self.assertEqual(json_path.name, "global_attn_80bus_p_calendar.json")
        self.assertEqual(md_path.name, "global_attn_80bus_p_calendar.md")

    def test_load_buses_output_path_has_loadsrc_suffix(self) -> None:
        json_path, md_path = output_paths(
            "80_bus_rural_reference_grid", Path("results/centralized"), "load_buses"
        )
        self.assertEqual(json_path.name, "global_attn_80bus_p_calendar_loadsrc.json")
        self.assertEqual(md_path.name, "global_attn_80bus_p_calendar_loadsrc.md")

    def test_parser_default_scope_is_all_nodes(self) -> None:
        self.assertEqual(build_parser().parse_args([]).attention_scope, "all_nodes")

    def test_hop2_mask_uses_loader_distance_directly(self) -> None:
        hop_distance = np.asarray([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
        mask = build_physical_hop2_mask(hop_distance)
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(np.all(np.diag(mask)))
        self.assertTrue(mask[0, 2])
        self.assertFalse(mask[0, 3])

    def test_hop2_output_path_and_invalid_combination(self) -> None:
        json_path, md_path = output_paths(
            "80_bus_rural_reference_grid", Path("results/centralized"), "all_nodes", "physical_hop2"
        )
        self.assertEqual(json_path.name, "global_attn_80bus_p_calendar_hop2.json")
        self.assertEqual(md_path.name, "global_attn_80bus_p_calendar_hop2.md")
        with self.assertRaises(ValueError):
            output_paths("80_bus_rural_reference_grid", Path("results/centralized"), "load_buses", "physical_hop2")

    def test_electrical_load_knn3_mask(self) -> None:
        distances = np.array([
            [0, 1, 1, 4, 0.5],
            [1, 0, 2, 3, 0.2],
            [1, 2, 0, 1, 0.3],
            [4, 3, 1, 0, 0.1],
            [0.5, 0.2, 0.3, 0.1, 0],
        ], dtype=float)
        load = np.array([True, True, True, True, False])
        mask = build_electrical_load_knn_mask(distances, load, k=3)
        self.assertEqual(mask.shape, (5, 5))
        self.assertTrue(np.all(np.diag(mask)))
        self.assertTrue(np.all(mask.sum(axis=1) >= 1))
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [0, 1, 2, 3])
        self.assertEqual(np.flatnonzero(mask[4]).tolist(), [4])
        self.assertFalse(mask[0, 4])
        self.assertFalse(mask[4, 0])

    def test_electrical_knn_tie_breaks_by_node_index_and_path(self) -> None:
        distances = np.array([[0, 1, 1, 1, 1], [1, 0, 2, 3, 4], [1, 2, 0, 3, 4], [1, 3, 3, 0, 4], [1, 4, 4, 4, 0]], dtype=float)
        mask = build_electrical_load_knn_mask(distances, np.ones(5, dtype=bool), k=3)
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [0, 1, 2, 3])

    def test_electrical_knn_scope_options_and_output(self) -> None:
        validate_attention_options("electrical_load_knn3", "all_nodes")
        with self.assertRaises(ValueError):
            validate_attention_options("electrical_load_knn3", "load_buses")
        json_path, md_path = output_paths("80_bus_rural_reference_grid", Path("results/centralized"), "all_nodes", "electrical_load_knn3")
        self.assertEqual(json_path.name, "global_attn_80bus_p_calendar_elecknn3.json")
        self.assertEqual(md_path.name, "global_attn_80bus_p_calendar_elecknn3.md")


if __name__ == "__main__":
    unittest.main()
