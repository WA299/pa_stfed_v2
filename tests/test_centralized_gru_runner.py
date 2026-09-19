from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_centralized_gru import (
    GRID_NAMES,
    build_parser,
    input_size_for_feature_mode,
    output_paths,
    parse_grid_name,
)


class CentralizedGRURunnerTest(unittest.TestCase):
    def test_four_grid_names_parse(self) -> None:
        self.assertEqual(len(GRID_NAMES), 4)
        for grid_name in GRID_NAMES:
            self.assertEqual(parse_grid_name(grid_name), grid_name)

    def test_feature_mode_input_sizes(self) -> None:
        self.assertEqual(input_size_for_feature_mode("p_calendar"), 6)
        self.assertEqual(input_size_for_feature_mode("pq_calendar"), 7)

    def test_output_paths_are_grid_mode_specific(self) -> None:
        json_path, md_path = output_paths(
            "80_bus_rural_reference_grid",
            "pq_calendar",
            Path("results") / "centralized",
        )
        self.assertEqual(json_path.name, "gru_80bus_pq_calendar.json")
        self.assertEqual(md_path.name, "gru_80bus_pq_calendar.md")
        self.assertNotIn("gru_smoke_39bus", json_path.name)
        self.assertNotIn("gru_smoke_39bus", md_path.name)

    def test_parser_defaults_match_formal_baseline(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.max_epochs, 50)
        self.assertEqual(args.patience, 8)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.learning_rate, 1e-3)


if __name__ == "__main__":
    unittest.main()
