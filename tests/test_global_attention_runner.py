from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_global_attention import build_parser, output_paths


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


if __name__ == "__main__":
    unittest.main()
