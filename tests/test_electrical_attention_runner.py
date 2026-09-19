from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_electrical_attention import build_parser, output_paths


class ElectricalAttentionRunnerTest(unittest.TestCase):
    def test_parser_default_lambda_initial(self) -> None:
        self.assertEqual(build_parser().parse_args([]).lambda_initial, 1.0)

    def test_output_path_preserves_default_name(self) -> None:
        json_path, md_path = output_paths(
            "39_bus_semi_urban_reference_grid", Path("results/centralized"), 1.0
        )
        self.assertEqual(json_path.name, "electrical_attn_39bus_p_calendar.json")
        self.assertEqual(md_path.name, "electrical_attn_39bus_p_calendar.md")

    def test_output_path_adds_lambda_suffix_for_ablation(self) -> None:
        json_path, md_path = output_paths(
            "80_bus_rural_reference_grid", Path("results/centralized"), 0.1
        )
        self.assertEqual(json_path.name, "electrical_attn_80bus_p_calendar_lambda0p1.json")
        self.assertEqual(md_path.name, "electrical_attn_80bus_p_calendar_lambda0p1.md")

    def test_lambda_initial_is_passed_to_model_constructor(self) -> None:
        try:
            import numpy as np
            from code.models.electrical_attention import TORCH_AVAILABLE, ElectricalAttentionBaseline
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed")
        model = ElectricalAttentionBaseline(np.asarray([[0.0, 1.0], [1.0, 0.0]]), lambda_initial=0.1)
        self.assertAlmostEqual(float(model.lambda_value.detach()), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
