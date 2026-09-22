import unittest
from pathlib import Path

from scripts.run_puc_rstattn_v2_ablation import VARIANTS, build_parser, output_paths, report_metadata


class TestPUCRSTAttnV2AblationRunner(unittest.TestCase):
    def test_defaults_identical_and_filenames_deterministic(self):
        args = build_parser().parse_args(["--variant", "gru_anchor_only"])
        self.assertEqual((args.seed, args.max_epochs, args.patience, args.batch_size, args.learning_rate), (42, 50, 8, 32, 1e-3))
        for variant in VARIANTS:
            json_path, md_path = output_paths(variant, args.grid, Path("results/ablations"))
            self.assertTrue(json_path.name.endswith(f"{variant}_39bus_p_calendar.json"))
            self.assertTrue(md_path.name.endswith(f"{variant}_39bus_p_calendar.md"))

    def test_metadata_flags_and_no_test(self):
        expected = {
            "gru_anchor_only": (False, False, False, False, False, False, False),
            "temporal_residual_only": (True, False, False, False, False, False, False),
            "utility_uniform_spatial": (True, True, True, False, False, False, True),
            "utility_prior_no_physics": (True, True, True, True, True, False, True),
        }
        for variant, flags in expected.items():
            metadata = report_metadata(variant)
            self.assertEqual(tuple(metadata[key] for key in ("temporal_residual_enabled", "spatial_residual_enabled", "utility_candidate_selection_enabled", "utility_magnitude_prior_enabled", "dynamic_attention_enabled", "physical_relation_bias_enabled", "aggregate_preserving_spatial_residual")), flags)
            self.assertEqual(metadata["experiment"], "puc_rstattn_v2_formal_ablation")
            self.assertTrue(metadata["independent_training"])
            self.assertFalse(metadata["validation_used_for_graph"])
            self.assertFalse(metadata["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
