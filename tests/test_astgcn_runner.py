import unittest
from pathlib import Path

from scripts.run_astgcn import build_parser, model_metadata, output_paths


class ASTGCNRunnerTest(unittest.TestCase):
    def test_defaults_and_output(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.max_epochs, 50)
        self.assertEqual(args.patience, 8)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.learning_rate, 1e-3)
        json_path, markdown_path = output_paths(
            "80_bus_rural_reference_grid", Path("results/centralized")
        )
        self.assertEqual(json_path.name, "astgcn_80bus_p_calendar.json")
        self.assertEqual(markdown_path.name, "astgcn_80bus_p_calendar.md")

    def test_model_metadata(self):
        metadata = model_metadata(build_parser().parse_args([]))
        self.assertEqual(metadata["model"], "ASTGCN-r")
        self.assertEqual(
            metadata["implementation"], "ASTGCN_recent_component_adapted"
        )
        self.assertEqual(metadata["reference"], "Guo_et_al_AAAI_2019")
        self.assertFalse(metadata["full_three_component_astgcn"])
        self.assertEqual(metadata["graph_type"], "physical_binary_undirected")
        self.assertFalse(metadata["laplacian_self_loop_added"])
        self.assertFalse(metadata["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
