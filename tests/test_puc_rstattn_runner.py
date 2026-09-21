import unittest
from pathlib import Path

from scripts.run_puc_rstattn import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    build_parser,
    output_paths,
    report_metadata,
)


class TestPUCRSTAttnRunner(unittest.TestCase):
    def test_defaults_filename_and_metadata(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.seed, args.max_epochs, args.patience, args.batch_size, args.learning_rate), (DEFAULT_SEED, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE))
        json_path, md_path = output_paths(args.grid, Path("results/centralized"))
        self.assertTrue(str(json_path).endswith("puc_rstattn_39bus_p_calendar.json"))
        self.assertTrue(str(md_path).endswith("puc_rstattn_39bus_p_calendar.md"))
        metadata = report_metadata()
        self.assertEqual(metadata["model"], "PUC-RSTAttn")
        self.assertEqual(metadata["utility_fit_samples"], 4771)
        self.assertEqual(metadata["utility_selection_samples"], 1193)
        self.assertEqual(metadata["anchor_loss_weight"], 0.2)
        self.assertFalse(metadata["validation_used_for_graph"])
        self.assertFalse(metadata["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
