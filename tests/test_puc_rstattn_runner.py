import unittest
from pathlib import Path

from scripts.run_puc_rstattn import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    build_parser,
    build_epoch_record,
    output_paths,
    report_metadata,
)
from code.models.centralized_gru import best_epoch_summary


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

    def test_training_history_matches_best_summary_contract(self):
        history = [
            build_epoch_record(
                epoch=1,
                primary_losses=[0.8, 0.6],
                anchor_losses=[0.7, 0.5],
                total_losses=[0.94, 0.7],
                validation={"node_macro": {"mae": 2.0}},
                temporal_anchor={"node_macro": {"mae": 2.1}},
            ),
            {
                "epoch": 2,
                "train_scaled_mae": 0.4,
                "train_anchor_scaled_mae": 0.45,
                "train_total_loss": 0.49,
                "validation": {"node_macro": {"mae": 1.0}},
                "temporal_anchor": {"node_macro": {"mae": 1.1}},
            },
        ]
        summary = best_epoch_summary(history)
        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["best_validation_node_macro_mae"], 1.0)
        self.assertEqual(summary["best_train_scaled_mae"], 0.4)
        self.assertEqual(summary["epochs_run"], 2)


if __name__ == "__main__":
    unittest.main()
