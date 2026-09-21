import unittest
from pathlib import Path

from code.models.centralized_gru import best_epoch_summary
from scripts.run_puc_rstattn_v2 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    build_epoch_record,
    build_parser,
    output_paths,
    report_metadata,
    render_markdown,
)


class TestPUCRSTAttnV2Runner(unittest.TestCase):
    def test_defaults_outputs_metadata_and_summary_contract(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.seed, args.max_epochs, args.patience, args.batch_size, args.learning_rate), (DEFAULT_SEED, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE))
        json_path, md_path = output_paths(args.grid, Path("results/centralized"))
        self.assertTrue(str(json_path).endswith("puc_rstattn_v2_39bus_p_calendar.json"))
        self.assertTrue(str(md_path).endswith("puc_rstattn_v2_39bus_p_calendar.md"))
        metadata = report_metadata()
        self.assertEqual(metadata["model"], "PUC-RSTAttn-V2")
        self.assertTrue(metadata["gru_anchor_preserved"])
        self.assertTrue(metadata["utility_attention_prior"])
        self.assertEqual(metadata["physical_relation_features"], ["normalized_impedance_abs_distance", "normalized_hop_distance", "direct_physical_adjacency"])
        self.assertEqual(metadata["dynamic_attention_scale_initial"], 0.0)
        self.assertEqual(metadata["anchor_loss_weight"], 0.2)
        self.assertFalse(metadata["validation_used_for_graph"])
        self.assertFalse(metadata["test_evaluated"])
        self.assertEqual(
            metadata["physical_relation_features"],
            ["normalized_impedance_abs_distance", "normalized_hop_distance", "direct_physical_adjacency"],
        )
        validation = {
            method: {
                scope: {"mae": 1.0, "rmse": 1.0, "wape_pct": 1.0, "smape_pct": 1.0}
                for scope in ("node_macro", "grid_aggregate")
            }
            for method in ("final", "temporal_residual", "gru_anchor", "persistence_1h")
        }
        diagnostics = {
            "mean_temporal_gate": 0.0,
            "mean_spatial_gate_active": 0.0,
            "mean_abs_temporal_correction": 0.0,
            "mean_abs_centered_spatial_residual": 0.0,
            "mean_spatial_attention_entropy": 0.0,
            "aggregate_preservation_max_abs_error": 0.0,
        }
        markdown = render_markdown({"config": {"grid_name": args.grid}, "validation": validation, "validation_diagnostics": diagnostics})
        for method in validation:
            for scope in validation[method]:
                self.assertIn(f"| {method} | {scope} |", markdown)
        for key in diagnostics:
            self.assertIn(key, markdown)
        history = [build_epoch_record(1, [0.4], [0.5], [0.5], {"node_macro": {"mae": 1.0}}, {"node_macro": {"mae": 1.1}}, {"node_macro": {"mae": 1.2}})]
        summary = best_epoch_summary(history)
        self.assertEqual(summary["best_epoch"], 1)
        self.assertEqual(summary["best_train_scaled_mae"], 0.4)


if __name__ == "__main__":
    unittest.main()
