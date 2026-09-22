import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_puc_rstattn_v2_ablation import VARIANTS
from scripts.summarize_puc_rstattn_v2_ablation import summarize
from scripts.summarize_centralized_results import summarize as summarize_centralized


def metric(value):
    return {"mae": value, "rmse": value * 2, "wape_pct": value * 10, "smape_pct": value * 20}


def validation(value):
    return {scope: metric(value if scope == "node_macro" else value * 2) for scope in ("node_macro", "grid_aggregate")}


class TestPUCRSTAttnV2AblationSummary(unittest.TestCase):
    def _write_reports(self, root):
        centralized = root / "centralized"
        ablations = root / "ablations"
        centralized.mkdir(); ablations.mkdir()
        for index, grid in enumerate(("39bus", "50bus", "56bus", "80bus"), start=1):
            full = {"config": {"model": "PUC-RSTAttn-V2", "feature_mode": "p_calendar", "test_evaluated": False}, "validation": {"final": validation(10.0 + index)}, "test_evaluated": False}
            (centralized / f"puc_rstattn_v2_{grid}_p_calendar.json").write_text(json.dumps(full))
            for variant_index, variant in enumerate(VARIANTS, start=1):
                ablation = {
                    "config": {"experiment": "puc_rstattn_v2_formal_ablation", "variant": variant, "independent_training": True, "seed": 42, "feature_mode": "p_calendar", "history_length": 168, "forecast_horizon": 1, "utility_top_k": 3, "utility_positive_only": True, "validation_used_for_graph": False, "test_evaluated": False},
                    "validation": {"final": validation(float(variant_index + index))}, "test_evaluated": False,
                }
                (ablations / f"puc_rstattn_v2_{variant}_{grid}_p_calendar.json").write_text(json.dumps(ablation))
        return centralized, ablations

    def test_summary_macro_and_deltas(self):
        with tempfile.TemporaryDirectory() as td:
            centralized, ablations = self._write_reports(Path(td))
            report = summarize(centralized, ablations)
            self.assertEqual(report["metadata"]["test_evaluated"], False)
            self.assertEqual(report["macro_across_grids"]["node_macro"]["gru_anchor_only"]["metrics"]["mae"], 3.5)
            self.assertAlmostEqual(report["incremental_deltas"]["temporal_residual_only_vs_gru_anchor_only"]["node_macro_mae_delta_unweighted_macro"], 1.0)
            self.assertAlmostEqual(report["incremental_deltas"]["full_puc_rstattn_v2_vs_utility_prior_no_physics"]["node_macro_mae_delta_unweighted_macro"], 6.0)

    def test_centralized_summary_includes_all_frozen_v2_files(self):
        with tempfile.TemporaryDirectory() as td:
            centralized, _ = self._write_reports(Path(td))
            report = summarize_centralized(centralized)
            self.assertEqual(report["standard_baselines"]["39bus"]["puc_rstattn_v2"]["node_macro"]["mae"], 11.0)
            self.assertEqual(report["macro_across_grids"]["node_macro"]["puc_rstattn_v2"]["available_grid_count"], 4)

    def test_rejects_top_level_test_evaluation_and_graph_validation(self):
        with tempfile.TemporaryDirectory() as td:
            centralized, ablations = self._write_reports(Path(td))
            path = ablations / "puc_rstattn_v2_gru_anchor_only_39bus_p_calendar.json"
            data = json.loads(path.read_text())
            data["test_evaluated"] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                summarize(centralized, ablations)
            data["test_evaluated"] = False
            data["config"]["validation_used_for_graph"] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                summarize(centralized, ablations)

    def test_rejects_top_level_full_model_test_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            centralized, ablations = self._write_reports(Path(td))
            path = centralized / "puc_rstattn_v2_39bus_p_calendar.json"
            data = json.loads(path.read_text())
            data["test_evaluated"] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                summarize(centralized, ablations)


if __name__ == "__main__":
    unittest.main()
