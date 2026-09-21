import json, tempfile, unittest
from pathlib import Path
from scripts.summarize_centralized_results import render_markdown, summarize

def payload(model, value, persistence=None):
    m={"mae":value,"rmse":value*2,"wape_pct":value*10,"smape_pct":value*20}
    validation={model:{"node_macro":m,"grid_aggregate":{k:v*2 for k,v in m.items()}}}
    if persistence is not None: validation["persistence_1h"]={"node_macro":persistence,"grid_aggregate":persistence}
    return {"validation":validation,"test":{"astgcn":{"node_macro":{"mae":999}}}}

class SummaryTest(unittest.TestCase):
    @staticmethod
    def stgcn_payload(value, *, adjacency_mode="binary_physical", adjacency_type="binary_physical_normalized", topology=True, electrical=False, feature_mode="p_calendar"):
        metric={"mae":value,"rmse":value*2,"wape_pct":value*10,"smape_pct":value*20}
        return {"config":{"feature_mode":feature_mode,"topology_used":topology,"electrical_edge_features_used":electrical,"adjacency_mode":adjacency_mode,"adjacency_type":adjacency_type},"validation":{"stgcn":{"node_macro":metric,"grid_aggregate":metric},"persistence_1h":{"node_macro":metric,"grid_aggregate":metric}}}

    def test_stgcn_legacy_and_binary_physical_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            root.joinpath("stgcn_39bus_p_calendar.json").write_text(json.dumps(self.stgcn_payload(1., adjacency_mode="", adjacency_type="binary_physical_normalized")))
            root.joinpath("stgcn_50bus_p_calendar_binary_physical.json").write_text(json.dumps(self.stgcn_payload(2.)))
            report=summarize(root)
            self.assertEqual(report["standard_baselines"]["39bus"]["stgcn"]["node_macro"]["mae"],1.)
            self.assertEqual(report["standard_baselines"]["50bus"]["stgcn"]["node_macro"]["mae"],2.)

    def test_stgcn_diagnostics_are_excluded_and_four_grid_availability(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for grid,value in zip(("39bus","50bus","56bus","80bus"),(1.,2.,3.,4.)):
                root.joinpath(f"stgcn_{grid}_p_calendar_binary_physical.json").write_text(json.dumps(self.stgcn_payload(value)))
            root.joinpath("stgcn_39bus_p_calendar_identity.json").write_text(json.dumps(self.stgcn_payload(99., adjacency_mode="identity", adjacency_type="identity", topology=False)))
            report=summarize(root)
            self.assertEqual(report["macro_across_grids"]["node_macro"]["stgcn"]["available_grid_count"],4)
            self.assertEqual(report["macro_across_grids"]["node_macro"]["stgcn"]["expected_grid_count"],4)
            self.assertEqual(report["standard_baselines"]["39bus"]["stgcn"]["node_macro"]["mae"],1.)

    def test_stgcn_ambiguous_candidates_raise(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            root.joinpath("stgcn_39bus_p_calendar_binary_physical.json").write_text(json.dumps(self.stgcn_payload(1.)))
            root.joinpath("stgcn_39bus_p_calendar.json").write_text(json.dumps(self.stgcn_payload(2., adjacency_mode="", adjacency_type="binary_physical_normalized")))
            with self.assertRaisesRegex(ValueError,"ambiguous standard STGCN results"):
                summarize(root)

    def test_stgcn_binary_candidate_has_deterministic_priority(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            root.joinpath("stgcn_39bus_p_calendar_binary_physical.json").write_text(json.dumps(self.stgcn_payload(1.)))
            root.joinpath("stgcn_39bus_p_calendar.json").write_text(json.dumps(self.stgcn_payload(1., adjacency_mode="", adjacency_type="binary_physical_normalized")))
            report=summarize(root)
            self.assertEqual(report["standard_baselines"]["39bus"]["stgcn"]["node_macro"]["mae"],1.)

    def test_stgcn_invalid_binary_candidate_falls_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            root.joinpath("stgcn_39bus_p_calendar_binary_physical.json").write_text(json.dumps(self.stgcn_payload(99., topology=False)))
            root.joinpath("stgcn_39bus_p_calendar.json").write_text(json.dumps(self.stgcn_payload(3., adjacency_mode="", adjacency_type="binary_physical_normalized")))
            report=summarize(root)
            self.assertEqual(report["standard_baselines"]["39bus"]["stgcn"]["node_macro"]["mae"],3.)

    def test_stgcn_contradictory_identity_mode_is_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            root.joinpath("stgcn_39bus_p_calendar.json").write_text(json.dumps(self.stgcn_payload(99., adjacency_mode="identity", adjacency_type="binary_physical_normalized", topology=True)))
            report=summarize(root)
            self.assertIsNone(report["standard_baselines"]["39bus"]["stgcn"])

    def test_validation_only_missing_and_macro(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"astgcn_39bus_p_calendar.json"; p={"mae":1.,"rmse":2.,"wape_pct":10.,"smape_pct":20.}; path.write_text(json.dumps(payload("astgcn",1.,p)))
            r=summarize(Path(td)); self.assertEqual(r["standard_baselines"]["39bus"]["astgcn"]["node_macro"]["mae"],1.); self.assertIsNone(r["standard_baselines"]["50bus"]["astgcn"]); self.assertEqual(r["macro_across_grids"]["node_macro"]["astgcn"]["metrics"]["wape_pct"],10.); self.assertEqual(r["macro_across_grids"]["grid_aggregate"]["astgcn"]["metrics"]["mae"],2.); self.assertEqual(r["macro_across_grids"]["node_macro"]["astgcn"]["available_grid_count"],1); self.assertEqual(r["macro_across_grids"]["node_macro"]["astgcn"]["expected_grid_count"],4); self.assertFalse(r["metadata"]["test_evaluated"])
    def test_shared_gru_filename_and_validation_key(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td,"gru_39bus_p_calendar.json").write_text(json.dumps(payload("gru",3.)))
            r=summarize(Path(td)); self.assertEqual(r["standard_baselines"]["39bus"]["shared_gru"]["node_macro"]["mae"],3.)
    def test_persistence_loaded_and_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            p={"mae":1.,"rmse":2.,"wape_pct":10.,"smape_pct":20.}; Path(td,"astgcn_39bus_p_calendar.json").write_text(json.dumps(payload("astgcn",1.,p))); stgcn=self.stgcn_payload(2., adjacency_mode="", adjacency_type="binary_physical_normalized"); stgcn["validation"]["persistence_1h"]={"node_macro":p,"grid_aggregate":p}; Path(td,"stgcn_39bus_p_calendar.json").write_text(json.dumps(stgcn))
            self.assertIsNotNone(summarize(Path(td))["standard_baselines"]["39bus"]["persistence_1h"])
    def test_inconsistent_persistence_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p={"mae":1.,"rmse":2.,"wape_pct":10.,"smape_pct":20.}; q=dict(p); q["mae"]=3.; Path(td,"astgcn_39bus_p_calendar.json").write_text(json.dumps(payload("astgcn",1.,p))); stgcn=self.stgcn_payload(2., adjacency_mode="", adjacency_type="binary_physical_normalized"); stgcn["validation"]["persistence_1h"]["node_macro"]["mae"]=3.; Path(td,"stgcn_39bus_p_calendar.json").write_text(json.dumps(stgcn))
            with self.assertRaises(ValueError): summarize(Path(td))
    def test_diagnostic_separate_and_metadata(self):
        r=summarize(Path("does-not-exist")); self.assertIn("diagnostic_variants",r); self.assertNotIn("global_attention",r["standard_baselines"]["39bus"]); self.assertFalse(r["metadata"]["micro_wape_computed"]); self.assertEqual(r["metadata"]["wape_unit"],"percent"); self.assertIn("missing",render_markdown(r))

if __name__=="__main__": unittest.main()
