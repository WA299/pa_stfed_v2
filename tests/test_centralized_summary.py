import json, tempfile, unittest
from pathlib import Path
from scripts.summarize_centralized_results import render_markdown, summarize

def payload(model, value, persistence=None):
    m={"mae":value,"rmse":value*2,"wape_pct":value*10,"smape_pct":value*20}
    validation={model:{"node_macro":m,"grid_aggregate":{k:v*2 for k,v in m.items()}}}
    if persistence is not None: validation["persistence_1h"]={"node_macro":persistence,"grid_aggregate":persistence}
    return {"validation":validation,"test":{"astgcn":{"node_macro":{"mae":999}}}}

class SummaryTest(unittest.TestCase):
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
            p={"mae":1.,"rmse":2.,"wape_pct":10.,"smape_pct":20.}; Path(td,"astgcn_39bus_p_calendar.json").write_text(json.dumps(payload("astgcn",1.,p))); Path(td,"stgcn_39bus_p_calendar.json").write_text(json.dumps(payload("stgcn",2.,p)))
            self.assertIsNotNone(summarize(Path(td))["standard_baselines"]["39bus"]["persistence_1h"])
    def test_inconsistent_persistence_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p={"mae":1.,"rmse":2.,"wape_pct":10.,"smape_pct":20.}; q=dict(p); q["mae"]=3.; Path(td,"astgcn_39bus_p_calendar.json").write_text(json.dumps(payload("astgcn",1.,p))); Path(td,"stgcn_39bus_p_calendar.json").write_text(json.dumps(payload("stgcn",2.,q)))
            with self.assertRaises(ValueError): summarize(Path(td))
    def test_diagnostic_separate_and_metadata(self):
        r=summarize(Path("does-not-exist")); self.assertIn("diagnostic_variants",r); self.assertNotIn("global_attention",r["standard_baselines"]["39bus"]); self.assertFalse(r["metadata"]["micro_wape_computed"]); self.assertEqual(r["metadata"]["wape_unit"],"percent"); self.assertIn("missing",render_markdown(r))

if __name__=="__main__": unittest.main()
