import json, tempfile, unittest
from pathlib import Path
from scripts.summarize_centralized_results import summarize

class SummaryTest(unittest.TestCase):
 def test_validation_only_missing_and_macro(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'astgcn_39bus_p_calendar.json'; metric={'mae':1.,'rmse':2.,'wape_pct':10.,'smape_pct':20.}
   p.write_text(json.dumps({'validation':{'astgcn':{'node_macro':metric,'grid_aggregate':metric},'test':{'node_macro':{'mae':999}}}}))
   r=summarize(Path(td)); self.assertEqual(r['standard_baselines']['39bus']['astgcn']['node_macro']['mae'],1.); self.assertIsNone(r['standard_baselines']['50bus']['astgcn']); self.assertEqual(r['macro_across_grids_node']['astgcn']['wape_pct'],10.); self.assertFalse(r['metadata']['test_evaluated'])
 def test_diagnostic_separate(self):
  r=summarize(Path('does-not-exist')); self.assertIn('diagnostic_variants',r); self.assertNotIn('global_attention',r['standard_baselines']['39bus'])
