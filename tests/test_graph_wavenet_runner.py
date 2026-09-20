import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_graph_wavenet import build_parser, model_config, output_paths, render_markdown


class GraphWaveNetRunnerTest(unittest.TestCase):
 def test_defaults_and_output(self):
  args=build_parser().parse_args([])
  self.assertEqual(args.seed,42); self.assertEqual(args.max_epochs,50)
  self.assertEqual(args.patience,8); self.assertEqual(args.batch_size,32)
  self.assertEqual(args.learning_rate,1e-3)
  json_path,markdown_path=output_paths('80_bus_rural_reference_grid',Path('results/centralized'))
  self.assertEqual(json_path.name,'graph_wavenet_80bus_p_calendar.json')
  self.assertEqual(markdown_path.name,'graph_wavenet_80bus_p_calendar.md')

 def test_metadata_and_report_are_graph_wavenet_validation_only(self):
  args=build_parser().parse_args([])
  config=model_config(args,'cpu',SimpleNamespace(receptive_field=13))
  self.assertEqual(config['model'],'GraphWaveNet')
  self.assertEqual(config['implementation'],'GraphWaveNet_core_adaptation')
  self.assertEqual(config['reference'],'Wu_et_al_IJCAI_2019')
  self.assertEqual(config['diffusion_order'],2)
  self.assertEqual(config['dropout'],0.3)
  self.assertEqual(config['receptive_field'],13)
  self.assertFalse(config['test_evaluated'])
  metric={'mae':1.0,'rmse':2.0,'wape_pct':3.0,'smape_pct':4.0}
  report={
   'config':config,
   'validation':{'graph_wavenet':{'node_macro':metric,'grid_aggregate':metric},'persistence_1h':{'node_macro':metric,'grid_aggregate':metric}},
   'comparison':{'node_macro':{key:'equal' for key in metric},'grid_aggregate':{key:'equal' for key in metric}},
   'summary':{'best_epoch':1,'best_validation_node_macro_mae':1.0,'best_train_scaled_mae':1.0,'epochs_run':1},
  }
  markdown=render_markdown(report)
  self.assertIn('# Graph WaveNet baseline:',markdown)
  self.assertNotIn('STGCN',markdown)
