import unittest
from pathlib import Path
from scripts.run_graph_wavenet import build_parser, output_paths
class GraphWaveNetRunnerTest(unittest.TestCase):
 def test_defaults_and_output(self):
  a=build_parser().parse_args([]); self.assertEqual(a.seed,42); j,m=output_paths('80_bus_rural_reference_grid',Path('results/centralized')); self.assertEqual(j.name,'graph_wavenet_80bus_p_calendar.json')
