import unittest
from pathlib import Path
from scripts.run_astgcn import build_parser, output_paths

class ASTGCNRunnerTest(unittest.TestCase):
    def test_defaults_and_output(self):
        args=build_parser().parse_args([]); self.assertEqual(args.seed,42); self.assertEqual(args.max_epochs,50); self.assertEqual(args.patience,8); self.assertEqual(args.batch_size,32)
        j,m=output_paths('80_bus_rural_reference_grid',Path('results/centralized')); self.assertEqual(j.name,'astgcn_80bus_p_calendar.json'); self.assertEqual(m.name,'astgcn_80bus_p_calendar.md')

if __name__=='__main__': unittest.main()
