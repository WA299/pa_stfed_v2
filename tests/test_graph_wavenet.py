import unittest, numpy as np
from code.models.graph_wavenet_baseline import TORCH_AVAILABLE, GraphWaveNetBaseline, normalized_physical_support
class GraphWaveNetTest(unittest.TestCase):
 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_support_adaptive_and_forward(self):
  import torch
  n=5; e=np.vstack([np.arange(n-1),np.arange(1,n)]); m=GraphWaveNetBaseline(e,n); a=m.adaptive_adjacency(); self.assertEqual(tuple(a.shape),(n,n)); self.assertTrue(torch.isfinite(a).all()); self.assertTrue(torch.allclose(a.sum(1),torch.ones(n))); y=m(torch.randn(1,168,n,6)); self.assertEqual(tuple(y.shape),(1,n)); y.mean().backward(); self.assertIsNotNone(m.nodevec1.grad)
 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_predefined_support_shape(self):
  import torch
  s=normalized_physical_support(np.asarray([[0,1],[1,2]]),3); self.assertEqual(tuple(s.shape),(3,3)); self.assertTrue(torch.isfinite(s).all())
