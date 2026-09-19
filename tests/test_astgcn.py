import unittest
import numpy as np
from code.models.astgcn_baseline import TORCH_AVAILABLE, ASTGCNBaseline, build_scaled_laplacian, build_chebyshev_supports
from code.models.centralized_gru import masked_scaled_mae

class ASTGCNTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is not installed')
    def test_graph_supports_and_forward(self):
        import torch
        for n in (39,50,56,80):
            edges=np.vstack([np.arange(n-1),np.arange(1,n)])
            lap=build_scaled_laplacian(edges,n); self.assertTrue(torch.isfinite(lap).all()); np.testing.assert_allclose(lap.cpu(),lap.cpu().T,atol=1e-6)
            sup=build_chebyshev_supports(lap,3); self.assertEqual(tuple(sup.shape),(3,n,n)); self.assertTrue(torch.isfinite(sup).all())
            model=ASTGCNBaseline(edges,n); x=torch.randn(1,168,n,6); y=model(x); self.assertEqual(tuple(y.shape),(1,n)); self.assertTrue(torch.isfinite(y).all()); masked_scaled_mae(y,torch.randn_like(y),torch.tensor([True]+[False]*(n-1))).backward()
    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is not installed')
    def test_attention_tensors_finite(self):
        import torch
        model=ASTGCNBaseline(np.asarray([[0,1],[1,2]]),3); _=model(torch.randn(2,168,3,6))
        for block in model.blocks:
            self.assertEqual(tuple(block.temporal_attention.shape),(2,168,168)); self.assertEqual(tuple(block.spatial_attention.shape),(2,3,3)); self.assertTrue(torch.isfinite(block.temporal_attention).all()); self.assertTrue(torch.isfinite(block.spatial_attention).all())
