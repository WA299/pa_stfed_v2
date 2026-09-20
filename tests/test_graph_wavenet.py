import unittest

import numpy as np

import code.models.graph_wavenet_baseline as graph_wavenet
from code.models.centralized_gru import masked_scaled_mae


TORCH_AVAILABLE = graph_wavenet.TORCH_AVAILABLE


class GraphWaveNetTest(unittest.TestCase):
 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_support_adaptive_and_all_grid_sizes_forward(self):
  import torch
  for n in (39, 50, 56, 80):
   edges=np.vstack([np.arange(n-1),np.arange(1,n)])
   model=graph_wavenet.GraphWaveNetBaseline(edges,n).eval()
   adaptive=model.adaptive_adjacency()
   self.assertEqual(tuple(adaptive.shape),(n,n))
   self.assertTrue(torch.isfinite(adaptive).all())
   self.assertTrue(torch.allclose(adaptive.sum(1),torch.ones(n),atol=1e-6))
   with torch.no_grad():
    output=model(torch.randn(1,168,n,6))
   self.assertEqual(tuple(output.shape),(1,n))
   self.assertTrue(torch.isfinite(output).all())

 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_predefined_support_shape(self):
  import torch
  support=graph_wavenet.normalized_physical_support(np.asarray([[0,1],[1,2]]),3)
  self.assertEqual(tuple(support.shape),(3,3))
  self.assertTrue(torch.isfinite(support).all())
  self.assertTrue(torch.allclose(support,support.T))

 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_diffusion_order_two_and_projection_gradient(self):
  import torch
  conv=graph_wavenet.DiffusionGraphConv(1,1,num_supports=1,order=2)
  with torch.no_grad():
   conv.projection.weight.zero_(); conv.projection.bias.zero_()
   conv.projection.weight[0,2,0,0]=1.0
  support=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
  x=torch.arange(3,dtype=torch.float32).view(1,1,3,1)
  output=conv(x,(support,))
  expected=torch.einsum('ij,bcjt->bcit',support,torch.einsum('ij,bcjt->bcit',support,x))
  self.assertTrue(torch.allclose(output,expected))
  output.sum().backward()
  self.assertIsNotNone(conv.projection.weight.grad)

 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_graph_parameters_gradients_and_residual_skip_shapes(self):
  import torch
  n=5; edges=np.vstack([np.arange(n-1),np.arange(1,n)])
  model=graph_wavenet.GraphWaveNetBaseline(edges,n,blocks=1,layers_per_block=2)
  x=torch.randn(2,20,n,6)
  hidden,skips=model.temporal_forward(x)
  self.assertEqual(tuple(hidden.shape),(2,32,n,20))
  self.assertEqual(len(skips),2)
  self.assertTrue(all(tuple(skip.shape)==(2,64,n,20) for skip in skips))
  model(x).sum().backward()
  graph_parameters=[parameter for name,parameter in model.named_parameters() if name.startswith('graph_convs.')]
  self.assertTrue(graph_parameters)
  self.assertTrue(all(parameter.grad is not None for parameter in graph_parameters))
  self.assertIsNotNone(model.nodevec1.grad)
  self.assertIsNotNone(model.nodevec2.grad)

 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_temporal_forward_is_causal_in_eval_mode(self):
  import torch
  n=4; edges=np.vstack([np.arange(n-1),np.arange(1,n)])
  model=graph_wavenet.GraphWaveNetBaseline(edges,n,blocks=1,layers_per_block=2).eval()
  original=torch.randn(1,16,n,6); changed=original.clone(); cut=7
  changed[:,cut+1:]+=100.0
  with torch.no_grad():
   first,_=model.temporal_forward(original); second,_=model.temporal_forward(changed)
  self.assertTrue(torch.allclose(first[...,cut],second[...,cut],atol=1e-6))

 @unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is not installed')
 def test_masked_loss_excludes_non_load_and_backward(self):
  import torch
  prediction=torch.tensor([[2.,1000.,4.]],requires_grad=True)
  target=torch.tensor([[1.,-1000.,7.]])
  mask=torch.tensor([True,False,True])
  loss=masked_scaled_mae(prediction,target,mask)
  self.assertAlmostEqual(float(loss.detach()),2.0)
  loss.backward()
  self.assertEqual(float(prediction.grad[0,1]),0.0)
