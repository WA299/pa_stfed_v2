import unittest

import numpy as np

import code.models.astgcn_baseline as astgcn
from code.models.centralized_gru import masked_scaled_mae


TORCH_AVAILABLE = astgcn.TORCH_AVAILABLE


def chain_edges(num_nodes):
    return np.vstack([np.arange(num_nodes - 1), np.arange(1, num_nodes)])


class ASTGCNTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_attention_layers_match_reference_bilinear_equations(self):
        import torch

        batch, nodes, channels, timesteps = 2, 3, 2, 4
        x = torch.randn(batch, nodes, channels, timesteps)
        temporal_layer = astgcn.TemporalAttentionLayer(nodes, channels, timesteps)
        temporal = temporal_layer(x)
        temporal_lhs = (x.permute(0, 3, 2, 1) @ temporal_layer.U1) @ temporal_layer.U2
        temporal_rhs = torch.einsum("f,bnft->bnt", temporal_layer.U3, x)
        temporal_expected = torch.softmax(
            temporal_layer.Ve
            @ torch.sigmoid(temporal_lhs @ temporal_rhs + temporal_layer.be),
            dim=1,
        )
        self.assertEqual(tuple(temporal_layer.U1.shape), (nodes,))
        self.assertEqual(tuple(temporal_layer.U2.shape), (channels, nodes))
        self.assertEqual(tuple(temporal_layer.U3.shape), (channels,))
        self.assertEqual(tuple(temporal_layer.be.shape), (1, timesteps, timesteps))
        self.assertEqual(tuple(temporal_layer.Ve.shape), (timesteps, timesteps))
        self.assertTrue(torch.allclose(temporal, temporal_expected))

        spatial_layer = astgcn.SpatialAttentionLayer(nodes, channels, timesteps)
        spatial = spatial_layer(x)
        spatial_lhs = (x @ spatial_layer.W1) @ spatial_layer.W2
        spatial_rhs = torch.einsum(
            "f,bnft->bnt", spatial_layer.W3, x
        ).transpose(1, 2)
        spatial_expected = torch.softmax(
            spatial_layer.Vs
            @ torch.sigmoid(spatial_lhs @ spatial_rhs + spatial_layer.bs),
            dim=1,
        )
        self.assertEqual(tuple(spatial_layer.W1.shape), (timesteps,))
        self.assertEqual(tuple(spatial_layer.W2.shape), (channels, timesteps))
        self.assertEqual(tuple(spatial_layer.W3.shape), (channels,))
        self.assertEqual(tuple(spatial_layer.bs.shape), (1, nodes, nodes))
        self.assertEqual(tuple(spatial_layer.Vs.shape), (nodes, nodes))
        self.assertTrue(torch.allclose(spatial, spatial_expected))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_graph_construction_and_chebyshev_supports(self):
        import torch

        adjacency = astgcn.build_physical_adjacency(chain_edges(5), 5)
        self.assertEqual(tuple(adjacency.shape), (5, 5))
        self.assertTrue(torch.equal(adjacency, adjacency.T))
        self.assertTrue(torch.equal(torch.diag(adjacency), torch.zeros(5)))

        laplacian = astgcn.build_scaled_laplacian(adjacency)
        self.assertTrue(torch.isfinite(laplacian).all())
        self.assertTrue(torch.allclose(laplacian, laplacian.T, atol=1e-6))
        supports = astgcn.chebyshev_supports(laplacian, 3)
        self.assertIsInstance(supports, list)
        self.assertEqual(len(supports), 3)
        self.assertTrue(all(tuple(support.shape) == (5, 5) for support in supports))
        self.assertTrue(torch.equal(supports[0], torch.eye(5)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_cheb_propagation_uses_transposed_attention_weighted_support(self):
        import torch

        convolution = astgcn.ChebConvWithSpatialAttention(1, 1, 1)
        with torch.no_grad():
            convolution.Theta[0].fill_(1.0)
        support = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        spatial_attention = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        graph_signal = torch.tensor([[[[5.0]], [[7.0]]]])
        output = convolution(graph_signal, spatial_attention, [support])
        expected = (
            (support.unsqueeze(0) * spatial_attention).transpose(1, 2)
            @ graph_signal[..., 0]
        ).unsqueeze(-1)
        self.assertTrue(torch.allclose(output, expected))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_cheb_convolution_applies_relu_to_negative_preactivation(self):
        import torch

        convolution = astgcn.ChebConvWithSpatialAttention(1, 1, 1)
        with torch.no_grad():
            convolution.Theta[0].fill_(-1.0)
        support = torch.eye(2)
        spatial_attention = torch.ones(1, 2, 2)
        graph_signal = torch.tensor([[[[2.0]], [[3.0]]]])
        preactivation = (
            (support.unsqueeze(0) * spatial_attention).transpose(1, 2)
            @ graph_signal[..., 0]
        ) @ convolution.Theta[0]
        self.assertTrue((preactivation < 0).all())
        output = convolution(graph_signal, spatial_attention, [support])
        self.assertTrue((output >= 0).all())
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_block_uses_temporal_input_for_attention_and_original_input_for_cheb(self):
        import torch

        block = astgcn.ASTGCNBlock(1, 2, num_nodes=2, timesteps=3, chebyshev_k=1)
        x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 1, 3)
        temporal_attention = torch.eye(3).flip(1).unsqueeze(0)
        captured = {}

        block.temporal_attention.forward = lambda _: temporal_attention

        def capture_spatial(value):
            captured["spatial_input"] = value.detach().clone()
            return torch.ones(1, 2, 2) / 2.0

        def capture_cheb(value, spatial, supports):
            captured["cheb_input"] = value.detach().clone()
            return torch.zeros(1, 2, 2, 3)

        block.spatial_attention.forward = capture_spatial
        block.cheb_conv.forward = capture_cheb
        block(x, [torch.eye(2)])
        expected_temporal = torch.matmul(x.reshape(1, 2, 3), temporal_attention)
        expected_temporal = expected_temporal.reshape_as(x)
        self.assertTrue(torch.equal(captured["spatial_input"], expected_temporal))
        self.assertTrue(torch.equal(captured["cheb_input"], x))
        self.assertFalse(torch.equal(captured["cheb_input"], expected_temporal))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_all_grid_sizes_forward(self):
        import torch

        for num_nodes in (39, 50, 56, 80):
            model = astgcn.ASTGCNBaseline(chain_edges(num_nodes), num_nodes).eval()
            with torch.no_grad():
                output = model(torch.randn(1, 168, num_nodes, 6))
            self.assertEqual(tuple(output.shape), (1, num_nodes))
            self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_layer_norm_and_final_convolution_match_reference_shapes(self):
        model = astgcn.ASTGCNBaseline(chain_edges(4), 4)
        self.assertTrue(
            all(block.layer_norm.normalized_shape == (32,) for block in model.blocks)
        )
        self.assertEqual(model.final_conv.in_channels, 168)
        self.assertEqual(model.final_conv.out_channels, 1)
        self.assertEqual(model.final_conv.kernel_size, (1, 32))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_attention_shapes_finite_and_normalized(self):
        import torch

        model = astgcn.ASTGCNBaseline(chain_edges(4), 4).eval()
        with torch.no_grad():
            model(torch.randn(2, 168, 4, 6))
        for block in model.blocks:
            temporal = block.temporal_attention_weights
            spatial = block.spatial_attention_weights
            self.assertEqual(tuple(temporal.shape), (2, 168, 168))
            self.assertEqual(tuple(spatial.shape), (2, 4, 4))
            self.assertTrue(torch.isfinite(temporal).all())
            self.assertTrue(torch.isfinite(spatial).all())
            self.assertTrue(
                torch.allclose(temporal.sum(dim=1), torch.ones(2, 168), atol=1e-5)
            )
            self.assertTrue(
                torch.allclose(spatial.sum(dim=1), torch.ones(2, 4), atol=1e-5)
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_prediction_backward_reaches_attention_and_cheb_parameters(self):
        import torch

        model = astgcn.ASTGCNBaseline(chain_edges(4), 4)
        prediction = model(torch.randn(1, 168, 4, 6))
        prediction.square().mean().backward()
        for block in model.blocks:
            temporal_parameters = (
                block.temporal_attention.U1,
                block.temporal_attention.U2,
                block.temporal_attention.U3,
                block.temporal_attention.be,
                block.temporal_attention.Ve,
            )
            spatial_parameters = (
                block.spatial_attention.W1,
                block.spatial_attention.W2,
                block.spatial_attention.W3,
                block.spatial_attention.bs,
                block.spatial_attention.Vs,
            )
            self.assertTrue(all(parameter.grad is not None for parameter in temporal_parameters))
            self.assertTrue(all(parameter.grad is not None for parameter in spatial_parameters))
            self.assertTrue(
                all(theta.grad is not None for theta in block.cheb_conv.Theta)
            )
        self.assertIsNotNone(model.final_conv.weight.grad)
        self.assertIsNotNone(model.final_conv.bias.grad)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_masked_loss_excludes_non_load_nodes(self):
        import torch

        prediction = torch.tensor([[2.0, 1000.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.0, -1000.0, 7.0]])
        load_mask = torch.tensor([True, False, True])
        loss = masked_scaled_mae(prediction, target, load_mask)
        self.assertAlmostEqual(float(loss.detach()), 2.0)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
