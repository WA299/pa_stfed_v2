from __future__ import annotations

import unittest

import numpy as np

from code.models.centralized_gru import masked_scaled_mae
from code.models.stgcn_baseline import (
    TORCH_AVAILABLE,
    STGCNBaseline,
    build_adjacency,
    build_binary_physical_normalized_adjacency,
)


class STGCNTopologyTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_identity_adjacency_is_identity(self) -> None:
        import torch

        adjacency = build_adjacency(np.asarray([[0], [1]], dtype=np.int64), 4, "identity")
        self.assertTrue(torch.equal(adjacency, torch.eye(4)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_adjacency_is_bidirectional_self_looped_symmetric_and_shaped(self) -> None:
        import torch

        edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        adjacency = build_binary_physical_normalized_adjacency(edge_index, num_nodes=4)
        self.assertEqual(tuple(adjacency.shape), (4, 4))
        np.testing.assert_allclose(adjacency.cpu().numpy(), adjacency.cpu().numpy().T, atol=1e-7)
        self.assertTrue(torch.all(adjacency.diag() > 0))
        self.assertGreater(float(adjacency[1, 0]), 0.0)
        self.assertGreater(float(adjacency[0, 1]), 0.0)
        self.assertGreater(float(adjacency[2, 1]), 0.0)
        self.assertGreater(float(adjacency[1, 2]), 0.0)
        self.assertGreater(float(adjacency[3, 3]), 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_forward_supports_different_node_counts_and_backward(self) -> None:
        import torch

        for node_count in (3, 5):
            edge_index = np.asarray(
                [np.arange(node_count - 1), np.arange(1, node_count)],
                dtype=np.int64,
            )
            model = STGCNBaseline(edge_index, num_nodes=node_count)
            x = torch.randn(2, 168, node_count, 6)
            target = torch.randn(2, node_count)
            mask = torch.tensor([True] + [False] * (node_count - 1))
            prediction = model(x)
            self.assertEqual(tuple(prediction.shape), (2, node_count))
            loss = masked_scaled_mae(prediction, target, mask)
            loss.backward()
            self.assertIsNotNone(model.input_projection.weight.grad)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_binary_mode_matches_original_adjacency_builder(self) -> None:
        import torch

        edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        original = build_binary_physical_normalized_adjacency(edge_index, 3)
        selected = build_adjacency(edge_index, 3, "binary_physical")
        self.assertTrue(torch.equal(original, selected))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_both_adjacency_modes_have_same_output_shape(self) -> None:
        import torch

        edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        x = torch.randn(2, 168, 3, 6)
        binary = STGCNBaseline(edge_index, 3, adjacency_mode="binary_physical")
        identity = STGCNBaseline(edge_index, 3, adjacency_mode="identity")
        self.assertEqual(tuple(binary(x).shape), tuple(identity(x).shape))


if __name__ == "__main__":
    unittest.main()
