from __future__ import annotations

import unittest

import numpy as np

from code.models.centralized_gru import masked_scaled_mae
from code.models.gat_st_baseline import (
    TORCH_AVAILABLE,
    GATSTBaseline,
    build_bidirectional_self_loop_edge_index,
    build_physical_attention_mask,
)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class GATSTBaselineTest(unittest.TestCase):
    def test_directed_edges_expand_and_self_loops_exist(self) -> None:
        import torch

        expanded = build_bidirectional_self_loop_edge_index(np.asarray([[0, 1], [1, 2]]), 3)
        pairs = {tuple(pair) for pair in expanded.t().tolist()}
        self.assertTrue({(0, 1), (1, 0), (1, 2), (2, 1)}.issubset(pairs))
        self.assertTrue({(0, 0), (1, 1), (2, 2)}.issubset(pairs))
        self.assertEqual(expanded.dtype, torch.long)

    def test_attention_mask_and_weights_are_local_and_normalized(self) -> None:
        import torch

        edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        mask = build_physical_attention_mask(edge_index, 4)
        expected = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask, expected))
        model = GATSTBaseline(edge_index, num_nodes=4)
        prediction, weights = model(torch.randn(2, 168, 4, 6), return_attention=True)
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6))
        self.assertTrue(torch.all(weights.masked_select(~mask.unsqueeze(0)) == 0))

    def test_different_node_counts_output_and_masked_backward(self) -> None:
        import torch

        for node_count in (3, 5):
            edge_index = np.asarray([np.arange(node_count - 1), np.arange(1, node_count)])
            model = GATSTBaseline(edge_index, num_nodes=node_count)
            prediction = model(torch.randn(2, 168, node_count, 6))
            self.assertEqual(tuple(prediction.shape), (2, node_count))
            target = torch.randn(2, node_count)
            loss = masked_scaled_mae(prediction, target, torch.tensor([True] + [False] * (node_count - 1)))
            loss.backward()
            self.assertIsNotNone(model.gru.weight_ih_l0.grad)


if __name__ == "__main__":
    unittest.main()
