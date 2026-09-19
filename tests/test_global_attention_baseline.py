from __future__ import annotations

import unittest

import numpy as np

from code.models.centralized_gru import masked_scaled_mae
from code.models.global_attention_baseline import TORCH_AVAILABLE, GlobalAttentionBaseline


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class GlobalAttentionBaselineTest(unittest.TestCase):
    def test_global_attention_shape_rows_normalized_and_topology_free(self) -> None:
        import torch

        model = GlobalAttentionBaseline()
        self.assertFalse(hasattr(model, "attention_mask"))
        self.assertFalse(hasattr(model, "edge_index"))
        prediction, weights = model(torch.randn(2, 168, 4, 6), return_attention=True)
        self.assertEqual(tuple(weights.shape), (2, 4, 4))
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6))
        self.assertTrue(torch.all(weights[:, 0, 3] > 0))

    def test_load_source_mask_zeroes_non_load_sources_but_keeps_all_queries(self) -> None:
        import torch

        model = GlobalAttentionBaseline()
        source_mask = torch.tensor([True, False, True, False])
        prediction, weights = model(
            torch.randn(2, 168, 4, 6),
            source_mask=source_mask,
            return_attention=True,
        )
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertEqual(tuple(weights.shape), (2, 4, 4))
        self.assertTrue(torch.all(weights[..., ~source_mask] == 0))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6))
        self.assertTrue(torch.all(weights[..., source_mask] > 0))

    def test_different_node_counts_and_masked_backward(self) -> None:
        import torch

        for node_count in (3, 5):
            model = GlobalAttentionBaseline()
            prediction = model(torch.randn(2, 168, node_count, 6))
            self.assertEqual(tuple(prediction.shape), (2, node_count))
            target = torch.randn(2, node_count)
            mask = torch.tensor([True] + [False] * (node_count - 1))
            loss = masked_scaled_mae(prediction, target, mask)
            loss.backward()
            self.assertIsNotNone(model.gru.weight_ih_l0.grad)


if __name__ == "__main__":
    unittest.main()
