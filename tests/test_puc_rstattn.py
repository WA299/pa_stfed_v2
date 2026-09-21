import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from code.models.centralized_gru import masked_scaled_mae
from code.models.puc_rstattn import (
    ANCHOR_LOSS_WEIGHT,
    PUCRSTAttn,
    build_relation_features,
    normalized_distance_matrices,
    select_utility_neighbors,
    split_utility_targets,
)


class _Grid:
    num_nodes = 5
    edge_index = np.asarray([[0, 1, 2], [1, 2, 3]])
    load_bus_mask = np.asarray([True, True, True, True, False])
    distance_matrices = {
        "impedance_abs_distance": np.asarray([[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]], dtype=float),
        "hop_distance": np.asarray([[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]], dtype=float),
    }


@unittest.skipIf(torch is None, "PyTorch unavailable")
class TestPUCRSTAttn(unittest.TestCase):
    def test_utility_split_and_selection(self):
        indices = np.arange(5964)
        fit, selection = split_utility_targets(indices)
        self.assertEqual((len(fit), len(selection)), (4771, 1193))
        utilities = {(source, target): 0.0 for target in range(4) for source in range(4) if source != target}
        utilities.update({(1, 0): 0.5, (2, 0): 0.5, (3, 0): -1.0})
        selected = select_utility_neighbors(utilities, [0, 1, 2, 3])
        self.assertEqual([item[0] for item in selected[0]], [1, 2])
        self.assertTrue(all(value > 0 for values in selected.values() for _, value in values))

    def test_relation_features_are_four_dimensional_and_no_forced_bias(self):
        edges = np.asarray([[1, 2], [0, 0]])
        utilities = {(1, 0): 0.2, (2, 0): 0.1}
        relation = build_relation_features(_Grid(), edges, utilities, [0, 1, 2, 3])
        self.assertEqual(relation.shape, (2, 4))
        self.assertTrue(np.isfinite(relation).all())
        self.assertAlmostEqual(float(relation[0, 0]), 0.2)
        self.assertAlmostEqual(float(relation[1, 0]), 0.1)
        normalized, _ = normalized_distance_matrices(_Grid(), [0, 1, 2, 3])
        self.assertTrue(np.isfinite(normalized).all())

    def test_forward_attention_zero_neighbor_and_backward(self):
        edges = np.asarray([[1, 2], [0, 1]])
        relation = np.asarray([[0.4, 1.0, 1.0, 1.0], [0.2, 1.0, 1.0, 1.0]], dtype=np.float32)
        model = PUCRSTAttn(4, edges, relation)
        x = torch.randn(2, 168, 4, 6)
        final, temporal, details = model(x, return_details=True)
        self.assertEqual(tuple(final.shape), (2, 4))
        self.assertEqual(tuple(temporal.shape), (2, 4))
        self.assertTrue(torch.allclose(details["temporal_attention"].sum(-1), torch.ones(2, 4), atol=1e-6))
        self.assertTrue(torch.allclose(details["spatial_attention"][:, 0], torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(details["spatial_attention"][:, 1], torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(details["gate"][:, 2:], torch.zeros(2, 2), atol=0.0))
        self.assertTrue(torch.allclose(final[:, 2:], temporal[:, 2:], atol=0.0))
        loss = masked_scaled_mae(final, torch.zeros_like(final), torch.tensor([True, True, False, False])) + ANCHOR_LOSS_WEIGHT * masked_scaled_mae(temporal, torch.zeros_like(temporal), torch.tensor([True, True, False, False]))
        loss.backward()
        for module in (model.gru, model.temporal_score, model.q_projection, model.k_projection, model.v_projection, model.relation_encoder, model.correction_mlp, model.gate_mlp):
            self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()))
        self.assertTrue(torch.equal(model.correction_mlp[-1].weight.detach(), torch.zeros_like(model.correction_mlp[-1].weight)))


if __name__ == "__main__":
    unittest.main()
