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

    def test_utility_graph_invariants_on_synthetic_grid(self):
        loads = np.flatnonzero(_Grid.load_bus_mask)
        utilities = {
            (source, target): float(10 - source - target)
            for target in loads
            for source in loads
            if source != target
        }
        utilities[(3, 0)] = -1.0
        utilities[(0, 1)] = -1.0
        neighbors = select_utility_neighbors(utilities, loads)
        sources = []
        targets = []
        for target in loads:
            for source, utility in neighbors[int(target)]:
                self.assertGreater(utility, 0.0)
                sources.append(source)
                targets.append(target)
        edge_index = np.asarray([sources, targets], dtype=np.int64)
        relation = build_relation_features(_Grid(), edge_index, utilities, loads)

        load_set = set(loads.tolist())
        edge_pairs = {(int(source), int(target)) for source, target in edge_index.T}
        self.assertIn((1, 0), edge_pairs)
        self.assertNotIn((0, 1), edge_pairs)
        self.assertTrue(all(int(source) in load_set for source in edge_index[0]))
        self.assertTrue(all(int(target) in load_set for target in edge_index[1]))
        self.assertTrue(np.all(edge_index[0] != edge_index[1]))
        self.assertTrue(all(len(neighbors[int(target)]) <= 3 for target in loads))
        self.assertNotIn(4, edge_index)
        self.assertEqual(relation.shape, (edge_index.shape[1], 4))

    def test_forward_attention_zero_neighbor_and_backward(self):
        torch.manual_seed(7)
        edges = np.asarray([[1, 2, 0, 2], [0, 0, 1, 1]])
        relation = np.asarray(
            [[0.4, 1.0, 1.0, 1.0], [0.2, 1.0, 1.0, 1.0],
             [0.3, 1.0, 1.0, 1.0], [0.1, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        model = PUCRSTAttn(4, edges, relation)
        x = torch.randn(2, 168, 4, 6)
        final, temporal, details = model(x, return_details=True)
        self.assertEqual(tuple(final.shape), (2, 4))
        self.assertEqual(tuple(temporal.shape), (2, 4))
        self.assertTrue(torch.allclose(details["temporal_attention"].sum(-1), torch.ones(2, 4), atol=1e-6))
        self.assertTrue(torch.allclose(details["spatial_attention"][:, :2].sum(-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(details["spatial_attention"][:, 2:].sum(-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(details["gate"][:, 2:], torch.zeros(2, 2), atol=0.0))
        self.assertTrue(torch.allclose(final[:, 2:], temporal[:, 2:], atol=0.0))
        self.assertTrue(torch.equal(model.correction_mlp[-1].weight.detach(), torch.zeros_like(model.correction_mlp[-1].weight)))
        self.assertTrue(torch.equal(model.correction_mlp[-1].bias.detach(), torch.zeros_like(model.correction_mlp[-1].bias)))

        target = torch.randn_like(final)
        load_mask = torch.tensor([True, True, True, False])
        loss = masked_scaled_mae(final, target, load_mask) + ANCHOR_LOSS_WEIGHT * masked_scaled_mae(temporal, target, load_mask)
        loss.backward()
        correction_grad = model.correction_mlp[-1].weight.grad
        self.assertIsNotNone(correction_grad)
        self.assertTrue(torch.isfinite(correction_grad).all())
        self.assertGreater(float(correction_grad.norm()), 0.0)
        self.assertGreater(float(model.gru.weight_ih_l0.grad.norm()), 0.0)
        self.assertGreater(float(model.temporal_score.weight.grad.norm()), 0.0)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        final_second, temporal_second, _ = model(x, return_details=True)
        second_loss = masked_scaled_mae(final_second, target, load_mask) + ANCHOR_LOSS_WEIGHT * masked_scaled_mae(temporal_second, target, load_mask)
        second_loss.backward()

        def module_grad_norm(module):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
            return sum(float(gradient.norm()) for gradient in gradients)

        for module in (model.q_projection, model.k_projection, model.v_projection, model.relation_encoder, model.gate_mlp):
            self.assertGreater(module_grad_norm(module), 0.0)


if __name__ == "__main__":
    unittest.main()
