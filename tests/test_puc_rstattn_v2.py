import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from code.models.centralized_gru import masked_scaled_mae
from code.models.puc_rstattn import ANCHOR_LOSS_WEIGHT, build_relation_features, formal_train_target_indices, select_utility_neighbors, split_utility_targets
from code.models.puc_rstattn_v2 import PUCRSTAttnV2


@unittest.skipIf(torch is None, "PyTorch unavailable")
class TestPUCRSTAttnV2(unittest.TestCase):
    def _model(self):
        edges = np.asarray([[1, 2, 0], [0, 0, 1]], dtype=np.int64)
        physical = np.asarray([[1.0, 2.0, 0.0], [2.0, 1.0, 1.0], [1.5, 1.5, 0.0]], dtype=np.float32)
        utility = np.asarray([0.8, 0.2, 0.4], dtype=np.float32)
        return PUCRSTAttnV2(4, edges, physical, np.asarray([True, True, True, False]), utility)

    def test_utility_split_and_graph_contract(self):
        fit, selection = split_utility_targets(np.arange(5964))
        self.assertEqual((len(fit), len(selection)), (4771, 1193))
        model = self._model()
        self.assertTrue(np.all(model.utility_edge_index[0].numpy() != model.utility_edge_index[1].numpy()))
        self.assertTrue(np.all(model.utility_prior.numpy() > 0))
        for target in range(model.num_nodes):
            positions = model.utility_edge_index[1] == target
            if bool(positions.any()):
                self.assertAlmostEqual(float(model.utility_prior[positions].sum()), 1.0, places=6)

    def test_utility_graph_is_train_only_directed_load_only_top3(self):
        class Grid:
            num_nodes = 5
            load_bus_mask = np.asarray([True, True, True, True, False])
            edge_index = np.asarray([[0, 1, 2], [1, 2, 3]])
            splits = {"train": type("Split", (), {"start_index": 0, "end_index": 5964 + 168})()}
            distance_matrices = {
                "impedance_abs_distance": np.ones((5, 5), dtype=float),
                "hop_distance": np.ones((5, 5), dtype=float),
            }

        formal = formal_train_target_indices(Grid())
        self.assertEqual((formal[0], formal[-1], len(formal)), (168, 6131, 5964))
        loads = np.flatnonzero(Grid.load_bus_mask)
        utilities = {(int(source), int(target)): float(100 - 10 * int(target) - int(source)) for target in loads for source in loads if source != target}
        utilities[(0, 1)] = -1.0
        neighbors = select_utility_neighbors(utilities, loads)
        sources = [source for target in loads for source, _ in neighbors[int(target)]]
        targets = [target for target in loads for _source, _ in neighbors[int(target)]]
        edges = np.asarray([sources, targets], dtype=np.int64)
        relation = build_relation_features(Grid(), edges, utilities, loads)
        load_set = set(loads.tolist())
        self.assertTrue(all(int(source) in load_set and int(target) in load_set for source, target in edges.T))
        self.assertTrue(np.all(edges[0] != edges[1]))
        self.assertTrue(all(len(neighbors[int(target)]) <= 3 for target in loads))
        self.assertNotIn(4, edges)
        self.assertIn((1, 0), {(int(source), int(target)) for source, target in edges.T})
        self.assertNotIn((0, 1), {(int(source), int(target)) for source, target in edges.T})
        self.assertEqual(relation.shape, (edges.shape[1], 4))

    def test_integrated_utility_graph_uses_only_formal_train_prefix(self):
        from code.models.puc_rstattn import build_utility_graph

        class Grid:
            num_nodes = 5
            node_ids = np.asarray(["0", "1", "2", "3", "4"], dtype=object)
            load_bus_mask = np.asarray([True, True, True, True, False])
            edge_index = np.asarray([[0, 1, 2], [1, 2, 3]])
            p = np.zeros((6132, 5), dtype=np.float32)
            timestamps = None
            splits = {"train": type("Split", (), {"start_index": 0, "end_index": 6132})()}
            distance_matrices = {
                "impedance_abs_distance": np.asarray([[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]], dtype=float),
                "hop_distance": np.asarray([[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]], dtype=float),
            }

        calls = {"feature": [], "predict": []}

        def fake_feature(_p, _timestamps, indices, _target, _sources=()):
            calls["feature"].append(np.asarray(indices).copy())
            return np.zeros((len(indices), 8), dtype=float)

        def fake_predict(_model, features):
            call_index = len(calls["predict"])
            calls["predict"].append(len(features))
            return np.ones(len(features)) if call_index % 4 == 0 else np.zeros(len(features))

        with patch("code.models.puc_rstattn.feature_matrix", side_effect=fake_feature), patch("code.models.puc_rstattn.fit_scaled_ridge", return_value={}), patch("code.models.puc_rstattn.predict_scaled_ridge", side_effect=fake_predict):
            graph = build_utility_graph(Grid())
        self.assertEqual(len(calls["feature"]), 32)
        self.assertTrue(all(int(indices.min()) >= 168 and int(indices.max()) < 6132 for indices in calls["feature"]))
        self.assertTrue(np.all(graph.edge_index[0] != graph.edge_index[1]))
        self.assertTrue(np.all(graph.edge_index[0] < 4))
        self.assertTrue(np.all(graph.edge_index[1] < 4))
        self.assertEqual(graph.relation_features.shape[1], 4)
        self.assertTrue(all(int(np.sum(graph.edge_index[1] == target)) <= 3 for target in range(4)))

    def test_architecture_initialization_and_prior_attention(self):
        model = self._model()
        self.assertIsInstance(model.gru_head, nn.Linear)
        self.assertEqual((model.gru_head.in_features, model.gru_head.out_features), (32, 1))
        self.assertEqual(model.physical_encoder[0].in_features, 3)
        self.assertEqual(model.physical_encoder[0].out_features, 16)
        self.assertEqual(float(model.dynamic_scale.detach()), 0.0)
        self.assertTrue(torch.equal(model.temporal_correction[-1].weight, torch.zeros_like(model.temporal_correction[-1].weight)))
        self.assertTrue(torch.equal(model.temporal_correction[-1].bias, torch.zeros_like(model.temporal_correction[-1].bias)))
        self.assertTrue(torch.equal(model.physical_encoder[-1].weight, torch.zeros_like(model.physical_encoder[-1].weight)))
        self.assertTrue(torch.equal(model.physical_encoder[-1].bias, torch.zeros_like(model.physical_encoder[-1].bias)))
        self.assertTrue(torch.equal(model.spatial_correction[-1].weight, torch.zeros_like(model.spatial_correction[-1].weight)))
        self.assertTrue(torch.equal(model.spatial_correction[-1].bias, torch.zeros_like(model.spatial_correction[-1].bias)))

        torch.manual_seed(3)
        final, temporal, details = model(torch.randn(2, 168, 4, 6), return_details=True)
        self.assertTrue(torch.allclose(details["temporal_attention"].sum(-1), torch.ones(2, 4), atol=1e-6))
        self.assertTrue(torch.allclose(details["y_gru"], temporal, atol=0.0))
        self.assertTrue(torch.allclose(details["spatial_attention"][:, :2], model.utility_prior[:2].expand(2, -1), atol=1e-6))
        self.assertTrue(torch.allclose(details["spatial_attention"][:, 2], model.utility_prior[2].expand(2), atol=1e-6))
        self.assertTrue(torch.allclose(details["temporal_gate"], torch.zeros_like(details["temporal_gate"]) + 0.5, atol=0.5))
        self.assertTrue(torch.allclose(details["spatial_gate"][:, 3], torch.zeros(2), atol=0.0))

    def test_first_backward_unblocks_only_zero_initialized_correction_outputs(self):
        torch.manual_seed(4)
        model = self._model()
        x = torch.randn(2, 168, 4, 6)
        target = torch.randn(2, 4)
        mask = torch.tensor([True, True, True, False])
        final, _, details = model(x, return_details=True)
        loss = masked_scaled_mae(final, target, mask) + ANCHOR_LOSS_WEIGHT * masked_scaled_mae(details["y_gru"], target, mask)
        loss.backward()
        correction_grad = model.temporal_correction[-1].weight.grad
        spatial_grad = model.spatial_correction[-1].weight.grad
        self.assertIsNotNone(correction_grad)
        self.assertIsNotNone(spatial_grad)
        self.assertGreater(float(correction_grad.norm()), 0.0)
        self.assertTrue(torch.isfinite(spatial_grad).all())
        self.assertGreater(float(model.gru.weight_ih_l0.grad.norm()), 0.0)
        self.assertEqual(float(model.dynamic_scale.grad.abs()), 0.0)
        self.assertEqual(float(model.q_projection.weight.grad.abs().sum()), 0.0)

    def test_aggregate_preservation_zero_neighbors_and_non_load(self):
        model = self._model()
        final, temporal, details = model(torch.randn(3, 168, 4, 6), return_details=True)
        load = model.load_bus_mask
        self.assertTrue(torch.allclose(details["raw_spatial_residual"][:, 2], torch.zeros(3), atol=0.0))
        self.assertTrue(torch.allclose(details["centered_spatial_residual"][:, 3], torch.zeros(3), atol=0.0))
        self.assertTrue(torch.allclose(details["centered_spatial_residual"][:, load].sum(-1), torch.zeros(3), atol=1e-7))
        self.assertTrue(torch.allclose(final[:, load].sum(-1), temporal[:, load].sum(-1), atol=1e-7))

    def test_staged_gradients_reach_temporal_and_spatial_modules(self):
        torch.manual_seed(8)
        model = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randn(2, 168, 4, 6)
        target = torch.randn(2, 4)
        mask = torch.tensor([True, True, True, False])
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            final, _, details = model(x, return_details=True)
            loss = masked_scaled_mae(final, target, mask) + ANCHOR_LOSS_WEIGHT * masked_scaled_mae(details["y_gru"], target, mask)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        final, _, details = model(x, return_details=True)
        loss = masked_scaled_mae(final, target, mask) + ANCHOR_LOSS_WEIGHT * masked_scaled_mae(details["y_gru"], target, mask)
        loss.backward()

        modules = (model.temporal_query, model.temporal_key, model.temporal_score, model.q_projection, model.k_projection, model.v_projection, model.physical_encoder, model.temporal_correction, model.temporal_gate, model.spatial_correction, model.spatial_gate)
        for module in modules:
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
            self.assertGreater(sum(float(gradient.norm()) for gradient in gradients), 0.0)
        self.assertIsNotNone(model.dynamic_scale.grad)
        self.assertTrue(torch.isfinite(model.dynamic_scale.grad).all())
        self.assertGreater(float(model.dynamic_scale.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
