import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from code.models.puc_rstattn_v2_ablation import PUCRSTAttnV2Ablation, VARIANT_SPECS


@unittest.skipIf(torch is None, "PyTorch unavailable")
class TestPUCRSTAttnV2Ablation(unittest.TestCase):
    def _model(self, variant, edges=None, utilities=None):
        if edges is None:
            edges = np.asarray([[1, 2, 0], [0, 0, 1]], dtype=np.int64)
        relation = np.asarray([[0.8, 1.0, 0.0, 1.0], [0.2, 2.0, 1.0, 0.0], [0.4, 1.5, 1.5, 0.0]], dtype=np.float32)
        if utilities is not None:
            relation[:, 0] = utilities
        return PUCRSTAttnV2Ablation(variant, 4, edges, relation, np.asarray([True, True, True, False]))

    def test_variant_flags_are_exact(self):
        self.assertEqual(VARIANT_SPECS["gru_anchor_only"].__dict__, {"temporal_residual_enabled": False, "spatial_residual_enabled": False, "utility_candidate_selection_enabled": False, "utility_magnitude_prior_enabled": False, "dynamic_attention_enabled": False, "physical_relation_bias_enabled": False, "aggregate_preserving_spatial_residual": False})
        self.assertTrue(VARIANT_SPECS["temporal_residual_only"].temporal_residual_enabled)
        self.assertFalse(VARIANT_SPECS["temporal_residual_only"].spatial_residual_enabled)
        self.assertTrue(VARIANT_SPECS["utility_uniform_spatial"].spatial_residual_enabled)
        self.assertTrue(VARIANT_SPECS["utility_uniform_spatial"].utility_candidate_selection_enabled)
        self.assertFalse(VARIANT_SPECS["utility_uniform_spatial"].utility_magnitude_prior_enabled)
        self.assertFalse(VARIANT_SPECS["utility_uniform_spatial"].dynamic_attention_enabled)
        self.assertFalse(VARIANT_SPECS["utility_uniform_spatial"].physical_relation_bias_enabled)
        self.assertTrue(VARIANT_SPECS["utility_prior_no_physics"].utility_magnitude_prior_enabled)
        self.assertTrue(VARIANT_SPECS["utility_prior_no_physics"].dynamic_attention_enabled)
        self.assertFalse(VARIANT_SPECS["utility_prior_no_physics"].physical_relation_bias_enabled)

    def test_anchor_and_temporal_only_outputs(self):
        x = torch.randn(2, 168, 4, 6)
        anchor = self._model("gru_anchor_only")
        final, temporal, details = anchor(x, return_details=True)
        self.assertTrue(torch.equal(final, details["y_gru"]))
        self.assertTrue(torch.equal(temporal, details["y_gru"]))
        temporal_only = self._model("temporal_residual_only")
        final, temporal, details = temporal_only(x, return_details=True)
        self.assertTrue(torch.equal(final, temporal))
        self.assertTrue(torch.equal(temporal, details["y_temporal"]) if "y_temporal" in details else torch.allclose(temporal, details["y_gru"] + details["temporal_gate"] * details["temporal_correction"]))
        self.assertTrue(torch.equal(details["centered_spatial_residual"], torch.zeros_like(details["centered_spatial_residual"])))

    def test_uniform_attention_ignores_utility_magnitude(self):
        x = torch.randn(2, 168, 4, 6)
        first = self._model("utility_uniform_spatial", utilities=np.asarray([0.99, 0.01, 0.5], dtype=np.float32))
        _, _, details = first(x, return_details=True)
        self.assertTrue(torch.allclose(details["spatial_attention"][:, :2], torch.full((2, 2), 0.5), atol=1e-6))
        second = self._model("utility_uniform_spatial", utilities=np.asarray([0.01, 0.99, 0.5], dtype=np.float32))
        _, _, second_details = second(x, return_details=True)
        self.assertTrue(torch.allclose(details["spatial_attention"], second_details["spatial_attention"], atol=1e-6))

    def test_no_physics_initial_attention_is_utility_prior_and_physics_is_disabled(self):
        model = self._model("utility_prior_no_physics")
        x = torch.randn(2, 168, 4, 6)
        _, _, details = model(x, return_details=True)
        self.assertTrue(torch.allclose(details["spatial_attention"][:, :2], model.utility_prior[:2].expand(2, -1), atol=1e-6))
        self.assertTrue(torch.equal(model.physical_encoder[-1].weight, torch.zeros_like(model.physical_encoder[-1].weight)))
        self.assertTrue(torch.equal(model.physical_encoder[-1].bias, torch.zeros_like(model.physical_encoder[-1].bias)))
        self.assertEqual(float(model.dynamic_scale.detach()), 0.0)

    def test_spatial_variants_preserve_load_aggregate_and_zero_neighbors(self):
        x = torch.randn(3, 168, 4, 6)
        for variant in ("utility_uniform_spatial", "utility_prior_no_physics"):
            model = self._model(variant)
            final, temporal, details = model(x, return_details=True)
            load = model.load_bus_mask
            self.assertTrue(torch.allclose(final[:, load].sum(-1), temporal[:, load].sum(-1), atol=1e-7))
            self.assertTrue(torch.allclose(details["centered_spatial_residual"][:, load].sum(-1), torch.zeros(3), atol=1e-7))
            self.assertTrue(torch.allclose(details["spatial_gate"][:, 3], torch.zeros(3), atol=0.0))
            self.assertTrue(torch.allclose(details["centered_spatial_residual"][:, 3], torch.zeros(3), atol=0.0))


if __name__ == "__main__":
    unittest.main()
