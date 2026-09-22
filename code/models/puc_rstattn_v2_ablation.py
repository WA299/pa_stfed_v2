"""Formal independently-trainable PUC-RSTAttn V2 ablation variants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from code.models.puc_rstattn_v2 import PUCRSTAttnV2

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class AblationSpec:
    temporal_residual_enabled: bool
    spatial_residual_enabled: bool
    utility_candidate_selection_enabled: bool
    utility_magnitude_prior_enabled: bool
    dynamic_attention_enabled: bool
    physical_relation_bias_enabled: bool
    aggregate_preserving_spatial_residual: bool


VARIANT_SPECS = {
    "gru_anchor_only": AblationSpec(False, False, False, False, False, False, False),
    "temporal_residual_only": AblationSpec(True, False, False, False, False, False, False),
    "utility_uniform_spatial": AblationSpec(True, True, True, False, False, False, True),
    "utility_prior_no_physics": AblationSpec(True, True, True, True, True, False, True),
}


def variant_spec(variant: str) -> AblationSpec:
    try:
        return VARIANT_SPECS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown PUC-RSTAttn V2 ablation variant: {variant}") from exc


if TORCH_AVAILABLE:

    class PUCRSTAttnV2Ablation(PUCRSTAttnV2):
        """V2 with frozen architecture and an explicit ablation policy."""

        def __init__(self, variant: str, *args: Any, **kwargs: Any) -> None:
            self.variant = variant
            self.spec = variant_spec(variant)
            super().__init__(*args, **kwargs)

        def forward(self, x: Tensor, return_details: bool = False) -> Any:
            if x.ndim != 4:
                raise ValueError("x must have shape (batch, history, nodes, features)")
            batch, history, nodes, features = x.shape
            if nodes != self.num_nodes or features != 6:
                raise ValueError("input node/feature dimensions do not match model")
            sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, history, features)
            encoded, _ = self.gru(sequence)
            encoded = encoded.reshape(batch, nodes, history, 32)
            h_last = encoded[:, :, -1, :]
            y_gru = self.gru_head(h_last).squeeze(-1)

            if self.spec.temporal_residual_enabled:
                temporal_scores = self.temporal_score(torch.tanh(self.temporal_key(encoded) + self.temporal_query(h_last).unsqueeze(2))).squeeze(-1)
                temporal_attention = torch.softmax(temporal_scores, dim=-1)
                context = torch.sum(temporal_attention.unsqueeze(-1) * encoded, dim=2)
                temporal_input = torch.cat([h_last, context, torch.abs(h_last - context)], dim=-1)
                temporal_correction = self.temporal_correction(temporal_input).squeeze(-1)
                temporal_gate = self.temporal_gate(temporal_input).squeeze(-1)
                y_temporal = y_gru + temporal_gate * temporal_correction
            else:
                temporal_attention = torch.zeros((batch, nodes, history), device=x.device, dtype=x.dtype)
                context = torch.zeros_like(h_last)
                temporal_correction = torch.zeros_like(y_gru)
                temporal_gate = torch.zeros_like(y_gru)
                y_temporal = y_gru

            edge_count = int(self.utility_edge_index.shape[1])
            spatial_message = torch.zeros_like(h_last)
            spatial_attention = torch.zeros((batch, edge_count), device=x.device, dtype=x.dtype)
            selected_count = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            utility_max = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            utility_mean = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            if edge_count and self.spec.spatial_residual_enabled and self.spec.utility_candidate_selection_enabled:
                edge = self.utility_edge_index.to(x.device)
                prior = self.utility_prior.to(device=x.device, dtype=x.dtype)
                selected_utility = self.selected_utility.to(device=x.device, dtype=x.dtype)
                physical = self.physical_relation_features.to(device=x.device, dtype=x.dtype)
                values = self.v_projection(h_last[:, edge[0], :])
                logits = torch.zeros((batch, edge_count), device=x.device, dtype=x.dtype)
                if self.spec.utility_magnitude_prior_enabled:
                    logits = logits + torch.log(prior.clamp_min(1e-12)).unsqueeze(0)
                if self.spec.dynamic_attention_enabled:
                    q = self.q_projection(h_last[:, edge[1], :])
                    k = self.k_projection(h_last[:, edge[0], :])
                    dynamic_score = (q * k).sum(dim=-1) / np.sqrt(32.0)
                    logits = logits + self.dynamic_scale * dynamic_score
                if self.spec.physical_relation_bias_enabled:
                    logits = logits + self.physical_encoder(physical).squeeze(-1).unsqueeze(0)
                for target in range(nodes):
                    positions = torch.nonzero(edge[1] == target, as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    weights = torch.softmax(logits[:, positions], dim=-1)
                    spatial_attention[:, positions] = weights
                    spatial_message[:, target, :] = torch.sum(weights.unsqueeze(-1) * values[:, positions, :], dim=1)
                    selected_count[target] = float(positions.numel())
                    target_utility = selected_utility[positions]
                    utility_max[target] = target_utility.max()
                    utility_mean[target] = target_utility.mean()

            if self.variant == "utility_uniform_spatial":
                reliability = torch.stack([torch.zeros_like(selected_count), torch.zeros_like(selected_count), selected_count / 3.0], dim=-1)
            else:
                reliability = torch.stack([utility_max, utility_mean, selected_count / 3.0], dim=-1)
            spatial_input = torch.cat([h_last, spatial_message], dim=-1)
            spatial_correction = self.spatial_correction(spatial_input).squeeze(-1)
            spatial_gate = self.spatial_gate(
                torch.cat([h_last, spatial_message, torch.abs(h_last - spatial_message), reliability.unsqueeze(0).expand(batch, -1, -1)], dim=-1)
            ).squeeze(-1)
            spatial_gate = spatial_gate * (
                (selected_count > 0) & self.load_bus_mask.to(x.device)
            ).to(dtype=spatial_gate.dtype).unsqueeze(0)
            raw_residual = spatial_gate * spatial_correction if self.spec.spatial_residual_enabled else torch.zeros_like(spatial_gate)
            centered_residual = torch.zeros_like(raw_residual)
            active = self.load_bus_mask.to(x.device) & (selected_count > 0)
            if self.spec.aggregate_preserving_spatial_residual and bool(active.any()):
                centered_residual[:, active] = raw_residual[:, active] - raw_residual[:, active].mean(dim=1, keepdim=True)
            y_final = y_temporal + centered_residual
            details = {
                "h_last": h_last, "y_gru": y_gru, "y_temporal": y_temporal, "context": context,
                "temporal_attention": temporal_attention, "temporal_correction": temporal_correction,
                "temporal_gate": temporal_gate, "spatial_attention": spatial_attention,
                "spatial_message": spatial_message, "spatial_correction": spatial_correction,
                "spatial_gate": spatial_gate, "raw_spatial_residual": raw_residual,
                "centered_spatial_residual": centered_residual, "reliability": reliability,
            }
            self.last_details = details
            if return_details:
                return y_final, y_temporal, details
            return y_final

else:

    class PUCRSTAttnV2Ablation(PUCRSTAttnV2):  # type: ignore[no-redef]
        def __init__(self, variant: str, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PUCRSTAttnV2Ablation requires PyTorch")


__all__ = ["AblationSpec", "VARIANT_SPECS", "PUCRSTAttnV2Ablation", "variant_spec"]
