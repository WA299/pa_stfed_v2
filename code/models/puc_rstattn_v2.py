"""Utility-anchored hierarchical residual spatio-temporal attention V2."""
from __future__ import annotations

from typing import Any

import numpy as np

from code.models.puc_rstattn import (
    ANCHOR_LOSS_WEIGHT,
    UTILITY_FIT_SAMPLES,
    UTILITY_SELECTION_SAMPLES,
    UTILITY_TOP_K,
    build_utility_graph,
    formal_train_target_indices,
    normalized_distance_matrices,
    select_utility_neighbors,
    split_utility_targets,
)

try:
    import torch
    from torch import Tensor, nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _as_edge_and_relations(
    utility_edge_index: np.ndarray | Tensor | None,
    relation_features: np.ndarray | Tensor | None,
    utility_prior: np.ndarray | Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if utility_edge_index is None:
        utility_edge_index = np.empty((2, 0), dtype=np.int64)
    edge = torch.as_tensor(utility_edge_index, dtype=torch.long)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("utility_edge_index must have shape (2,E)")
    if relation_features is None:
        relation_features = np.empty((edge.shape[1], 3), dtype=np.float32)
    relation = torch.as_tensor(relation_features, dtype=torch.float32)
    if relation.ndim != 2 or relation.shape[0] != edge.shape[1] or relation.shape[1] not in (3, 4):
        raise ValueError("relation_features must have shape (E,3) or legacy (E,4)")
    if relation.shape[1] == 4:
        if utility_prior is None:
            utility_prior = relation[:, 0]
        physical = relation[:, 1:]
    else:
        physical = relation
        if utility_prior is None and edge.shape[1] == 0:
            utility_prior = np.empty((0,), dtype=np.float32)
        if utility_prior is None:
            raise ValueError("utility_prior is required with three physical relation features")
    prior_values = torch.as_tensor(utility_prior, dtype=torch.float32).reshape(-1)
    if prior_values.shape[0] != edge.shape[1] or bool((prior_values <= 0).any()):
        raise ValueError("utility_prior must contain one positive value per edge")
    normalized = torch.zeros_like(prior_values)
    for target in torch.unique(edge[1]):
        positions = torch.nonzero(edge[1] == target, as_tuple=False).flatten()
        normalized[positions] = prior_values[positions] / prior_values[positions].sum()
    return edge, physical, normalized, prior_values


if TORCH_AVAILABLE:

    class PUCRSTAttnV2(nn.Module):
        """Baseline-preserving hierarchical residual attention model."""

        def __init__(
            self,
            num_nodes: int,
            utility_edge_index: np.ndarray | Tensor | None = None,
            relation_features: np.ndarray | Tensor | None = None,
            load_bus_mask: np.ndarray | Tensor | None = None,
            utility_prior: np.ndarray | Tensor | None = None,
            input_size: int = 6,
            hidden_size: int = 32,
        ) -> None:
            super().__init__()
            self.num_nodes = int(num_nodes)
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            if self.hidden_size != 32 or self.input_size != 6:
                raise ValueError("PUC-RSTAttn V2 fixes input_size=6 and hidden_size=32")
            if load_bus_mask is None:
                raise ValueError("load_bus_mask is required for aggregate-preserving residuals")
            load_mask = torch.as_tensor(load_bus_mask, dtype=torch.bool).reshape(-1)
            if load_mask.shape[0] != self.num_nodes or not bool(load_mask.any()):
                raise ValueError("load_bus_mask must match num_nodes and select at least one bus")
            self.register_buffer("load_bus_mask", load_mask)

            self.gru = nn.GRU(6, 32, num_layers=1, batch_first=True)
            self.gru_head = nn.Linear(32, 1)
            self.temporal_query = nn.Linear(32, 32)
            self.temporal_key = nn.Linear(32, 32)
            self.temporal_score = nn.Linear(32, 1, bias=False)
            self.temporal_correction = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))
            self.temporal_gate = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            self.q_projection = nn.Linear(32, 32)
            self.k_projection = nn.Linear(32, 32)
            self.v_projection = nn.Linear(32, 32)
            self.dynamic_scale = nn.Parameter(torch.zeros(()))
            self.physical_encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
            self.spatial_correction = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            self.spatial_gate = nn.Sequential(nn.Linear(99, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

            nn.init.zeros_(self.temporal_correction[-1].weight)
            nn.init.zeros_(self.temporal_correction[-1].bias)
            nn.init.zeros_(self.physical_encoder[-1].weight)
            nn.init.zeros_(self.physical_encoder[-1].bias)
            nn.init.zeros_(self.spatial_correction[-1].weight)
            nn.init.zeros_(self.spatial_correction[-1].bias)

            edge, physical, prior, utility_values = _as_edge_and_relations(utility_edge_index, relation_features, utility_prior)
            self.register_buffer("utility_edge_index", edge)
            self.register_buffer("physical_relation_features", physical)
            self.register_buffer("utility_prior", prior)
            self.register_buffer("selected_utility", utility_values)
            self.last_details: dict[str, Tensor] = {}

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

            temporal_scores = self.temporal_score(
                torch.tanh(self.temporal_key(encoded) + self.temporal_query(h_last).unsqueeze(2))
            ).squeeze(-1)
            temporal_attention = torch.softmax(temporal_scores, dim=-1)
            context = torch.sum(temporal_attention.unsqueeze(-1) * encoded, dim=2)
            temporal_input = torch.cat([h_last, context, torch.abs(h_last - context)], dim=-1)
            temporal_correction = self.temporal_correction(temporal_input).squeeze(-1)
            temporal_gate = self.temporal_gate(temporal_input).squeeze(-1)
            y_temporal = y_gru + temporal_gate * temporal_correction

            edge_count = int(self.utility_edge_index.shape[1])
            spatial_message = torch.zeros_like(h_last)
            spatial_attention = torch.zeros((batch, edge_count), device=x.device, dtype=x.dtype)
            selected_count = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            utility_max = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            utility_mean = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            if edge_count:
                edge = self.utility_edge_index.to(x.device)
                prior = self.utility_prior.to(device=x.device, dtype=x.dtype)
                selected_utility = self.selected_utility.to(device=x.device, dtype=x.dtype)
                physical = self.physical_relation_features.to(device=x.device, dtype=x.dtype)
                q = self.q_projection(h_last[:, edge[1], :])
                k = self.k_projection(h_last[:, edge[0], :])
                values = self.v_projection(h_last[:, edge[0], :])
                dynamic_score = (q * k).sum(dim=-1) / np.sqrt(32.0)
                physical_bias = self.physical_encoder(physical).squeeze(-1)
                logits = torch.log(prior.clamp_min(1e-12)).unsqueeze(0) + self.dynamic_scale * dynamic_score + physical_bias.unsqueeze(0)
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

            active = self.load_bus_mask.to(x.device) & (selected_count > 0)
            reliability = torch.stack(
                [utility_max, utility_mean, selected_count / float(UTILITY_TOP_K)], dim=-1
            )
            spatial_input = torch.cat([h_last, spatial_message], dim=-1)
            spatial_correction = self.spatial_correction(spatial_input).squeeze(-1)
            spatial_gate = self.spatial_gate(
                torch.cat([h_last, spatial_message, torch.abs(h_last - spatial_message), reliability.unsqueeze(0).expand(batch, -1, -1)], dim=-1)
            ).squeeze(-1)
            spatial_gate = spatial_gate * (
                (selected_count > 0) & self.load_bus_mask.to(x.device)
            ).to(dtype=spatial_gate.dtype).unsqueeze(0)
            raw_residual = spatial_gate * spatial_correction
            centered_residual = torch.zeros_like(raw_residual)
            if bool(active.any()):
                centered_residual[:, active] = raw_residual[:, active] - raw_residual[:, active].mean(dim=1, keepdim=True)
            y_final = y_temporal + centered_residual
            self.last_details = {
                "h_last": h_last,
                "y_gru": y_gru,
                "context": context,
                "temporal_attention": temporal_attention,
                "temporal_correction": temporal_correction,
                "temporal_gate": temporal_gate,
                "spatial_attention": spatial_attention,
                "spatial_message": spatial_message,
                "spatial_correction": spatial_correction,
                "spatial_gate": spatial_gate,
                "raw_spatial_residual": raw_residual,
                "centered_spatial_residual": centered_residual,
                "reliability": reliability,
            }
            if return_details:
                return y_final, y_temporal, self.last_details
            return y_final

else:

    class PUCRSTAttnV2:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PUCRSTAttnV2 requires PyTorch")


PUCRSTAttnV2Model = PUCRSTAttnV2

__all__ = [
    "ANCHOR_LOSS_WEIGHT",
    "UTILITY_FIT_SAMPLES",
    "UTILITY_SELECTION_SAMPLES",
    "UTILITY_TOP_K",
    "PUCRSTAttnV2",
    "PUCRSTAttnV2Model",
    "build_utility_graph",
    "formal_train_target_indices",
    "normalized_distance_matrices",
    "select_utility_neighbors",
    "split_utility_targets",
]
