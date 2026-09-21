"""Predictive-Utility-Calibrated Residual Spatio-Temporal Attention.

The utility graph is deliberately constructed outside the neural model from the
formal training prefix.  The model itself only consumes directed source->target
edges and their four canonical relation features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from code.audits.predictive_utility_graph_audit import (
    LAGS,
    feature_matrix,
    fit_scaled_ridge,
    load_bus_indices,
    predict_scaled_ridge,
    signed_utility,
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


UTILITY_FIT_SAMPLES = 4771
UTILITY_SELECTION_SAMPLES = 1193
UTILITY_TOP_K = 3
ANCHOR_LOSS_WEIGHT = 0.2


def formal_train_target_indices(grid: Any) -> np.ndarray:
    train = grid.splits["train"]
    indices = np.arange(train.start_index + max(LAGS), train.end_index, dtype=int)
    if len(indices) != UTILITY_FIT_SAMPLES + UTILITY_SELECTION_SAMPLES:
        raise ValueError(
            "PUC-RSTAttn requires 5964 valid formal-train targets; "
            f"got {len(indices)}"
        )
    return indices


def split_utility_targets(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    expected = UTILITY_FIT_SAMPLES + UTILITY_SELECTION_SAMPLES
    if len(indices) != expected or (len(indices) > 1 and not np.all(np.diff(indices) > 0)):
        raise ValueError("utility targets must be 5964 strictly chronological indices")
    return indices[:UTILITY_FIT_SAMPLES].copy(), indices[UTILITY_FIT_SAMPLES:].copy()


def _select_positive_utility_neighbors(
    utilities: dict[tuple[int, int], float],
    loads: Iterable[int],
    top_k: int = UTILITY_TOP_K,
) -> dict[int, list[tuple[int, float]]]:
    result: dict[int, list[tuple[int, float]]] = {}
    load_list = [int(v) for v in loads]
    for target in load_list:
        candidates = [
            (source, float(utilities[(source, target)]))
            for source in load_list
            if source != target and float(utilities[(source, target)]) > 0.0
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        result[target] = candidates[:top_k]
    return result


# Public names used by lightweight audits/tests.
select_utility_neighbors = _select_positive_utility_neighbors


def normalized_distance_matrices(grid: Any, load_indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Normalize canonical distances using graph-only load-load statistics."""
    loads = np.asarray(list(load_indices), dtype=int)
    out = []
    for name in ("impedance_abs_distance", "hop_distance"):
        distance = np.asarray(grid.distance_matrices[name], dtype=float)
        if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
            raise ValueError(f"{name} must be square")
        values = distance[np.ix_(loads, loads)]
        off_diag = ~np.eye(len(loads), dtype=bool)
        finite_positive = values[off_diag & np.isfinite(values)]
        if finite_positive.size == 0:
            raise ValueError(f"{name} has no finite load-load off-diagonal values")
        scale = float(np.median(finite_positive))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name} normalization scale must be positive")
        normalized = distance / (scale + 1e-12)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{name} normalization produced non-finite values")
        out.append(normalized)
    return out[0], out[1]


def build_relation_features(
    grid: Any,
    edges: np.ndarray,
    utilities: dict[tuple[int, int], float],
    load_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Create the four relation values for an existing directed edge list."""
    loads = load_bus_indices(grid) if load_indices is None else np.asarray(list(load_indices), dtype=int)
    impedance_norm, hop_norm = normalized_distance_matrices(grid, loads)
    adjacency = _direct_adjacency(grid)
    edge_array = np.asarray(edges, dtype=int)
    if edge_array.ndim != 2 or edge_array.shape[0] != 2:
        raise ValueError("edges must have shape (2,E)")
    rows = []
    for source, target in edge_array.T:
        utility = float(utilities[(int(source), int(target))])
        rows.append([max(utility, 0.0), float(impedance_norm[source, target]), float(hop_norm[source, target]), float(adjacency[source, target])])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def _direct_adjacency(grid: Any) -> np.ndarray:
    n = int(grid.num_nodes)
    adjacency = np.zeros((n, n), dtype=bool)
    edge_index = np.asarray(grid.edge_index, dtype=int)
    for source, target in edge_index.T:
        adjacency[source, target] = True
        adjacency[target, source] = True
    return adjacency


@dataclass(frozen=True)
class UtilityGraph:
    edge_index: np.ndarray  # (2, E), source -> target
    relation_features: np.ndarray  # (E, 4)
    utilities: dict[tuple[int, int], float]
    neighbors: dict[int, list[tuple[int, float]]]
    load_indices: np.ndarray
    diagnostics: dict[str, Any]


def build_utility_graph(grid: Any) -> UtilityGraph:
    """Fit pairwise Ridge utilities on formal train only and build directed graph."""
    formal = formal_train_target_indices(grid)
    fit_indices, selection_indices = split_utility_targets(formal)
    loads = load_bus_indices(grid)
    utilities: dict[tuple[int, int], float] = {}
    for target in loads:
        x_self_fit = feature_matrix(grid.p, grid.timestamps, fit_indices, int(target))
        x_self_sel = feature_matrix(grid.p, grid.timestamps, selection_indices, int(target))
        self_model = fit_scaled_ridge(x_self_fit, grid.p[fit_indices, target])
        mae_self = float(np.mean(np.abs(predict_scaled_ridge(self_model, x_self_sel) - grid.p[selection_indices, target])))
        for source in loads:
            if int(source) == int(target):
                continue
            x_pair_fit = feature_matrix(grid.p, grid.timestamps, fit_indices, int(target), [int(source)])
            x_pair_sel = feature_matrix(grid.p, grid.timestamps, selection_indices, int(target), [int(source)])
            pair_model = fit_scaled_ridge(x_pair_fit, grid.p[fit_indices, target])
            mae_pair = float(np.mean(np.abs(predict_scaled_ridge(pair_model, x_pair_sel) - grid.p[selection_indices, target])))
            utilities[(int(source), int(target))] = signed_utility(mae_self, mae_pair)

    neighbors = _select_positive_utility_neighbors(utilities, loads)
    impedance_norm, hop_norm = normalized_distance_matrices(grid, loads)
    adjacency = _direct_adjacency(grid)
    sources: list[int] = []
    targets: list[int] = []
    relations: list[list[float]] = []
    for target in loads:
        for source, utility in neighbors[int(target)]:
            sources.append(int(source)); targets.append(int(target))
            relations.append([
                max(float(utility), 0.0),
                float(impedance_norm[source, target]),
                float(hop_norm[source, target]),
                float(adjacency[source, target]),
            ])
    edge_index = np.asarray([sources, targets], dtype=np.int64) if sources else np.empty((2, 0), dtype=np.int64)
    relation_features = np.asarray(relations, dtype=np.float32).reshape(-1, 4)
    counts = {str(k): int(sum(len(neighbors[int(t)]) == k for t in loads)) for k in range(4)}
    positive = [max(value, 0.0) for value in utilities.values() if value > 0]
    diagnostics = {
        "positive_edge_count": int(len(sources)),
        "targets_with_selected_neighbors": counts,
        "mean_positive_utility": float(np.mean(positive)) if positive else 0.0,
        "mean_normalized_impedance": float(np.mean(relation_features[:, 1])) if relations else 0.0,
        "mean_normalized_hop": float(np.mean(relation_features[:, 2])) if relations else 0.0,
        "utility_fit_samples": UTILITY_FIT_SAMPLES,
        "utility_selection_samples": UTILITY_SELECTION_SAMPLES,
    }
    return UtilityGraph(edge_index, relation_features, utilities, neighbors, loads, diagnostics)


if TORCH_AVAILABLE:

    class PUCRSTAttn(nn.Module):
        """Shared-GRU temporal anchor with sparse utility-calibrated correction."""

        def __init__(
            self,
            num_nodes: int,
            utility_edge_index: np.ndarray | Tensor | None = None,
            relation_features: np.ndarray | Tensor | None = None,
            input_size: int = 6,
            hidden_size: int = 32,
        ) -> None:
            super().__init__()
            self.num_nodes = int(num_nodes)
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
            self.temporal_query = nn.Linear(hidden_size, hidden_size)
            self.temporal_key = nn.Linear(hidden_size, hidden_size)
            self.temporal_score = nn.Linear(hidden_size, 1, bias=False)
            self.temporal_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
            self.q_projection = nn.Linear(hidden_size, hidden_size)
            self.k_projection = nn.Linear(hidden_size, hidden_size)
            self.v_projection = nn.Linear(hidden_size, hidden_size)
            self.relation_encoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
            self.correction_mlp = nn.Sequential(nn.Linear(64, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
            self.gate_mlp = nn.Sequential(nn.Linear(99, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid())
            nn.init.zeros_(self.correction_mlp[-1].weight)
            nn.init.zeros_(self.correction_mlp[-1].bias)
            if utility_edge_index is None:
                utility_edge_index = np.empty((2, 0), dtype=np.int64)
            if relation_features is None:
                relation_features = np.empty((0, 4), dtype=np.float32)
            edge = torch.as_tensor(utility_edge_index, dtype=torch.long)
            relation = torch.as_tensor(relation_features, dtype=torch.float32)
            if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != relation.shape[0]:
                raise ValueError("utility_edge_index must be (2,E) aligned with relation_features (E,4)")
            if relation.ndim != 2 or relation.shape[1] != 4:
                raise ValueError("relation_features must have shape (E,4)")
            self.register_buffer("utility_edge_index", edge)
            self.register_buffer("relation_features", relation)
            self.last_temporal_attention: Tensor | None = None
            self.last_spatial_attention: Tensor | None = None
            self.last_spatial_message: Tensor | None = None
            self.last_gate: Tensor | None = None

        def forward(self, x: Tensor, return_details: bool = False) -> Any:
            if x.ndim != 4:
                raise ValueError("x must have shape (batch, history, nodes, features)")
            batch, history, nodes, features = x.shape
            if nodes != self.num_nodes or features != self.input_size:
                raise ValueError("input node/feature dimensions do not match model")
            sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, history, features)
            encoded, _ = self.gru(sequence)
            encoded = encoded.reshape(batch, nodes, history, self.hidden_size)
            last = encoded[:, :, -1, :]
            temporal_scores = self.temporal_score(torch.tanh(self.temporal_key(encoded) + self.temporal_query(last).unsqueeze(2))).squeeze(-1)
            temporal_attention = torch.softmax(temporal_scores, dim=-1)
            z = torch.sum(temporal_attention.unsqueeze(-1) * encoded, dim=2)
            y_temporal = self.temporal_head(z).squeeze(-1)

            edge_count = int(self.utility_edge_index.shape[1])
            spatial_message = torch.zeros_like(z)
            spatial_attention = torch.zeros((batch, edge_count), device=x.device, dtype=x.dtype)
            if edge_count:
                edge = self.utility_edge_index.to(x.device)
                relation = self.relation_features.to(device=x.device, dtype=x.dtype)
                q = self.q_projection(z[:, edge[1], :])
                k = self.k_projection(z[:, edge[0], :])
                values = self.v_projection(z[:, edge[0], :])
                logits = (q * k).sum(dim=-1) / np.sqrt(self.hidden_size)
                logits = logits + self.relation_encoder(relation).squeeze(-1).unsqueeze(0)
                for target in range(nodes):
                    positions = torch.nonzero(edge[1] == target, as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    weights = torch.softmax(logits[:, positions], dim=-1)
                    spatial_attention[:, positions] = weights
                    spatial_message[:, target, :] = torch.sum(weights.unsqueeze(-1) * values[:, positions, :], dim=1)

            selected_count = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            utility_sum = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            utility_max = torch.zeros(nodes, device=x.device, dtype=x.dtype)
            if edge_count:
                edge = self.utility_edge_index.to(x.device)
                utility = self.relation_features[:, 0].to(device=x.device, dtype=x.dtype)
                for target in range(nodes):
                    values = utility[edge[1] == target]
                    if values.numel():
                        selected_count[target] = float(values.numel())
                        utility_sum[target] = values.sum()
                        utility_max[target] = values.max()
            has_neighbors = selected_count > 0
            reliability = torch.stack([utility_max, utility_sum / selected_count.clamp_min(1.0), selected_count / float(UTILITY_TOP_K)], dim=-1)
            correction_input = torch.cat([z, spatial_message], dim=-1)
            correction = self.correction_mlp(correction_input).squeeze(-1)
            gate_input = torch.cat([z, spatial_message, torch.abs(z - spatial_message), reliability.unsqueeze(0).expand(batch, -1, -1)], dim=-1)
            gate = self.gate_mlp(gate_input).squeeze(-1)
            gate = gate * has_neighbors.to(dtype=gate.dtype).unsqueeze(0)
            prediction = y_temporal + gate * correction
            self.last_temporal_attention = temporal_attention
            self.last_spatial_attention = spatial_attention
            self.last_spatial_message = spatial_message
            self.last_gate = gate
            if return_details:
                return prediction, y_temporal, {"temporal_attention": temporal_attention, "spatial_attention": spatial_attention, "spatial_message": spatial_message, "gate": gate, "reliability": reliability}
            return prediction

else:
    class PUCRSTAttn:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PUCRSTAttn requires PyTorch")


PUCRSTAttnModel = PUCRSTAttn


def anchor_loss_weight() -> float:
    return ANCHOR_LOSS_WEIGHT
