"""25%-history bridge from directed temporal transfer to the full V2-CU model."""
from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from code.audits.federated_transfer_benefit import (
    ANCHOR_WEIGHT,
    BATCH_SIZE,
    HISTORY,
    LEARNING_RATE,
    PATIENCE,
    SEED,
    IndexDataset,
    _evaluate,
    fit_fit_only_scaler,
    make_proxy,
    scarce_split,
    train_proxy,
    transfer_temporal_parameters,
)
from code.audits.strong_temporal_conditional_utility_audit import (
    compute_conditional_utilities,
    load_bus_indices,
    select_conditional_neighbors,
)
from code.models.centralized_gru import masked_scaled_mae
from code.models.puc_rstattn import build_relation_features
from code.models.puc_rstattn_v2_conditional_utility import PUCRSTAttnV2ConditionalUtility

GRAPH_TOP_K = 3


def build_scarce_target_graph(grid: Any, split: Any | None = None) -> Any:
    """Build a UtilityGraph from target FIT only, never calibration onward."""
    if split is None:
        split = scarce_split(grid)
    fit_count = (2 * len(split.fit_indices)) // 3
    graph_fit = np.asarray(split.fit_indices[:fit_count], dtype=np.int64)
    graph_selection = np.asarray(split.fit_indices[fit_count:], dtype=np.int64)
    if not len(graph_fit) or not len(graph_selection):
        raise ValueError("target FIT is too short for graph fit/selection")
    if graph_fit[-1] >= graph_selection[0] or graph_selection[-1] >= split.calibration_indices[0]:
        raise AssertionError("scarce graph split is not chronological or leaks calibration")
    loads = load_bus_indices(grid)
    utilities, _ = compute_conditional_utilities(grid, graph_fit, graph_selection, loads)
    neighbors = select_conditional_neighbors(utilities, loads, top_k=GRAPH_TOP_K)
    sources, targets = [], []
    for target in loads:
        for source, _utility in neighbors[int(target)]:
            sources.append(int(source)); targets.append(int(target))
    edge_index = np.asarray([sources, targets], dtype=np.int64) if sources else np.empty((2, 0), dtype=np.int64)
    relation = build_relation_features(grid, edge_index, utilities, loads)
    from code.models.puc_rstattn import UtilityGraph
    counts = {str(k): int(sum(len(neighbors[int(t)]) == k for t in loads)) for k in range(GRAPH_TOP_K + 1)}
    diagnostics = {
        "graph_fit_target_count": int(len(graph_fit)),
        "graph_selection_target_count": int(len(graph_selection)),
        "graph_uses_target_fit_only": True,
        "graph_uses_calibration": False,
        "graph_uses_audit": False,
        "graph_uses_validation": False,
        "graph_uses_test": False,
        "selected_edge_count": int(edge_index.shape[1]),
        "targets_with_selected_neighbors": counts,
        "graph_fit_first_target_index": int(graph_fit[0]),
        "graph_fit_last_target_index": int(graph_fit[-1]),
        "graph_selection_first_target_index": int(graph_selection[0]),
        "graph_selection_last_target_index": int(graph_selection[-1]),
        "graph_top_k": GRAPH_TOP_K,
        "graph_positive_only": True,
    }
    return UtilityGraph(edge_index, relation, utilities, neighbors, loads, diagnostics)


def full_model(grid: Any, graph: Any, device: str = "cpu") -> Any:
    import torch
    torch.manual_seed(SEED)
    model = PUCRSTAttnV2ConditionalUtility(
        grid.num_nodes, graph.edge_index, graph.relation_features[:, 1:],
        grid.load_bus_mask, graph.relation_features[:, 0],
    )
    return model.to(device)


def temporal_state_from_model(model: Any) -> dict[str, Any]:
    names = set(__import__("code.federated.parameter_groups", fromlist=["parameter_groups"]).parameter_groups(model)["temporal"])
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if name in names}


def inject_temporal_state(model: Any, state: Mapping[str, Any]) -> tuple[str, ...]:
    names = __import__("code.federated.parameter_groups", fromlist=["parameter_groups"]).parameter_groups(model)["temporal"]
    params = dict(model.named_parameters())
    with __import__("torch").no_grad():
        for name in names:
            if name not in state: raise KeyError(name)
            params[name].copy_(state[name].to(params[name].device))
    return names


def train_full_model(model: Any, grid: Any, fit_indices: np.ndarray, calibration_indices: np.ndarray,
                     scaler: Any, device: str = "cpu", max_epochs: int = 50,
                     patience: int = PATIENCE, batch_size: int = BATCH_SIZE) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    model = model.to(device)
    fit_ds = IndexDataset(grid, fit_indices)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mask = torch.as_tensor(grid.load_bus_mask, dtype=torch.bool, device=device)
    best_state, best_mae, stale = None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in __import__("scripts.run_puc_rstattn_v2", fromlist=["_batch_indices"])._batch_indices(len(fit_ds), batch_size):
            x, y = __import__("scripts.run_puc_rstattn_v2", fromlist=["_to_batch"])._to_batch(fit_ds, scaler, batch)
            optimizer.zero_grad(set_to_none=True)
            final, _, details = model(torch.from_numpy(x).to(device), return_details=True)
            target = torch.from_numpy(y).to(device)
            loss = masked_scaled_mae(final, target, mask) + ANCHOR_WEIGHT * masked_scaled_mae(details["y_gru"], target, mask)
            loss.backward(); optimizer.step()
        calibration = _evaluate(model, grid, calibration_indices, scaler, device, batch_size=batch_size)
        current = calibration["node_macro"]["mae"]
        if current < best_mae:
            best_mae = current; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= patience: break
    if best_state is None: raise RuntimeError("full model did not produce a checkpoint")
    model.load_state_dict(best_state)
    return best_state, {"best_calibration_node_mae": float(best_mae), "epochs_run": int(epoch)}


def relative_improvement(local: Mapping[str, Any], transferred: Mapping[str, Any]) -> float:
    return float((local["node_macro"]["mae"] - transferred["node_macro"]["mae"]) / (local["node_macro"]["mae"] + 1e-12))


__all__ = ["GRAPH_TOP_K", "build_scarce_target_graph", "full_model", "temporal_state_from_model", "inject_temporal_state", "train_full_model", "relative_improvement"]
