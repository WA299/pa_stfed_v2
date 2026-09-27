"""FedFomo-style personalized comparator for heterogeneous-grid FL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model
from code.audits.federated_transfer_benefit import _evaluate, donor_split, fit_fit_only_scaler, scarce_split
from code.federated.formal_btd import CLIENT_NAMES, _indices, _validation_indices, _make_client
from code.federated.formal_btd_ablations import load_frozen_reference
from code.federated.parameter_groups import parameter_groups
from code.models.centralized_gru import masked_scaled_mae
from scripts.run_puc_rstattn_v2 import _batch_indices, _to_batch

EPSILON = 1e-12
ANCHOR_WEIGHT = 0.2


def fedfomo_raw_weight(self_loss: float, peer_loss: float, distance: float, epsilon: float = EPSILON) -> float:
    values = (float(self_loss), float(peer_loss), float(distance), float(epsilon))
    if not all(np.isfinite(value) for value in values) or distance < 0 or epsilon <= 0:
        raise ValueError("FedFomo weight inputs must be finite, distance non-negative, epsilon positive")
    return (self_loss - peer_loss) / (distance + epsilon)


def normalize_positive_weights(raw_weights: Mapping[str, float]) -> tuple[dict[str, float], bool]:
    if not all(np.isfinite(float(value)) for value in raw_weights.values()):
        raise ValueError("FedFomo raw weights must be finite")
    positive = {name: max(float(value), 0.0) for name, value in raw_weights.items()}
    total = sum(positive.values())
    if total <= 0.0:
        return {}, True
    return {name: value / total for name, value in positive.items()}, False


def _trainable_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def parameter_l2_distance(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    names = set(left) & set(right)
    if names != set(left) or names != set(right):
        raise ValueError("parameter maps must contain the same trainable names")
    squared = sum(torch.sum((left[name].float() - right[name].float()) ** 2) for name in names)
    result = float(torch.sqrt(squared).item())
    if not np.isfinite(result):
        raise ValueError("FedFomo parameter distance is not finite")
    return result


def _load_trainable_state(model: Any, state: Mapping[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if name not in params or tuple(params[name].shape) != tuple(value.shape):
                raise ValueError(f"incompatible trainable parameter: {name}")
            params[name].copy_(value.to(params[name].device))


def _local_train(model: Any, client: Any, epochs: int, batch_size: int, device: str) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mask = torch.as_tensor(client.load_bus_mask, dtype=torch.bool, device=device)
    losses = []
    for _ in range(epochs):
        model.train()
        for batch in _batch_indices(len(client.train_dataset), batch_size):
            x, y = _to_batch(client.train_dataset, client.scaler, batch)
            optimizer.zero_grad(set_to_none=True)
            final, _, details = model(torch.from_numpy(x).to(device), return_details=True)
            target = torch.from_numpy(y).to(device)
            loss = masked_scaled_mae(final, target, mask) + ANCHOR_WEIGHT * masked_scaled_mae(details["y_gru"], target, mask)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    if not losses:
        raise ValueError(f"client {client.grid_name} has no training data")
    return {"mean_train_loss": float(np.mean(losses))}


def _evaluate_peer(client: Any, state: Mapping[str, torch.Tensor], batch_size: int, device: str) -> float:
    model = full_model(client.grid, client.graph, device)
    _load_trainable_state(model, state)
    indices = np.asarray(client.calibration_indices, dtype=np.int64)
    return float(_evaluate(model, client.grid, indices, client.scaler, device, client.history_start, batch_size)["node_macro"]["mae"])


def run_fedfomo(
    grids: Mapping[str, Any], reference: Path | str | Mapping[str, Any],
    scarce_target: str,
    rounds: int = 10, local_epochs: int = 5, batch_size: int = 32,
    device: str = "cpu", epsilon: float = EPSILON,
) -> dict[str, Any]:
    frozen = load_frozen_reference(reference)
    clients = []
    if scarce_target not in CLIENT_NAMES:
        raise ValueError(f"unknown scarce target: {scarce_target}")
    for name in CLIENT_NAMES:
        grid = grids[name]
        if name == scarce_target:
            split = scarce_split(grid)
            train_indices = split.fit_indices
            scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
            graph = build_scarce_target_graph(grid, split)
            history_start = split.available_start
            calibration_indices = split.calibration_indices
        else:
            train_indices, calibration_indices = donor_split(grid)
            scaler = fit_fit_only_scaler(grid, train_indices)
            from code.models.puc_rstattn_v2_conditional_utility import build_conditional_utility_graph
            graph = build_conditional_utility_graph(grid)
            history_start = None
        client = _make_client(grid, graph, train_indices, _validation_indices(grid), scaler, history_start, device)
        client.grid = grid
        client.graph = graph
        client.calibration_indices = calibration_indices
        client.history_start = history_start
        clients.append(client)

    states = {client.grid_name: _trainable_state(client.model) for client in clients}
    round_history = []
    for round_index in range(1, rounds + 1):
        previous = {name: {key: value.clone() for key, value in state.items()} for name, state in states.items()}
        local_training = {}
        for client in clients:
            _load_trainable_state(client.model, previous[client.grid_name])
            local_training[client.grid_name] = _local_train(client.model, client, local_epochs, batch_size, device)
            states[client.grid_name] = _trainable_state(client.model)
        round_weights, round_diag = {}, {}
        for target in clients:
            self_state = previous[target.grid_name]
            self_loss = _evaluate_peer(target, self_state, batch_size, device)
            raw, distances, peer_losses = {}, {}, {}
            for peer in clients:
                if peer.grid_name == target.grid_name:
                    continue
                distances[peer.grid_name] = parameter_l2_distance(states[peer.grid_name], self_state)
                peer_losses[peer.grid_name] = _evaluate_peer(target, states[peer.grid_name], batch_size, device)
                raw[peer.grid_name] = fedfomo_raw_weight(self_loss, peer_losses[peer.grid_name], distances[peer.grid_name], epsilon)
            normalized, fallback = normalize_positive_weights(raw)
            round_weights[target.grid_name] = normalized
            round_diag[target.grid_name] = {
                "is_scarce_target": target.grid_name == scarce_target,
                "baseline_self_calibration_loss": self_loss,
                "peer_calibration_loss_by_donor": peer_losses,
                "trainable_parameter_l2_distance_by_donor": distances,
                "raw_weight_by_donor": raw,
                "positive_clipped_weight_by_donor": {name: max(value, 0.0) for name, value in raw.items()},
                "normalized_weight_by_donor": normalized,
                "positive_peer_count": len(normalized),
                "no_positive_peer_fallback": fallback,
            }
        updated = {}
        for target in clients:
            own = states[target.grid_name]
            weights = round_weights[target.grid_name]
            if not weights:
                updated[target.grid_name] = own
                continue
            result = {name: own[name].clone() for name in own}
            for name in result:
                delta = sum(weights[peer] * (states[peer][name] - previous[target.grid_name][name]) for peer in weights)
                result[name] = own[name] + delta
            updated[target.grid_name] = result
        states = updated
        for client in clients:
            _load_trainable_state(client.model, states[client.grid_name])
        round_history.append({"round": round_index, "local_training": local_training, "peer_weights": round_weights, "diagnostics": round_diag})

    scenario_metrics = {}
    for client in clients:
        grid = client.grid
        split = scarce_split(grid) if client.grid_name == scarce_target else None
        scenario_metrics[client.grid_name] = {
            "audit": _evaluate(client.model, grid, split.audit_indices if split is not None else client.calibration_indices, client.scaler, device, split.available_start if split is not None else None, batch_size),
            "validation": _evaluate(client.model, grid, _validation_indices(grid), client.scaler, device, None, batch_size),
        }
    macro = {split: {scope: {metric: float(np.mean([scenario_metrics[name][split][scope][metric] for name in CLIENT_NAMES])) for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for split in ("audit", "validation")}
    comparisons, wins = {}, {}
    for comparator in ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl"):
        comparisons[comparator] = {split: float((frozen["four_scenario_macro"][split][comparator]["node_macro"]["mae"] - macro[split]["node_macro"]["mae"]) / (frozen["four_scenario_macro"][split][comparator]["node_macro"]["mae"] + 1e-12)) for split in ("audit", "validation")}
        wins[comparator] = sum(scenario_metrics[target]["validation"]["node_macro"]["mae"] < frozen["scenarios"][target]["methods"][comparator]["validation"]["node_macro"]["mae"] for target in CLIENT_NAMES)
    return {"experiment": "formal_fedfomo_style_25pct", "method_name": "FedFomo-style", "client_grid_names": list(CLIENT_NAMES), "scarce_target": scarce_target, "rounds": rounds, "local_epochs": local_epochs, "batch_size": batch_size, "learning_rate": 1e-3, "seed": 42, "fedfomo_epsilon": epsilon, "heterogeneous_grid_adaptation": "peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged", "validation_used_for_peer_weighting": False, "audit_used_for_peer_weighting": False, "test_evaluated": False, "communication_round_selection": "fixed_final_round", "round_history": round_history, "final_round_peer_weights": round_history[-1]["peer_weights"], "scenarios": scenario_metrics, "four_scenario_unweighted_macro": macro, "frozen_comparison_metrics": {}, "relative_improvement_vs_frozen_comparators": comparisons, "win_counts_vs_frozen_comparators": wins, "frozen_reference": {"path": "results/federated/formal_25pct/btd_fl_25pct_seed42.json", "direct_transfer_path": "results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json", "test_evaluated": frozen["test_evaluated"]}}


__all__ = ["fedfomo_raw_weight", "normalize_positive_weights", "parameter_l2_distance", "run_fedfomo"]
