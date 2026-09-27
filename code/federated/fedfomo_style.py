"""FedFomo-style personalized comparator for heterogeneous-grid FL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model
from code.audits.federated_transfer_benefit import _evaluate, donor_split, fit_fit_only_scaler, scarce_split
from code.federated.formal_btd import CLIENT_NAMES, _validation_indices, _make_client
from code.federated.formal_btd_ablations import load_frozen_reference
from code.federated.seeding import reproducibility_metadata, set_global_seed
from code.models.centralized_gru import masked_scaled_mae
from scripts.run_puc_rstattn_v2 import _batch_indices, _to_batch

EPSILON = 1e-12
ANCHOR_WEIGHT = 0.2
TOPOLOGY_BUFFERS = {"load_bus_mask", "utility_edge_index", "physical_relation_features", "utility_prior", "selected_utility"}


def fedfomo_raw_weight(self_loss: float, candidate_loss: float, distance: float, epsilon: float = EPSILON) -> float:
    values = (float(self_loss), float(candidate_loss), float(distance), float(epsilon))
    if not all(np.isfinite(value) for value in values) or distance < 0 or epsilon <= 0:
        raise ValueError("FedFomo inputs must be finite, non-negative distance, positive epsilon")
    return (self_loss - candidate_loss) / (distance + epsilon)


def normalize_positive_weights(raw_weights: Mapping[str, float]) -> tuple[dict[str, float], bool]:
    if not all(np.isfinite(float(value)) for value in raw_weights.values()):
        raise ValueError("FedFomo weights must be finite")
    positive = {name: float(value) for name, value in raw_weights.items() if float(value) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}, True
    return {name: value / total for name, value in positive.items()}, False


def parameter_l2_distance(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if set(left) != set(right):
        raise ValueError("parameter maps must contain the same trainable names")
    squared = sum(torch.sum((left[name].float() - right[name].float()) ** 2) for name in left)
    result = float(torch.sqrt(squared).item())
    if not np.isfinite(result):
        raise ValueError("FedFomo distance is not finite")
    return result


def _trainable_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def _load_trainable_state(model: Any, state: Mapping[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if name not in params or tuple(params[name].shape) != tuple(value.shape):
                raise ValueError(f"incompatible trainable parameter: {name}")
            params[name].copy_(value.to(params[name].device))


def _local_train(model: Any, client: Any, epochs: int, batch_size: int, device: str) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mask = torch.as_tensor(client.load_bus_mask, dtype=torch.bool, device=device)
    for _ in range(epochs):
        model.train()
        for batch in _batch_indices(len(client.train_dataset), batch_size):
            x, y = _to_batch(client.train_dataset, client.scaler, batch)
            optimizer.zero_grad(set_to_none=True)
            final, _, details = model(torch.from_numpy(x).to(device), return_details=True)
            target = torch.from_numpy(y).to(device)
            loss = masked_scaled_mae(final, target, mask) + ANCHOR_WEIGHT * masked_scaled_mae(details["y_gru"], target, mask)
            loss.backward(); optimizer.step()


def _evaluate_candidate(client: Any, state: Mapping[str, torch.Tensor], batch_size: int, device: str, seed: int = 42) -> float:
    model = full_model(client.grid, client.graph, device, seed=seed)
    _load_trainable_state(model, state)
    return float(_evaluate(model, client.grid, client.calibration_indices, client.scaler, device, client.history_start, batch_size)["node_macro"]["mae"])


def run_fedfomo_scenario(grids: Mapping[str, Any], scarce_target: str, rounds: int, local_epochs: int, batch_size: int, device: str, epsilon: float, seed: int = 42) -> dict[str, Any]:
    clients = []
    for name in CLIENT_NAMES:
        grid = grids[name]
        if name == scarce_target:
            split = scarce_split(grid); train_indices = split.fit_indices; calibration = split.calibration_indices; scaler = fit_fit_only_scaler(grid, train_indices, split.available_start); graph = build_scarce_target_graph(grid, split); history_start = split.available_start
        else:
            train_indices, calibration = donor_split(grid); scaler = fit_fit_only_scaler(grid, train_indices)
            from code.models.puc_rstattn_v2_conditional_utility import build_conditional_utility_graph
            graph = build_conditional_utility_graph(grid); history_start = None
        client = _make_client(grid, graph, train_indices, _validation_indices(grid), scaler, history_start, device, seed)
        client.grid = grid; client.graph = graph; client.calibration_indices = calibration; client.history_start = history_start
        clients.append(client)
    set_global_seed(seed)
    states = {client.grid_name: _trainable_state(client.model) for client in clients}
    history = []
    for round_index in range(1, rounds + 1):
        previous = {name: {key: value.clone() for key, value in state.items()} for name, state in states.items()}
        for client in clients:
            _load_trainable_state(client.model, previous[client.grid_name])
            set_global_seed(seed)
            _local_train(client.model, client, local_epochs, batch_size, device)
            states[client.grid_name] = _trainable_state(client.model)
        round_weights, diagnostics = {}, {}
        for target in clients:
            anchor = previous[target.grid_name]
            self_loss = _evaluate_candidate(target, anchor, batch_size, device, seed)
            losses, distances, raw = {}, {}, {}
            for candidate in clients:
                distances[candidate.grid_name] = parameter_l2_distance(states[candidate.grid_name], anchor)
                losses[candidate.grid_name] = _evaluate_candidate(target, states[candidate.grid_name], batch_size, device, seed)
                raw[candidate.grid_name] = fedfomo_raw_weight(self_loss, losses[candidate.grid_name], distances[candidate.grid_name], epsilon)
            normalized, fallback = normalize_positive_weights(raw)
            round_weights[target.grid_name] = normalized
            diagnostics[target.grid_name] = {
                "is_scarce_target": target.grid_name == scarce_target,
                "baseline_self_calibration_loss": self_loss,
                "candidate_calibration_loss_by_client": losses,
                "peer_calibration_loss_by_donor": {name: value for name, value in losses.items() if name != target.grid_name},
                "trainable_parameter_l2_distance_by_client": distances,
                "raw_weight_by_client": raw,
                "raw_weight_by_donor": {name: value for name, value in raw.items() if name != target.grid_name},
                "positive_clipped_weight_by_client": {name: max(value, 0.0) for name, value in raw.items()},
                "normalized_weight_by_client": normalized,
                "normalized_weight_by_donor": {name: value for name, value in normalized.items() if name != target.grid_name},
                "positive_candidate_count": len(normalized),
                "peer_only_positive_count": sum(name != target.grid_name for name in normalized),
                "no_positive_candidate_fallback": fallback,
            }
        updated = {}
        for target in clients:
            anchor = previous[target.grid_name]; weights = round_weights[target.grid_name]
            updated[target.grid_name] = {name: anchor[name] + sum(weights[candidate] * (states[candidate][name] - anchor[name]) for candidate in weights) for name in anchor} if weights else {name: value.clone() for name, value in anchor.items()}
        states = updated
        for client in clients:
            _load_trainable_state(client.model, states[client.grid_name])
        history.append({"round": round_index, "peer_weights": round_weights, "diagnostics": diagnostics})
    split = scarce_split(grids[scarce_target])
    primary = {"audit": _evaluate(clients[CLIENT_NAMES.index(scarce_target)].model, grids[scarce_target], split.audit_indices, clients[CLIENT_NAMES.index(scarce_target)].scaler, device, split.available_start, batch_size), "validation": _evaluate(clients[CLIENT_NAMES.index(scarce_target)].model, grids[scarce_target], _validation_indices(grids[scarce_target]), clients[CLIENT_NAMES.index(scarce_target)].scaler, device, None, batch_size)}
    return {"scarce_target_metrics": primary, "round_history": history, "final_round_peer_weights": history[-1]["peer_weights"], "metadata": {"scarce_target": scarce_target, "validation_used_for_peer_weighting": False, "audit_used_for_peer_weighting": False, "test_evaluated": False, "communication_round_selection": "fixed_final_round"}}


def _macro_from_metrics(metrics_by_target: Mapping[str, Any]) -> dict[str, Any]:
    return {split: {scope: {metric: float(np.mean([metrics_by_target[target][split][scope][metric] for target in CLIENT_NAMES])) for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for split in ("audit", "validation")}


def run_fedfomo(grids: Mapping[str, Any], reference: Path | str | Mapping[str, Any], rounds: int = 10, local_epochs: int = 5, batch_size: int = 32, device: str = "cpu", epsilon: float = EPSILON, direct_transfer_reference: Path | str | Mapping[str, Any] | None = None, seed: int = 42) -> dict[str, Any]:
    set_global_seed(seed)
    frozen = load_frozen_reference(reference)
    scenarios = {target: run_fedfomo_scenario(grids, target, rounds, local_epochs, batch_size, device, epsilon, seed) for target in CLIENT_NAMES}
    primary = {target: scenarios[target]["scarce_target_metrics"] for target in CLIENT_NAMES}
    macro = _macro_from_metrics(primary)
    comparisons, wins = {}, {}
    for comparator in ("scarce_local", "fedavg", "fedprox", "fedper"):
        comparisons[comparator] = {split: float((frozen["four_scenario_macro"][split][comparator]["node_macro"]["mae"] - macro[split]["node_macro"]["mae"]) / (frozen["four_scenario_macro"][split][comparator]["node_macro"]["mae"] + 1e-12)) for split in ("audit", "validation")}
        wins[comparator] = sum(primary[target]["validation"]["node_macro"]["mae"] < frozen["scenarios"][target]["methods"][comparator]["validation"]["node_macro"]["mae"] for target in CLIENT_NAMES)
    result = {"experiment": "formal_fedfomo_style_25pct", "method_name": "FedFomo-style", "client_grid_names": list(CLIENT_NAMES), "rounds": rounds, "local_epochs": local_epochs, "max_epochs": 50, "patience": 8, "batch_size": batch_size, "learning_rate": 1e-3, "seed": int(seed), "history_fraction": 0.25, "train_calibration_audit_split": "60/20/20 for scarce target; donor train-internal fit/calibration", "fedfomo_epsilon": epsilon, "heterogeneous_grid_adaptation": "peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged", "validation_used_for_peer_weighting": False, "audit_used_for_peer_weighting": False, "test_evaluated": False, "communication_round_selection": "fixed_final_round", "canonical_validation_evaluated_during_training": False, "scenarios": scenarios, "four_scenario_unweighted_macro": macro, "relative_improvement_vs_frozen_comparators": comparisons, "win_counts_vs_frozen_comparators": wins, "frozen_reference": {"path": "results/federated/formal_25pct/btd_fl_25pct_seed42.json", "test_evaluated": frozen["test_evaluated"]}, "reproducibility": reproducibility_metadata(seed, device)}
    if direct_transfer_reference is not None:
        direct = json.loads(Path(direct_transfer_reference).read_text(encoding="utf-8")) if not isinstance(direct_transfer_reference, Mapping) else dict(direct_transfer_reference)
        if direct.get("btd_variant") != "btd_fl_direct_transfer" or direct.get("test_evaluated") is not False:
            raise ValueError("direct-transfer frozen result violates guardrails")
        direct_metrics = {target: direct["scenarios"][target]["metrics"] for target in CLIENT_NAMES}
        direct_macro = direct.get("four_target_unweighted_macro") or _macro_from_metrics(direct_metrics)
        direct_comp = {split: float((direct_macro[split]["node_macro"]["mae"] - macro[split]["node_macro"]["mae"]) / (direct_macro[split]["node_macro"]["mae"] + 1e-12)) for split in ("audit", "validation")}
        result["relative_improvement_vs_final_btd_fl_direct_transfer"] = direct_comp
        result["win_counts_vs_final_btd_fl_direct_transfer"] = {split: sum(primary[target][split]["node_macro"]["mae"] < direct_metrics[target][split]["node_macro"]["mae"] for target in CLIENT_NAMES) for split in ("audit", "validation")}
        result["per_target_comparison_vs_final_btd_fl_direct_transfer"] = {target: {split: float((direct_metrics[target][split]["node_macro"]["mae"] - primary[target][split]["node_macro"]["mae"]) / (direct_metrics[target][split]["node_macro"]["mae"] + 1e-12)) for split in ("audit", "validation")} for target in CLIENT_NAMES}
        result["frozen_reference"]["direct_transfer_path"] = "results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json"
    return result


__all__ = ["fedfomo_raw_weight", "normalize_positive_weights", "parameter_l2_distance", "run_fedfomo_scenario", "run_fedfomo"]
