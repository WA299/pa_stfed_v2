"""Self-contained formal 25%-history BTD-FL benchmark orchestration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model, inject_temporal_state, train_full_model
from code.audits.federated_transfer_benefit import IndexDataset, _evaluate, donor_split, fit_fit_only_scaler, make_proxy, scarce_split, train_proxy, benefit
from code.data.forecast_dataset import ForecastFeatureScaler
from code.federated.trainer import FederatedClient, FederatedTrainer
from code.federated.parameter_groups import parameter_groups
from code.models.puc_rstattn_v2_conditional_utility import PUCRSTAttnV2ConditionalUtility, build_conditional_utility_graph

CLIENT_NAMES = ("39_bus_semi_urban_reference_grid", "50_bus_rural_reference_grid", "56_bus_semi_urban_reference_grid", "80_bus_rural_reference_grid")
METHODS = ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl")


def select_formal_donor(
    calibration_benefits: Mapping[str, float], variant: str = "btd_fl"
) -> tuple[str | None, float, bool, str]:
    """Select a donor from current-run calibration benefits only.

    This small pure helper is shared by the runner and focused contract tests,
    making the zero-transfer and ablation semantics explicit.
    """
    if not calibration_benefits:
        return None, 0.0, variant == "btd_fl", "maximum_current_calibration_benefit_with_zero_transfer"
    maximum_donor, maximum_benefit = max(
        calibration_benefits.items(), key=lambda item: (float(item[1]), item[0])
    )
    maximum_benefit = float(maximum_benefit)
    if variant == "btd_no_benefit_selection":
        donor = sorted(calibration_benefits)[0]
        return donor, float(calibration_benefits[donor]), False, "fixed_deterministic_no_benefit"
    if variant == "btd_no_zero_transfer":
        return maximum_donor, maximum_benefit, False, "maximum_current_calibration_benefit_no_zero_fallback"
    if maximum_benefit <= 0.0:
        return None, 0.0, True, "maximum_current_calibration_benefit_with_zero_transfer"
    return maximum_donor, maximum_benefit, False, "maximum_current_calibration_benefit_with_zero_transfer"


def _indices(grid: Any) -> np.ndarray:
    train = grid.splits["train"]
    return np.arange(train.start_index + 168, train.end_index, dtype=np.int64)


def _validation_indices(grid: Any) -> np.ndarray:
    split = grid.splits["validation"]
    return np.arange(split.start_index, split.end_index, dtype=np.int64)


def _model_from_graph(grid: Any, graph: Any, device: str) -> Any:
    import torch
    torch.manual_seed(42)
    return PUCRSTAttnV2ConditionalUtility(grid.num_nodes, graph.edge_index, graph.relation_features[:, 1:], grid.load_bus_mask, graph.relation_features[:, 0]).to(device)


def _make_client(grid: Any, graph: Any, train_indices: np.ndarray, validation_indices: np.ndarray, scaler: ForecastFeatureScaler, history_start: int | None, device: str) -> FederatedClient:
    return FederatedClient(grid_name=grid.grid_name, model=_model_from_graph(grid, graph, device), train_dataset=IndexDataset(grid, train_indices, history_start), validation_dataset=IndexDataset(grid, validation_indices), scaler=scaler, load_bus_mask=np.asarray(grid.load_bus_mask, dtype=bool), graph_metadata=dict(graph.diagnostics))


def load_selected_donors(audit_json: Path) -> dict[str, str | None]:
    """Load prior selections for optional consistency checks, never decisions."""
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    for key in ("validation_used_for_selection", "audit_used_for_selection", "test_evaluated"):
        if report.get(key) is not False:
            raise ValueError(f"historical audit violates {key}")
    return {name: item.get("selected_donor") for name, item in report.get("selected_policy", {}).items()}


def load_selection_metadata(audit_json: Path) -> dict[str, Any]:
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    for key in ("validation_used_for_selection", "audit_used_for_selection", "test_evaluated"):
        if report.get(key) is not False:
            raise ValueError(f"historical audit violates {key}")
    names = report.get("client_grid_names", list(CLIENT_NAMES))
    return {
        target: {
            "selected_donor": item.get("selected_donor"),
            "candidate_donors": [name for name in names if name != target],
            "calibration_benefit_by_donor": {
                name: value for name, value in report.get("calibration_benefit", {}).get(target, {}).items()
                if name != target
            },
        }
        for target, item in report.get("selected_policy", {}).items()
    }


def prepare_scenario_clients(grids: dict[str, Any], target: str, device: str = "cpu") -> tuple[list[FederatedClient], Any, Any, dict[str, Any]]:
    target_grid = grids[target]
    split = scarce_split(target_grid)
    target_scaler = fit_fit_only_scaler(target_grid, split.fit_indices, split.available_start)
    target_graph = build_scarce_target_graph(target_grid, split)
    clients = []
    metadata = {}
    for name in CLIENT_NAMES:
        grid = grids[name]
        if name == target:
            train_indices, scaler, graph, history_start = split.fit_indices, target_scaler, target_graph, split.available_start
        else:
            train_indices, scaler, graph, history_start = _indices(grid), fit_fit_only_scaler(grid, _indices(grid)), build_conditional_utility_graph(grid), None
        metadata[name] = dict(graph.diagnostics)
        clients.append(_make_client(grid, graph, train_indices, _validation_indices(grid), scaler, history_start, device))
    return clients, split, target_graph, metadata


def _method_metrics(model: Any, grid: Any, split: Any, scaler: Any, device: str, batch_size: int = 32) -> dict[str, Any]:
    model = model.to(device)
    return {"audit": _evaluate(model, grid, split.audit_indices, scaler, device, split.available_start, batch_size), "validation": _evaluate(model, grid, _validation_indices(grid), scaler, device, None, batch_size)}


def _cpu_state(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_proxy_once(grid: Any, fit_indices: np.ndarray, calibration_indices: np.ndarray, scaler: Any, device: str, max_epochs: int, batch_size: int, initial_state: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    model, training = train_proxy(grid, fit_indices, calibration_indices, scaler, initial_state=initial_state, device=device, max_epochs=max_epochs, history_start_index=getattr(scaler, "fit_start_index", None), batch_size=batch_size)
    state = _cpu_state(model)
    del model
    return state, dict(training)


def _spatial_buffer_snapshot(model: Any) -> dict[str, Any]:
    groups = parameter_groups(model)
    snapshot = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if name in groups["spatial"]}
    snapshot.update({name: value.detach().cpu().clone() for name, value in model.named_buffers() if name in {"load_bus_mask", "utility_edge_index", "physical_relation_features", "utility_prior", "selected_utility"}})
    return snapshot


def _same_snapshot(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(bool(np.array_equal(left[name].numpy(), right[name].numpy())) for name in left)


def _train_full_donor_states(grids: dict[str, Any], device: str, max_epochs: int, batch_size: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states, metadata = {}, {}
    for name in CLIENT_NAMES:
        grid = grids[name]; fit, calibration = donor_split(grid); scaler = fit_fit_only_scaler(grid, fit); graph = build_conditional_utility_graph(grid); model = full_model(grid, graph, device)
        state, training = train_full_model(model, grid, fit, calibration, scaler, device=device, max_epochs=max_epochs, batch_size=batch_size)
        params = dict(model.named_parameters())
        states[name] = {key: value.detach().cpu().clone() for key, value in state.items() if key in params}
        metadata[name] = {"source": "donor_full_history_model", **training}
        del model
    return states, metadata


def _copy_compatible_trainables(model: Any, donor_state: Mapping[str, Any]) -> tuple[str, ...]:
    import torch
    params = dict(model.named_parameters()); copied = []
    with torch.no_grad():
        for name, value in donor_state.items():
            if name in params and tuple(params[name].shape) == tuple(value.shape):
                params[name].copy_(value.to(params[name].device)); copied.append(name)
    return tuple(sorted(copied))


def _run_federated(grids: dict[str, Any], target: str, method: str, device: str, rounds: int, local_epochs: int, batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    clients, split, _graph, graph_metadata = prepare_scenario_clients(grids, target, device)
    algorithm = {"fedavg": "standard", "fedprox": "fedprox", "fedper": "fedper"}[method]
    report = FederatedTrainer(clients, "fedavg_all", rounds=rounds, local_epochs=local_epochs, batch_size=batch_size, learning_rate=1e-3, seed=42, device=device, algorithm=algorithm, evaluate_validation_during_training=False).run()
    target_client = next(client for client in clients if client.grid_name == target)
    metrics = _method_metrics(target_client.model, grids[target], split, target_client.scaler, device, batch_size)
    counts, weights = report["local_sample_counts"], report["round_history"][-1]["aggregation_weights"]
    total = float(sum(counts.values()))
    expected = {name: counts[name] / total for name in counts}
    if not np.isclose(sum(weights.values()), 1.0) or any(not np.isclose(weights[name], expected[name]) for name in counts):
        raise AssertionError("aggregation weights are not sample-count weighted")
    return metrics, {"federated_report": report, "local_sample_counts": counts, "aggregation_weights": weights, "graph_metadata": graph_metadata}


def run_formal_benchmark(grids: dict[str, Any], selected_donors: dict[str, str | None] | None = None, device: str = "cpu", rounds: int = 10, local_epochs: int = 5, max_epochs: int = 50, btd_variant: str = "btd_fl", batch_size: int = 32, selection_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    variants = {"btd_fl", "btd_no_benefit_selection", "btd_no_zero_transfer", "btd_full_model_transfer"}
    if btd_variant not in variants:
        raise ValueError("unknown BTD variant")
    donor_states, donor_metadata = {}, {}
    for name in CLIENT_NAMES:
        grid = grids[name]; fit, calibration = donor_split(grid); scaler = fit_fit_only_scaler(grid, fit)
        state, training = _train_proxy_once(grid, fit, calibration, scaler, device, max_epochs, batch_size)
        proxy = make_proxy(grid); proxy.load_state_dict(state)
        from code.audits.btd_full_backbone_bridge import temporal_state_from_model
        donor_states[name] = temporal_state_from_model(proxy)
        donor_metadata[name] = {"donor_fit_target_count": int(len(fit)), "donor_calibration_target_count": int(len(calibration)), "scaler_fit_start_index": int(scaler.fit_start_index), "scaler_fit_end_index": int(scaler.fit_end_index), "scaler_fit_timestamp_end": scaler.fit_timestamp_end, **training, "source": "current_formal_run_full_history_proxy"}
        del proxy
    donor_full_states, donor_full_metadata = ({}, {})
    if btd_variant == "btd_full_model_transfer":
        donor_full_states, donor_full_metadata = _train_full_donor_states(grids, device, max_epochs, batch_size)

    scenarios = {}
    for target in CLIENT_NAMES:
        grid = grids[target]; split = scarce_split(grid); scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start); graph = build_scarce_target_graph(grid, split)
        local_proxy_state, local_proxy_training = _train_proxy_once(grid, split.fit_indices, split.calibration_indices, scaler, device, max_epochs, batch_size)
        local_proxy = make_proxy(grid); local_proxy.load_state_dict(local_proxy_state); local_calibration_mae = float(local_proxy_training["best_calibration_node_mae"])
        adapted_by_donor, calibration_benefits = {}, {}
        for donor in CLIENT_NAMES:
            if donor == target: continue
            donor_temporal = {name: value for name, value in donor_states[donor].items() if name in parameter_groups(local_proxy)["temporal"]}
            initial_proxy = make_proxy(grid); inject_temporal_state(initial_proxy, donor_temporal); initial_state = _cpu_state(initial_proxy); del initial_proxy
            adapted_state, adaptation_training = _train_proxy_once(grid, split.fit_indices, split.calibration_indices, scaler, device, max_epochs, batch_size, initial_state)
            adapted_proxy = make_proxy(grid); adapted_proxy.load_state_dict(adapted_state); adapted_mae = float(adaptation_training["best_calibration_node_mae"]); del adapted_proxy
            calibration_benefits[donor] = benefit(local_calibration_mae, adapted_mae)
            adapted_by_donor[donor] = {"state": adapted_state, "calibration_node_mae": adapted_mae, "training": {"donor": donor, "source": "current_formal_run_donor_proxy", **adaptation_training}}
        selected_donor, selected_benefit, fallback, selection_rule = select_formal_donor(calibration_benefits, btd_variant)

        local_model = full_model(grid, graph, device); local_spatial = _spatial_buffer_snapshot(local_model); inject_temporal_state(local_model, {name: value for name, value in local_proxy_state.items() if name in parameter_groups(local_model)["temporal"]})
        local_state, local_full_training = train_full_model(local_model, grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, batch_size=batch_size); local_model.load_state_dict(local_state); local_metrics = _method_metrics(local_model, grid, split, scaler, device, batch_size)
        if fallback:
            btd_metrics, btd_training = copy.deepcopy(local_metrics), {"fallback_exact_local": True, "full": copy.deepcopy(local_full_training)}
        elif btd_variant == "btd_full_model_transfer":
            btd_model = full_model(grid, graph, device); assert _same_snapshot(_spatial_buffer_snapshot(btd_model), local_spatial)
            transferred = _copy_compatible_trainables(btd_model, donor_full_states[selected_donor]); state, training = train_full_model(btd_model, grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, batch_size=batch_size); btd_model.load_state_dict(state); btd_metrics = _method_metrics(btd_model, grid, split, scaler, device, batch_size); btd_training = {"full": training, "transferred_parameter_names": list(transferred)}; del btd_model
        else:
            btd_model = full_model(grid, graph, device); assert _same_snapshot(_spatial_buffer_snapshot(btd_model), local_spatial)
            inject_temporal_state(btd_model, {name: value for name, value in adapted_by_donor[selected_donor]["state"].items() if name in parameter_groups(btd_model)["temporal"]}); state, training = train_full_model(btd_model, grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, batch_size=batch_size); btd_model.load_state_dict(state); btd_metrics = _method_metrics(btd_model, grid, split, scaler, device, batch_size); btd_training = {"adaptation": adapted_by_donor[selected_donor]["training"], "full": training, "transferred_parameter_names": list(parameter_groups(btd_model)["temporal"])}; del btd_model
        methods = {"scarce_local": local_metrics}; federated = {}
        for method in ("fedavg", "fedprox", "fedper"):
            methods[method], federated[method] = _run_federated(grids, target, method, device, rounds, local_epochs, batch_size)
        methods["btd_fl"] = btd_metrics
        selected_adaptation = adapted_by_donor.get(selected_donor)
        train = grid.splits["train"]
        scenarios[target] = {"methods": methods, "metadata": {
            "canonical_train_raw_hours": int(train.end_index - train.start_index),
            "available_raw_hours": int(split.available_end - split.available_start),
            "history_fraction": 0.25,
            "available_raw_start_index": int(split.available_start),
            "available_raw_end_index": int(split.available_end),
            "eligible_target_count": int(len(split.eligible_indices)),
            "fit_target_count": int(len(split.fit_indices)),
            "calibration_target_count": int(len(split.calibration_indices)),
            "audit_target_count": int(len(split.audit_indices)),
            "candidate_donors": [name for name in CLIENT_NAMES if name != target],
            "calibration_benefit_by_donor": calibration_benefits,
            "donor_adaptation_by_donor": {
                donor: {
                    "donor_proxy_source": "current_formal_run_full_history_proxy",
                    "target_adaptation_best_calibration_node_mae": item["calibration_node_mae"],
                    "target_adaptation_epochs_run": item["training"]["epochs_run"],
                    "training": item["training"],
                }
                for donor, item in adapted_by_donor.items()
            },
            "selected_donor": selected_donor,
            "selected_calibration_benefit": float(selected_benefit),
            "zero_transfer_fallback": bool(fallback),
            "donor_selection_rule": selection_rule,
            "selection_matches_previous_audit": (selected_donor == (selected_donors or {}).get(target)) if selected_donors is not None else None,
            "target_graph_metadata": graph.diagnostics,
            "local_proxy_training": local_proxy_training,
            "selected_donor_proxy_training": donor_metadata.get(selected_donor),
            "selected_donor_target_adaptation_training": selected_adaptation["training"] if selected_adaptation else None,
            "scarce_local_full_model_training": local_full_training,
            "btd_target_full_model_training": btd_training["full"],
            "btd_training": btd_training,
            "federated": federated,
            "validation_used_for_selection": False,
            "audit_used_for_selection": False,
            "test_evaluated": False,
        }}
        del local_proxy, local_model

    macros = {}
    for split_name in ("audit", "validation"):
        macros[split_name] = {method: {scope: {metric: float(np.mean([scenarios[name]["methods"][method][split_name][scope][metric] for name in CLIENT_NAMES])) for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for method in METHODS}
    wins = {method: sum(scenarios[name]["methods"]["btd_fl"]["validation"]["node_macro"]["mae"] < scenarios[name]["methods"][method]["validation"]["node_macro"]["mae"] for name in CLIENT_NAMES) for method in METHODS[:-1]}
    relative = {method: {split: float((macros[split][method]["node_macro"]["mae"] - macros[split]["btd_fl"]["node_macro"]["mae"]) / (macros[split][method]["node_macro"]["mae"] + 1e-12)) for split in ("audit", "validation")} for method in METHODS[:-1]}
    return {"experiment": "formal_btd_fl_25pct_benchmark", "method_name": "BTD-FL", "btd_variant": btd_variant, "methods": list(METHODS), "client_grid_names": list(CLIENT_NAMES), "rounds": rounds, "local_epochs": local_epochs, "max_epochs": max_epochs, "batch_size": batch_size, "learning_rate": 1e-3, "seed": 42, "history_fraction": 0.25, "validation_used_for_selection": False, "audit_used_for_selection": False, "test_evaluated": False, "communication_round_selection": "fixed_final_round", "canonical_validation_evaluated_during_training": False, "donor_proxy_metadata_by_grid": donor_metadata, "donor_full_model_metadata_by_grid": donor_full_metadata, "scenarios": scenarios, "four_scenario_macro": macros, "relative_btd_fl_improvement_vs": relative, "btd_fl_win_counts": wins}


__all__ = ["CLIENT_NAMES", "METHODS", "load_selected_donors", "load_selection_metadata", "prepare_scenario_clients", "select_formal_donor", "run_formal_benchmark"]
