"""Formal 25%-history leave-one-scarce BTD-FL benchmark orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from code.audits.btd_full_backbone_bridge import (
    build_scarce_target_graph,
    full_model,
    inject_temporal_state,
    relative_improvement,
    temporal_state_from_model,
    train_full_model,
)
from code.audits.federated_transfer_benefit import (
    IndexDataset,
    _evaluate,
    donor_split,
    fit_fit_only_scaler,
    make_proxy,
    scarce_split,
    train_proxy,
    transfer_temporal_parameters,
)
from code.data.forecast_dataset import ForecastFeatureScaler
from code.federated.trainer import FederatedClient, FederatedTrainer
from code.models.puc_rstattn_v2_conditional_utility import (
    PUCRSTAttnV2ConditionalUtility,
    build_conditional_utility_graph,
)

CLIENT_NAMES = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
METHODS = ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl")


def _indices(grid: Any) -> np.ndarray:
    train = grid.splits["train"]
    return np.arange(train.start_index + 168, train.end_index, dtype=np.int64)


def _validation_indices(grid: Any) -> np.ndarray:
    split = grid.splits["validation"]
    return np.arange(split.start_index, split.end_index, dtype=np.int64)


def _model_from_graph(grid: Any, graph: Any, device: str) -> Any:
    import torch
    torch.manual_seed(42)
    return PUCRSTAttnV2ConditionalUtility(
        grid.num_nodes, graph.edge_index, graph.relation_features[:, 1:],
        grid.load_bus_mask, graph.relation_features[:, 0],
    ).to(device)


def _grid_metadata(graph: Any) -> dict[str, Any]:
    return dict(graph.diagnostics)


def _make_client(grid: Any, graph: Any, train_indices: np.ndarray, validation_indices: np.ndarray,
                 scaler: ForecastFeatureScaler, history_start: int | None, device: str) -> FederatedClient:
    from code.audits.federated_transfer_benefit import IndexDataset
    model = _model_from_graph(grid, graph, device)
    return FederatedClient(
        grid_name=grid.grid_name,
        model=model,
        train_dataset=IndexDataset(grid, train_indices, history_start),
        validation_dataset=IndexDataset(grid, validation_indices),
        scaler=scaler,
        load_bus_mask=np.asarray(grid.load_bus_mask, dtype=bool),
        graph_metadata=_grid_metadata(graph),
    )


def load_selected_donors(audit_json: Path) -> dict[str, str | None]:
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    if report.get("validation_used_for_selection") is not False:
        raise ValueError("validation was used for donor selection")
    if report.get("audit_used_for_selection") is not False:
        raise ValueError("audit was used for donor selection")
    if report.get("test_evaluated") is not False:
        raise ValueError("completed audit evaluated test")
    return {name: item.get("selected_donor") for name, item in report["selected_policy"].items()}


def load_selection_metadata(audit_json: Path) -> dict[str, Any]:
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    if report.get("validation_used_for_selection") is not False or report.get("audit_used_for_selection") is not False or report.get("test_evaluated") is not False:
        raise ValueError("audit selection metadata violates leakage guardrails")
    result = {}
    for target, item in report["selected_policy"].items():
        result[target] = {
            "candidate_donors": [name for name in report["client_grid_names"] if name != target],
            "calibration_benefit_by_donor": {name: value for name, value in report["calibration_benefit"][target].items() if name != target},
            "selected_donor": item.get("selected_donor"),
            "selected_calibration_benefit": item.get("selected_calibration_benefit", 0.0),
            "zero_transfer_fallback": bool(item.get("zero_transfer_fallback", item.get("selected_donor") is None)),
        }
    return result


def prepare_scenario_clients(grids: dict[str, Any], target: str, device: str = "cpu") -> tuple[list[FederatedClient], Any, Any, dict[str, Any]]:
    target_grid = grids[target]
    target_split = scarce_split(target_grid)
    target_scaler = fit_fit_only_scaler(target_grid, target_split.fit_indices, target_split.available_start)
    target_graph = build_scarce_target_graph(target_grid, target_split)
    validation = _validation_indices(target_grid)
    clients: list[FederatedClient] = []
    graph_metadata: dict[str, Any] = {target: target_graph.diagnostics}
    for name in CLIENT_NAMES:
        grid = grids[name]
        if name == target:
            train_indices, scaler, history_start = target_split.fit_indices, target_scaler, target_split.available_start
            graph = target_graph
        else:
            train_indices = _indices(grid)
            scaler = fit_fit_only_scaler(grid, train_indices)
            graph = build_conditional_utility_graph(grid)
            history_start = None
        graph_metadata[name] = graph.diagnostics
        clients.append(_make_client(grid, graph, train_indices, validation, scaler, history_start, device))
    return clients, target_split, target_graph, graph_metadata


def _method_metrics(model: Any, grid: Any, split: Any, scaler: Any, device: str, batch_size: int = 32) -> dict[str, Any]:
    validation = _evaluate(model, grid, _validation_indices(grid), scaler, device, batch_size=batch_size)
    audit = _evaluate(model, grid, split.audit_indices, scaler, device, split.available_start, batch_size)
    return {"audit": audit, "validation": validation}


def _run_scarce_local(grid: Any, split: Any, graph: Any, scaler: Any, device: str, max_epochs: int, batch_size: int = 32) -> tuple[dict[str, Any], dict[str, Any]]:
    proxy, proxy_training = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, history_start_index=split.available_start, batch_size=batch_size)
    model = full_model(grid, graph, device); inject_temporal_state(model, temporal_state_from_model(proxy)); state, full_training = train_full_model(model, grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, batch_size=batch_size)
    model.load_state_dict(state)
    metrics = _method_metrics(model, grid, split, scaler, device, batch_size)
    del proxy, model
    return metrics, {"proxy": proxy_training, "full": full_training}


def _run_btd(grid: Any, split: Any, graph: Any, scaler: Any, donor_state: dict[str, Any] | None, device: str, max_epochs: int, variant: str = "btd_fl", donor_full_state: dict[str, Any] | None = None, batch_size: int = 32) -> tuple[dict[str, Any], dict[str, Any]]:
    if donor_state is None:
        return _run_scarce_local(grid, split, graph, scaler, device, max_epochs, batch_size)
    target_proxy = make_proxy(grid)
    initial_state = target_proxy.state_dict()
    if variant == "btd_full_model_transfer" and donor_full_state is not None:
        for name, value in donor_full_state.items():
            if name in initial_state: initial_state[name] = value.clone()
    else:
        inject_temporal_state(target_proxy, donor_state)
        initial_state = target_proxy.state_dict()
    adapted, adaptation = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, initial_state=initial_state, device=device, max_epochs=max_epochs, history_start_index=split.available_start, batch_size=batch_size)
    model = full_model(grid, graph, device); inject_temporal_state(model, temporal_state_from_model(adapted)); state, full_training = train_full_model(model, grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, batch_size=batch_size); model.load_state_dict(state)
    metrics = _method_metrics(model, grid, split, scaler, device, batch_size)
    del target_proxy, adapted, model
    return metrics, {"adaptation": adaptation, "full": full_training}


def _run_federated(grids: dict[str, Any], target: str, method: str, device: str, rounds: int, local_epochs: int, batch_size: int = 32) -> tuple[dict[str, Any], dict[str, Any]]:
    clients, split, _graph, metadata = prepare_scenario_clients(grids, target, device)
    algorithm = {"fedavg": "standard", "fedprox": "fedprox", "fedper": "fedper"}[method]
    report = FederatedTrainer(clients, "fedavg_all", rounds=rounds, local_epochs=local_epochs, batch_size=batch_size, learning_rate=1e-3, seed=42, device=device, algorithm=algorithm).run()
    target_client = next(client for client in clients if client.grid_name == target)
    metrics = _method_metrics(target_client.model, grids[target], split, target_client.scaler, device, batch_size)
    return metrics, {"federated_report": report, "aggregation_weights": report["round_history"][-1]["aggregation_weights"], "graph_metadata": metadata}


def run_formal_benchmark(grids: dict[str, Any], selected_donors: dict[str, str | None], device: str = "cpu", rounds: int = 10, local_epochs: int = 5, max_epochs: int = 50, btd_variant: str = "btd_fl", batch_size: int = 32, selection_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if btd_variant not in {"btd_fl", "btd_no_benefit_selection", "btd_no_zero_transfer", "btd_full_model_transfer"}:
        raise ValueError("unknown BTD variant")
    donor_states: dict[str, dict[str, Any]] = {}
    donor_full_states: dict[str, dict[str, Any]] = {}
    donor_metadata: dict[str, Any] = {}
    for name in CLIENT_NAMES:
        grid = grids[name]; fit, calibration = donor_split(grid); scaler = fit_fit_only_scaler(grid, fit); model, training = train_proxy(grid, fit, calibration, scaler, device=device, max_epochs=max_epochs, batch_size=batch_size); donor_states[name] = temporal_state_from_model(model); donor_full_states[name] = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}; donor_metadata[name] = {"fit_target_count": len(fit), "calibration_target_count": len(calibration), **training}; del model
    scenarios: dict[str, Any] = {}
    for target in CLIENT_NAMES:
        grid = grids[target]; split = scarce_split(grid); scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start); graph = build_scarce_target_graph(grid, split); methods: dict[str, Any] = {}; selection = (selection_metadata or {}).get(target, {}); metadata: dict[str, Any] = {"selected_donor": selected_donors.get(target), "candidate_donors": selection.get("candidate_donors", [name for name in CLIENT_NAMES if name != target]), "calibration_benefit_by_donor": selection.get("calibration_benefit_by_donor", {}), "selected_calibration_benefit": selection.get("selected_calibration_benefit", 0.0), "zero_transfer_fallback": selection.get("zero_transfer_fallback", selected_donors.get(target) is None), "target_graph_metadata": graph.diagnostics, "validation_used_for_selection": False, "audit_used_for_selection": False, "test_evaluated": False}
        local_metrics, local_training = _run_scarce_local(grid, split, graph, scaler, device, max_epochs, batch_size); methods["scarce_local"] = local_metrics; metadata["scarce_local_training"] = local_training
        donor = selected_donors.get(target)
        if btd_variant == "btd_no_benefit_selection":
            donor = next((name for name in CLIENT_NAMES if name != target), None)
        if btd_variant == "btd_no_zero_transfer" and donor is None:
            donor = next((name for name in CLIENT_NAMES if name != target), None)
        donor_state = donor_states.get(donor) if donor else None; btd_metrics, btd_training = _run_btd(grid, split, graph, scaler, donor_state, device, max_epochs, btd_variant, donor_full_states.get(donor) if donor else None, batch_size); methods["btd_fl"] = btd_metrics; metadata["btd_training"] = btd_training; metadata["selected_donor_proxy_training"] = donor_metadata.get(donor); metadata["btd_variant"] = btd_variant
        for method in ("fedavg", "fedprox", "fedper"):
            methods[method], metadata[method] = _run_federated(grids, target, method, device, rounds, local_epochs, batch_size)
        scenarios[target] = {"methods": methods, "metadata": metadata}
    macros = {}
    for split_name in ("audit", "validation"):
        macros[split_name] = {method: {scope: {metric: float(np.mean([scenarios[name]["methods"][method][split_name][scope][metric] for name in CLIENT_NAMES])) for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for method in METHODS}
    wins = {method: sum(scenarios[name]["methods"]["btd_fl"]["validation"]["node_macro"]["mae"] < scenarios[name]["methods"][method]["validation"]["node_macro"]["mae"] for name in CLIENT_NAMES) for method in ("scarce_local", "fedavg", "fedprox", "fedper")}
    relative = {}
    for method in ("scarce_local", "fedavg", "fedprox", "fedper"):
        relative[method] = {
            split: float((macros[split][method]["node_macro"]["mae"] - macros[split]["btd_fl"]["node_macro"]["mae"]) / (macros[split][method]["node_macro"]["mae"] + 1e-12))
            for split in ("audit", "validation")
        }
    return {"experiment": "formal_btd_fl_25pct_benchmark", "method_name": "BTD-FL", "btd_variant": btd_variant, "methods": list(METHODS), "client_grid_names": list(CLIENT_NAMES), "rounds": rounds, "local_epochs": local_epochs, "max_epochs": max_epochs, "batch_size": batch_size, "learning_rate": 1e-3, "seed": 42, "validation_used_for_selection": False, "audit_used_for_selection": False, "test_evaluated": False, "donor_proxy_metadata_by_grid": donor_metadata, "scenarios": scenarios, "four_scenario_macro": macros, "relative_btd_fl_improvement_vs": relative, "btd_fl_win_counts": wins}


__all__ = ["CLIENT_NAMES", "METHODS", "load_selected_donors", "load_selection_metadata", "prepare_scenario_clients", "run_formal_benchmark"]
