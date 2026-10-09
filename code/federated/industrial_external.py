"""Frozen five-client external-domain benchmark orchestration.

The four reference grids remain full-history source clients.  The industrial
MV/LV grid is the only scarce target.  This module contains no new learning
rule; it wires the accepted BTD, federated baselines, and FedFomo-style
primitives to that external client contract.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from code.audits.btd_full_backbone_bridge import (
    build_scarce_target_graph,
    full_model,
    inject_temporal_state,
    temporal_state_from_model,
    train_full_model,
)
from code.audits.federated_transfer_benefit import (
    _evaluate,
    benefit,
    donor_split,
    fit_fit_only_scaler,
    make_proxy,
    scarce_split,
    train_proxy,
)
from code.data.industrial_mvlv_loader import load_industrial_mvlv
from code.data.lv_grid_loader import LVGridLoader
from code.federated.parameter_groups import parameter_groups
from code.federated.checkpointing import export_model_checkpoint, matrix_cells
from code.federated.trainer import FederatedTrainer
from code.models.puc_rstattn_v2_conditional_utility import build_conditional_utility_graph

REFERENCE_CLIENT_NAMES = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
INDUSTRIAL_TARGET = "norway_industrial_mvlv"
EXTERNAL_CLIENT_NAMES = REFERENCE_CLIENT_NAMES + (INDUSTRIAL_TARGET,)
BASELINE_METHODS = ("industrial_scarce_local", "fedavg", "fedprox", "fedper")
ALL_METHODS = BASELINE_METHODS + ("fedfomo_style", "btd_fl_direct_transfer")
METRICS = ("mae", "rmse", "wape_pct", "smape_pct")


def select_external_donor(benefits: Mapping[str, float]) -> tuple[str | None, float, bool]:
    """Select the current-run positive argmax or exact local fallback."""
    if set(benefits) != set(REFERENCE_CLIENT_NAMES):
        raise ValueError("external BTD requires exactly four reference donors")
    selected, value = max(((name, float(score)) for name, score in benefits.items()), key=lambda item: (item[1], item[0]))
    if not np.isfinite(value) or any(not np.isfinite(float(score)) for score in benefits.values()):
        raise ValueError("external donor benefits must be finite")
    return (None, 0.0, True) if value <= 0 else (selected, value, False)


def sample_count_weights(counts: Mapping[str, int]) -> dict[str, float]:
    if not counts or any(int(value) <= 0 for value in counts.values()):
        raise ValueError("sample counts must be positive")
    total = float(sum(int(value) for value in counts.values()))
    return {name: int(value) / total for name, value in counts.items()}


def load_external_grids(data_root: str | Path, mapping_json: str | Path) -> dict[str, Any]:
    loader = LVGridLoader(data_root, mapping_json)
    grids = {name: loader.load(name) for name in REFERENCE_CLIENT_NAMES}
    data_root = Path(data_root)
    industrial_root = data_root / "norway_industrial_mvlv" if (data_root / "norway_industrial_mvlv").is_dir() else data_root.parent / "norway_industrial_mvlv"
    grids[INDUSTRIAL_TARGET] = load_industrial_mvlv(industrial_root)
    return grids


def _indices(grid: Any) -> np.ndarray:
    train = grid.splits["train"]
    return np.arange(train.start_index + 168, train.end_index, dtype=np.int64)


def _validation_indices(grid: Any) -> np.ndarray:
    split = grid.splits["validation"]
    return np.arange(split.start_index, split.end_index, dtype=np.int64)


def _formal_helpers() -> tuple[Any, Any]:
    from code.federated.formal_btd import _cpu_state, _make_client
    return _cpu_state, _make_client


def _seed_helpers() -> tuple[Any, Any]:
    from code.federated.seeding import reproducibility_metadata, set_global_seed
    return reproducibility_metadata, set_global_seed


def _train_proxy_state(
    grid: Any,
    fit: np.ndarray,
    calibration: np.ndarray,
    scaler: Any,
    device: str,
    max_epochs: int,
    batch_size: int,
    seed: int,
    initial_state: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, training = train_proxy(
        grid,
        fit,
        calibration,
        scaler,
        initial_state=initial_state,
        device=device,
        max_epochs=max_epochs,
        history_start_index=getattr(scaler, "fit_start_index", None),
        batch_size=batch_size,
        seed=seed,
    )
    cpu_state, _ = _formal_helpers()
    return cpu_state(model), dict(training)


def _client_bundle(
    grids: Mapping[str, Any],
    history_fraction: float,
    device: str,
    seed: int,
) -> tuple[list[Any], Any, Any, dict[str, Any]]:
    target_grid = grids[INDUSTRIAL_TARGET]
    target_split = scarce_split(target_grid, history_fraction)
    target_scaler = fit_fit_only_scaler(target_grid, target_split.fit_indices, target_split.available_start)
    target_graph = build_scarce_target_graph(target_grid, target_split)
    clients: list[Any] = []
    metadata: dict[str, Any] = {}
    _, make_client = _formal_helpers()
    for name in EXTERNAL_CLIENT_NAMES:
        grid = grids[name]
        if name == INDUSTRIAL_TARGET:
            train_indices, calibration, scaler, graph, history_start = (
                target_split.fit_indices,
                target_split.calibration_indices,
                target_scaler,
                target_graph,
                target_split.available_start,
            )
        else:
            # Standard FL sources follow the accepted formal baseline: all
            # complete canonical-TRAIN targets are optimized locally. The
            # donor fit/calibration split remains reserved for BTD proxies
            # and FedFomo's calibration objective.
            train_indices = _indices(grid)
            _, calibration = donor_split(grid)
            scaler = fit_fit_only_scaler(grid, train_indices)
            graph = build_conditional_utility_graph(grid)
            history_start = None
        client = make_client(
            grid,
            graph,
            train_indices,
            _validation_indices(grid),
            scaler,
            history_start,
            device,
            seed,
        )
        client.grid = grid
        client.graph = graph
        client.calibration_indices = calibration
        client.history_start = history_start
        clients.append(client)
        metadata[name] = dict(graph.diagnostics)
        metadata[name]["training_index_contract"] = (
            "industrial_scarce_fit_only" if name == INDUSTRIAL_TARGET else "full_canonical_train_all_complete_targets"
        )
        metadata[name]["train_target_count"] = int(len(train_indices))
        metadata[name]["scaler_fit_target_count"] = int(len(train_indices))
    return clients, target_split, target_graph, metadata


def _target_metrics(model: Any, grid: Any, split: Any, scaler: Any, device: str, batch_size: int) -> dict[str, Any]:
    return {
        "audit": _evaluate(model, grid, split.audit_indices, scaler, device, split.available_start, batch_size),
        "validation": _evaluate(model, grid, _validation_indices(grid), scaler, device, None, batch_size),
    }


def _local_target(
    grid: Any,
    split: Any,
    graph: Any,
    device: str,
    max_epochs: int,
    batch_size: int,
    seed: int,
    checkpoint_export: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
    local_proxy_state, proxy_training = _train_proxy_state(
        grid, split.fit_indices, split.calibration_indices, scaler, device, max_epochs, batch_size, seed
    )
    model = full_model(grid, graph, device, seed=seed)
    inject_temporal_state(model, local_proxy_state)
    best_state, full_training = train_full_model(
        model,
        grid,
        split.fit_indices,
        split.calibration_indices,
        scaler,
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    model.load_state_dict(best_state)
    if checkpoint_export is not None:
        export_model_checkpoint(
            root=Path(checkpoint_export["root"]), cell=checkpoint_export["cell"], model=model,
            grid=grid, scaler=scaler, graph_metadata=graph.diagnostics,
            source_artifact=checkpoint_export["source_artifact"],
            frozen_commits=checkpoint_export.get("frozen_commits", {}),
            selected_epoch=full_training.get("selected_epoch"), epochs_run=full_training.get("epochs_run"),
        )
    return _target_metrics(model, grid, split, scaler, device, batch_size), {
        "proxy_training": proxy_training,
        "full_model_training": full_training,
        "scaler_fit_start_index": int(scaler.fit_start_index),
        "scaler_fit_end_index": int(scaler.fit_end_index),
        "scaler_fit_timestamp_end": scaler.fit_timestamp_end,
        "transferred_parameter_names": list(parameter_groups(model)["temporal"]),
    }


def _run_federated(
    grids: Mapping[str, Any],
    method: str,
    history_fraction: float,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    checkpoint_root: Path | None = None,
    source_artifact: str = "industrial_external_runner",
    frozen_commits: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clients, split, _graph, graph_metadata = _client_bundle(grids, history_fraction, device, seed)
    algorithm = {"fedavg": "standard", "fedprox": "fedprox", "fedper": "fedper"}[method]
    report = FederatedTrainer(
        clients,
        "fedavg_all",
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        seed=seed,
        device=device,
        algorithm=algorithm,
        evaluate_validation_during_training=False,
        expected_client_count=5,
    ).run()
    target_client = next(client for client in clients if client.grid_name == INDUSTRIAL_TARGET)
    metrics = _target_metrics(target_client.model, grids[INDUSTRIAL_TARGET], split, target_client.scaler, device, batch_size)
    if checkpoint_root is not None:
        cell = next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == history_fraction and item["method"] == method)
        export_model_checkpoint(
            root=checkpoint_root, cell=cell, model=target_client.model,
            grid=grids[INDUSTRIAL_TARGET], scaler=target_client.scaler,
            graph_metadata=graph_metadata[INDUSTRIAL_TARGET], source_artifact=source_artifact,
            frozen_commits=frozen_commits or {}, final_round=rounds,
        )
    counts = report["local_sample_counts"]
    weights = report["round_history"][-1]["aggregation_weights"]
    expected = {name: count / float(sum(counts.values())) for name, count in counts.items()}
    if any(not np.isclose(weights[name], expected[name]) for name in counts):
        raise AssertionError("external aggregation is not sample-count weighted")
    return metrics, {
        "federated_report": report,
        "local_sample_counts": counts,
        "aggregation_weights": weights,
        "graph_metadata": graph_metadata,
        "training_index_contract_by_client": {
            name: ("industrial_scarce_fit_only" if name == INDUSTRIAL_TARGET else "full_canonical_train_all_complete_targets")
            for name in EXTERNAL_CLIENT_NAMES
        },
        "scaler_fit_target_count_by_client": {
            client.grid_name: int(len(client.train_dataset)) for client in clients
        },
        "fedprox_mu": 0.01 if method == "fedprox" else None,
        "primary_target": INDUSTRIAL_TARGET,
    }


def _macro(method_metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        split: {
            scope: {
                metric: float(method_metrics[split][scope][metric])
                for metric in METRICS
            }
            for scope in ("node_macro", "grid_aggregate")
        }
        for split in ("audit", "validation")
    }


def run_external_baselines(
    grids: Mapping[str, Any],
    history_fraction: float = 0.25,
    rounds: int = 10,
    local_epochs: int = 5,
    max_epochs: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_root: str | Path | None = None,
    source_artifact: str = "industrial_external_baselines",
    frozen_commits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _, set_global_seed = _seed_helpers()
    set_global_seed(seed)
    grid = grids[INDUSTRIAL_TARGET]
    split = scarce_split(grid, history_fraction)
    graph = build_scarce_target_graph(grid, split)
    scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
    methods: dict[str, Any] = {}
    local_cell = next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == history_fraction and item["method"] == "local")
    local_metrics, local_training = _local_target(
        grid, split, graph, device, max_epochs, batch_size, seed,
        {"root": checkpoint_root, "cell": local_cell, "source_artifact": source_artifact, "frozen_commits": frozen_commits or {}} if checkpoint_root is not None else None,
    )
    methods["industrial_scarce_local"] = local_metrics
    federated: dict[str, Any] = {}
    for method in ("fedavg", "fedprox", "fedper"):
        methods[method], federated[method] = _run_federated(
            grids, method, history_fraction, rounds, local_epochs, batch_size, device, seed,
            Path(checkpoint_root) if checkpoint_root is not None else None, source_artifact, frozen_commits,
        )
    scenario = {
        "methods": methods,
        "metadata": {
            "target": INDUSTRIAL_TARGET,
            "history_fraction": history_fraction,
            "available_raw_hours": int(split.available_raw_hours),
            "eligible_target_count": int(len(split.eligible_indices)),
            "fit_target_count": int(len(split.fit_indices)),
            "calibration_target_count": int(len(split.calibration_indices)),
            "audit_target_count": int(len(split.audit_indices)),
            "target_graph_metadata": graph.diagnostics,
            "target_local_training": local_training,
            "federated": federated,
            "validation_used_for_selection": False,
            "audit_used_for_selection": False,
            "test_evaluated": False,
            "industrial_test_locked": True,
            "industrial_test_evaluated": False,
            "reference_grid_tests_evaluated": False,
        },
    }
    return _base_report(
        "industrial_baselines",
        BASELINE_METHODS,
        history_fraction,
        rounds,
        local_epochs,
        max_epochs,
        batch_size,
        seed,
        {INDUSTRIAL_TARGET: scenario},
        {method: _macro(methods[method]) for method in BASELINE_METHODS},
        device,
    )


def run_external_btd(
    grids: Mapping[str, Any],
    history_fraction: float = 0.25,
    max_epochs: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_root: str | Path | None = None,
    source_artifact: str = "industrial_external_btd",
    frozen_commits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _, set_global_seed = _seed_helpers()
    set_global_seed(seed)
    target_grid = grids[INDUSTRIAL_TARGET]
    target_split = scarce_split(target_grid, history_fraction)
    target_scaler = fit_fit_only_scaler(target_grid, target_split.fit_indices, target_split.available_start)
    target_graph = build_scarce_target_graph(target_grid, target_split)

    donor_states: dict[str, dict[str, Any]] = {}
    donor_metadata: dict[str, Any] = {}
    for donor in REFERENCE_CLIENT_NAMES:
        donor_grid = grids[donor]
        fit, calibration = donor_split(donor_grid)
        scaler = fit_fit_only_scaler(donor_grid, fit)
        state, training = _train_proxy_state(donor_grid, fit, calibration, scaler, device, max_epochs, batch_size, seed)
        proxy = make_proxy(donor_grid, seed=seed)
        proxy.load_state_dict(state)
        donor_states[donor] = temporal_state_from_model(proxy)
        donor_metadata[donor] = {"fit_target_count": len(fit), "calibration_target_count": len(calibration), **training}

    local_state, local_training = _train_proxy_state(
        target_grid,
        target_split.fit_indices,
        target_split.calibration_indices,
        target_scaler,
        device,
        max_epochs,
        batch_size,
        seed,
    )
    local_proxy = make_proxy(target_grid, seed=seed)
    local_proxy.load_state_dict(local_state)
    local_calibration_mae = float(local_training["best_calibration_node_mae"])
    benefits: dict[str, float] = {}
    probes: dict[str, Any] = {}
    for donor in REFERENCE_CLIENT_NAMES:
        initial = make_proxy(target_grid, seed=seed)
        inject_temporal_state(initial, donor_states[donor])
        adapted_state, adaptation = _train_proxy_state(
            target_grid,
            target_split.fit_indices,
            target_split.calibration_indices,
            target_scaler,
            device,
            max_epochs,
            batch_size,
            seed,
            initial_state=_formal_helpers()[0](initial),
        )
        adapted_mae = float(adaptation["best_calibration_node_mae"])
        benefits[donor] = benefit(local_calibration_mae, adapted_mae)
        probes[donor] = {"adapted_calibration_node_mae": adapted_mae, "training": adaptation}
    selected_donor, maximum_benefit, fallback = select_external_donor(benefits)

    local_model = full_model(target_grid, target_graph, device, seed=seed)
    inject_temporal_state(local_model, local_state)
    local_best, local_full_training = train_full_model(
        local_model,
        target_grid,
        target_split.fit_indices,
        target_split.calibration_indices,
        target_scaler,
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    local_model.load_state_dict(local_best)
    local_metrics = _target_metrics(local_model, target_grid, target_split, target_scaler, device, batch_size)
    if fallback:
        metrics = copy.deepcopy(local_metrics)
        full_training = {"fallback_exact_local": True, **local_full_training}
        transferred_names: list[str] = []
        final_model = local_model
    else:
        model = full_model(target_grid, target_graph, device, seed=seed)
        transferred_names = list(inject_temporal_state(model, donor_states[selected_donor]))
        best_state, full_training = train_full_model(
            model,
            target_grid,
            target_split.fit_indices,
            target_split.calibration_indices,
            target_scaler,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        model.load_state_dict(best_state)
        metrics = _target_metrics(model, target_grid, target_split, target_scaler, device, batch_size)
        final_model = model
    if checkpoint_root is not None:
        cell = next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == history_fraction and item["method"] == "btd_fl_direct_transfer")
        export_model_checkpoint(
            root=Path(checkpoint_root), cell=cell, model=final_model, grid=target_grid,
            scaler=target_scaler, graph_metadata=target_graph.diagnostics,
            source_artifact=source_artifact, frozen_commits=frozen_commits or {},
            selected_epoch=full_training.get("selected_epoch"), epochs_run=full_training.get("epochs_run"),
            selection={
                "selected_donor": selected_donor,
                "selected_calibration_benefit": float(maximum_benefit),
                "zero_transfer_fallback": bool(fallback),
                "raw_selected_donor_temporal_state_used": not fallback,
                "adapted_probe_states_used_for_final_initialization": False,
                "transferred_parameter_names": transferred_names,
            },
        )
    scenario = {
        "metrics": metrics,
        "metadata": {
            "target": INDUSTRIAL_TARGET,
            "history_fraction": history_fraction,
            "candidate_donors": list(REFERENCE_CLIENT_NAMES),
            "calibration_benefit_by_donor": benefits,
            "selected_donor": selected_donor,
            "selected_calibration_benefit": float(maximum_benefit),
            "positive_benefit_count": int(sum(value > 0 for value in benefits.values())),
            "non_positive_benefit_count": int(sum(value <= 0 for value in benefits.values())),
            "zero_transfer_fallback": fallback,
            "donor_proxy_metadata_by_grid": donor_metadata,
            "benefit_probe_metadata_by_donor": probes,
            "adapted_probe_states_used_for_final_initialization": False,
            "raw_selected_donor_temporal_state_used": not fallback,
            "transferred_parameter_names": transferred_names,
            "target_graph_metadata": target_graph.diagnostics,
            "target_scaler_fit_start_index": int(target_scaler.fit_start_index),
            "target_scaler_fit_end_index": int(target_scaler.fit_end_index),
            "target_scaler_fit_timestamp_end": target_scaler.fit_timestamp_end,
            "available_raw_hours": int(target_split.available_raw_hours),
            "eligible_target_count": int(len(target_split.eligible_indices)),
            "fit_target_count": int(len(target_split.fit_indices)),
            "calibration_target_count": int(len(target_split.calibration_indices)),
            "audit_target_count": int(len(target_split.audit_indices)),
            "target_full_model_training": full_training,
            "industrial_test_locked": True,
            "industrial_test_evaluated": False,
            "reference_grid_tests_evaluated": False,
            "validation_used_for_selection": False,
            "audit_used_for_selection": False,
            "test_evaluated": False,
        },
    }
    return _base_report(
        "industrial_btd_direct_transfer",
        ("btd_fl_direct_transfer",),
        history_fraction,
        0,
        0,
        max_epochs,
        batch_size,
        seed,
        {INDUSTRIAL_TARGET: scenario},
        {"btd_fl_direct_transfer": _macro(metrics)},
        device,
        btd_variant="btd_fl_direct_transfer",
    )


def run_external_fedfomo(
    grids: Mapping[str, Any],
    history_fraction: float = 0.25,
    rounds: int = 10,
    local_epochs: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 42,
    max_epochs_metadata: int = 50,
    checkpoint_root: str | Path | None = None,
    checkpoint_domain: str = "industrial_external",
    source_artifact: str = "industrial_external_fedfomo",
    frozen_commits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    from code.federated.fedfomo_style import run_fedfomo_scenario

    _, set_global_seed = _seed_helpers()
    set_global_seed(seed)
    scenario = run_fedfomo_scenario(
        grids,
        INDUSTRIAL_TARGET,
        rounds,
        local_epochs,
        batch_size,
        device,
        1e-12,
        seed,
        history_fraction,
        EXTERNAL_CLIENT_NAMES,
        checkpoint_root=Path(checkpoint_root) if checkpoint_root is not None else None,
        checkpoint_domain=checkpoint_domain,
        source_artifact=source_artifact,
        frozen_commits=frozen_commits,
    )
    scenario["metadata"].update(
        {
            "industrial_test_locked": True,
            "industrial_test_evaluated": False,
            "reference_grid_tests_evaluated": False,
            "target_graph_is_local": True,
        }
    )
    metrics = scenario["scarce_target_metrics"]
    return _base_report(
        "industrial_fedfomo_style",
        ("fedfomo_style",),
        history_fraction,
        rounds,
        local_epochs,
        max_epochs_metadata,
        batch_size,
        seed,
        {INDUSTRIAL_TARGET: scenario},
        {"fedfomo_style": _macro(metrics)},
        device,
    )


def _base_report(
    experiment: str,
    methods: tuple[str, ...],
    history_fraction: float,
    rounds: int,
    local_epochs: int,
    max_epochs: int,
    batch_size: int,
    seed: int,
    scenarios: dict[str, Any],
    macro: dict[str, Any],
    device: str,
    btd_variant: str | None = None,
) -> dict[str, Any]:
    report = {
        "experiment": experiment,
        "artifact_type": experiment,
        "method_name": experiment,
        "methods": list(methods),
        "client_grid_names": list(EXTERNAL_CLIENT_NAMES),
        "primary_targets": [INDUSTRIAL_TARGET],
        "rounds": rounds,
        "local_epochs": local_epochs,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "patience": 8,
        "seed": int(seed),
        "history_fraction": float(history_fraction),
        "train_calibration_audit_split": "60/20/20 industrial scarce target; full-history donor train-internal fit/calibration",
        "validation_used_for_selection": False,
        "audit_used_for_selection": False,
        "test_evaluated": False,
        "industrial_test_locked": True,
        "industrial_test_evaluated": False,
        "reference_grid_tests_evaluated": False,
        "communication_round_selection": "fixed_final_round",
        "canonical_validation_evaluated_during_training": False,
        "scenarios": scenarios,
        "industrial_target_macro": macro,
        "reproducibility": _seed_helpers()[0](seed, device),
    }
    if btd_variant is not None:
        report["btd_variant"] = btd_variant
    return report


__all__ = [
    "INDUSTRIAL_TARGET",
    "REFERENCE_CLIENT_NAMES",
    "EXTERNAL_CLIENT_NAMES",
    "BASELINE_METHODS",
    "ALL_METHODS",
    "load_external_grids",
    "run_external_baselines",
    "run_external_btd",
    "run_external_fedfomo",
    "select_external_donor",
    "sample_count_weights",
]
