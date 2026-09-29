"""Self-contained direct-temporal-transfer BTD-FL candidate runner."""
from __future__ import annotations

import json
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
    benefit,
    donor_split,
    fit_fit_only_scaler,
    make_proxy,
    scarce_split,
)
from code.federated.formal_btd import (
    CLIENT_NAMES,
    _cpu_state,
    _method_metrics,
    select_formal_donor,
)
from code.federated.formal_btd_ablations import load_frozen_reference, _frozen_selected_donors
from code.federated.parameter_groups import parameter_groups
from code.federated.seeding import reproducibility_metadata, set_global_seed
from code.models.puc_rstattn_v2_conditional_utility import build_conditional_utility_graph


def _train_proxy_state(
    grid: Any, fit: np.ndarray, calibration: np.ndarray, scaler: Any,
    device: str, max_epochs: int, batch_size: int,
    initial_state: Mapping[str, Any] | None = None, seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from code.audits.federated_transfer_benefit import train_proxy

    model, metadata = train_proxy(
        grid, fit, calibration, scaler, initial_state=initial_state,
        device=device, max_epochs=max_epochs,
        history_start_index=getattr(scaler, "fit_start_index", None),
        batch_size=batch_size, seed=seed,
    )
    state = _cpu_state(model)
    del model
    return state, dict(metadata)


def _train_full_history_donors(
    grids: Mapping[str, Any], device: str, max_epochs: int, batch_size: int, seed: int = 42,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states, metadata = {}, {}
    for donor in CLIENT_NAMES:
        grid = grids[donor]
        fit, calibration = donor_split(grid)
        scaler = fit_fit_only_scaler(grid, fit)
        state, training = _train_proxy_state(
            grid, fit, calibration, scaler, device, max_epochs, batch_size, seed=seed
        )
        proxy = make_proxy(grid, seed=seed)
        proxy.load_state_dict(state)
        states[donor] = temporal_state_from_model(proxy)
        metadata[donor] = {
            "source": "current_direct_transfer_run_full_history_temporal_proxy",
            "fit_target_count": int(len(fit)),
            "calibration_target_count": int(len(calibration)),
            "scaler_fit_start_index": int(scaler.fit_start_index),
            "scaler_fit_end_index": int(scaler.fit_end_index),
            "scaler_fit_timestamp_end": scaler.fit_timestamp_end,
            **training,
        }
        del proxy
    return states, metadata


def _equal_tensor_maps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(left[name].detach().cpu().numpy(), right[name].detach().cpu().numpy())
        for name in left
    )


def _snapshot_non_temporal(model: Any) -> dict[str, Any]:
    groups = parameter_groups(model)
    values = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if name in groups["spatial"]
    }
    values.update({
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
        if name in {"load_bus_mask", "utility_edge_index", "physical_relation_features", "utility_prior", "selected_utility"}
    })
    return values


def run_direct_transfer(
    grids: Mapping[str, Any], reference: Path | str | Mapping[str, Any],
    device: str = "cpu", max_epochs: int = 50, batch_size: int = 32, seed: int = 42,
    history_fraction: float = 0.25,
) -> dict[str, Any]:
    """Recompute benefit probes and train only revised BTD target models."""
    # Historical seed-42 synthetic callers relied on the pre-existing process
    # state; model/proxy constructors still receive the explicit seed. New
    # seeds are reset at the run boundary for reproducibility.
    if int(seed) != 42:
        set_global_seed(seed)
    frozen = load_frozen_reference(reference)
    old_selected = _frozen_selected_donors(frozen)
    donor_states, donor_metadata = _train_full_history_donors(
        grids, device, max_epochs, batch_size, seed
    )
    scenarios: dict[str, Any] = {}

    for target in CLIENT_NAMES:
        grid = grids[target]
        split = scarce_split(grid, history_fraction)
        scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
        graph = build_scarce_target_graph(grid, split)

        local_state, local_training = _train_proxy_state(
            grid, split.fit_indices, split.calibration_indices, scaler,
            device, max_epochs, batch_size, seed=seed,
        )
        local_calibration_mae = float(local_training["best_calibration_node_mae"])

        benefits: dict[str, float] = {}
        probe_metadata: dict[str, Any] = {}
        for donor in CLIENT_NAMES:
            if donor == target:
                continue
            probe = make_proxy(grid, seed=seed)
            inject_temporal_state(probe, donor_states[donor])
            initial = _cpu_state(probe)
            del probe
            adapted_state, adaptation = _train_proxy_state(
                grid, split.fit_indices, split.calibration_indices, scaler,
                device, max_epochs, batch_size, initial, seed=seed,
            )
            probe_mae = float(adaptation["best_calibration_node_mae"])
            benefits[donor] = benefit(local_calibration_mae, probe_mae)
            probe_metadata[donor] = {
                "donor_proxy_source": "current_direct_transfer_run_full_history_temporal_proxy",
                "adapted_probe_calibration_node_mae": probe_mae,
                "training": adaptation,
            }
            # Adapted states are deliberately discarded: they only estimate benefit.
            del adapted_state

        selected, selected_benefit, fallback, selection_rule = select_formal_donor(benefits)
        local_model = full_model(grid, graph, device, seed=seed)
        initial_snapshot = _snapshot_non_temporal(local_model)
        if fallback:
            model = local_model
            training = {"fallback_exact_local": True}
            # Recreate the same seed-specific full model, then inject local temporal state.
            local_proxy = make_proxy(grid, seed=seed)
            local_proxy.load_state_dict(local_state)
            from code.audits.btd_full_backbone_bridge import temporal_state_from_model
            inject_temporal_state(model, temporal_state_from_model(local_proxy))
            del local_proxy
            best_state, full_training = train_full_model(
                model, grid, split.fit_indices, split.calibration_indices, scaler,
                device=device, max_epochs=max_epochs, batch_size=batch_size, seed=seed,
            )
            model.load_state_dict(best_state)
            metrics = _method_metrics(model, grid, split, scaler, device, batch_size)
            training["full"] = full_training
        else:
            model = local_model
            selected_state = donor_states[selected]
            transferred_names = inject_temporal_state(model, selected_state)
            after_snapshot = _snapshot_non_temporal(model)
            if not _equal_tensor_maps(initial_snapshot, after_snapshot):
                raise AssertionError("direct temporal transfer changed target spatial parameters or topology buffers")
            best_state, full_training = train_full_model(
                model, grid, split.fit_indices, split.calibration_indices, scaler,
                device=device, max_epochs=max_epochs, batch_size=batch_size, seed=seed,
            )
            model.load_state_dict(best_state)
            metrics = _method_metrics(model, grid, split, scaler, device, batch_size)
            training = {"full": full_training, "transferred_parameter_names": list(transferred_names)}

        main_target = frozen["scenarios"][target]["methods"]
        scenarios[target] = {
            "metrics": metrics,
            "frozen_comparator_metrics": {
                name: main_target[name] for name in ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl")
            },
            "metadata": {
                "candidate_donors": [name for name in CLIENT_NAMES if name != target],
                "local_proxy_training": local_training,
                "local_proxy_calibration_node_mae": local_calibration_mae,
                "calibration_benefit_by_donor": benefits,
                "benefit_probe_training_by_donor": probe_metadata,
                "selected_donor": selected,
                "selected_calibration_benefit": float(selected_benefit),
                "zero_transfer_fallback": bool(fallback),
                "selection_rule": selection_rule,
                "selection_matches_previous_formal_run": selected == old_selected[target],
                "matches_historical_seed42_selection": selected == old_selected[target],
                "previous_formal_selected_donor": old_selected[target],
                "raw_donor_temporal_source": (
                    donor_metadata[selected]["source"] if selected is not None else None
                ),
                "donor_proxy_training": donor_metadata.get(selected),
                "transferred_parameter_names": training.get("transferred_parameter_names", []),
                "target_graph_metadata": graph.diagnostics,
                "target_scaler_metadata": {
                    "fit_start_index": int(scaler.fit_start_index),
                    "fit_end_index": int(scaler.fit_end_index),
                    "fit_timestamp_end": scaler.fit_timestamp_end,
                    "fit_split": scaler.fit_split,
                },
                "target_full_model_training": training["full"],
                "validation_used_for_selection": False,
                "audit_used_for_selection": False,
                "test_evaluated": False,
            },
        }
        del model

    metrics_by_target = {name: item["metrics"] for name, item in scenarios.items()}
    macro = {
        split_name: {
            scope: {
                metric: float(np.mean([
                    metrics_by_target[name][split_name][scope][metric]
                    for name in CLIENT_NAMES
                ]))
                for metric in ("mae", "rmse", "wape_pct", "smape_pct")
            }
            for scope in ("node_macro", "grid_aggregate")
        }
        for split_name in ("audit", "validation")
    }
    comparisons = {}
    win_counts = {}
    for comparator in ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl"):
        comparisons[comparator] = {}
        for split_name in ("audit", "validation"):
            base = frozen["four_scenario_macro"][split_name][comparator]["node_macro"]["mae"]
            value = macro[split_name]["node_macro"]["mae"]
            comparisons[comparator][split_name] = float((base - value) / (base + 1e-12))
        win_counts[comparator] = sum(
            scenarios[target]["metrics"]["validation"]["node_macro"]["mae"]
            < frozen["scenarios"][target]["methods"][comparator]["validation"]["node_macro"]["mae"]
            for target in CLIENT_NAMES
        )

    expected_regression = {"validation_node_mae": 0.000343768, "audit_node_mae": 0.000313559}
    return {
        "experiment": f"formal_btd_fl_direct_temporal_transfer_{int(history_fraction * 100)}pct",
        "method_name": "BTD-FL Direct Temporal Transfer",
        "btd_variant": "btd_fl_direct_transfer",
        "client_grid_names": list(CLIENT_NAMES),
        "seed": int(seed),
        "rounds": 0,
        "local_epochs": 0,
        "history_fraction": history_fraction,
        "train_calibration_audit_split": "60/20/20 for scarce target; donor train-internal fit/calibration",
        "max_epochs": max_epochs,
        "patience": 8,
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "validation_used_for_selection": False,
        "audit_used_for_selection": False,
        "test_evaluated": False,
        "canonical_validation_evaluated_during_training": False,
        "selection_semantics": "current-run target-calibration benefit probes; raw selected donor temporal state is transferred",
        "adapted_probe_states_used_for_final_initialization": False,
        "donor_proxy_metadata_by_grid": donor_metadata,
        "scenarios": scenarios,
        "four_target_unweighted_macro": macro,
        "relative_improvement_vs_frozen_comparators": comparisons,
        "win_counts_vs_frozen_comparators": win_counts,
        "previous_no_target_adaptation_regression_expectation": expected_regression,
        "difference_from_previous_no_target_adaptation_expectation": {
            "validation_node_mae": float(macro["validation"]["node_macro"]["mae"] - expected_regression["validation_node_mae"]),
            "audit_node_mae": float(macro["audit"]["node_macro"]["mae"] - expected_regression["audit_node_mae"]),
        },
        "selection_matches_previous_formal_run_by_target": {
            target: scenarios[target]["metadata"]["selection_matches_previous_formal_run"]
            for target in CLIENT_NAMES
        },
        "reproducibility": reproducibility_metadata(seed, device),
    }


__all__ = ["run_direct_transfer"]
