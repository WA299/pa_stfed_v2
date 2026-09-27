"""Efficient mechanism-only ablations for the frozen formal BTD-FL result."""
from __future__ import annotations

import copy
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
    donor_split,
    fit_fit_only_scaler,
    make_proxy,
    scarce_split,
    train_proxy,
)
from code.federated.formal_btd import (
    CLIENT_NAMES,
    _cpu_state,
    _method_metrics,
    _same_snapshot,
    _spatial_buffer_snapshot,
    _train_proxy_once,
    _copy_compatible_trainables,
)

ABLATIONS = (
    "btd_no_benefit_selection",
    "btd_full_model_transfer",
    "btd_no_target_adaptation",
)


def load_frozen_reference(path: Path | str) -> dict[str, Any]:
    """Load and validate the accepted main formal result as read-only input."""
    report = json.loads(Path(path).read_text(encoding="utf-8")) if not isinstance(path, Mapping) else dict(path)
    required = {
        "btd_variant": "btd_fl",
        "seed": 42,
        "history_fraction": 0.25,
        "validation_used_for_selection": False,
        "audit_used_for_selection": False,
        "test_evaluated": False,
        "canonical_validation_evaluated_during_training": False,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"frozen formal reference has invalid {key}: {report.get(key)!r}")
    if tuple(report.get("client_grid_names", ())) != CLIENT_NAMES:
        raise ValueError("frozen formal reference has unexpected client grid names")
    return report


def _frozen_selected_donors(reference: Mapping[str, Any]) -> dict[str, str]:
    selected = {}
    for target in CLIENT_NAMES:
        donor = reference["scenarios"][target]["metadata"].get("selected_donor")
        if donor is None or donor == target or donor not in CLIENT_NAMES:
            raise ValueError(f"frozen reference has invalid selected donor for {target}: {donor!r}")
        selected[target] = donor
    return selected


def _train_donor_temporal_states(
    grids: Mapping[str, Any], device: str, max_epochs: int, batch_size: int,
    donor_names: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states, metadata = {}, {}
    for name in donor_names or list(CLIENT_NAMES):
        grid = grids[name]
        fit, calibration = donor_split(grid)
        scaler = fit_fit_only_scaler(grid, fit)
        state, training = _train_proxy_once(
            grid, fit, calibration, scaler, device, max_epochs, batch_size
        )
        proxy = make_proxy(grid)
        proxy.load_state_dict(state)
        states[name] = temporal_state_from_model(proxy)
        metadata[name] = {
            "source": "current_ablation_run_full_history_temporal_proxy",
            "fit_target_count": int(len(fit)),
            "calibration_target_count": int(len(calibration)),
            "scaler_fit_start_index": int(scaler.fit_start_index),
            "scaler_fit_end_index": int(scaler.fit_end_index),
            "scaler_fit_timestamp_end": scaler.fit_timestamp_end,
            **training,
        }
        del proxy
    return states, metadata


def _train_selected_full_donor_states(
    grids: Mapping[str, Any], selected: Mapping[str, str], device: str,
    max_epochs: int, batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states, metadata = {}, {}
    for donor in sorted(set(selected.values())):
        grid = grids[donor]
        fit, calibration = donor_split(grid)
        scaler = fit_fit_only_scaler(grid, fit)
        from code.models.puc_rstattn_v2_conditional_utility import build_conditional_utility_graph
        graph = build_conditional_utility_graph(grid)
        model = full_model(grid, graph, device)
        state, training = train_full_model(
            model, grid, fit, calibration, scaler, device=device,
            max_epochs=max_epochs, batch_size=batch_size,
        )
        params = dict(model.named_parameters())
        states[donor] = {
            name: value.detach().cpu().clone()
            for name, value in state.items() if name in params
        }
        metadata[donor] = {
            "source": "current_ablation_run_full_history_full_model",
            "fit_target_count": int(len(fit)),
            "calibration_target_count": int(len(calibration)),
            "transferred_state_type": "full_puc_rstattn_v2_trainable_parameters",
            **training,
        }
        del model
    return states, metadata


def _train_target_full(
    grid: Any, split: Any, graph: Any, scaler: Any, temporal_state: Mapping[str, Any],
    device: str, max_epochs: int, batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    model = full_model(grid, graph, device)
    inject_temporal_state(model, temporal_state)
    state, training = train_full_model(
        model, grid, split.fit_indices, split.calibration_indices, scaler,
        device=device, max_epochs=max_epochs, batch_size=batch_size,
    )
    model.load_state_dict(state)
    return _method_metrics(model, grid, split, scaler, device, batch_size), training, list(
        dict.fromkeys(temporal_state.keys())
    )


def _macro(metrics_by_target: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        split: {
            scope: {
                metric: float(np.mean([
                    metrics_by_target[target][split][scope][metric]
                    for target in CLIENT_NAMES
                ]))
                for metric in ("mae", "rmse", "wape_pct", "smape_pct")
            }
            for scope in ("node_macro", "grid_aggregate")
        }
        for split in ("audit", "validation")
    }


def _relative(ablation: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    return {
        split: float(
            (reference[split]["node_macro"]["mae"] - ablation[split]["node_macro"]["mae"])
            / (reference[split]["node_macro"]["mae"] + 1e-12)
        )
        for split in ("audit", "validation")
    }


def run_ablation(
    grids: Mapping[str, Any], reference: Path | str | Mapping[str, Any],
    variant: str, device: str = "cpu", max_epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run one mechanism ablation without rerunning invariant baselines."""
    if variant not in ABLATIONS:
        raise ValueError(f"unknown ablation: {variant}")
    frozen = load_frozen_reference(reference)
    selected = _frozen_selected_donors(frozen)
    if variant == "btd_full_model_transfer":
        needed_temporal_donors = []
    elif variant == "btd_no_benefit_selection":
        needed_temporal_donors = sorted(
            {sorted(name for name in CLIENT_NAMES if name != target)[0] for target in CLIENT_NAMES}
        )
    else:
        needed_temporal_donors = sorted(set(selected.values()))
    if needed_temporal_donors:
        temporal_states, donor_proxy_metadata = _train_donor_temporal_states(
            grids, device, max_epochs, batch_size, needed_temporal_donors
        )
    else:
        temporal_states, donor_proxy_metadata = {}, {}
    full_donor_states, full_donor_metadata = ({}, {})
    if variant == "btd_full_model_transfer":
        full_donor_states, full_donor_metadata = _train_selected_full_donor_states(
            grids, selected, device, max_epochs, batch_size
        )

    per_target: dict[str, Any] = {}
    for target in CLIENT_NAMES:
        grid = grids[target]
        split = scarce_split(grid)
        scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
        graph = build_scarce_target_graph(grid, split)
        if variant == "btd_no_benefit_selection":
            donor = sorted(name for name in CLIENT_NAMES if name != target)[0]
            selection_rule = "fixed_deterministic_no_benefit"
        else:
            donor = selected[target]
            selection_rule = "frozen_main_btd_fl_selected_donor"

        target_proxy_adaptation = None
        if variant == "btd_no_benefit_selection":
            initial_proxy = make_proxy(grid)
            inject_temporal_state(initial_proxy, temporal_states[donor])
            initial_state = _cpu_state(initial_proxy)
            del initial_proxy
            adapted_state, adaptation_training = _train_proxy_once(
                grid, split.fit_indices, split.calibration_indices, scaler,
                device, max_epochs, batch_size, initial_state,
            )
            adapted_proxy = make_proxy(grid)
            adapted_proxy.load_state_dict(adapted_state)
            adapted_temporal = temporal_state_from_model(adapted_proxy)
            target_proxy_adaptation = {
                "performed": True,
                "donor": donor,
                **adaptation_training,
            }
            del adapted_proxy
        elif variant == "btd_no_target_adaptation":
            adapted_temporal = temporal_states[donor]
            adaptation_training = None
            target_proxy_adaptation = {"performed": False, "donor": donor, "steps": 0}
        else:
            adapted_temporal = None
            adaptation_training = None

        if variant == "btd_full_model_transfer":
            model = full_model(grid, graph, device)
            transferred = _copy_compatible_trainables(model, full_donor_states[donor])
            state, full_training = train_full_model(
                model, grid, split.fit_indices, split.calibration_indices, scaler,
                device=device, max_epochs=max_epochs, batch_size=batch_size,
            )
            model.load_state_dict(state)
            metrics = _method_metrics(model, grid, split, scaler, device, batch_size)
            transferred_names = list(transferred)
            del model
        else:
            temporal = adapted_temporal if adapted_temporal is not None else temporal_states[donor]
            metrics, full_training, transferred_names = _train_target_full(
                grid, split, graph, scaler, temporal, device, max_epochs, batch_size
            )

        main_metrics = frozen["scenarios"][target]["methods"]["btd_fl"]
        local_metrics = frozen["scenarios"][target]["methods"]["scarce_local"]
        per_target[target] = {
            "donor_used": donor,
            "selection_rule": selection_rule,
            "metrics": metrics,
            "main_btd_fl_metrics": main_metrics,
            "scarce_local_metrics": local_metrics,
            "main_btd_calibration_benefit_by_donor": frozen["scenarios"][target]["metadata"].get("calibration_benefit_by_donor", {}),
            "relative_change_vs_main_btd_fl": _relative(metrics, main_metrics),
            "relative_change_vs_scarce_local": _relative(metrics, local_metrics),
            "target_graph_metadata": graph.diagnostics,
            "target_scarcity_metadata": {
                "canonical_train_raw_hours": int(split.raw_train_hours),
                "available_raw_hours": int(split.available_raw_hours),
                "available_raw_start_index": int(split.available_start),
                "available_raw_end_index": int(split.available_end),
                "fit_target_count": int(len(split.fit_indices)),
                "calibration_target_count": int(len(split.calibration_indices)),
                "audit_target_count": int(len(split.audit_indices)),
                "history_fraction": 0.25,
            },
            "target_scaler_metadata": {
                "fit_start_index": int(scaler.fit_start_index),
                "fit_end_index": int(scaler.fit_end_index),
                "fit_timestamp_end": scaler.fit_timestamp_end,
                "fit_split": scaler.fit_split,
            },
            "selected_donor_proxy_training": donor_proxy_metadata.get(donor),
            "selected_donor_full_model_training": full_donor_metadata.get(donor),
            "target_proxy_adaptation": target_proxy_adaptation,
            "target_full_model_training": full_training,
            "transferred_parameter_names": transferred_names,
            "test_evaluated": False,
        }

    metrics_by_target = {target: item["metrics"] for target, item in per_target.items()}
    macro = _macro(metrics_by_target)
    macro_reference = {
        "main_btd_fl": {
            "audit": frozen["four_scenario_macro"]["audit"]["btd_fl"],
            "validation": frozen["four_scenario_macro"]["validation"]["btd_fl"],
        },
        "scarce_local": {
            "audit": frozen["four_scenario_macro"]["audit"]["scarce_local"],
            "validation": frozen["four_scenario_macro"]["validation"]["scarce_local"],
        },
    }
    wins = {
        "main_btd_fl": sum(
            per_target[target]["metrics"]["validation"]["node_macro"]["mae"]
            < per_target[target]["main_btd_fl_metrics"]["validation"]["node_macro"]["mae"]
            for target in CLIENT_NAMES
        ),
        "scarce_local": sum(
            per_target[target]["metrics"]["validation"]["node_macro"]["mae"]
            < per_target[target]["scarce_local_metrics"]["validation"]["node_macro"]["mae"]
            for target in CLIENT_NAMES
        ),
    }
    return {
        "experiment": "formal_btd_fl_25pct_mechanism_ablation",
        "ablation": variant,
        "mechanism_changed": {
            "btd_no_benefit_selection": "donor identity uses fixed lexicographic rule",
            "btd_full_model_transfer": "all compatible trainable neural parameters transferred",
            "btd_no_target_adaptation": "donor temporal state injected without target proxy adaptation",
        }[variant],
        "client_grid_names": list(CLIENT_NAMES),
        "seed": 42,
        "history_fraction": 0.25,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": 1e-3,
        "validation_used_for_selection": False,
        "audit_used_for_selection": False,
        "test_evaluated": False,
        "canonical_validation_evaluated_during_training": False,
        "frozen_reference": {
            "path": "results/federated/formal_25pct/btd_fl_25pct_seed42.json",
            "btd_variant": frozen["btd_variant"],
            "selected_donors": selected,
        },
        "donor_proxy_metadata_by_grid": donor_proxy_metadata,
        "donor_full_model_metadata_by_grid": full_donor_metadata,
        "scenarios": per_target,
        "four_target_unweighted_macro": macro,
        "reference_macro": macro_reference,
        "relative_change_vs_main_btd_fl": _relative(macro, {
            "audit": frozen["four_scenario_macro"]["audit"]["btd_fl"],
            "validation": frozen["four_scenario_macro"]["validation"]["btd_fl"],
        }),
        "relative_change_vs_scarce_local": _relative(macro, {
            "audit": frozen["four_scenario_macro"]["audit"]["scarce_local"],
            "validation": frozen["four_scenario_macro"]["validation"]["scarce_local"],
        }),
        "win_counts": wins,
    }


__all__ = ["ABLATIONS", "load_frozen_reference", "run_ablation"]
