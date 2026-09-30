"""Strict structural validation for industrial external benchmark artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from code.federated.industrial_external import (
    BASELINE_METHODS,
    EXTERNAL_CLIENT_NAMES,
    INDUSTRIAL_TARGET,
    REFERENCE_CLIENT_NAMES,
    select_external_donor,
)
from code.federated.parameter_groups import frozen_temporal_parameter_names

METRICS = ("mae", "rmse", "wape_pct", "smape_pct")
TARGET_COUNTS = {
    0.25: (2335, 2167, 1300, 433, 434),
    0.50: (4670, 4502, 2701, 900, 901),
}
CONTRACTS = {
    "industrial_baselines": (10, 5, 50, 32),
    "industrial_btd_direct_transfer": (0, 0, 50, 32),
    "industrial_fedfomo_style": (10, 5, 50, 32),
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _reject_nonfinite(value: Any, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is not finite")


def _metric_tree(item: Any, label: str) -> None:
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} is missing")
    for scope in ("node_macro", "grid_aggregate"):
        scope_item = item.get(scope)
        if not isinstance(scope_item, Mapping):
            raise ValueError(f"{label}.{scope} is missing")
        for metric in METRICS:
            if not _finite(scope_item.get(metric)):
                raise ValueError(f"{label}.{scope}.{metric} is not finite")


def _split_metrics(item: Any, label: str) -> None:
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} is missing")
    for split in ("audit", "validation"):
        _metric_tree(item.get(split), f"{label}.{split}")


def _same_number(left: Any, right: Any) -> bool:
    return _finite(left) and _finite(right) and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _assert_macro_matches(macro: Mapping[str, Any], metrics: Mapping[str, Any], label: str) -> None:
    _split_metrics(metrics, label)
    _split_metrics(macro, f"industrial_target_macro.{label}")
    for split in ("audit", "validation"):
        for scope in ("node_macro", "grid_aggregate"):
            for metric in METRICS:
                if not _same_number(macro[split][scope][metric], metrics[split][scope][metric]):
                    raise ValueError(f"industrial_target_macro mismatch for {label}.{split}.{scope}.{metric}")


def _fraction_key(fraction: float) -> float:
    for candidate in TARGET_COUNTS:
        if math.isclose(float(fraction), candidate, rel_tol=0.0, abs_tol=1e-12):
            return candidate
    raise ValueError(f"unsupported history fraction: {fraction!r}")


def _require_graph_guardrails(metadata: Mapping[str, Any], label: str) -> None:
    graph = metadata.get("target_graph_metadata")
    if not isinstance(graph, Mapping):
        raise ValueError(f"{label}.target_graph_metadata is missing")
    expected = {
        "graph_uses_target_fit_only": True,
        "graph_uses_calibration": False,
        "graph_uses_audit": False,
        "graph_uses_validation": False,
        "graph_uses_test": False,
    }
    for key, value in expected.items():
        if graph.get(key) is not value:
            raise ValueError(f"{label}.{key} violates graph guardrail")


def _require_counts(metadata: Mapping[str, Any], fraction: float) -> None:
    expected = TARGET_COUNTS[_fraction_key(fraction)]
    fields = ("available_raw_hours", "eligible_target_count", "fit_target_count", "calibration_target_count", "audit_target_count")
    actual = tuple(metadata.get(field) for field in fields)
    if actual != expected:
        raise ValueError(f"wrong industrial scarce counts: {actual!r} != {expected!r}")


def _validate_report(
    report: Mapping[str, Any], *, artifact_type: str, seed: int, fraction: float,
    run_mode: str, rounds: int, local_epochs: int, max_epochs: int, batch_size: int,
) -> None:
    if artifact_type not in CONTRACTS:
        raise ValueError(f"unknown artifact type: {artifact_type}")
    if run_mode == "real" and (int(rounds), int(local_epochs), int(max_epochs), int(batch_size)) != CONTRACTS[artifact_type]:
        raise ValueError("validator called with a non-frozen expected contract")
    expected = {
        "artifact_type": artifact_type,
        "seed": int(seed),
        "history_fraction": float(fraction),
        "run_mode": run_mode,
        "rounds": int(rounds),
        "local_epochs": int(local_epochs),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "learning_rate": 1e-3,
        "patience": 8,
        "test_evaluated": False,
        "industrial_test_locked": True,
        "industrial_test_evaluated": False,
        "reference_grid_tests_evaluated": False,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if isinstance(value, float):
            if not _same_number(actual, value):
                raise ValueError(f"wrong {key}: {actual!r}")
        elif actual != value:
            raise ValueError(f"wrong {key}: {actual!r}")
    for field in ("validation_used_for_selection", "audit_used_for_selection", "canonical_validation_evaluated_during_training"):
        if report.get(field) is not False:
            raise ValueError(f"top-level guardrail {field} is invalid")
    if report.get("communication_round_selection") != "fixed_final_round":
        raise ValueError("communication round selection is not fixed-final-round")
    if tuple(report.get("client_grid_names", ())) != EXTERNAL_CLIENT_NAMES:
        raise ValueError("external client set is incomplete")
    if tuple(report.get("primary_targets", ())) != (INDUSTRIAL_TARGET,):
        raise ValueError("only the industrial target may be primary")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != {INDUSTRIAL_TARGET}:
        raise ValueError("industrial target scenario is incomplete")
    methods = tuple(report.get("methods", ()))
    expected_methods = {
        "industrial_baselines": BASELINE_METHODS,
        "industrial_btd_direct_transfer": ("btd_fl_direct_transfer",),
        "industrial_fedfomo_style": ("fedfomo_style",),
    }[artifact_type]
    if methods != expected_methods:
        raise ValueError("artifact has wrong method set")
    macro = report.get("industrial_target_macro")
    if not isinstance(macro, Mapping) or set(macro) != set(expected_methods):
        raise ValueError("industrial_target_macro is incomplete")
    scenario = scenarios[INDUSTRIAL_TARGET]
    if not isinstance(scenario, Mapping):
        raise ValueError("industrial target scenario is invalid")

    if artifact_type == "industrial_baselines":
        method_map = scenario.get("methods")
        if not isinstance(method_map, Mapping) or set(method_map) != set(BASELINE_METHODS):
            raise ValueError("baseline scenario is incomplete")
        for method in BASELINE_METHODS:
            _assert_macro_matches(macro[method], method_map[method], method)
        metadata = scenario.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("baseline metadata is missing")
        if run_mode == "real":
            _require_counts(metadata, fraction)
        _require_graph_guardrails(metadata, "baseline")
        for field in ("validation_used_for_selection", "audit_used_for_selection", "test_evaluated", "industrial_test_evaluated", "reference_grid_tests_evaluated"):
            if metadata.get(field) is not False:
                raise ValueError(f"baseline guardrail {field} is invalid")
        if metadata.get("canonical_validation_evaluated_during_training") not in (False, None):
            raise ValueError("canonical validation was used during FL training")
        federated = metadata.get("federated")
        if not isinstance(federated, Mapping) or set(federated) != {"fedavg", "fedprox", "fedper"}:
            raise ValueError("baseline federated reports are incomplete")
        for method, item in federated.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"missing federated report for {method}")
            trainer_report = item.get("federated_report")
            if not isinstance(trainer_report, Mapping):
                raise ValueError(f"{method} trainer report is missing")
            if trainer_report.get("rounds") != int(rounds) or trainer_report.get("local_epochs") != int(local_epochs) or trainer_report.get("batch_size") != int(batch_size):
                raise ValueError(f"{method} trainer budget is not frozen")
            optimizer = trainer_report.get("optimizer")
            if not isinstance(optimizer, Mapping) or optimizer.get("name") != "Adam" or not _same_number(optimizer.get("learning_rate"), 1e-3):
                raise ValueError(f"{method} trainer optimizer contract is invalid")
            if trainer_report.get("communication_round_selection") != "fixed_final_round":
                raise ValueError(f"{method} did not use fixed final round")
            for field in ("canonical_validation_evaluated_during_training", "validation_used_for_selection", "test_evaluated"):
                if trainer_report.get(field) is not False:
                    raise ValueError(f"{method} trainer guardrail {field} is invalid")
            if tuple(trainer_report.get("client_grid_names", ())) != EXTERNAL_CLIENT_NAMES:
                raise ValueError(f"{method} trainer client set is incomplete")
            counts = item.get("local_sample_counts")
            if not isinstance(counts, Mapping) or tuple(counts) != EXTERNAL_CLIENT_NAMES:
                raise ValueError(f"{method} local sample-count clients are incomplete")
            trainer_counts = trainer_report.get("local_sample_counts")
            if not isinstance(trainer_counts, Mapping) or dict(trainer_counts) != dict(counts):
                raise ValueError(f"{method} trainer/wrapper sample counts disagree")
            if run_mode == "real":
                expected_counts = {name: (5964 if name in REFERENCE_CLIENT_NAMES else TARGET_COUNTS[_fraction_key(fraction)][2]) for name in EXTERNAL_CLIENT_NAMES}
                if any(not isinstance(counts[name], int) or isinstance(counts[name], bool) for name in EXTERNAL_CLIENT_NAMES):
                    raise ValueError(f"{method} local sample counts are not integer")
                if {name: int(counts[name]) for name in EXTERNAL_CLIENT_NAMES} != expected_counts:
                    raise ValueError(f"{method} source sample-count contract is invalid")
            contracts = item.get("training_index_contract_by_client")
            if not isinstance(contracts, Mapping):
                raise ValueError(f"{method} training index contracts are missing")
            for name in EXTERNAL_CLIENT_NAMES:
                expected_contract = "industrial_scarce_fit_only" if name == INDUSTRIAL_TARGET else "full_canonical_train_all_complete_targets"
                if contracts.get(name) != expected_contract:
                    raise ValueError(f"{method} has invalid index contract for {name}")
            if method == "fedprox" and not _same_number(item.get("fedprox_mu"), 0.01):
                raise ValueError("FedProx mu is not frozen at 0.01")
            round_history = trainer_report.get("round_history")
            if not isinstance(round_history, list) or len(round_history) != int(rounds):
                raise ValueError(f"{method} trainer round history is incomplete")
            if [round_item.get("round") for round_item in round_history if isinstance(round_item, Mapping)] != list(range(1, int(rounds) + 1)):
                raise ValueError(f"{method} trainer round numbers are invalid")
            aggregation_weights = item.get("aggregation_weights")
            if not isinstance(aggregation_weights, Mapping) or round_history[-1].get("aggregation_weights") != dict(aggregation_weights):
                raise ValueError(f"{method} final aggregation weights disagree with wrapper")
    elif artifact_type == "industrial_btd_direct_transfer":
        if report.get("btd_variant") != "btd_fl_direct_transfer":
            raise ValueError("wrong BTD variant")
        metrics = scenario.get("metrics")
        _assert_macro_matches(macro["btd_fl_direct_transfer"], metrics, "btd_fl_direct_transfer")
        metadata = scenario.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("BTD metadata is missing")
        if run_mode == "real":
            _require_counts(metadata, fraction)
        for field in ("target_scaler_fit_start_index", "target_scaler_fit_end_index"):
            if not isinstance(metadata.get(field), int) or isinstance(metadata.get(field), bool):
                raise ValueError(f"BTD {field} is missing or not an integer")
        benefits = metadata.get("calibration_benefit_by_donor")
        if not isinstance(benefits, Mapping) or tuple(benefits) != REFERENCE_CLIENT_NAMES:
            raise ValueError("BTD requires exactly the four reference donor benefits")
        if any(not _finite(value) for value in benefits.values()):
            raise ValueError("BTD donor benefit is not finite")
        selected, maximum, fallback = select_external_donor(benefits)
        recorded_fallback = metadata.get("zero_transfer_fallback")
        if not isinstance(recorded_fallback, bool):
            raise ValueError("BTD fallback flag is not boolean")
        if metadata.get("selected_donor") != selected or recorded_fallback != fallback:
            raise ValueError("BTD donor selection/fallback is inconsistent with benefits")
        expected_benefit = 0.0 if fallback else maximum
        if not _same_number(metadata.get("selected_calibration_benefit"), expected_benefit):
            raise ValueError("BTD selected calibration benefit is inconsistent")
        if metadata.get("positive_benefit_count") != sum(float(v) > 0 for v in benefits.values()) or metadata.get("non_positive_benefit_count") != sum(float(v) <= 0 for v in benefits.values()):
            raise ValueError("BTD benefit counts are inconsistent")
        if not isinstance(metadata.get("raw_selected_donor_temporal_state_used"), bool) or metadata.get("raw_selected_donor_temporal_state_used") is not (not fallback):
            raise ValueError("BTD raw temporal transfer flag is inconsistent")
        if metadata.get("adapted_probe_states_used_for_final_initialization") is not False:
            raise ValueError("adapted probe state was used for final initialization")
        transferred = metadata.get("transferred_parameter_names")
        if not isinstance(transferred, list):
            raise ValueError("BTD transferred parameter names are missing")
        if fallback:
            if transferred:
                raise ValueError("fallback BTD must transfer no parameters")
        elif tuple(transferred) != frozen_temporal_parameter_names():
            raise ValueError("BTD transferred parameter group is not the complete frozen temporal group")
        for field in ("validation_used_for_selection", "audit_used_for_selection", "test_evaluated", "industrial_test_evaluated", "reference_grid_tests_evaluated"):
            if metadata.get(field) is not False:
                raise ValueError(f"BTD guardrail {field} is invalid")
    else:
        _assert_macro_matches(macro["fedfomo_style"], scenario.get("scarce_target_metrics"), "fedfomo_style")
        history = scenario.get("round_history")
        if not isinstance(history, list) or len(history) != int(rounds):
            raise ValueError("FedFomo round history length does not match the requested contract")
        if [item.get("round") for item in history if isinstance(item, Mapping)] != list(range(1, int(rounds) + 1)):
            raise ValueError("FedFomo round numbers are not sequential")
        metadata = scenario.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("FedFomo metadata is missing")
        if metadata.get("target_graph_is_local") is not True:
            raise ValueError("FedFomo target topology is not marked local")
        for field in ("validation_used_for_peer_weighting", "audit_used_for_peer_weighting", "test_evaluated", "industrial_test_evaluated", "reference_grid_tests_evaluated"):
            if metadata.get(field) is not False:
                raise ValueError(f"FedFomo guardrail {field} is invalid")
        for round_item in history:
            weights = round_item.get("peer_weights")
            diagnostics = round_item.get("diagnostics")
            if not isinstance(weights, Mapping) or not isinstance(diagnostics, Mapping) or INDUSTRIAL_TARGET not in weights or INDUSTRIAL_TARGET not in diagnostics:
                raise ValueError("FedFomo industrial target round diagnostics are incomplete")
            target_weights = weights[INDUSTRIAL_TARGET]
            target_diag = diagnostics[INDUSTRIAL_TARGET]
            if not isinstance(target_weights, Mapping) or not isinstance(target_diag, Mapping):
                raise ValueError("FedFomo industrial target diagnostics are invalid")
            normalized = target_diag.get("normalized_weight_by_client", target_weights)
            if not isinstance(normalized, Mapping) or any(not _finite(v) or float(v) < 0 for v in normalized.values()):
                raise ValueError("FedFomo normalized weights are invalid")
            if dict(target_weights) != dict(normalized):
                raise ValueError("FedFomo peer weights do not match normalized diagnostics")
            if normalized and not math.isclose(sum(float(v) for v in normalized.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("FedFomo normalized weights do not sum to one")
            if not normalized and target_diag.get("no_positive_candidate_fallback") is not True:
                raise ValueError("FedFomo empty normalized weights require fallback")
            if normalized and target_diag.get("no_positive_candidate_fallback") is True:
                raise ValueError("FedFomo positive weights cannot report fallback")
            if target_diag.get("positive_candidate_count") != len(normalized):
                raise ValueError("FedFomo positive candidate count is inconsistent")
            for key in ("baseline_self_calibration_loss", "positive_candidate_count", "peer_only_positive_count"):
                if key in target_diag and not _finite(target_diag[key]):
                    raise ValueError(f"FedFomo diagnostic {key} is not finite")
            for key in ("candidate_calibration_loss_by_client", "trainable_parameter_l2_distance_by_client", "raw_weight_by_client"):
                values = target_diag.get(key)
                if not isinstance(values, Mapping) or any(not _finite(v) for v in values.values()):
                    raise ValueError(f"FedFomo diagnostic {key} is invalid")
        if scenario.get("final_round_peer_weights") != history[-1].get("peer_weights"):
            raise ValueError("FedFomo final round weights do not match round history")


def load_valid_external_result(path: Path, **contract: Any) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("artifact root must be an object")
    _reject_nonfinite(report)
    _validate_report(report, **contract)
    return dict(report)


def resumable_external_result(path: Path, **contract: Any) -> bool:
    try:
        load_valid_external_result(path, **contract)
    except (FileNotFoundError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


__all__ = ["CONTRACTS", "load_valid_external_result", "resumable_external_result"]
