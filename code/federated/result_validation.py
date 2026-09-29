"""Strict validation for resumable formal benchmark artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from code.federated.formal_btd import CLIENT_NAMES

_METRIC_NAMES = ("mae", "rmse", "wape_pct", "smape_pct")
_SPLITS = ("audit", "validation")
_SCOPES = ("node_macro", "grid_aggregate")
_BASELINE_METHODS = ("scarce_local", "fedavg", "fedprox", "fedper")


def _finite_number(value: Any) -> bool:
    """Return true for finite real-valued metrics, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_metric_tree(metrics: Any, context: str) -> None:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{context} must be a mapping")
    for split in _SPLITS:
        split_value = metrics.get(split)
        if not isinstance(split_value, Mapping):
            raise ValueError(f"{context}.{split} is missing")
        for scope in _SCOPES:
            scope_value = split_value.get(scope)
            if not isinstance(scope_value, Mapping):
                raise ValueError(f"{context}.{split}.{scope} is missing")
            for metric in _METRIC_NAMES:
                value = scope_value.get(metric)
                if not _finite_number(value):
                    raise ValueError(f"{context}.{split}.{scope}.{metric} is not finite")


def _validate_metric_split(split_value: Any, context: str) -> None:
    """Validate one split of a macro tree (scope -> scalar metrics)."""
    if not isinstance(split_value, Mapping):
        raise ValueError(f"{context} is missing")
    for scope in _SCOPES:
        scope_value = split_value.get(scope)
        if not isinstance(scope_value, Mapping):
            raise ValueError(f"{context}.{scope} is missing")
        for metric in _METRIC_NAMES:
            value = scope_value.get(metric)
            if not _finite_number(value):
                raise ValueError(f"{context}.{scope}.{metric} is not finite")


def _validate_macro(report: Mapping[str, Any], methods: tuple[str, ...], key: str) -> None:
    macro = report.get(key)
    if not isinstance(macro, Mapping):
        raise ValueError(f"artifact is missing {key}")
    for split in _SPLITS:
        split_macro = macro.get(split)
        if not isinstance(split_macro, Mapping):
            raise ValueError(f"{key}.{split} is missing")
        if set(split_macro) != set(methods):
            raise ValueError(f"{key}.{split} has the wrong method set")
        for method in methods:
            _validate_metric_split(split_macro[method], f"{key}.{split}.{method}")


def _validate_baseline_scenarios(report: Mapping[str, Any]) -> None:
    methods = _BASELINE_METHODS
    for target, scenario in report["scenarios"].items():
        if not isinstance(scenario, Mapping) or not isinstance(scenario.get("methods"), Mapping):
            raise ValueError(f"baseline scenario {target} is missing methods")
        method_map = scenario["methods"]
        if set(method_map) != set(methods):
            raise ValueError(f"baseline scenario {target} has the wrong method set")
        for method in methods:
            _validate_metric_tree(method_map[method], f"scenarios.{target}.methods.{method}")
    _validate_macro(report, methods, "four_scenario_macro")


def _validate_direct_transfer_scenarios(report: Mapping[str, Any]) -> None:
    for target, scenario in report["scenarios"].items():
        if not isinstance(scenario, Mapping):
            raise ValueError(f"direct-transfer scenario {target} is not a mapping")
        _validate_metric_tree(scenario.get("metrics"), f"scenarios.{target}.metrics")
        metadata = scenario.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"direct-transfer scenario {target} is missing metadata")
        for field in ("calibration_benefit_by_donor", "selected_donor",
                      "selected_calibration_benefit", "zero_transfer_fallback"):
            if field not in metadata:
                raise ValueError(f"direct-transfer scenario {target} is missing {field}")
        for field in ("validation_used_for_selection", "audit_used_for_selection", "test_evaluated"):
            if metadata.get(field) is not False:
                raise ValueError(f"direct-transfer scenario {target}.{field} must be false")
        benefits = metadata["calibration_benefit_by_donor"]
        expected_donors = set(CLIENT_NAMES) - {target}
        if not isinstance(benefits, Mapping) or set(benefits) != expected_donors:
            raise ValueError(f"direct-transfer scenario {target} must expose exactly three donor benefits")
        for donor, value in benefits.items():
            if not _finite_number(value):
                raise ValueError(f"direct-transfer benefit {donor}->{target} is not finite")
        if not _finite_number(metadata["selected_calibration_benefit"]):
            raise ValueError(f"direct-transfer scenario {target} selected benefit is not finite")
        fallback = metadata["zero_transfer_fallback"]
        if not isinstance(fallback, bool):
            raise ValueError(f"direct-transfer scenario {target} fallback flag is invalid")
        selected = metadata["selected_donor"]
        if fallback and selected is not None:
            raise ValueError(f"direct-transfer fallback scenario {target} selected a donor")
        if not fallback and selected not in expected_donors:
            raise ValueError(f"direct-transfer scenario {target} selected an invalid donor")
    # Direct-transfer reports use a target-unweighted name, unlike formal BTD.
    macro = report.get("four_target_unweighted_macro")
    if macro is not None:
        _validate_metric_tree(macro, "four_target_unweighted_macro")


def _validate_fedfomo_scenarios(report: Mapping[str, Any], rounds: int | None) -> None:
    if report.get("validation_used_for_peer_weighting") is not False:
        raise ValueError("FedFomo validation weighting guardrail is invalid")
    if report.get("audit_used_for_peer_weighting") is not False:
        raise ValueError("FedFomo audit weighting guardrail is invalid")
    if report.get("test_evaluated") is not False:
        raise ValueError("FedFomo test guardrail is invalid")
    if report.get("canonical_validation_evaluated_during_training") is not False:
        raise ValueError("FedFomo canonical validation must be evaluation-only")
    if report.get("communication_round_selection") != "fixed_final_round":
        raise ValueError("FedFomo must use fixed-final-round selection")
    expected_rounds = report.get("rounds") if rounds is None else rounds
    for target, scenario in report["scenarios"].items():
        if not isinstance(scenario, Mapping):
            raise ValueError(f"FedFomo scenario {target} is not a mapping")
        _validate_metric_tree(scenario.get("scarce_target_metrics"), f"scenarios.{target}.scarce_target_metrics")
        history = scenario.get("round_history")
        if not isinstance(history, list) or len(history) != expected_rounds:
            raise ValueError(f"FedFomo scenario {target} has the wrong round history length")
        for index, round_item in enumerate(history):
            if not isinstance(round_item, Mapping) or "peer_weights" not in round_item or "diagnostics" not in round_item:
                raise ValueError(f"FedFomo scenario {target} round {index} is incomplete")
        metadata = scenario.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"FedFomo scenario {target} is missing metadata")
        for field in ("validation_used_for_peer_weighting", "audit_used_for_peer_weighting", "test_evaluated"):
            if metadata.get(field) is not False:
                raise ValueError(f"FedFomo scenario {target}.{field} must be false")
    macro = report.get("four_scenario_unweighted_macro")
    if macro is not None:
        _validate_metric_tree(macro, "four_scenario_unweighted_macro")


def load_valid_result(path: Path, *, seed: int, artifact_type: str, run_mode: str,
                      methods: tuple[str, ...], rounds: int | None = None,
                      local_epochs: int | None = None, batch_size: int = 32,
                      learning_rate: float = 1e-3, max_epochs: int | None = None,
                      patience: int | None = None, history_fraction: float = 0.25) -> dict[str, Any]:
    """Load only a complete result matching the requested execution contract."""
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_mode": run_mode, "seed": int(seed), "artifact_type": artifact_type,
        "history_fraction": float(history_fraction), "batch_size": int(batch_size),
        "learning_rate": float(learning_rate), "test_evaluated": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"artifact {path} has {key}={report.get(key)!r}; expected {value!r}")
    for key, value in (("rounds", rounds), ("local_epochs", local_epochs),
                       ("max_epochs", max_epochs), ("patience", patience)):
        if value is not None and report.get(key) != value:
            raise ValueError(f"artifact {path} has {key}={report.get(key)!r}; expected {value!r}")
    if tuple(report.get("client_grid_names", ())) != CLIENT_NAMES:
        raise ValueError(f"artifact {path} does not expose the four scarce targets")
    if set(report.get("methods", ())) != set(methods):
        raise ValueError(f"artifact {path} has method set {report.get('methods')!r}; expected {methods!r}")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(CLIENT_NAMES):
        raise ValueError(f"artifact {path} is missing one or more scarce-target scenarios")
    expected_methods = tuple(methods)
    if artifact_type == "formal_baselines":
        if expected_methods != _BASELINE_METHODS:
            raise ValueError("formal_baselines requires the four baseline methods")
        _validate_baseline_scenarios(report)
    elif artifact_type == "btd_fl_direct_transfer":
        _validate_direct_transfer_scenarios(report)
    elif artifact_type == "fedfomo_style":
        _validate_fedfomo_scenarios(report, rounds)
    elif artifact_type == "formal_btd":
        # Retain safety for the legacy combined formal runner while keeping
        # the baseline-only and direct-transfer schemas independently strict.
        for target, scenario in scenarios.items():
            if not isinstance(scenario, Mapping) or not isinstance(scenario.get("methods"), Mapping):
                raise ValueError(f"formal BTD scenario {target} is missing methods")
            method_map = scenario["methods"]
            if set(method_map) != set(expected_methods):
                raise ValueError(f"formal BTD scenario {target} has the wrong method set")
            for method in expected_methods:
                _validate_metric_tree(method_map[method], f"scenarios.{target}.methods.{method}")
        _validate_macro(report, expected_methods, "four_scenario_macro")
    else:
        raise ValueError(f"unsupported artifact type {artifact_type!r}")
    return report


def resumable_result(path: Path, **contract: Any) -> bool:
    """Return true only for a complete matching result; invalid artifacts are not resumable."""
    try:
        load_valid_result(path, **contract)
    except (FileNotFoundError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True
