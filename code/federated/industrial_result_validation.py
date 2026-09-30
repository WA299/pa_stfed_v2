"""Strict resume validation for the industrial external benchmark artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from code.federated.industrial_external import BASELINE_METHODS, EXTERNAL_CLIENT_NAMES, INDUSTRIAL_TARGET, REFERENCE_CLIENT_NAMES

METRICS = ("mae", "rmse", "wape_pct", "smape_pct")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _metric_tree(item: Any, label: str) -> None:
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} is missing")
    for scope in ("node_macro", "grid_aggregate"):
        if not isinstance(item.get(scope), Mapping):
            raise ValueError(f"{label}.{scope} is missing")
        for metric in METRICS:
            if not _finite(item[scope].get(metric)):
                raise ValueError(f"{label}.{scope}.{metric} is not finite")


def _split_metrics(item: Any, label: str) -> None:
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} is missing")
    for split in ("audit", "validation"):
        _metric_tree(item.get(split), f"{label}.{split}")


def _validate_report(report: Mapping[str, Any], *, artifact_type: str, seed: int, fraction: float, run_mode: str, rounds: int, local_epochs: int, max_epochs: int, batch_size: int) -> None:
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
            if not isinstance(actual, (int, float)) or abs(float(actual) - value) > 1e-12:
                raise ValueError(f"wrong {key}: {actual!r}")
        elif actual != value:
            raise ValueError(f"wrong {key}: {actual!r}")
    if tuple(report.get("client_grid_names", ())) != EXTERNAL_CLIENT_NAMES:
        raise ValueError("external client set is incomplete")
    if set(report.get("scenarios", {})) != {INDUSTRIAL_TARGET}:
        raise ValueError("industrial target scenario is incomplete")
    methods = tuple(report.get("methods", ()))
    if artifact_type == "industrial_baselines" and methods != BASELINE_METHODS:
        raise ValueError("baseline artifact has wrong method set")
    if artifact_type == "industrial_btd_direct_transfer" and methods != ("btd_fl_direct_transfer",):
        raise ValueError("BTD artifact has wrong method set")
    if artifact_type == "industrial_fedfomo_style" and methods != ("fedfomo_style",):
        raise ValueError("FedFomo artifact has wrong method set")
    scenario = report["scenarios"][INDUSTRIAL_TARGET]
    if artifact_type == "industrial_baselines":
        method_map = scenario.get("methods")
        if not isinstance(method_map, Mapping) or set(method_map) != set(BASELINE_METHODS):
            raise ValueError("baseline scenario is incomplete")
        for method in BASELINE_METHODS:
            _split_metrics(method_map[method], f"{method}")
    elif artifact_type == "industrial_btd_direct_transfer":
        _split_metrics(scenario.get("metrics"), "BTD metrics")
        metadata = scenario.get("metadata", {})
        benefits = metadata.get("calibration_benefit_by_donor")
        if not isinstance(benefits, Mapping) or set(benefits) != set(REFERENCE_CLIENT_NAMES):
            raise ValueError("BTD requires exactly four donor benefits")
        if any(not _finite(value) for value in benefits.values()):
            raise ValueError("BTD donor benefit is not finite")
        selected = metadata.get("selected_donor")
        fallback = metadata.get("zero_transfer_fallback")
        if not isinstance(fallback, bool) or (fallback and selected is not None) or (not fallback and selected not in REFERENCE_CLIENT_NAMES):
            raise ValueError("BTD selected donor/fallback contract is invalid")
        for field in ("validation_used_for_selection", "audit_used_for_selection", "test_evaluated"):
            if metadata.get(field) is not False:
                raise ValueError(f"BTD guardrail {field} is invalid")
    else:
        _split_metrics(scenario.get("scarce_target_metrics"), "FedFomo metrics")
        history = scenario.get("round_history")
        if not isinstance(history, list) or len(history) != rounds:
            raise ValueError("FedFomo round history is incomplete")
        if any(not isinstance(item, Mapping) or "peer_weights" not in item or "diagnostics" not in item for item in history):
            raise ValueError("FedFomo diagnostics are incomplete")
        metadata = scenario.get("metadata", {})
        for field in ("validation_used_for_peer_weighting", "audit_used_for_peer_weighting", "test_evaluated"):
            if metadata.get(field) is not False:
                raise ValueError(f"FedFomo guardrail {field} is invalid")


def load_valid_external_result(path: Path, **contract: Any) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_report(report, **contract)
    return report


def resumable_external_result(path: Path, **contract: Any) -> bool:
    try:
        load_valid_external_result(path, **contract)
    except (FileNotFoundError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


__all__ = ["load_valid_external_result", "resumable_external_result"]
