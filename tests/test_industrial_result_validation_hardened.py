from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from code.federated.industrial_external import (
    BASELINE_METHODS,
    EXTERNAL_CLIENT_NAMES,
    INDUSTRIAL_TARGET,
    REFERENCE_CLIENT_NAMES,
)
from code.federated.industrial_result_validation import resumable_external_result


def _metric(value: float = 1.0) -> dict:
    scope = {"mae": value, "rmse": value + 1, "wape_pct": value + 2, "smape_pct": value + 3}
    return {"audit": {"node_macro": dict(scope), "grid_aggregate": dict(scope)}, "validation": {"node_macro": dict(scope), "grid_aggregate": dict(scope)}}


def _graph() -> dict:
    return {"graph_uses_target_fit_only": True, "graph_uses_calibration": False, "graph_uses_audit": False, "graph_uses_validation": False, "graph_uses_test": False}


def _base(artifact_type: str, methods: tuple[str, ...], rounds: int, local_epochs: int, max_epochs: int = 50, batch_size: int = 32) -> dict:
    return {
        "artifact_type": artifact_type, "seed": 123, "history_fraction": 0.25, "run_mode": "real",
        "rounds": rounds, "local_epochs": local_epochs, "max_epochs": max_epochs, "batch_size": batch_size,
        "learning_rate": 1e-3, "patience": 8, "test_evaluated": False,
        "industrial_test_locked": True, "industrial_test_evaluated": False, "reference_grid_tests_evaluated": False,
        "validation_used_for_selection": False, "audit_used_for_selection": False,
        "canonical_validation_evaluated_during_training": False, "communication_round_selection": "fixed_final_round",
        "client_grid_names": list(EXTERNAL_CLIENT_NAMES), "primary_targets": [INDUSTRIAL_TARGET], "methods": list(methods),
    }


def _baseline() -> dict:
    report = _base("industrial_baselines", BASELINE_METHODS, 10, 5)
    methods = {name: _metric() for name in BASELINE_METHODS}
    federated = {}
    counts = {name: 5964 for name in REFERENCE_CLIENT_NAMES} | {INDUSTRIAL_TARGET: 1300}
    contracts = {name: "full_canonical_train_all_complete_targets" for name in REFERENCE_CLIENT_NAMES} | {INDUSTRIAL_TARGET: "industrial_scarce_fit_only"}
    for method in ("fedavg", "fedprox", "fedper"):
        federated[method] = {
            "communication_round_selection": "fixed_final_round",
            "canonical_validation_evaluated_during_training": False,
            "local_sample_counts": counts,
            "training_index_contract_by_client": contracts,
            "fedprox_mu": 0.01 if method == "fedprox" else None,
        }
    report["scenarios"] = {INDUSTRIAL_TARGET: {"methods": methods, "metadata": {
        "available_raw_hours": 2335, "eligible_target_count": 2167, "fit_target_count": 1300,
        "calibration_target_count": 433, "audit_target_count": 434, "target_graph_metadata": _graph(),
        "federated": federated, "validation_used_for_selection": False, "audit_used_for_selection": False,
        "test_evaluated": False, "industrial_test_evaluated": False, "reference_grid_tests_evaluated": False,
    }}}
    report["industrial_target_macro"] = {name: copy.deepcopy(metrics) for name, metrics in methods.items()}
    return report


def _btd(fallback: bool = False) -> dict:
    report = _base("industrial_btd_direct_transfer", ("btd_fl_direct_transfer",), 0, 0)
    benefits = {name: (-0.1 if fallback else (0.1 if name == REFERENCE_CLIENT_NAMES[0] else -0.1)) for name in REFERENCE_CLIENT_NAMES}
    metrics = _metric()
    metadata = {
        "calibration_benefit_by_donor": benefits, "selected_donor": None if fallback else REFERENCE_CLIENT_NAMES[0],
        "selected_calibration_benefit": 0.0 if fallback else 0.1, "positive_benefit_count": 0 if fallback else 1,
        "non_positive_benefit_count": 4 if fallback else 3, "zero_transfer_fallback": fallback,
        "raw_selected_donor_temporal_state_used": not fallback, "adapted_probe_states_used_for_final_initialization": False,
        "transferred_parameter_names": [] if fallback else ["gru.weight"], "validation_used_for_selection": False,
        "audit_used_for_selection": False, "test_evaluated": False, "industrial_test_evaluated": False,
        "reference_grid_tests_evaluated": False,
    }
    report["btd_variant"] = "btd_fl_direct_transfer"
    report["scenarios"] = {INDUSTRIAL_TARGET: {"metrics": metrics, "metadata": metadata}}
    report["industrial_target_macro"] = {"btd_fl_direct_transfer": copy.deepcopy(metrics)}
    return report


def _fedfomo() -> dict:
    report = _base("industrial_fedfomo_style", ("fedfomo_style",), 10, 5)
    rounds = []
    for index in range(1, 11):
        weights = {REFERENCE_CLIENT_NAMES[0]: 1.0}
        diagnostics = {"baseline_self_calibration_loss": 1.0, "candidate_calibration_loss_by_client": {name: 1.0 for name in EXTERNAL_CLIENT_NAMES}, "trainable_parameter_l2_distance_by_client": {name: 1.0 for name in EXTERNAL_CLIENT_NAMES}, "raw_weight_by_client": {name: 1.0 for name in EXTERNAL_CLIENT_NAMES}, "normalized_weight_by_client": weights, "positive_candidate_count": 1, "peer_only_positive_count": 1, "no_positive_candidate_fallback": False}
        rounds.append({"round": index, "peer_weights": {INDUSTRIAL_TARGET: weights}, "diagnostics": {INDUSTRIAL_TARGET: diagnostics}})
    metrics = _metric()
    report["scenarios"] = {INDUSTRIAL_TARGET: {"scarce_target_metrics": metrics, "round_history": rounds, "final_round_peer_weights": rounds[-1]["peer_weights"], "metadata": {"validation_used_for_peer_weighting": False, "audit_used_for_peer_weighting": False, "test_evaluated": False, "industrial_test_evaluated": False, "reference_grid_tests_evaluated": False}}}
    report["industrial_target_macro"] = {"fedfomo_style": copy.deepcopy(metrics)}
    return report


def _write(report: dict, name: str = "tmp_external_validation_fixture.json") -> Path:
    path = Path(name)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _contract(artifact_type: str) -> dict:
    return {"artifact_type": artifact_type, "seed": 123, "fraction": 0.25, "run_mode": "real", "rounds": {"industrial_baselines": 10, "industrial_btd_direct_transfer": 0, "industrial_fedfomo_style": 10}[artifact_type], "local_epochs": {"industrial_baselines": 5, "industrial_btd_direct_transfer": 0, "industrial_fedfomo_style": 5}[artifact_type], "max_epochs": 50, "batch_size": 32}


@pytest.mark.parametrize("artifact_type, builder", [("industrial_baselines", _baseline), ("industrial_btd_direct_transfer", _btd), ("industrial_fedfomo_style", _fedfomo)])
def test_valid_complete_external_artifacts_resume(artifact_type: str, builder):
    path = _write(builder())
    try:
        assert resumable_external_result(path, **_contract(artifact_type))
    finally:
        path.unlink(missing_ok=True)


def test_baseline_macro_mismatch_and_nan_do_not_resume():
    report = _baseline()
    report["industrial_target_macro"]["fedavg"]["validation"]["node_macro"]["mae"] = 99.0
    path = _write(report)
    assert not resumable_external_result(path, **_contract("industrial_baselines"))
    report = _baseline()
    report["scenarios"][INDUSTRIAL_TARGET]["methods"]["fedavg"]["audit"]["node_macro"]["mae"] = float("nan")
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not resumable_external_result(path, **_contract("industrial_baselines"))
    path.unlink(missing_ok=True)


def test_wrong_budget_and_incomplete_fedfomo_do_not_resume():
    report = _baseline()
    contract = _contract("industrial_baselines")
    contract["rounds"] = 9
    path = _write(report)
    assert not resumable_external_result(path, **contract)
    report = _fedfomo()
    report["scenarios"][INDUSTRIAL_TARGET]["round_history"].pop()
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not resumable_external_result(path, **_contract("industrial_fedfomo_style"))
    path.unlink(missing_ok=True)


def test_btd_selection_and_fallback_contracts():
    report = _btd()
    report["scenarios"][INDUSTRIAL_TARGET]["metadata"]["selected_donor"] = REFERENCE_CLIENT_NAMES[1]
    path = _write(report)
    assert not resumable_external_result(path, **_contract("industrial_btd_direct_transfer"))
    path.write_text(json.dumps(_btd(fallback=True)), encoding="utf-8")
    assert resumable_external_result(path, **_contract("industrial_btd_direct_transfer"))
    path.unlink(missing_ok=True)
