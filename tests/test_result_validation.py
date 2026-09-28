import json
import math
import shutil
from pathlib import Path

import pytest

from code.federated.formal_btd import CLIENT_NAMES
from code.federated.result_validation import resumable_result


METRICS = ("mae", "rmse", "wape_pct", "smape_pct")
BASELINES = ("scarce_local", "fedavg", "fedprox", "fedper")


def _metric(value=1.0):
    return {
        split: {
            scope: {name: value for name in METRICS}
            for scope in ("node_macro", "grid_aggregate")
        }
        for split in ("audit", "validation")
    }


def _macro(methods):
    return {
        split: {
            method: {
                scope: {name: 1.0 for name in METRICS}
                for scope in ("node_macro", "grid_aggregate")
            }
            for method in methods
        }
        for split in ("audit", "validation")
    }


def _common(artifact_type, methods, **extra):
    result = {
        "run_mode": "real",
        "seed": 123,
        "artifact_type": artifact_type,
        "history_fraction": 0.25,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "test_evaluated": False,
        "client_grid_names": list(CLIENT_NAMES),
        "methods": list(methods),
    }
    result.update(extra)
    return result


def _baseline_report():
    report = _common(
        "formal_baselines", BASELINES, rounds=10, local_epochs=5,
        max_epochs=50, patience=8,
    )
    report["scenarios"] = {
        target: {"methods": {method: _metric() for method in BASELINES}}
        for target in CLIENT_NAMES
    }
    report["four_scenario_macro"] = _macro(BASELINES)
    return report


def _direct_report():
    report = _common(
        "btd_fl_direct_transfer", ("btd_fl_direct_transfer",),
        btd_variant="btd_fl_direct_transfer", max_epochs=50, patience=8,
    )
    report["scenarios"] = {}
    for index, target in enumerate(CLIENT_NAMES):
        donors = {donor: 0.1 for donor in CLIENT_NAMES if donor != target}
        report["scenarios"][target] = {
            "metrics": _metric(),
            "metadata": {
                "calibration_benefit_by_donor": donors,
                "selected_donor": max(donors),
                "selected_calibration_benefit": 0.1,
                "zero_transfer_fallback": False,
                "validation_used_for_selection": False,
                "audit_used_for_selection": False,
                "test_evaluated": False,
            },
        }
    report["four_target_unweighted_macro"] = _metric()
    return report


def _fomo_report(rounds=2):
    report = _common(
        "fedfomo_style", ("fedfomo_style",), rounds=rounds,
        local_epochs=5, max_epochs=50, patience=8,
        validation_used_for_peer_weighting=False,
        audit_used_for_peer_weighting=False,
        canonical_validation_evaluated_during_training=False,
        communication_round_selection="fixed_final_round",
    )
    report["scenarios"] = {
        target: {
            "scarce_target_metrics": _metric(),
            "round_history": [
                {"peer_weights": {}, "diagnostics": {}}
                for _ in range(rounds)
            ],
            "metadata": {
                "validation_used_for_peer_weighting": False,
                "audit_used_for_peer_weighting": False,
                "test_evaluated": False,
            },
        }
        for target in CLIENT_NAMES
    }
    report["four_scenario_unweighted_macro"] = _metric()
    return report


def _write(path, report):
    path.write_text(json.dumps(report), encoding="utf-8")


def _contract(artifact_type, methods, **extra):
    return dict(
        seed=123, artifact_type=artifact_type, run_mode="real",
        methods=methods, batch_size=32, learning_rate=1e-3, **extra,
    )


def _fixture_dir():
    root = Path("tmp_result_validation_fixture")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def test_empty_scenarios_are_not_resumable_for_all_artifacts():
    tmp_path = _fixture_dir()
    reports = [
        (_baseline_report(), _contract("formal_baselines", BASELINES, rounds=10, local_epochs=5, max_epochs=50, patience=8)),
        (_direct_report(), _contract("btd_fl_direct_transfer", ("btd_fl_direct_transfer",), max_epochs=50, patience=8)),
        (_fomo_report(), _contract("fedfomo_style", ("fedfomo_style",), rounds=2, local_epochs=5, max_epochs=50, patience=8)),
    ]
    for index, (report, contract) in enumerate(reports):
        report["scenarios"] = {target: {} for target in CLIENT_NAMES}
        path = tmp_path / f"empty{index}.json"
        _write(path, report)
        assert resumable_result(path, **contract) is False
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.parametrize("artifact_index", [0, 1, 2])
def test_complete_artifacts_are_resumable(artifact_index):
    tmp_path = _fixture_dir()
    reports = [
        (_baseline_report(), _contract("formal_baselines", BASELINES, rounds=10, local_epochs=5, max_epochs=50, patience=8)),
        (_direct_report(), _contract("btd_fl_direct_transfer", ("btd_fl_direct_transfer",), max_epochs=50, patience=8)),
        (_fomo_report(), _contract("fedfomo_style", ("fedfomo_style",), rounds=2, local_epochs=5, max_epochs=50, patience=8)),
    ]
    report, contract = reports[artifact_index]
    path = tmp_path / f"valid{artifact_index}.json"
    _write(path, report)
    assert resumable_result(path, **contract) is True
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_structural_metric_and_protocol_errors_are_not_resumable():
    tmp_path = _fixture_dir()
    report = _baseline_report()
    path = tmp_path / "invalid.json"
    contract = _contract("formal_baselines", BASELINES, rounds=10, local_epochs=5, max_epochs=50, patience=8)
    del report["scenarios"][CLIENT_NAMES[0]]["methods"]["fedavg"]["audit"]
    _write(path, report)
    assert resumable_result(path, **contract) is False
    report = _baseline_report()
    report["scenarios"][CLIENT_NAMES[0]]["methods"]["fedavg"]["validation"]["node_macro"]["mae"] = math.nan
    _write(path, report)
    assert resumable_result(path, **contract) is False
    report = _baseline_report()
    report["rounds"] = 9
    _write(path, report)
    assert resumable_result(path, **contract) is False
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_direct_transfer_requires_exact_three_donors_and_fallback_consistency():
    tmp_path = _fixture_dir()
    report = _direct_report()
    path = tmp_path / "direct.json"
    contract = _contract("btd_fl_direct_transfer", ("btd_fl_direct_transfer",), max_epochs=50, patience=8)
    report["scenarios"][CLIENT_NAMES[0]]["metadata"]["calibration_benefit_by_donor"].pop(CLIENT_NAMES[1])
    _write(path, report)
    assert resumable_result(path, **contract) is False
    report = _direct_report()
    report["scenarios"][CLIENT_NAMES[0]]["metadata"]["zero_transfer_fallback"] = True
    _write(path, report)
    assert resumable_result(path, **contract) is False
    shutil.rmtree(tmp_path, ignore_errors=True)
