from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.federated.industrial_external import (
    BASELINE_METHODS,
    EXTERNAL_CLIENT_NAMES,
    REFERENCE_CLIENT_NAMES,
    sample_count_weights,
    select_external_donor,
)
from code.federated.industrial_result_validation import resumable_external_result


def test_external_client_contract_and_donor_selection():
    assert len(REFERENCE_CLIENT_NAMES) == 4
    assert EXTERNAL_CLIENT_NAMES[-1] == "norway_industrial_mvlv"
    assert select_external_donor({name: index / 10 for index, name in enumerate(REFERENCE_CLIENT_NAMES)}) == (REFERENCE_CLIENT_NAMES[-1], 0.3, False)
    assert select_external_donor({name: -0.1 for name in REFERENCE_CLIENT_NAMES}) == (None, 0.0, True)


def test_external_sample_count_weights():
    weights = sample_count_weights({"a": 10, "b": 20, "c": 70, "target": 5})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["b"] == pytest.approx(20 / 105)


def test_external_baseline_methods_are_exact():
    assert BASELINE_METHODS == ("industrial_scarce_local", "fedavg", "fedprox", "fedper")


def test_incomplete_external_artifact_does_not_resume():
    path = Path(".industrial_incomplete_result_test.json")
    path.write_text(json.dumps({"run_mode": "real", "seed": 123, "history_fraction": 0.25, "scenarios": {}}))
    try:
        assert not resumable_external_result(
            path,
            artifact_type="industrial_baselines",
            seed=123,
            fraction=0.25,
            run_mode="real",
            rounds=10,
            local_epochs=5,
            max_epochs=50,
            batch_size=32,
        )
    finally:
        path.unlink(missing_ok=True)
