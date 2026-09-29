import json
import shutil
from pathlib import Path

import numpy as np

from code.audits.federated_transfer_benefit import scarce_split
from code.federated.result_validation import resumable_result
from code.federated.result_validation import is_historical_25pct
from scripts.run_federated import build_synthetic_clients
from scripts.run_fedfomo_style import _formal_grid_from_client
from scripts.summarize_history_fraction import summarize as summarize_history


def _grid():
    return _formal_grid_from_client(build_synthetic_clients(123)[0])


def test_only_seed42_at_25pct_is_historical():
    assert is_historical_25pct(42, 0.25)
    assert not is_historical_25pct(42, 0.50)
    assert not is_historical_25pct(123, 0.25)


def test_50pct_uses_recent_train_history_and_chronological_60_20_20():
    grid = _grid()
    split = scarce_split(grid, 0.50)
    train = grid.splits["train"]
    expected_start = train.end_index - int(np.ceil(0.50 * (train.end_index - train.start_index)))
    assert split.available_start == expected_start
    assert split.available_end == train.end_index
    assert split.eligible_indices[0] >= expected_start + 168
    assert split.eligible_indices[-1] < train.end_index
    assert len(split.fit_indices) == int(0.60 * len(split.eligible_indices))
    assert len(split.calibration_indices) == int(0.80 * len(split.eligible_indices)) - int(0.60 * len(split.eligible_indices))
    assert len(split.audit_indices) == len(split.eligible_indices) - len(split.fit_indices) - len(split.calibration_indices)
    assert split.fit_indices[-1] < split.calibration_indices[0] < split.audit_indices[0]


def test_25pct_indices_remain_unchanged():
    grid = _grid()
    split = scarce_split(grid, 0.25)
    train = grid.splits["train"]
    expected_start = train.end_index - int(np.ceil(0.25 * (train.end_index - train.start_index)))
    assert split.available_start == expected_start
    assert split.eligible_indices[0] >= expected_start + 168


def test_history_fraction_is_part_of_resume_contract():
    report = {
        "run_mode": "real", "seed": 123, "artifact_type": "formal_baselines",
        "history_fraction": 0.50, "batch_size": 32, "learning_rate": 1e-3,
        "test_evaluated": False, "rounds": 10, "local_epochs": 5,
        "max_epochs": 50, "patience": 8,
        "client_grid_names": [
            "39_bus_semi_urban_reference_grid", "50_bus_rural_reference_grid",
            "56_bus_semi_urban_reference_grid", "80_bus_rural_reference_grid",
        ],
        "methods": ["scarce_local", "fedavg", "fedprox", "fedper"],
        "scenarios": {},
    }
    path = Path("tmp_history_fraction_resume_result.json")
    path.write_text(json.dumps(report), encoding="utf-8")
    try:
        assert not resumable_result(path, seed=123, artifact_type="formal_baselines", run_mode="real", methods=("scarce_local", "fedavg", "fedprox", "fedper"), rounds=10, local_epochs=5, batch_size=32, learning_rate=1e-3, max_epochs=50, patience=8, history_fraction=0.25)
    finally:
        path.unlink(missing_ok=True)


def test_50pct_reference_contract_rejects_25pct_artifacts():
    report = {
        "run_mode": "real", "seed": 42, "artifact_type": "formal_baselines",
        "history_fraction": 0.25, "batch_size": 32, "learning_rate": 1e-3,
        "test_evaluated": False, "client_grid_names": [], "methods": [], "scenarios": {},
    }
    path = Path("tmp_25pct_reference_for_50pct.json")
    path.write_text(json.dumps(report), encoding="utf-8")
    try:
        assert not resumable_result(path, seed=42, artifact_type="formal_baselines", run_mode="real", methods=("scarce_local", "fedavg", "fedprox", "fedper"), history_fraction=0.50)
    finally:
        path.unlink(missing_ok=True)


def test_cross_history_summary_reads_both_fractions_exactly():
    methods = ("scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer")
    def report(fraction):
        metrics = {}
        for method in methods:
            metrics[method] = {}
            for split in ("audit", "validation"):
                metrics[method][split] = {"node_macro": {"mae": {"mean": fraction, "std": 0.1, "per_seed": {"42": fraction, "123": fraction, "2026": fraction}}}}
        return {
            "history_fraction": fraction,
            "metrics": metrics,
            "btd_vs_comparators": {
                "scarce_local": {"mean_relative_improvement": fraction, "std_relative_improvement": 0.1, "seeds_btd_wins": 2, "scarce_target_validation_wins_out_of_12": 8},
                "fedavg": {"mean_relative_improvement": fraction, "std_relative_improvement": 0.1, "seeds_btd_wins": 2, "scarce_target_validation_wins_out_of_12": 8},
                "fedprox": {"mean_relative_improvement": fraction, "std_relative_improvement": 0.1, "seeds_btd_wins": 2, "scarce_target_validation_wins_out_of_12": 8},
                "fedper": {"mean_relative_improvement": fraction, "std_relative_improvement": 0.1, "seeds_btd_wins": 2, "scarce_target_validation_wins_out_of_12": 8},
                "fedfomo_style": {"mean_relative_improvement": fraction, "std_relative_improvement": 0.1, "seeds_btd_wins": 2, "scarce_target_validation_wins_out_of_12": 8},
            },
            "donor_selection_robustness": {"positive_directed_benefit_count": 1, "non_positive_directed_benefit_count": 2, "fallback_occurrence_count": 0},
        }
    old, new = Path("tmp_history_summary_old.json"), Path("tmp_history_summary_new.json")
    old.write_text(json.dumps(report(0.25)), encoding="utf-8")
    new.write_text(json.dumps(report(0.5)), encoding="utf-8")
    try:
        result = summarize_history(old, new)
        assert result["metrics"]["btd_fl_direct_transfer"]["validation"]["25pct"]["mean"] == 0.25
        assert result["metrics"]["btd_fl_direct_transfer"]["validation"]["50pct"]["mean"] == 0.5
        assert result["btd_vs_comparators"]["fedavg"]["50pct"]["target_x_seed_validation_wins"] == 8
    finally:
        old.unlink(missing_ok=True)
        new.unlink(missing_ok=True)
