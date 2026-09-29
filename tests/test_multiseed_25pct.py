import json
import shutil
import inspect
from pathlib import Path

import numpy as np
import pytest

from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model
from code.audits.federated_transfer_benefit import scarce_split
from code.federated.seeding import set_global_seed
from scripts.run_federated import build_synthetic_clients
from scripts.run_fedfomo_style import _formal_grid_from_client
from scripts.run_multiseed_25pct import DEFAULT_SEEDS
from scripts.summarize_multiseed_25pct import METHODS, TARGETS, summarize
from code.federated.result_validation import resumable_result
from code.federated.formal_btd import run_formal_baselines


def _metric(value: float) -> dict:
    return {split: {scope: {name: value for name in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for split in ("audit", "validation")}


def _write_mock_results(root, seed: int, offset: float, history_fraction: float = 0.25) -> None:
    percent = int(round(history_fraction * 100))
    baseline = {
        "seed": seed, "run_mode": "real", "artifact_type": "formal_baselines", "history_fraction": history_fraction, "btd_variant": "btd_fl", "methods": ["scarce_local", "fedavg", "fedprox", "fedper"] if not (seed == 42 and history_fraction == 0.25) else ["scarce_local", "fedavg", "fedprox", "fedper", "btd_fl"], "test_evaluated": False,
        "client_grid_names": list(TARGETS),
        "scenarios": {target: {"methods": {name: _metric(offset + index + 1) for index, name in enumerate(("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl"))}} for target in TARGETS},
    }
    direct = {
        "seed": seed, "run_mode": "real", "artifact_type": "btd_fl_direct_transfer", "history_fraction": history_fraction, "methods": ["btd_fl_direct_transfer"], "btd_variant": "btd_fl_direct_transfer", "test_evaluated": False,
        "client_grid_names": list(TARGETS),
        "scenarios": {target: {"metrics": _metric(offset), "metadata": {"selected_donor": TARGETS[(index + 1) % len(TARGETS)], "selected_calibration_benefit": 0.1, "zero_transfer_fallback": False, "calibration_benefit_by_donor": {donor: 0.1 for donor in TARGETS if donor != target}}} for index, target in enumerate(TARGETS)},
    }
    fomo = {
        "seed": seed, "run_mode": "real", "artifact_type": "fedfomo_style", "history_fraction": history_fraction, "methods": ["fedfomo_style"], "method_name": "FedFomo-style", "test_evaluated": False,
        "client_grid_names": list(TARGETS),
        "scenarios": {target: {"scarce_target_metrics": _metric(offset + 2)} for target in TARGETS},
    }
    root.mkdir(parents=True, exist_ok=True)
    if seed == 42 and history_fraction == 0.25:
        paths = (root.parent / "btd_fl_25pct_seed42.json", root.parent / "btd_fl_direct_transfer_25pct_seed42.json", root.parent / "fedfomo_style_25pct_seed42.json")
    else:
        paths = (root / f"formal_baselines_{percent}pct_seed{seed}.json", root / f"btd_fl_direct_transfer_{percent}pct_seed{seed}.json", root / f"fedfomo_style_{percent}pct_seed{seed}.json")
    for path, data in zip(paths, (baseline, direct, fomo)):
        path.write_text(json.dumps(data), encoding="utf-8")


def test_default_training_seeds_exclude_historical_seed42():
    assert DEFAULT_SEEDS == (123, 2026)
    assert 42 not in DEFAULT_SEEDS


def test_seeded_full_model_initialization_is_reproducible_and_varies_across_seeds():
    client = build_synthetic_clients(42)[0]
    grid = _formal_grid_from_client(client)
    graph = build_scarce_target_graph(grid, scarce_split(grid))
    a = full_model(grid, graph, seed=123)
    b = full_model(grid, graph, seed=123)
    c = full_model(grid, graph, seed=2026)
    assert all(np.array_equal(x.detach().cpu().numpy(), y.detach().cpu().numpy()) for x, y in zip(a.parameters(), b.parameters()))
    assert any(not np.array_equal(x.detach().cpu().numpy(), y.detach().cpu().numpy()) for x, y in zip(a.parameters(), c.parameters()))


def test_multiseed_summary_macro_std_and_pairwise_arithmetic():
    root = Path("tmp_multiseed_summary_fixture") / "formal_25pct" / "multiseed"
    shutil.rmtree(root, ignore_errors=True)
    # The historical seed is represented by the accepted schema but is still read-only.
    _write_mock_results(root, 42, 1.0)
    _write_mock_results(root, 123, 2.0)
    _write_mock_results(root, 2026, 3.0)
    report = summarize(root)
    values = report["metrics"]["btd_fl_direct_transfer"]["validation"]["node_macro"]["mae"]
    assert values["per_seed"] == {"42": 1.0, "123": 2.0, "2026": 3.0}
    assert values["mean"] == pytest.approx(2.0)
    assert values["std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    pair = report["btd_vs_comparators"]["scarce_local"]
    assert pair["scarce_target_validation_wins_out_of_12"] == 12
    assert pair["seeds_btd_wins"] == 3
    shutil.rmtree(root, ignore_errors=True)


def test_summary_requires_all_methods_and_scenarios():
    root = Path("tmp_multiseed_summary_fixture_broken") / "formal_25pct" / "multiseed"
    shutil.rmtree(root, ignore_errors=True)


def test_50pct_summary_accepts_seed42_baseline_only_schema_for_all_seeds():
    root = Path("tmp_multiseed_summary_fixture_50pct") / "formal_50pct" / "multiseed"
    shutil.rmtree(root, ignore_errors=True)
    for seed, offset in ((42, 1.0), (123, 2.0), (2026, 3.0)):
        _write_mock_results(root, seed, offset, history_fraction=0.50)
    report = summarize(root, history_fraction=0.50)
    assert report["history_fraction"] == 0.50
    assert report["guardrails"]["seed42_source"] == "current_50pct_run"
    shutil.rmtree(root, ignore_errors=True)
    _write_mock_results(root, 42, 1.0)
    _write_mock_results(root, 123, 2.0)
    _write_mock_results(root, 2026, 3.0)
    broken = root / "formal_baselines_25pct_seed123.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    del payload["scenarios"][TARGETS[0]]
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        summarize(root)
    shutil.rmtree(root, ignore_errors=True)


def test_synthetic_and_protocol_mismatch_never_resume():
    root = Path("tmp_resume_validation_fixture")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    payload = {
        "run_mode": "synthetic_smoke", "artifact_type": "formal_baselines", "seed": 123,
        "history_fraction": 0.25, "rounds": 1, "local_epochs": 1,
        "batch_size": 256, "learning_rate": 1e-3, "max_epochs": 1,
        "patience": 8, "test_evaluated": False, "client_grid_names": list(TARGETS),
        "methods": ["scarce_local", "fedavg", "fedprox", "fedper"],
        "scenarios": {target: {} for target in TARGETS},
    }
    path = root / "formal_baselines_25pct_seed123.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = dict(seed=123, artifact_type="formal_baselines", run_mode="real", methods=("scarce_local", "fedavg", "fedprox", "fedper"), rounds=10, local_epochs=5, batch_size=32, learning_rate=1e-3, max_epochs=50, patience=8)
    assert resumable_result(path, **contract) is False
    payload["run_mode"] = "real"; payload["rounds"] = 10; payload["local_epochs"] = 5; payload["batch_size"] = 32; payload["max_epochs"] = 50
    payload["scenarios"] = {
        target: {"methods": {name: _metric(1.0) for name in ("scarce_local", "fedavg", "fedprox", "fedper")}}
        for target in TARGETS
    }
    payload["four_scenario_macro"] = {
        split: {name: {scope: {metric: 1.0 for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for name in ("scarce_local", "fedavg", "fedprox", "fedper")}
        for split in ("audit", "validation")
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert resumable_result(path, **contract) is True
    shutil.rmtree(root, ignore_errors=True)


def test_baseline_only_entry_point_has_no_btd_training_calls():
    source = inspect.getsource(run_formal_baselines)
    assert "_train_full_donor_states" not in source
    assert "select_formal_donor" not in source
    assert "donor_split" not in source
