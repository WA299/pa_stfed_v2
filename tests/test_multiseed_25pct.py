import json
import shutil
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


def _metric(value: float) -> dict:
    return {split: {scope: {name: value for name in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for split in ("audit", "validation")}


def _write_mock_results(root, seed: int, offset: float) -> None:
    baseline = {
        "seed": seed, "btd_variant": "btd_fl", "methods": ["scarce_local", "fedavg", "fedprox", "fedper", "btd_fl"], "test_evaluated": False,
        "client_grid_names": list(TARGETS),
        "scenarios": {target: {"methods": {name: _metric(offset + index + 1) for index, name in enumerate(("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl"))}} for target in TARGETS},
    }
    direct = {
        "seed": seed, "btd_variant": "btd_fl_direct_transfer", "test_evaluated": False,
        "client_grid_names": list(TARGETS),
        "scenarios": {target: {"metrics": _metric(offset), "metadata": {"selected_donor": TARGETS[(index + 1) % len(TARGETS)], "selected_calibration_benefit": 0.1, "zero_transfer_fallback": False, "calibration_benefit_by_donor": {donor: 0.1 for donor in TARGETS if donor != target}}} for index, target in enumerate(TARGETS)},
    }
    fomo = {
        "seed": seed, "method_name": "FedFomo-style", "test_evaluated": False,
        "client_grid_names": list(TARGETS),
        "scenarios": {target: {"scarce_target_metrics": _metric(offset + 2)} for target in TARGETS},
    }
    root.mkdir(parents=True, exist_ok=True)
    if seed == 42:
        paths = (root.parent / "btd_fl_25pct_seed42.json", root.parent / "btd_fl_direct_transfer_25pct_seed42.json", root.parent / "fedfomo_style_25pct_seed42.json")
    else:
        paths = (root / f"formal_baselines_25pct_seed{seed}.json", root / f"btd_fl_direct_transfer_25pct_seed{seed}.json", root / f"fedfomo_style_25pct_seed{seed}.json")
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
