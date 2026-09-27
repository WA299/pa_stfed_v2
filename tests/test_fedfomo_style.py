from pathlib import Path

import pytest

from code.federated.fedfomo_style import (
    fedfomo_raw_weight,
    normalize_positive_weights,
    parameter_l2_distance,
)
from scripts.run_fedfomo_style import _formal_grid_from_client
from scripts.run_federated import build_synthetic_clients
from code.federated.formal_btd_ablations import load_frozen_reference
from code.federated.formal_btd import CLIENT_NAMES


def test_fedfomo_raw_weight_exact():
    assert fedfomo_raw_weight(2.0, 1.5, 2.0, 1e-12) == pytest.approx(0.25)


def test_fedfomo_clips_and_normalizes_positive_weights():
    normalized, fallback = normalize_positive_weights({"a": 2.0, "b": -1.0, "c": 1.0})
    assert fallback is False
    assert normalized == {"a": pytest.approx(2 / 3), "c": pytest.approx(1 / 3)}
    assert sum(normalized.values()) == pytest.approx(1.0)


def test_fedfomo_zero_positive_peer_fallback():
    normalized, fallback = normalize_positive_weights({"a": -1.0, "b": 0.0})
    assert normalized == {}
    assert fallback is True


def test_parameter_distance_uses_trainable_parameter_map_only():
    import torch
    left = {"weight": torch.tensor([1.0, 2.0])}
    right = {"weight": torch.tensor([2.0, 4.0])}
    assert parameter_l2_distance(left, right) == pytest.approx(5 ** 0.5)


def test_client_contract_has_four_scenarios():
    assert len(CLIENT_NAMES) == 4


def test_synthetic_fedfomo_scenario_has_scarce_peer_diagnostics():
    from code.federated.fedfomo_style import run_fedfomo
    clients = build_synthetic_clients(42)
    grids = {client.grid_name: _formal_grid_from_client(client) for client in clients}
    report = run_fedfomo(
        grids, load_frozen_reference(Path("results/federated/formal_25pct/btd_fl_25pct_seed42.json")),
        rounds=1, local_epochs=1, batch_size=256, device="cpu",
    )
    diagnostics = report["scenarios"][CLIENT_NAMES[0]]["round_history"][0]["diagnostics"][CLIENT_NAMES[0]]
    assert len(diagnostics["peer_calibration_loss_by_donor"]) == 3
    assert len(diagnostics["raw_weight_by_donor"]) == 3
    assert len(diagnostics["normalized_weight_by_donor"]) <= 3
    assert report["validation_used_for_peer_weighting"] is False
    assert report["audit_used_for_peer_weighting"] is False
    assert report["test_evaluated"] is False
