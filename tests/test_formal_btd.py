from pathlib import Path

import numpy as np
import pytest

from code.audits.federated_transfer_benefit import scarce_split
from code.federated.formal_btd import CLIENT_NAMES, METHODS, load_selected_donors, load_selection_metadata
from scripts.run_btd_full_backbone_bridge import _grid_from_client
from scripts.run_federated import build_synthetic_clients
from scripts.run_formal_btd_fl import render_markdown
from scripts.run_formal_btd_fl import _formal_grid_from_client
from code.federated.formal_btd import run_formal_benchmark


def test_leave_one_scarce_history_views_are_target_only():
    clients = build_synthetic_clients(42)
    grids = {client.grid_name: _grid_from_client(client) for client in clients}
    for target in CLIENT_NAMES:
        split = scarce_split(grids[target])
        assert split.fit_indices[0] >= split.available_start + 168
        assert split.audit_indices[-1] < grids[target].splits["train"].end_index
        assert all(np.diff(split.eligible_indices) == 1)


def test_audit_mapping_isolation_and_candidates_exclude_target():
    donors = load_selected_donors(Path("results/audits/federated_transfer_benefit_25pct.json"))
    assert set(donors) == set(CLIENT_NAMES)
    for target, donor in donors.items():
        assert donor is None or donor != target
    metadata = load_selection_metadata(Path("results/audits/federated_transfer_benefit_25pct.json"))
    for target, item in metadata.items():
        assert target not in item["candidate_donors"]
        assert set(item["calibration_benefit_by_donor"]) == set(item["candidate_donors"])


def test_formal_method_contract_and_macro_arithmetic():
    assert METHODS == ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl")
    values = {name: index + 1.0 for index, name in enumerate(CLIENT_NAMES)}
    macro = float(np.mean(list(values.values())))
    assert macro == pytest.approx(2.5)
    wins = sum(values[name] < 10 for name in CLIENT_NAMES)
    assert wins == 4


def test_markdown_rendering_contract():
    report = {
        "client_grid_names": list(CLIENT_NAMES),
        "scenarios": {name: {"methods": {method: {"validation": {"node_macro": {"mae": 1.0}}, "audit": {"node_macro": {"mae": 1.0}}} for method in METHODS}, "metadata": {"selected_donor": None, "candidate_donors": [], "target_graph_metadata": {"graph_fit_target_count": 1, "graph_selection_target_count": 1, "selected_edge_count": 0}}} for name in CLIENT_NAMES},
        "four_scenario_macro": {"audit": {method: {"node_macro": {"mae": 1.0}} for method in METHODS}, "validation": {method: {"node_macro": {"mae": 1.0}} for method in METHODS}},
        "btd_fl_win_counts": {method: 0 for method in METHODS[:-1]},
        "validation_used_for_selection": False, "audit_used_for_selection": False, "test_evaluated": False,
    }
    text = render_markdown(report)
    assert "BTD-FL" in text and "FedProx" in text and "Selected Donor" in text


def test_tiny_formal_leave_one_scarce_smoke_completes_all_methods():
    grids = {client.grid_name: _formal_grid_from_client(client) for client in build_synthetic_clients(42)}
    donors = load_selected_donors(Path("results/audits/federated_transfer_benefit_25pct.json"))
    report = run_formal_benchmark(grids, donors, device="cpu", rounds=1, local_epochs=1, max_epochs=1, batch_size=256)
    assert set(report["methods"]) == set(METHODS)
    assert len(report["scenarios"]) == 4
    assert report["test_evaluated"] is False
    assert set(report["relative_btd_fl_improvement_vs"]) == set(METHODS[0:4])
    assert set(report["btd_fl_win_counts"]) == set(METHODS[0:4])
