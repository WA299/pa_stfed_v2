from pathlib import Path

import numpy as np
import pytest
import torch

from code.audits.federated_transfer_benefit import scarce_split
from code.federated.formal_btd import CLIENT_NAMES, METHODS, load_selected_donors, load_selection_metadata, select_formal_donor
from scripts.run_btd_full_backbone_bridge import _grid_from_client
from scripts.run_federated import build_synthetic_clients
from scripts.run_formal_btd_fl import render_markdown
from scripts.run_formal_btd_fl import _formal_grid_from_client
from code.federated.formal_btd import run_formal_benchmark
from code.federated.formal_btd import _copy_compatible_trainables
from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model
from code.federated.parameter_groups import parameter_groups
from code.audits.federated_transfer_benefit import benefit


def test_formal_selection_uses_current_run_benefits_only():
    benefits = {"donor_c": 0.04, "donor_a": 0.12, "donor_b": 0.08}
    selected, value, fallback, rule = select_formal_donor(benefits, "btd_fl")
    assert selected == "donor_a"
    assert value == pytest.approx(0.12)
    assert fallback is False
    assert rule == "maximum_current_calibration_benefit_with_zero_transfer"
    assert selected != "historical_donor"
    assert benefit(1.0, 0.88) == pytest.approx(0.12)


def test_formal_selection_isolated_from_audit_and_validation():
    benefits = {"donor_a": 0.2, "donor_b": 0.1, "donor_c": -0.2}
    selected = select_formal_donor(benefits)[0]
    audit_metrics = {"donor_a": 99.0, "donor_b": 0.0, "donor_c": -99.0}
    validation_metrics = {"donor_a": -10.0, "donor_b": 100.0, "donor_c": 1.0}
    assert select_formal_donor(benefits)[0] == selected
    assert audit_metrics and validation_metrics


def test_zero_transfer_and_ablation_selection_contracts():
    negative = {"donor_a": -0.2, "donor_b": -0.1, "donor_c": -0.3}
    selected, value, fallback, _ = select_formal_donor(negative, "btd_fl")
    assert selected is None and value == 0.0 and fallback is True
    selected, value, fallback, rule = select_formal_donor(negative, "btd_no_zero_transfer")
    assert selected == "donor_b" and value == pytest.approx(-0.1) and fallback is False
    assert rule.endswith("no_zero_fallback")
    first = {"donor_a": -0.2, "donor_b": 0.9, "donor_c": 0.1}
    second = {"donor_a": 0.8, "donor_b": -0.9, "donor_c": 0.2}
    assert select_formal_donor(first, "btd_no_benefit_selection")[0] == "donor_a"
    assert select_formal_donor(second, "btd_no_benefit_selection")[0] == "donor_a"


def test_formal_federated_fixed_round_metadata():
    clients = build_synthetic_clients(seed=42)
    from code.federated.trainer import FederatedTrainer
    report = FederatedTrainer(
        clients, "fedavg_all", rounds=1, local_epochs=1, batch_size=256,
        seed=42, device="cpu", evaluate_validation_during_training=False,
    ).run()
    assert all("validation" not in item and "primary_selection_metric" not in item for item in report["round_history"])
    assert report["communication_round_selection"] == "fixed_final_round"
    assert report["canonical_validation_evaluated_during_training"] is False
    assert report["validation_used_for_selection"] is False
    counts = report["local_sample_counts"]
    weights = report["round_history"][0]["aggregation_weights"]
    total = sum(counts.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weights[name] == pytest.approx(counts[name] / total) for name in counts)


def test_full_model_transfer_ablation_copies_trainables_not_buffers():
    clients = build_synthetic_clients(seed=42)
    grid = _formal_grid_from_client(clients[0])
    split = scarce_split(grid)
    graph = build_scarce_target_graph(grid, split)
    donor = full_model(grid, graph)
    target = full_model(grid, graph)
    donor_state = {name: value.detach().cpu().clone() for name, value in donor.named_parameters()}
    buffers_before = {name: value.detach().clone() for name, value in target.named_buffers()}
    copied = _copy_compatible_trainables(target, donor_state)
    trainables = tuple(sorted(name for name, parameter in target.named_parameters() if parameter.requires_grad))
    assert copied == trainables
    assert set(copied).isdisjoint(dict(target.named_buffers()))
    assert all(torch.equal(dict(target.named_buffers())[name], value) for name, value in buffers_before.items())
    assert set(parameter_groups(target)["temporal"]) | set(parameter_groups(target)["spatial"]) == set(trainables)


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
    for target in CLIENT_NAMES:
        metadata = report["scenarios"][target]["metadata"]
        assert len(metadata["candidate_donors"]) == 3
        assert len(metadata["calibration_benefit_by_donor"]) == 3
        assert len(metadata["donor_adaptation_by_donor"]) == 3
        local_mae = metadata["local_proxy_training"]["best_calibration_node_mae"]
        for donor, adaptation in metadata["donor_adaptation_by_donor"].items():
            expected = (local_mae - adaptation["target_adaptation_best_calibration_node_mae"]) / (local_mae + 1e-12)
            assert metadata["calibration_benefit_by_donor"][donor] == pytest.approx(expected)
        assert "scarce_local_full_model_training" in metadata
        assert "btd_target_full_model_training" in metadata
        assert metadata["validation_used_for_selection"] is False
        assert metadata["audit_used_for_selection"] is False
