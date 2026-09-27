from pathlib import Path

import numpy as np
import pytest

from code.federated.direct_transfer_btd import run_direct_transfer
from code.federated.formal_btd import CLIENT_NAMES
from code.federated.formal_btd_ablations import load_frozen_reference, _frozen_selected_donors
from code.federated.parameter_groups import parameter_groups
from code.audits.btd_full_backbone_bridge import full_model, build_scarce_target_graph
from code.audits.federated_transfer_benefit import scarce_split
from scripts.run_btd_fl_direct_transfer import _formal_grid_from_client, render_markdown
from scripts.run_federated import build_synthetic_clients


REFERENCE = Path("results/federated/formal_25pct/btd_fl_25pct_seed42.json")


def test_direct_transfer_reference_guardrails_and_raw_state_contract():
    reference = load_frozen_reference(REFERENCE)
    assert reference["btd_variant"] == "btd_fl"
    assert reference["test_evaluated"] is False
    assert reference["canonical_validation_evaluated_during_training"] is False


def test_direct_transfer_synthetic_four_target_smoke():
    clients = build_synthetic_clients(42)
    grids = {client.grid_name: _formal_grid_from_client(client) for client in clients}
    reference = load_frozen_reference(REFERENCE)
    report = run_direct_transfer(grids, reference, device="cpu", max_epochs=1, batch_size=256)
    assert len(report["scenarios"]) == 4
    assert report["adapted_probe_states_used_for_final_initialization"] is False
    assert report["test_evaluated"] is False
    frozen = _frozen_selected_donors(reference)
    for target in CLIENT_NAMES:
        item = report["scenarios"][target]
        metadata = item["metadata"]
        assert len(metadata["calibration_benefit_by_donor"]) == 3
        assert len(metadata["benefit_probe_training_by_donor"]) == 3
        assert metadata["selected_donor"] == frozen[target]
        assert metadata["raw_donor_temporal_source"]
        assert metadata["transferred_parameter_names"]
        assert metadata["target_graph_metadata"]["graph_uses_target_fit_only"] is True
        assert metadata["validation_used_for_selection"] is False
        assert metadata["audit_used_for_selection"] is False


def test_direct_transfer_markdown_contains_probe_and_guardrails():
    report = {
        "client_grid_names": list(CLIENT_NAMES),
        "scenarios": {name: {"metadata": {"selected_donor": None, "selected_calibration_benefit": 0.0, "zero_transfer_fallback": True, "selection_matches_previous_formal_run": False, "calibration_benefit_by_donor": {donor: 0.0 for donor in CLIENT_NAMES if donor != name}}, "metrics": {"audit": {"node_macro": {"mae": 1, "rmse": 1, "wape_pct": 1, "smape_pct": 1}, "grid_aggregate": {"mae": 1, "rmse": 1, "wape_pct": 1, "smape_pct": 1}}, "validation": {"node_macro": {"mae": 1, "rmse": 1, "wape_pct": 1, "smape_pct": 1}, "grid_aggregate": {"mae": 1, "rmse": 1, "wape_pct": 1, "smape_pct": 1}}}} for name in CLIENT_NAMES},
        "four_target_unweighted_macro": {split: {scope: {metric: 1 for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")} for split in ("audit", "validation")},
        "relative_improvement_vs_frozen_comparators": {name: {"audit": 0.0, "validation": 0.0} for name in ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl")},
        "win_counts_vs_frozen_comparators": {name: 0 for name in ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl")},
    }
    text = render_markdown(report)
    assert "benefit probes" in text
    assert "adapted_probe_states_used_for_final_initialization: false" in text
    assert "test_evaluated: false" in text


def test_direct_transfer_final_object_is_temporal_group_only():
    clients = build_synthetic_clients(42)
    grid = _formal_grid_from_client(clients[0])
    model = full_model(grid, build_scarce_target_graph(grid, scarce_split(grid)))
    temporal = set(parameter_groups(model)["temporal"])
    assert temporal
    assert not temporal.intersection({"load_bus_mask", "utility_edge_index", "physical_relation_features", "utility_prior", "selected_utility"})
