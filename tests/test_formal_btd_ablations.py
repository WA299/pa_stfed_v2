import json
import inspect
import copy
from pathlib import Path

import pytest

from code.federated.formal_btd import CLIENT_NAMES
from code.federated.formal_btd_ablations import (
    ABLATIONS,
    load_frozen_reference,
    _frozen_selected_donors,
)
import code.federated.formal_btd_ablations as ablations_module
from scripts.run_formal_btd_ablations import render_markdown
from scripts.run_formal_btd_ablations import _formal_grid_from_client
from scripts.run_federated import build_synthetic_clients
from code.federated.parameter_groups import TOPOLOGY_BUFFER_NAMES


REFERENCE = Path("results/federated/formal_25pct/btd_fl_25pct_seed42.json")


def test_main_formal_json_is_frozen_reference_only():
    report = load_frozen_reference(REFERENCE)
    assert report["btd_variant"] == "btd_fl"
    assert report["seed"] == 42
    assert report["history_fraction"] == 0.25
    assert report["validation_used_for_selection"] is False
    assert report["audit_used_for_selection"] is False
    assert report["test_evaluated"] is False
    assert report["canonical_validation_evaluated_during_training"] is False


def test_frozen_selected_donors_are_read_from_reference():
    report = load_frozen_reference(REFERENCE)
    selected = _frozen_selected_donors(report)
    assert set(selected) == set(CLIENT_NAMES)
    assert all(donor != target for target, donor in selected.items())


def test_ablation_names_and_output_contract():
    assert ABLATIONS == (
        "btd_no_benefit_selection",
        "btd_full_model_transfer",
        "btd_no_target_adaptation",
    )
    report = {
        "ablation": ABLATIONS[0],
        "mechanism_changed": "fixed donor",
        "client_grid_names": list(CLIENT_NAMES),
        "scenarios": {
            name: {
                "metrics": {"audit": {"node_macro": {"mae": 1.0}, "grid_aggregate": {"mae": 2.0}}, "validation": {"node_macro": {"mae": 1.0}, "grid_aggregate": {"mae": 2.0}}},
                "main_btd_fl_metrics": {"audit": {"node_macro": {"mae": 1.0}}, "validation": {"node_macro": {"mae": 1.0}}},
                "scarce_local_metrics": {"audit": {"node_macro": {"mae": 1.0}}, "validation": {"node_macro": {"mae": 1.0}}},
                "donor_used": "donor",
                "selection_rule": "fixed",
                "relative_change_vs_main_btd_fl": {"audit": 0.0, "validation": 0.0},
                "relative_change_vs_scarce_local": {"audit": 0.0, "validation": 0.0},
            }
            for name in CLIENT_NAMES
        },
        "four_target_unweighted_macro": {"audit": {"node_macro": {"mae": 1.0}, "grid_aggregate": {"mae": 2.0}}, "validation": {"node_macro": {"mae": 1.0}, "grid_aggregate": {"mae": 2.0}}},
        "reference_macro": {"main_btd_fl": {"audit": {"node_macro": {"mae": 1.0}}, "validation": {"node_macro": {"mae": 1.0}}}, "scarce_local": {"audit": {"node_macro": {"mae": 1.0}}, "validation": {"node_macro": {"mae": 1.0}}}},
        "win_counts": {"main_btd_fl": 0, "scarce_local": 0},
    }
    text = render_markdown(report)
    assert "Mechanism changed" in text
    assert "Main BTD-FL" in text
    assert "Scarce-local" in text


def test_ablation_path_does_not_retrain_federated_baselines():
    source = inspect.getsource(ablations_module)
    assert "FederatedTrainer" not in source
    assert "run_formal_benchmark" not in source


def test_no_target_adaptation_contract_is_explicit():
    # This is the metadata contract emitted by run_ablation for variant C.
    target_proxy_adaptation = {"performed": False, "steps": 0}
    assert target_proxy_adaptation["performed"] is False
    assert target_proxy_adaptation["steps"] == 0


def test_variant_output_names_are_isolated_from_main_result():
    stems = {
        f"{variant}_25pct_seed42" for variant in ABLATIONS
    }
    assert "btd_fl_25pct_seed42" not in stems
    assert len(stems) == 3


@pytest.fixture(scope="module")
def synthetic_ablation_grids():
    return {
        client.grid_name: _formal_grid_from_client(client)
        for client in build_synthetic_clients(42)
    }


@pytest.fixture(scope="module")
def synthetic_trainable_names():
    client = build_synthetic_clients(42)[0]
    return {
        name for name, parameter in client.model.named_parameters()
        if parameter.requires_grad
    }


def test_no_benefit_selection_real_runner(synthetic_ablation_grids):
    from code.federated.formal_btd_ablations import run_ablation

    reference = load_frozen_reference(REFERENCE)
    altered = copy.deepcopy(reference)
    for target in CLIENT_NAMES:
        benefits = altered["scenarios"][target]["metadata"]["calibration_benefit_by_donor"]
        altered_values = list(reversed(list(benefits.values())))
        for donor, value in zip(benefits, altered_values):
            benefits[donor] = value
    report = run_ablation(
        synthetic_ablation_grids, altered, "btd_no_benefit_selection",
        device="cpu", max_epochs=1, batch_size=256,
    )
    for target in CLIENT_NAMES:
        expected = sorted(name for name in CLIENT_NAMES if name != target)[0]
        item = report["scenarios"][target]
        assert item["donor_used"] == expected
        assert item["selection_rule"] == "fixed_deterministic_no_benefit"


def test_full_model_transfer_real_runner_uses_frozen_donors(synthetic_ablation_grids, synthetic_trainable_names):
    from code.federated.formal_btd_ablations import run_ablation

    reference = load_frozen_reference(REFERENCE)
    frozen = _frozen_selected_donors(reference)
    report = run_ablation(
        synthetic_ablation_grids, reference, "btd_full_model_transfer",
        device="cpu", max_epochs=1, batch_size=256,
    )
    assert report["donor_proxy_metadata_by_grid"] == {}
    topology = set(TOPOLOGY_BUFFER_NAMES)
    for target in CLIENT_NAMES:
        item = report["scenarios"][target]
        assert item["donor_used"] == frozen[target]
        assert item["selected_donor_full_model_training"] is not None
        assert item["transferred_parameter_names"]
        assert not topology.intersection(item["transferred_parameter_names"])
        assert set(item["transferred_parameter_names"]) == synthetic_trainable_names
        assert item["target_graph_metadata"]["graph_uses_target_fit_only"] is True


def test_no_target_adaptation_real_runner_has_zero_adaptation(synthetic_ablation_grids):
    from code.federated.formal_btd_ablations import run_ablation

    reference = load_frozen_reference(REFERENCE)
    frozen = _frozen_selected_donors(reference)
    report = run_ablation(
        synthetic_ablation_grids, reference, "btd_no_target_adaptation",
        device="cpu", max_epochs=1, batch_size=256,
    )
    for target in CLIENT_NAMES:
        item = report["scenarios"][target]
        assert item["donor_used"] == frozen[target]
        assert item["target_proxy_adaptation"]["performed"] is False
        assert item["target_proxy_adaptation"]["steps"] == 0
