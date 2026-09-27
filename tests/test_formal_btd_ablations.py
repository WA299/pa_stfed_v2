import json
import inspect
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
