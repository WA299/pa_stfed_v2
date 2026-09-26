from types import SimpleNamespace
import numpy as np
import torch
import pytest
from pathlib import Path

from code.audits.federated_transfer_benefit import (
    HISTORY, benefit, donor_split, fit_fit_only_scaler, make_proxy, scarce_split,
    select_donor, transfer_temporal_parameters, IndexDataset,
)
from code.federated.parameter_groups import parameter_groups
from scripts import run_federated_transfer_benefit as runner

def _grid(nodes=3, train_end=1200):
    rng=np.random.default_rng(3); total=train_end+80; dynamic=rng.normal(size=(total,nodes,7)).astype(np.float32)
    return SimpleNamespace(num_nodes=nodes,load_bus_mask=np.ones(nodes,dtype=bool),dynamic_features=dynamic,p=dynamic[:,:,0].copy(),timestamps=np.arange(total),splits={'train':SimpleNamespace(start_index=100,end_index=train_end),'validation':SimpleNamespace(start_index=train_end,end_index=train_end+40),'test':SimpleNamespace(start_index=train_end+40,end_index=total)})

def test_scarcity_recent_raw_train_and_chronological_disjoint():
    grid=_grid(); split=scarce_split(grid)
    expected=train_end=grid.splits['train'].end_index; assert split.available_start == train_end-int(np.ceil(.25*(train_end-grid.splits['train'].start_index))); assert split.eligible_indices[0] >= split.available_start+HISTORY; assert split.eligible_indices[-1] < train_end
    assert len(set(split.fit_indices)&set(split.calibration_indices))==0 and len(set(split.calibration_indices)&set(split.audit_indices))==0
    assert np.array_equal(np.concatenate([split.fit_indices, split.calibration_indices, split.audit_indices]), split.eligible_indices)
    assert all(IndexDataset(grid, split.eligible_indices)[i]["x"].shape[0] == HISTORY for i in range(len(split.eligible_indices)))
    assert all(IndexDataset(grid, split.eligible_indices)[i]["history_start_index"] >= split.available_start for i in range(len(split.eligible_indices)))

def test_fit_only_scaler_and_benefit_selection():
    grid=_grid(); split=scarce_split(grid); scaler=fit_fit_only_scaler(grid,split.fit_indices); assert scaler.fit_split=='train_fit_only'; assert scaler.fit_end_index == int(split.fit_indices.max())+1; assert benefit(10,8) == pytest.approx(.2); assert select_donor({'a':-.1,'b':0}) == (None,0.0); assert select_donor({'a':-.1,'b':.2}) == ('b',.2)
    donor_fit, donor_cal = donor_split(grid)
    assert donor_fit[-1] < donor_cal[0] < grid.splits["train"].end_index
    assert fit_fit_only_scaler(grid, donor_fit).fit_end_index <= donor_cal[0]

def test_temporal_transfer_only_preserves_topology_and_excludes_spatial():
    target, donor = _grid(3), _grid(5)
    target_model, donor_model = make_proxy(target), make_proxy(donor)
    groups = parameter_groups(target_model)
    spatial_before = {name: dict(target_model.named_parameters())[name].clone() for name in groups["spatial"]}
    buffers_before = {name: value.clone() for name, value in target_model.named_buffers()}
    with torch.no_grad():
        for name in groups["temporal"]:
            dict(donor_model.named_parameters())[name].add_(2.0)
    names = transfer_temporal_parameters(target_model, donor_model)
    assert set(names) == set(groups["temporal"])
    assert all(torch.equal(dict(target_model.named_parameters())[name], dict(donor_model.named_parameters())[name]) for name in names)
    assert all(torch.equal(dict(target_model.named_parameters())[name], spatial_before[name]) for name in groups["spatial"])
    assert all(torch.equal(value, buffers_before[name]) for name, value in target_model.named_buffers())


def test_temporal_only_output_ignores_spatial_parameters_and_topology_buffers():
    grid = _grid(4); model = make_proxy(grid); x = torch.randn(2, HISTORY, 4, 6)
    before = model(x).detach().clone()
    spatial = parameter_groups(model)["spatial"]
    with torch.no_grad():
        for name in spatial:
            dict(model.named_parameters())[name].add_(torch.randn_like(dict(model.named_parameters())[name]) * 10)
        model.physical_relation_features.fill_(99)
        model.utility_prior.fill_(7)
        model.selected_utility.fill_(11)
    torch.testing.assert_close(model(x), before)


def test_selection_uses_calibration_only_and_nonpositive_falls_back():
    calibration = {"donor_a": 0.1, "donor_b": 0.2, "donor_c": -0.4}
    selected = select_donor(calibration)
    assert selected == ("donor_b", 0.2)
    audit = {"donor_a": 100.0, "donor_b": -100.0, "donor_c": 500.0}
    validation = {"donor_a": -100.0, "donor_b": -200.0, "donor_c": 900.0}
    assert select_donor(calibration) == selected
    assert audit and validation
    assert select_donor({"a": 0.0, "b": -0.1}) == (None, 0.0)


@pytest.fixture(scope="module")
def synthetic_report():
    return runner.run_audit(Path("unused"), Path("unused"), device="cpu", synthetic=True, max_epochs=1)


def test_end_to_end_synthetic_protocol_contract(synthetic_report):
    report = synthetic_report; names = report["client_grid_names"]
    assert len(names) == 4 and len(report["donor_proxy_metadata_by_grid"]) == 4
    assert sum(len(report["transfer_metrics"][name]) for name in names) == 12
    assert len(report["selected_policy"]) == 4
    assert report["validation_used_for_selection"] is False
    assert report["audit_used_for_selection"] is False
    assert report["test_evaluated"] is False
    for matrix in ("calibration_benefit", "audit_benefit", "validation_benefit"):
        assert all(report[matrix][name][name] is None for name in names)
    assert set(report["four_grid_macro"]) == {"train_audit", "canonical_validation"}
    for split in report["four_grid_macro"].values():
        assert "target_local_proxy" in split and "selected_transfer_policy" in split
    assert report["targets_with_positive_selected_calibration_benefit"] == len(report["positive_selected_calibration_targets"])
    assert report["targets_with_positive_selected_audit_benefit"] == len(report["positive_selected_audit_targets"])
    assert report["targets_with_positive_selected_validation_benefit"] == len(report["positive_selected_validation_targets"])
    for name in report["zero_transfer_fallback_targets"]:
        item = report["selected_policy"][name]
        assert item["local_audit_metrics"] == item["selected_audit_metrics"]
        assert item["local_validation_metrics"] == item["selected_validation_metrics"]
        assert item["selected_audit_benefit"] == item["selected_validation_benefit"] == 0.0


def test_donor_pretraining_runs_once_per_grid(monkeypatch):
    original = runner.train_proxy; donor_calls = []
    def counted(grid, fit, calibration, scaler, **kwargs):
        if len(fit) == 4636 and len(calibration) == 1160:
            donor_calls.append(grid.grid_name)
        return original(grid, fit, calibration, scaler, **kwargs)
    monkeypatch.setattr(runner, "train_proxy", counted)
    runner.run_audit(Path("unused"), Path("unused"), device="cpu", synthetic=True, max_epochs=1)
    assert donor_calls == list(runner.CLIENT_GRID_NAMES)


def test_stage_b_reconstructed_models_move_to_requested_device():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'local_eval = make_proxy(grid).to(device)' in source
    assert 'model = make_proxy(grid).to(device)' in source


def test_json_and_markdown_rendering_complete(synthetic_report):
    import json
    assert json.loads(json.dumps(synthetic_report))["test_evaluated"] is False
    markdown = runner.render_markdown(synthetic_report)
    assert "Directed Calibration Benefit Matrix" in markdown
    assert "Per-Target Train Audit Comparison" in markdown
    assert "Per-Target Canonical Validation Comparison" in markdown
    assert "Canonical test split is never evaluated" in markdown
