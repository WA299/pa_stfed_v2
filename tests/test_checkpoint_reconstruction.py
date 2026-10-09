from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

from code.federated.checkpointing import (
    HISTORY_FRACTIONS,
    INDUSTRIAL_TARGET,
    PUBLIC_METHODS,
    REFERENCE_TARGETS,
    SEEDS,
    canonical_method,
    export_checkpoint,
    load_checkpoint,
    matrix_cells,
)
from code.federated.accepted_results import resolve_all_accepted_results
from code.federated.accepted_results import direct_transfer_reference_input, resolve_accepted_result
from code.federated.parameter_groups import FROZEN_TEMPORAL_PARAMETER_NAMES
from scripts.reconstruct_frozen_checkpoints import build_reconstruction_plan, execute_reconstruction
from scripts.validate_frozen_checkpoints import expected_cell_map, validate_frozen_checkpoints


def test_exact_180_matrix_and_aliases():
    cells = matrix_cells()
    assert len(cells) == 180
    assert len({cell["cell_id"] for cell in cells}) == 180
    assert {cell["method"] for cell in cells} == set(PUBLIC_METHODS)
    assert {cell["seed"] for cell in cells} == set(SEEDS)
    assert {cell["history_fraction"] for cell in cells} == set(HISTORY_FRACTIONS)
    assert canonical_method("reference_grid", "local") == "scarce_local"
    assert canonical_method("industrial_external", "local") == "industrial_scarce_local"
    assert sum(cell["domain"] == "reference_grid" for cell in cells) == 144
    assert sum(cell["domain"] == "industrial_external" for cell in cells) == 36
    assert all(cell["target"] in REFERENCE_TARGETS for cell in cells if cell["domain"] == "reference_grid")
    assert all(cell["target"] == INDUSTRIAL_TARGET for cell in cells if cell["domain"] == "industrial_external")


def test_atomic_tensor_checkpoint_save_load_and_prediction_equivalence():
    torch = pytest.importorskip("torch")
    tmp_path = Path(".checkpoint_test_tmp")
    tmp_path.mkdir(exist_ok=True)
    model = torch.nn.Linear(3, 2)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "model.pt"
    metadata = {
        "method": "local", "domain": "industrial_external", "target": INDUSTRIAL_TARGET,
        "seed": 42, "history_fraction": 0.25, "canonical_method": "industrial_scarce_local",
        "model_class": "PUCRSTAttnV2ConditionalUtility", "test_accessed": False,
    }
    exported = export_checkpoint(path, state, metadata)
    loaded, loaded_meta = load_checkpoint(path, expected=metadata)
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(loaded)
    x = torch.randn(4, 3)
    assert torch.equal(model(x), restored(x))
    assert exported["sha256"] == loaded_meta["sha256"]
    assert path.with_suffix(".pt.json").is_file()
    for item in tmp_path.iterdir():
        item.unlink()
    tmp_path.rmdir()


def test_corrupt_checkpoint_and_wrong_identity_rejected():
    torch = pytest.importorskip("torch")
    tmp_path = Path(".checkpoint_test_tmp")
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "model.pt"
    state = {"weight": torch.ones(1, 1), "bias": torch.zeros(1)}
    metadata = {"method": "fedavg", "domain": "reference_grid", "target": REFERENCE_TARGETS[0], "seed": 123, "history_fraction": 0.5, "canonical_method": "fedavg", "model_class": "PUCRSTAttnV2ConditionalUtility", "test_accessed": False}
    export_checkpoint(path, state, metadata)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        load_checkpoint(path)
    export_checkpoint(path, state, metadata)
    with pytest.raises(ValueError):
        load_checkpoint(path, expected={"seed": 42})
    for item in tmp_path.iterdir():
        item.unlink()
    tmp_path.rmdir()


def test_missing_checkpoint_and_parity_failure_halt_readiness():
    tmp_path = Path(".checkpoint_test_tmp")
    tmp_path.mkdir(exist_ok=True)
    report = validate_frozen_checkpoints(tmp_path)
    assert report["total_cells"] == 180
    assert len(report["missing_cells"]) == 180
    assert report["test_unlock_ready"] is False
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"test_accessed": False, "parity_passed": False, "cells": []}), encoding="utf-8")
    report = validate_frozen_checkpoints(tmp_path, parity)
    assert report["test_unlock_ready"] is False
    for item in tmp_path.iterdir():
        item.unlink()
    tmp_path.rmdir()


def test_reconstruction_plan_is_plan_only():
    tmp_path = Path(".checkpoint_test_tmp")
    plan = build_reconstruction_plan(tmp_path)
    assert plan["cell_count"] == 180
    assert plan["test_accessed"] is False
    assert plan["reconstruction_executed"] is False
    assert plan["parity_required_before_test_unlock"] is True


def test_accepted_result_mapping_has_exact_180_per_seed_cells():
    mapping = resolve_all_accepted_results()
    assert len(mapping) == 180
    assert set(mapping) == set(expected_cell_map())
    assert all(item["source_artifact"].is_file() for item in mapping.values())
    assert all(set(item["source_json_key"]) == {"audit", "validation"} for item in mapping.values())


def test_parity_is_mandatory_and_top_level_flag_is_not_sufficient():
    tmp_path = Path(".checkpoint_parity_contract_tmp")
    tmp_path.mkdir(exist_ok=True)
    root = tmp_path / "checkpoints"
    root.mkdir()
    report = tmp_path / "parity.json"
    report.write_text(json.dumps({"parity_passed": True, "test_accessed": False, "test_evaluated": False, "reconstruction_executed": True, "historical_artifacts_modified": False, "cells": []}), encoding="utf-8")
    result = validate_frozen_checkpoints(root, report)
    assert result["test_unlock_ready"] is False
    assert any("missing cells" in item or "expected 180" in item for item in result["parity_failures"])
    assert validate_frozen_checkpoints(root)["test_unlock_ready"] is False
    for item in tmp_path.rglob("*"):
        if item.is_file(): item.unlink()
    root.rmdir(); tmp_path.rmdir()


def test_duplicate_and_extra_parity_cell_ids_fail_closed():
    tmp_path = Path(".checkpoint_parity_duplicate_tmp")
    tmp_path.mkdir(exist_ok=True)
    root = tmp_path / "checkpoints"
    root.mkdir()
    cell = matrix_cells()[0]
    item = {**cell, "test_accessed": False, "test_evaluated": False, "parity_passed": True, "metric_comparisons": []}
    report = tmp_path / "parity.json"
    report.write_text(json.dumps({"parity_passed": True, "test_accessed": False, "test_evaluated": False, "reconstruction_executed": True, "historical_artifacts_modified": False, "cells": [item, item]}), encoding="utf-8")
    result = validate_frozen_checkpoints(root, report)
    assert result["test_unlock_ready"] is False
    assert any("duplicate" in item for item in result["parity_failures"])
    for item in tmp_path.rglob("*"):
        if item.is_file(): item.unlink()
    root.rmdir(); tmp_path.rmdir()


def test_reconstruction_authorization_is_separate_from_test_and_output_isolated():
    with pytest.raises(PermissionError):
        execute_reconstruction(Path("artifacts/frozen_pretest_checkpoints/test"), authorize_reconstruction=False, cell_runner=lambda *_: {})
    with pytest.raises(PermissionError):
        execute_reconstruction(Path("artifacts/frozen_pretest_checkpoints/test"), authorize_reconstruction=True, authorize_test=True, cell_runner=lambda *_: {})


def test_seed42_direct_transfer_uses_formal_btd_input_but_direct_artifact_for_parity():
    reference = direct_transfer_reference_input(history_fraction=0.25, seed=42)
    assert reference is not None
    assert reference.name == "btd_fl_25pct_seed42.json"
    report = json.loads(reference.read_text(encoding="utf-8"))
    assert report["btd_variant"] == "btd_fl"
    assert report["seed"] == 42
    assert report["history_fraction"] == 0.25
    assert report["test_evaluated"] is False
    direct_document = json.loads(
        (reference.parent / "btd_fl_direct_transfer_25pct_seed42.json").read_text(encoding="utf-8")
    )
    assert {
        target: payload["metadata"]["selected_donor"]
        for target, payload in report["scenarios"].items()
    } == {
        target: payload["metadata"]["selected_donor"]
        for target, payload in direct_document["scenarios"].items()
    }
    pytest.importorskip("torch")
    from code.federated.formal_btd_ablations import load_frozen_reference
    frozen = load_frozen_reference(reference)
    assert frozen["btd_variant"] == "btd_fl"

    direct_cell = next(
        cell for cell in matrix_cells()
        if cell["domain"] == "reference_grid"
        and cell["target"] == REFERENCE_TARGETS[0]
        and cell["seed"] == 42
        and cell["history_fraction"] == 0.25
        and cell["method"] == "btd_fl_direct_transfer"
    )
    parity_source = resolve_accepted_result(direct_cell)["source_artifact"]
    assert parity_source.name == "btd_fl_direct_transfer_25pct_seed42.json"
    assert parity_source != reference


def test_parity_record_without_reloaded_checkpoint_cannot_pass():
    from scripts.reconstruct_frozen_checkpoints import build_parity_cell

    cell = matrix_cells()[0]
    accepted = resolve_accepted_result(cell)
    document = json.loads(accepted["source_artifact"].read_text(encoding="utf-8"))

    def walk(path):
        value = document
        for part in path.split("."):
            value = value[part]
        return value

    observed = {split: walk(path) for split, path in accepted["source_json_key"].items()}
    record = build_parity_cell(cell, observed)
    assert record["parity_passed"] is False
    assert record["checkpoint_sha256"] is None
    assert "checkpoint" in record["checkpoint_replay_error"]


def test_actual_frozen_model_checkpoint_replay_and_temporal_only_transfer():
    torch = pytest.importorskip("torch")
    from code.audits.btd_full_backbone_bridge import inject_temporal_state, temporal_state_from_model
    from code.federated.checkpointing import export_model_checkpoint, replay_checkpoint
    from code.models.puc_rstattn_v2_conditional_utility import PUCRSTAttnV2ConditionalUtility

    tmp_path = Path(".checkpoint_actual_model_tmp")
    tmp_path.mkdir(exist_ok=True)
    cell = next(
        cell for cell in matrix_cells()
        if cell["domain"] == "industrial_external"
        and cell["method"] == "local"
        and cell["seed"] == 42
        and cell["history_fraction"] == 0.25
    )
    mask = np.asarray([False, True, True, False], dtype=bool)
    grid = SimpleNamespace(num_nodes=4, load_bus_mask=mask, node_ids=["n0", "n1", "n2", "n3"])
    scaler = SimpleNamespace(
        fit_start_index=0, fit_end_index=10, fit_timestamp_end="t10",
        fit_split="train_fit_only", fit_load_bus_count=2, fit_value_count=20,
        p_mean_=1.0, p_scale_=2.0, q_mean_=None, q_scale_=None,
    )
    empty_edges = np.empty((2, 0), dtype=np.int64)
    empty_relations = np.empty((0, 3), dtype=np.float32)
    model = PUCRSTAttnV2ConditionalUtility(4, empty_edges, empty_relations, mask, np.empty(0, dtype=np.float32))
    donor = PUCRSTAttnV2ConditionalUtility(4, empty_edges, empty_relations, mask, np.empty(0, dtype=np.float32))
    with torch.no_grad():
        for name, parameter in donor.named_parameters():
            if name in FROZEN_TEMPORAL_PARAMETER_NAMES:
                parameter.add_(0.01)
    spatial_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
        if name not in FROZEN_TEMPORAL_PARAMETER_NAMES
    }
    buffer_before = {name: value.detach().clone() for name, value in model.named_buffers()}
    transferred = inject_temporal_state(model, temporal_state_from_model(donor))
    assert tuple(transferred) == tuple(FROZEN_TEMPORAL_PARAMETER_NAMES)
    assert all(torch.equal(model.state_dict()[name], donor.state_dict()[name]) for name in FROZEN_TEMPORAL_PARAMETER_NAMES)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in spatial_before.items())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in buffer_before.items())

    model.eval()
    x = torch.randn(2, 168, 4, 6)
    with torch.no_grad():
        expected = model(x).detach().clone()
    try:
        export_model_checkpoint(
            root=tmp_path, cell=cell, model=model, grid=grid, scaler=scaler,
            graph_metadata={"synthetic": True}, source_artifact="synthetic",
            frozen_commits={"reconstruction_code": "test"},
            selected_epoch=3, epochs_run=5,
        )
        restored = PUCRSTAttnV2ConditionalUtility(4, empty_edges, empty_relations, mask, np.empty(0, dtype=np.float32))
        replay_checkpoint(next(tmp_path.rglob("local.pt")), restored, cell=cell, grid=grid, scaler=scaler)
        restored.eval()
        with torch.no_grad():
            actual = restored(x)
        assert torch.equal(expected, actual)
    finally:
        for item in sorted(tmp_path.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        tmp_path.rmdir()
