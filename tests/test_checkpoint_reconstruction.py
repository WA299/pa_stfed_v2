from __future__ import annotations

import json
from pathlib import Path

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
from scripts.reconstruct_frozen_checkpoints import build_reconstruction_plan
from scripts.validate_frozen_checkpoints import validate_frozen_checkpoints


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
