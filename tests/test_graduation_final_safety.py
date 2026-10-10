from __future__ import annotations

import hashlib
import json
import math
import subprocess
import copy
from pathlib import Path

import pytest

from code.federated.checkpointing import (
    FROZEN_COMMITS,
    REFERENCE_TARGETS,
    checkpoint_path,
    matrix_cells,
    sha256_file,
)
from scripts.reconstruct_frozen_checkpoints import _promote_staged_block
from scripts.run_final_test_once import (
    COMPARATORS,
    FRACTIONS,
    METHODS,
    METRICS,
    SCOPES,
    SEEDS,
    _result_path,
    _validate_existing_result,
    summarize,
)


def _synthetic_cell_result(cell: dict, value: float) -> dict:
    return {
        **cell,
        "test_accessed": True,
        "test_evaluated": True,
        "metrics": {
            "test": {
                scope: {metric: value for metric in METRICS}
                for scope in SCOPES
            }
        },
    }


def test_summary_uses_four_grid_macro_per_seed_then_seed_statistics():
    synthetic = []
    for cell in matrix_cells():
        target_offset = (
            REFERENCE_TARGETS.index(cell["target"]) * 2
            if cell["domain"] == "reference_grid" else 0
        )
        seed_offset = {42: 0, 123: 2, 2026: 4}[cell["seed"]]
        method_offset = -METHODS.index(cell["method"])
        if cell["domain"] == "reference_grid":
            value = 1.0 + target_offset + seed_offset + method_offset
        else:
            value = 10.0 + seed_offset + method_offset
        synthetic.append(_synthetic_cell_result(cell, value))

    result = summarize(synthetic)
    reference = result["metrics"]["reference_grid"]["0.25"]["local"]["node_macro"]["mae"]
    # Per-seed four-grid means are 4, 6, and 8. The across-seed mean/std
    # must be computed from those three values, not from 12 individual rows.
    assert reference["mean"] == pytest.approx(6.0)
    assert reference["std"] == pytest.approx(math.sqrt(8.0 / 3.0))
    assert result["per_seed_metrics"]["reference_grid"]["0.25"]["local"]["42"]["node_macro"]["mae"] == pytest.approx(4.0)

    industrial = result["metrics"]["industrial_external"]["0.5"]["local"]["node_macro"]["mae"]
    assert industrial["mean"] == pytest.approx(12.0)
    assert industrial["std"] == pytest.approx(math.sqrt(8.0 / 3.0))
    comparison = result["btd_comparisons"]["reference_grid"]["0.25"]["fedavg"]
    assert comparison["comparison_cells"] == 3
    assert comparison["seed_wins"] == 3


def _valid_existing_result(cell: dict, checkpoint: Path, protocol_identity: dict) -> tuple[dict, dict, dict]:
    split = {
        "split_name": "test", "dataset": cell["target"],
        "start_index": 10, "end_index": 12, "target_count": 2,
        "start_time": "2030-01-01T00:00:00", "end_time": "2030-01-01T01:00:00",
        "target_index_sha256": hashlib.sha256(b"synthetic-test-indices").hexdigest(),
    }
    provenance = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "protocol_lock_identity": dict(protocol_identity),
        "accepted_result_artifact": "synthetic/accepted.json",
    }
    metrics = {
        "test": {
            scope: {metric: 1.25 for metric in METRICS}
            for scope in SCOPES
        }
    }
    item = {
        "result_schema_version": 2,
        **cell,
        "test_accessed": True,
        "test_evaluated": True,
        "industrial_test_evaluated": cell["domain"] == "industrial_external",
        "reference_grid_tests_evaluated": cell["domain"] == "reference_grid",
        "selection_used_test": False,
        "training_used_test": False,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "protocol_lock_identity": dict(protocol_identity),
        "evaluation_started_at_utc": "2030-01-01T00:00:00+00:00",
        "evaluation_completed_at_utc": "2030-01-01T00:00:01+00:00",
        "evaluation_elapsed_seconds": 1.0,
        "test_split_identity": split,
        "test_index_start": 10,
        "test_index_end": 12,
        "test_target_count": 2,
        "source_provenance": provenance,
        "metrics": metrics,
    }
    return item, split, provenance


def test_valid_prior_test_cell_is_accepted_for_resume_skip(tmp_path):
    cell = matrix_cells()[0]
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    protocol = {"commit_sha": "a" * 40, "blob_sha": "b" * 40, "content_sha256": "c" * 64}
    item, split, provenance = _valid_existing_result(cell, checkpoint, protocol)
    result_path = _result_path(tmp_path / "results", cell)
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(item), encoding="utf-8")

    accepted = _validate_existing_result(
        result_path, cell, checkpoint,
        protocol_lock_identity=protocol,
        expected_test_split=split,
        expected_source_provenance=provenance,
    )
    assert accepted is not None
    assert accepted["cell_id"] == cell["cell_id"]
    assert accepted["evaluation_started_at_utc"] == item["evaluation_started_at_utc"]


@pytest.mark.parametrize("corruption", ["invalid_json", "wrong_cell", "wrong_checkpoint", "wrong_lock", "wrong_split"])
def test_invalid_prior_test_result_stops_without_rewrite(tmp_path, corruption):
    cell = matrix_cells()[0]
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    protocol = {"commit_sha": "a" * 40, "blob_sha": "b" * 40, "content_sha256": "c" * 64}
    item, split, provenance = _valid_existing_result(cell, checkpoint, protocol)
    if corruption == "invalid_json":
        payload = b"{broken"
    else:
        split = copy.deepcopy(split)
        if corruption == "wrong_cell":
            item["seed"] = -1
        elif corruption == "wrong_checkpoint":
            item["checkpoint_sha256"] = "0" * 64
        elif corruption == "wrong_lock":
            item["protocol_lock_identity"] = {"commit_sha": "d" * 40}
        elif corruption == "wrong_split":
            item["test_split_identity"]["target_index_sha256"] = "e" * 64
        payload = json.dumps(item).encode("utf-8")
    result_path = _result_path(tmp_path / "results", cell)
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(payload)
    before = result_path.read_bytes()

    with pytest.raises(RuntimeError, match="refusing reevaluation or overwrite"):
        _validate_existing_result(
            result_path, cell, checkpoint,
            protocol_lock_identity=protocol,
            expected_test_split=split,
            expected_source_provenance=provenance,
        )
    assert result_path.read_bytes() == before


def test_checkpoint_promotion_never_replaces_valid_existing_weights(tmp_path, monkeypatch):
    import scripts.reconstruct_frozen_checkpoints as reconstruction

    cell = matrix_cells()[0]
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    final_path = checkpoint_path(output, cell)
    staged_path = checkpoint_path(staging, cell)
    final_path.parent.mkdir(parents=True)
    staged_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"validated existing weights")
    staged_path.write_bytes(b"newly reconstructed weights")
    staged_path.with_suffix(staged_path.suffix + ".json").write_text(
        json.dumps({"sha256": sha256_file(staged_path)}), encoding="utf-8"
    )

    def fake_validation(root, _cell):
        path = checkpoint_path(root, cell)
        if root.resolve() == output.resolve():
            return path.is_file() and path.read_bytes() == b"validated existing weights", "invalid"
        return path.is_file() and path.read_bytes() == b"newly reconstructed weights", "invalid"

    monkeypatch.setattr(reconstruction, "_checkpoint_valid_for_cell", fake_validation)
    promoted = _promote_staged_block(staging, output, [cell])
    assert promoted == []
    assert final_path.read_bytes() == b"validated existing weights"


def test_damaged_or_missing_checkpoint_can_be_promoted_from_valid_stage(tmp_path, monkeypatch):
    import scripts.reconstruct_frozen_checkpoints as reconstruction

    cell = matrix_cells()[1]
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    final_path = checkpoint_path(output, cell)
    staged_path = checkpoint_path(staging, cell)
    final_path.parent.mkdir(parents=True)
    staged_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"damaged old weights")
    staged_path.write_bytes(b"reconstructed weights")
    staged_path.with_suffix(staged_path.suffix + ".json").write_text(
        json.dumps({"sha256": sha256_file(staged_path)}), encoding="utf-8"
    )

    def fake_validation(root, _cell):
        path = checkpoint_path(root, cell)
        return path.is_file() and path.read_bytes() == b"reconstructed weights", "not complete"

    monkeypatch.setattr(reconstruction, "_checkpoint_valid_for_cell", fake_validation)
    promoted = _promote_staged_block(staging, output, [cell])
    assert promoted == [cell["cell_id"]]
    assert final_path.read_bytes() == b"reconstructed weights"
    sidecar = json.loads(final_path.with_suffix(final_path.suffix + ".json").read_text(encoding="utf-8"))
    assert sidecar["sha256"] == sha256_file(final_path)
    assert sidecar["file_size_bytes"] == final_path.stat().st_size


def test_complete_valid_checkpoint_block_is_skipped_without_rerun(tmp_path, monkeypatch):
    import scripts.reconstruct_frozen_checkpoints as reconstruction

    cell = matrix_cells()[0]
    calls = []
    monkeypatch.setattr(reconstruction, "_block_ready", lambda *_: (True, []))
    report = reconstruction._run_protected_block(
        tmp_path, [cell], lambda _staging: calls.append("reconstructed"), "synthetic complete block"
    )
    assert report is None
    assert calls == []


def test_protocol_lock_contract_rejects_unresolved_or_mismatched_lock(tmp_path):
    from scripts.manage_test_protocol_lock import SCHEMA_VERSION, TEST_SPLITS, validate_lock_contract
    from scripts.run_final_test_once import EXPECTED_REPORTING_COMMIT

    parity = tmp_path / "parity.json"
    parity.write_text("{}", encoding="utf-8")
    lock = {
        "schema_version": SCHEMA_VERSION,
        "lock_identity": "Git commit containing this tracked file, derived at runtime",
        "reporting_commit": EXPECTED_REPORTING_COMMIT,
        "scientific_method_commit": FROZEN_COMMITS["scientific_method"],
        "accepted_results_commit": FROZEN_COMMITS["accepted_results"],
        "paper_ready_reporting_commit": FROZEN_COMMITS["paper_ready_reporting"],
        "reconstruction_code_commit": "1" * 40,
        "matrix_cells": 180,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "history_fractions": list(FRACTIONS),
        "reference_targets": list(REFERENCE_TARGETS),
        "industrial_target": "norway_industrial_mvlv",
        "test_split_contract": dict(TEST_SPLITS),
        "checkpoint_set_sha256": "2" * 64,
        "parity_report_path": str(parity.resolve()),
        "parity_report_sha256": sha256_file(parity),
        "test_authorization": "separate explicit --authorize-test flag required; not granted by this lock",
        "test_opening_authorized": False,
        "test_evaluated": False,
    }
    kwargs = {
        "checkpoint_digest": "2" * 64,
        "reconstruction_commit": "1" * 40,
        "parity_path": parity,
    }
    validate_lock_contract(lock, **kwargs)
    for key, value in (("seeds", [42]), ("methods", ["local"]),
                       ("test_opening_authorized", True), ("reporting_commit", "<COMMIT_SHA>")):
        invalid = dict(lock)
        invalid[key] = value
        with pytest.raises(RuntimeError):
            validate_lock_contract(invalid, **kwargs)


def test_protocol_lock_git_identity_requires_committed_clean_lock(tmp_path):
    from scripts.manage_test_protocol_lock import LOCK_RELATIVE_PATH, git_protocol_identity

    root = tmp_path / "repo"
    lock_path = root / Path(LOCK_RELATIVE_PATH)
    lock_path.parent.mkdir(parents=True)
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Synthetic Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "synthetic@example.invalid"], check=True)
    lock_path.write_bytes(b"{}\r\n")
    subprocess.run(["git", "-C", str(root), "add", LOCK_RELATIVE_PATH], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "synthetic protocol lock"], check=True, capture_output=True)

    identity = git_protocol_identity(lock_path, repo_root=root)
    assert len(identity["commit_sha"]) == 40
    assert identity["path"] == LOCK_RELATIVE_PATH

    untracked_source = root / "scripts" / "uncommitted_runner.py"
    untracked_source.parent.mkdir(parents=True)
    untracked_source.write_text("# synthetic\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="untracked executable source"):
        git_protocol_identity(lock_path, repo_root=root)
    untracked_source.unlink()

    lock_path.write_bytes(b"{\"edited\":true}\r\n")
    with pytest.raises(RuntimeError, match="tracked changes|uncommitted"):
        git_protocol_identity(lock_path, repo_root=root)


def test_statistical_contract_keeps_reference_and_industrial_units_separate():
    assert set(METHODS) == {
        "local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer"
    }
    assert SEEDS == (42, 123, 2026)
    assert FRACTIONS == (0.25, 0.5)
    assert COMPARATORS == ("local", "fedavg", "fedprox", "fedper", "fedfomo_style")
