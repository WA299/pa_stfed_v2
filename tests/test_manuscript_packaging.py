"""Lightweight checks for manuscript and TEST preflight packaging."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_checkpoint_readiness import (
    HISTORY_FRACTIONS,
    INDUSTRIAL_TARGET,
    METHODS,
    REFERENCE_TARGETS,
    SEEDS,
    build_evaluation_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PREFLIGHT = PAPER / "preflight"


def test_evaluation_matrix_has_exact_180_unique_cells():
    rows = build_evaluation_matrix()
    assert len(rows) == 180
    assert len({row["cell_id"] for row in rows}) == 180
    assert sum(row["domain"] == "reference_grid" for row in rows) == 144
    assert sum(row["domain"] == "industrial_external" for row in rows) == 36


def test_evaluation_matrix_expected_seeds_fractions_methods_targets():
    rows = build_evaluation_matrix()
    assert {row["seed"] for row in rows} == set(SEEDS)
    assert {row["history_fraction"] for row in rows} == set(HISTORY_FRACTIONS)
    assert {row["method"] for row in rows} == set(METHODS)
    assert {row["target"] for row in rows if row["domain"] == "reference_grid"} == set(REFERENCE_TARGETS)
    assert {row["target"] for row in rows if row["domain"] == "industrial_external"} == {INDUSTRIAL_TARGET}


def test_generated_matrix_has_no_test_observations_or_metrics():
    path = PREFLIGHT / "evaluation_matrix_180.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 180
    assert all(row["test_data_accessed"] == "False" for row in rows)
    assert all(row["evaluation_status"] == "not_evaluated" for row in rows)
    assert all(row["test_metric_value"] == "" for row in rows)


def test_checkpoint_inventory_is_read_only_and_requires_reconstruction():
    data = json.loads((PREFLIGHT / "checkpoint_inventory.json").read_text(encoding="utf-8"))
    assert data["guardrails"]["test_accessed"] is False
    assert data["guardrails"]["test_observations_read"] is False
    assert data["guardrails"]["test_metrics_read"] is False
    assert data["guardrails"]["checkpoint_deserialization_performed"] is False
    assert data["guardrails"]["reconstruction_executed"] is False
    assert data["coverage"]["total_cells"] == 180
    assert data["coverage"]["accepted_180_cell_inference_possible"] is False
    assert data["coverage"]["pre_test_reconstruction_required"] is True


def test_report_traceability_files_are_present():
    manifest_path = ROOT / "results/paper_ready/manifests/paper_results_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["canonical_test_metrics_included"] is False
    assert manifest["entry_count"] == len(manifest["entries"])
    for entry in manifest["entries"]:
        assert (ROOT / entry["source_artifact_path"]).exists()
        assert entry["source_test_guardrail_value"] is False
    assert "results/paper_ready/manifests/paper_results_manifest.json" in (
        (PAPER / "drafts/results_draft_zh.md").read_text(encoding="utf-8")
        + (PAPER / "drafts/discussion_draft_zh.md").read_text(encoding="utf-8")
    )


def test_manuscript_drafts_state_the_scope_guardrails():
    results = (PAPER / "drafts/results_draft_zh.md").read_text(encoding="utf-8")
    discussion = (PAPER / "drafts/discussion_draft_zh.md").read_text(encoding="utf-8")
    combined = results + discussion
    for phrase in ("development validation", "external industrial validation", "统计显著性", "fallback", "WAPE", "sMAPE"):
        assert phrase in combined
    assert "不是完全受控的 WHO-only 消融" in results
