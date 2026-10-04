"""Reporting-only integrity checks for the paper-ready frozen-results package."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import build_paper_results as builder


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_ready"


@pytest.fixture(scope="module")
def package():
    return builder.build(root=ROOT, out=OUT, figures=False)


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_temporal_only_uses_explicit_ablation_source(package):
    rows = _rows(OUT / "tables/table_centralized_main.csv")
    row = next(item for item in rows if item["Method"] == "temporal_residual_only")
    assert float(row["Node MAE"]) == pytest.approx(0.00032925806226992394, rel=0, abs=1e-18)
    assert row["Source"] == "results/ablations/puc_rstattn_v2_ablation_summary.json"
    assert "temporal_residual_only" in row["JSON key path"]
    assert "puc_rstattn_v2" not in row["JSON key path"]


def test_ordinary_v2_is_not_mislabeled_conditional_utility(package):
    rows = _rows(OUT / "tables/table_centralized_main.csv")
    labels = [row["Method"] for row in rows]
    assert "conditional-utility V2" not in labels
    assert "PUC-RSTAttn V2" in labels


def test_reference_comparisons_are_separate_and_aligned(package):
    main = _rows(OUT / "tables/table_reference_federated_main.csv")
    assert all(not row["Method"].startswith("BTD improvement") for row in main)
    comparison = _rows(OUT / "tables/table_reference_federated_comparisons.csv")
    assert comparison[0]["50% mean relative validation-MAE improvement"]
    assert comparison[0]["50% seed wins / 3"] == "3"


def test_industrial_comparisons_have_exact_win_counts(package):
    rows = _rows(OUT / "tables/table_industrial_external_comparisons.csv")
    assert {row["50% seed wins / 3"] for row in rows} == {"3"}
    assert {row["25% seed wins / 3"] for row in rows} == {"3"}


def test_adapted_state_ablation_is_present(package):
    rows = _rows(OUT / "tables/table_btd_ablations.csv")
    row = next(item for item in rows if item["Method"] == "Historical adapted-state BTD")
    assert row["Probe adapted state reused in final initialization?"] == "yes"
    assert row["Source artifact"].endswith("btd_fl_25pct_seed42.json")


def test_manifest_paths_and_json_keys_resolve(package):
    manifest = json.loads((OUT / "manifests/paper_results_manifest.json").read_text(encoding="utf-8"))
    assert manifest["entry_count"] == len(manifest["entries"])
    assert manifest["entry_count"] > 0
    for entry in manifest["entries"]:
        assert (ROOT / entry["source_artifact_path"]).exists()
        assert entry["source_test_guardrail_value"] is False
    builder.validate_manifest_entries(ROOT, manifest["entries"])


def test_missing_test_guardrail_fails_closed():
    path = Path("centralized_validation_summary.json")
    with pytest.raises(ValueError, match="missing TEST guardrail"):
        builder.validate_source_test_guardrail(path, {"metadata": {}})


def test_industrial_counts_and_marker_data(package):
    marker_data = json.loads((OUT / "figures/fig_industrial_benefit_matrix_marker_data.json").read_text(encoding="utf-8")) if (OUT / "figures/fig_industrial_benefit_matrix_marker_data.json").exists() else []
    # The no-figure fixture does not regenerate marker data; the committed artifact is checked when present.
    if marker_data:
        assert len(marker_data) == 6
        assert all(item["selected_donor"] in builder.REFERENCE_CLIENTS for item in marker_data)
    ind25 = json.loads((ROOT / "results/federated/industrial_external/industrial_external_25pct_summary.json").read_text(encoding="utf-8"))
    ind50 = json.loads((ROOT / "results/federated/industrial_external/industrial_external_50pct_summary.json").read_text(encoding="utf-8"))
    assert (ind25["btd_donor_behavior"]["positive_benefit_count"], ind25["btd_donor_behavior"]["non_positive_benefit_count"], ind25["btd_donor_behavior"]["fallback_count"]) == (11, 1, 0)
    assert (ind50["btd_donor_behavior"]["positive_benefit_count"], ind50["btd_donor_behavior"]["non_positive_benefit_count"], ind50["btd_donor_behavior"]["fallback_count"]) == (12, 0, 0)


def test_digest_has_no_significance_or_fallback_success_claim(package):
    digest = (OUT / "paper_results_summary.md").read_text(encoding="utf-8").lower()
    assert "statistical significance" not in digest
    assert "fallback performance is not empirically established" in digest
    assert "test metric" not in digest
