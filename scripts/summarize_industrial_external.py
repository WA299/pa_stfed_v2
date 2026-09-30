"""Summarize frozen industrial external-domain results across seeds/history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.federated.industrial_result_validation import load_valid_external_result

SEEDS = (42, 123, 2026)
METHODS = ("industrial_scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer")
BASELINE_METHODS = METHODS[:4]
TARGET = "norway_industrial_mvlv"


def _read(path: Path, seed: int, fraction: float) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("run_mode") != "real":
        raise ValueError(f"non-real artifact cannot enter formal summary: {path}")
    if report.get("seed") != seed or abs(float(report.get("history_fraction", -1)) - fraction) > 1e-12:
        raise ValueError(f"seed/history mismatch in {path}")
    artifact_type = report.get("artifact_type")
    expected_methods = {
        "industrial_baselines": set(BASELINE_METHODS),
        "industrial_btd_direct_transfer": {"btd_fl_direct_transfer"},
        "industrial_fedfomo_style": {"fedfomo_style"},
    }.get(artifact_type)
    if expected_methods is None or set(report.get("methods", ())) != expected_methods:
        raise ValueError(f"wrong artifact type or method set in {path}")
    load_valid_external_result(
        path,
        artifact_type=artifact_type,
        seed=seed,
        fraction=fraction,
        run_mode="real",
        rounds=int(report.get("rounds", 0)),
        local_epochs=int(report.get("local_epochs", 0)),
        max_epochs=int(report.get("max_epochs", 0)),
        batch_size=int(report.get("batch_size", 0)),
    )
    for field in ("test_evaluated", "industrial_test_evaluated", "reference_grid_tests_evaluated"):
        if report.get(field) is not False:
            raise ValueError(f"{path} violates {field}")
    if TARGET not in report.get("scenarios", {}):
        raise ValueError(f"{path} is missing the industrial target scenario")
    return report


def _metrics(report: dict[str, Any], method: str) -> dict[str, Any]:
    if method in report.get("industrial_target_macro", {}):
        return report["industrial_target_macro"][method]
    scenario = report["scenarios"][TARGET]
    if method in scenario.get("methods", {}):
        return scenario["methods"][method]
    if method == "btd_fl_direct_transfer":
        return scenario["metrics"]
    if method == "fedfomo_style":
        return scenario["scarce_target_metrics"]
    raise ValueError(f"method {method} missing from {report.get('artifact_type')}")


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {"values": [float(value) for value in values], "mean": float(np.mean(values)), "std": float(np.std(values))}


def summarize_fraction(root: Path, fraction: float) -> dict[str, Any]:
    percent = int(fraction * 100)
    loaded: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        base = root / f"industrial_baselines_{percent}pct_seed{seed}.json"
        btd = root / f"industrial_btd_direct_transfer_{percent}pct_seed{seed}.json"
        fomo = root / f"industrial_fedfomo_style_{percent}pct_seed{seed}.json"
        reports = {
            "industrial_scarce_local": _read(base, seed, fraction),
            "fedavg": _read(base, seed, fraction),
            "fedprox": _read(base, seed, fraction),
            "fedper": _read(base, seed, fraction),
            "btd_fl_direct_transfer": _read(btd, seed, fraction),
            "fedfomo_style": _read(fomo, seed, fraction),
        }
        if set(reports) != set(METHODS):
            raise ValueError(f"missing method for seed {seed}")
        loaded[seed] = reports

    metrics: dict[str, Any] = {}
    for method in METHODS:
        metrics[method] = {
            split: {
                scope: {
                    metric: _mean_std([_metrics(loaded[seed][method], method)[split][scope][metric] for seed in SEEDS])
                    for metric in ("mae", "rmse", "wape_pct", "smape_pct")
                }
                for scope in ("node_macro", "grid_aggregate")
            }
            for split in ("audit", "validation")
        }
    btd = "btd_fl_direct_transfer"
    comparisons: dict[str, Any] = {}
    for comparator in METHODS:
        if comparator == btd:
            continue
        per_seed = {}
        for seed in SEEDS:
            comparator_mae = _metrics(loaded[seed][comparator], comparator)["validation"]["node_macro"]["mae"]
            btd_mae = _metrics(loaded[seed][btd], btd)["validation"]["node_macro"]["mae"]
            per_seed[str(seed)] = {
                "relative_improvement": float((comparator_mae - btd_mae) / (comparator_mae + 1e-12)),
                "btd_wins": bool(btd_mae < comparator_mae),
            }
        values = [item["relative_improvement"] for item in per_seed.values()]
        comparisons[comparator] = {"per_seed": per_seed, "mean_relative_improvement": float(np.mean(values)), "std_relative_improvement": float(np.std(values)), "seed_win_count": int(sum(item["btd_wins"] for item in per_seed.values()))}

    donor = {}
    positive = non_positive = fallback = 0
    for seed in SEEDS:
        meta = loaded[seed][btd]["scenarios"][TARGET]["metadata"]
        benefits = meta["calibration_benefit_by_donor"]
        positive += sum(float(value) > 0 for value in benefits.values())
        non_positive += sum(float(value) <= 0 for value in benefits.values())
        fallback += int(bool(meta["zero_transfer_fallback"]))
        donor[str(seed)] = {"selected_donor": meta["selected_donor"], "selected_calibration_benefit": meta["selected_calibration_benefit"], "zero_transfer_fallback": meta["zero_transfer_fallback"], "benefits": benefits}
    return {"history_fraction": fraction, "seeds": list(SEEDS), "methods": metrics, "comparisons_btd_vs_comparators": comparisons, "btd_donor_behavior": {"by_seed": donor, "positive_benefit_count": positive, "non_positive_benefit_count": non_positive, "fallback_count": fallback}}


def cross_history(root: Path, summary25: dict[str, Any], summary50: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for comparator in summary25["comparisons_btd_vs_comparators"]:
        comparisons[comparator] = {
            "25pct": summary25["comparisons_btd_vs_comparators"][comparator],
            "50pct": summary50["comparisons_btd_vs_comparators"][comparator],
        }
    return {
        "history_fractions": [0.25, 0.50],
        "methods": {
            method: {
                "25pct": {split: summary25["methods"][method][split]["node_macro"]["mae"] for split in ("audit", "validation")},
                "50pct": {split: summary50["methods"][method][split]["node_macro"]["mae"] for split in ("audit", "validation")},
            }
            for method in METHODS
        },
        "btd_comparisons": comparisons,
        "btd_donor_behavior": {"25pct": summary25["btd_donor_behavior"], "50pct": summary50["btd_donor_behavior"]},
        "interpretation_guardrail": "No monotonicity or significance claim is made from three seeds.",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [f"# Industrial External Summary ({int(report['history_fraction'] * 100)}%)", "", "| Method | Validation node-MAE mean | std | Audit node-MAE mean | std |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        item = report["methods"][method]
        lines.append(f"| {method} | {item['validation']['node_macro']['mae']['mean']:.6g} | {item['validation']['node_macro']['mae']['std']:.6g} | {item['audit']['node_macro']['mae']['mean']:.6g} | {item['audit']['node_macro']['mae']['std']:.6g} |")
    lines += ["", "No significance claim is made from three seeds. Industrial and source TEST splits were not evaluated."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results" / "federated" / "industrial_external")
    parser.add_argument("--history-fraction", type=float, choices=(0.25, 0.50), default=None)
    parser.add_argument("--cross-history", action="store_true")
    args = parser.parse_args()
    root = args.input_dir
    if args.cross_history:
        summary25 = json.loads((root / "industrial_external_25pct_summary.json").read_text(encoding="utf-8"))
        summary50 = json.loads((root / "industrial_external_50pct_summary.json").read_text(encoding="utf-8"))
        report = cross_history(root, summary25, summary50)
        json_path = root / "industrial_external_history_summary.json"
        md_path = root / "industrial_external_history_summary.md"
        md = "# Industrial External Cross-History Summary\n\nNo monotonicity or significance claim is made from three seeds."
    else:
        if args.history_fraction is None:
            raise ValueError("--history-fraction is required unless --cross-history is used")
        report = summarize_fraction(root, args.history_fraction)
        percent = int(args.history_fraction * 100)
        json_path = root / f"industrial_external_{percent}pct_summary.json"
        md_path = root / f"industrial_external_{percent}pct_summary.md"
        md = _markdown(report)
    root.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
