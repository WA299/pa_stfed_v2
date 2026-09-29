"""Compare frozen 25% and 50% robustness summaries without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

METHODS = ("scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer")
TARGETS = ("39_bus_semi_urban_reference_grid", "50_bus_rural_reference_grid", "56_bus_semi_urban_reference_grid", "80_bus_rural_reference_grid")
SEEDS = ("42", "123", "2026")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(summary_25: Path, summary_50: Path) -> dict:
    old, new = _load(summary_25), _load(summary_50)
    if float(old.get("history_fraction", 0.25)) != 0.25 or float(new.get("history_fraction", -1)) != 0.5:
        raise ValueError("cross-history inputs must be 25% and 50% summaries")
    result = {"experiment": "history_fraction_25_vs_50_summary", "history_fractions": [0.25, 0.5], "methods": list(METHODS), "primary_metric": "validation node-macro MAE", "metrics": {}, "btd_vs_comparators": {}, "donor_behavior": {"25pct": old.get("donor_selection_robustness"), "50pct": new.get("donor_selection_robustness")}}
    for method in METHODS:
        result["metrics"][method] = {}
        for split in ("validation", "audit"):
            values = {}
            for fraction, report in (("25pct", old), ("50pct", new)):
                item = report["metrics"][method][split]["node_macro"]["mae"]
                values[fraction] = {"mean": item["mean"], "std": item["std"], "per_seed": item["per_seed"]}
            result["metrics"][method][split] = values
    for comparator in ("scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style"):
        entries = {}
        for fraction, report in (("25pct", old), ("50pct", new)):
            item = report["btd_vs_comparators"][comparator]
            entries[fraction] = {"mean_relative_improvement": item["mean_relative_improvement"], "std_relative_improvement": item["std_relative_improvement"], "seeds_btd_wins": item["seeds_btd_wins"], "target_x_seed_validation_wins": item["scarce_target_validation_wins_out_of_12"]}
        result["btd_vs_comparators"][comparator] = entries
    return result


def render(summary: dict) -> str:
    lines = ["# 25% vs 50% History Robustness", "", "This comparison reuses frozen summaries only. No monotonicity or significance claim is implied.", "", "## Primary Validation/Audit Node-MAE", "", "| Method | 25% validation mean +/- std | 50% validation mean +/- std | 25% audit mean +/- std | 50% audit mean +/- std |", "|---|---:|---:|---:|---:|"]
    for method in summary["methods"]:
        m = summary["metrics"][method]
        lines.append(f"| {method} | {m['validation']['25pct']['mean']:.8g} +/- {m['validation']['25pct']['std']:.8g} | {m['validation']['50pct']['mean']:.8g} +/- {m['validation']['50pct']['std']:.8g} | {m['audit']['25pct']['mean']:.8g} +/- {m['audit']['25pct']['std']:.8g} | {m['audit']['50pct']['mean']:.8g} +/- {m['audit']['50pct']['std']:.8g} |")
    lines += ["", "## BTD-FL Direct Transfer Comparisons", "", "| Comparator | 25% mean improvement | 50% mean improvement | 25% seed wins | 50% seed wins | 25% target-seed wins / 12 | 50% target-seed wins / 12 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for comparator, item in summary["btd_vs_comparators"].items():
        old, new = item["25pct"], item["50pct"]
        lines.append(f"| {comparator} | {old['mean_relative_improvement']:.8g} | {new['mean_relative_improvement']:.8g} | {old['seeds_btd_wins']} | {new['seeds_btd_wins']} | {old['target_x_seed_validation_wins']} | {new['target_x_seed_validation_wins']} |")
    lines += ["", "## Guardrails", "", "test_evaluated: false", "canonical test remains locked", "donor direction and fallback counts are reported from each input summary"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "results" / "federated"
    parser.add_argument("--summary-25", type=Path, default=root / "formal_25pct" / "multiseed" / "multiseed_25pct_summary.json")
    parser.add_argument("--summary-50", type=Path, default=root / "formal_50pct" / "multiseed" / "multiseed_50pct_summary.json")
    parser.add_argument("--output-dir", type=Path, default=root)
    args = parser.parse_args()
    result = summarize(args.summary_25, args.summary_50)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "history_fraction_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "history_fraction_summary.md").write_text(render(result), encoding="utf-8")


if __name__ == "__main__":
    main()
