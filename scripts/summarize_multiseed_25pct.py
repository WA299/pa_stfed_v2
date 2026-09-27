"""Aggregate frozen 25%-history formal results across seeds 42, 123, and 2026."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

METHODS = ("scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer")
SEEDS = (42, 123, 2026)
TARGETS = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
METRICS = ("mae", "rmse", "wape_pct", "smape_pct")
SCOPES = ("node_macro", "grid_aggregate")
SPLITS = ("audit", "validation")


def _read(path: Path, seed: int, *, artifact_type: str, methods: tuple[str, ...]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required seed result is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if "seed" not in report or int(report["seed"]) != seed:
        raise ValueError(f"seed metadata does not match filename: {path}")
    if report.get("test_evaluated") is not False:
        raise ValueError(f"test guardrail violated: {path}")
    historical = seed == 42
    if not historical:
        if report.get("run_mode") != "real" or report.get("artifact_type") != artifact_type:
            raise ValueError(f"non-historical artifact lacks the required real-run schema: {path}")
    for key in ("validation_used_for_selection", "audit_used_for_selection", "canonical_validation_evaluated_during_training"):
        if key in report and report[key] is not False:
            raise ValueError(f"selection/evaluation guardrail violated ({key}): {path}")
    if tuple(report.get("client_grid_names", TARGETS)) != TARGETS:
        raise ValueError(f"scenario set mismatch: {path}")
    if len(report.get("scenarios", {})) != 4 or set(report["scenarios"]) != set(TARGETS):
        raise ValueError(f"all four scarce scenarios are required: {path}")
    if not historical and set(report.get("methods", ())) != set(methods):
        raise ValueError(f"artifact {path} has an unexpected method set")
    return report


def _metric_from(report: Mapping[str, Any], target: str, method: str) -> Mapping[str, Any]:
    scenario = report["scenarios"][target]
    if method in {"scarce_local", "fedavg", "fedprox", "fedper"}:
        return scenario["methods"][method]
    if method == "btd_fl_direct_transfer":
        return scenario["metrics"]
    return scenario["scarce_target_metrics"]


def _macro(target_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {split: {scope: {metric: mean(float(target_metrics[t][split][scope][metric]) for t in TARGETS) for metric in METRICS} for scope in SCOPES} for split in SPLITS}


def _result_paths(root: Path, seed: int) -> dict[str, Path]:
    if seed == 42:
        base = root.parent
        return {
            "baseline": base / "btd_fl_25pct_seed42.json",
            "btd_fl_direct_transfer": base / "btd_fl_direct_transfer_25pct_seed42.json",
            "fedfomo_style": base / "fedfomo_style_25pct_seed42.json",
        }
    return {
        "baseline": root / f"formal_baselines_25pct_seed{seed}.json",
        "btd_fl_direct_transfer": root / f"btd_fl_direct_transfer_25pct_seed{seed}.json",
        "fedfomo_style": root / f"fedfomo_style_25pct_seed{seed}.json",
    }


def _load_seed(root: Path, seed: int) -> dict[str, Any]:
    paths = _result_paths(root, seed)
    baseline_methods = ("scarce_local", "fedavg", "fedprox", "fedper") if seed != 42 else ("scarce_local", "fedavg", "fedprox", "fedper", "btd_fl")
    baseline = _read(paths["baseline"], seed, artifact_type="formal_baselines", methods=baseline_methods)
    direct = _read(paths["btd_fl_direct_transfer"], seed, artifact_type="btd_fl_direct_transfer", methods=("btd_fl_direct_transfer",))
    fomo = _read(paths["fedfomo_style"], seed, artifact_type="fedfomo_style", methods=("fedfomo_style",))
    if seed == 42 and baseline.get("btd_variant") != "btd_fl":
        raise ValueError(f"baseline artifact is not the accepted formal baseline: {paths['baseline']}")
    if seed != 42 and baseline.get("artifact_type") != "formal_baselines":
        raise ValueError(f"new-seed baseline artifact is not baseline-only: {paths['baseline']}")
    if not set(("scarce_local", "fedavg", "fedprox", "fedper") + (("btd_fl",) if seed == 42 else ())).issubset(set(baseline.get("methods", ()) )):
        raise ValueError(f"formal baseline is missing an accepted method: {paths['baseline']}")
    if direct.get("btd_variant") != "btd_fl_direct_transfer":
        raise ValueError(f"direct-transfer artifact has wrong variant: {paths['btd_fl_direct_transfer']}")
    if fomo.get("method_name") != "FedFomo-style":
        raise ValueError(f"FedFomo artifact has wrong method name: {paths['fedfomo_style']}")
    return {"baseline": baseline, "btd_fl_direct_transfer": direct, "fedfomo_style": fomo}


def summarize(root: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    if 42 not in seeds:
        raise ValueError("seed 42 is required as the frozen historical reference")
    loaded = {seed: _load_seed(root, seed) for seed in seeds}
    normalized: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    for seed, reports in loaded.items():
        for method in METHODS:
            source = reports["baseline"] if method in {"scarce_local", "fedavg", "fedprox", "fedper"} else reports[method]
            normalized[method][seed] = {target: dict(_metric_from(source, target, method)) for target in TARGETS}

    stats: dict[str, Any] = {}
    for method in METHODS:
        stats[method] = {}
        for split in SPLITS:
            stats[method][split] = {}
            for scope in SCOPES:
                for metric in METRICS:
                    values = [_macro(normalized[method][seed])[split][scope][metric] for seed in seeds]
                    stats[method][split].setdefault(scope, {})[metric] = {"mean": mean(values), "std": pstdev(values), "per_seed": {str(seed): value for seed, value in zip(seeds, values)}}

    btd = "btd_fl_direct_transfer"
    comparisons: dict[str, Any] = {}
    for comparator in ("scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style"):
        per_seed = {}
        target_win_count = 0
        seed_win_count = 0
        for seed in seeds:
            btd_macro = _macro(normalized[btd][seed]); comp_macro = _macro(normalized[comparator][seed])
            relative = {split: float((comp_macro[split]["node_macro"]["mae"] - btd_macro[split]["node_macro"]["mae"]) / (comp_macro[split]["node_macro"]["mae"] + 1e-12)) for split in SPLITS}
            target_wins = {split: sum(normalized[btd][seed][target][split]["node_macro"]["mae"] < normalized[comparator][seed][target][split]["node_macro"]["mae"] for target in TARGETS) for split in SPLITS}
            target_win_count += target_wins["validation"]
            seed_win_count += int(btd_macro["validation"]["node_macro"]["mae"] < comp_macro["validation"]["node_macro"]["mae"])
            per_seed[str(seed)] = {"relative_improvement": relative, "target_win_counts": target_wins}
        vals = [per_seed[str(seed)]["relative_improvement"]["validation"] for seed in seeds]
        comparisons[comparator] = {"per_seed": per_seed, "mean_relative_improvement": mean(vals), "std_relative_improvement": pstdev(vals), "seeds_btd_wins": seed_win_count, "scarce_target_validation_wins_out_of_12": target_win_count}

    donor_robustness: dict[str, Any] = {target: {} for target in TARGETS}
    target_summary: dict[str, Any] = {}
    positive = non_positive = fallback_count = 0
    for target in TARGETS:
        for seed in seeds:
            report = loaded[seed]["btd_fl_direct_transfer"]
            metadata = report["scenarios"][target]["metadata"]
            benefits = metadata.get("calibration_benefit_by_donor", {})
            selected = metadata.get("selected_donor")
            donor_robustness[target][str(seed)] = {"selected_donor": selected, "selected_calibration_benefit": metadata.get("selected_calibration_benefit"), "zero_transfer_fallback": metadata.get("zero_transfer_fallback", False), "calibration_benefit_by_donor": benefits}
            fallback_count += int(bool(metadata.get("zero_transfer_fallback", False)))
            for value in benefits.values():
                if float(value) > 0: positive += 1
                else: non_positive += 1
        selected_donors = [donor_robustness[target][str(seed)]["selected_donor"] for seed in seeds]
        target_summary[target] = {
            "selected_donor_by_seed": {str(seed): donor_robustness[target][str(seed)]["selected_donor"] for seed in seeds},
            "unique_selected_donor_count": len({donor for donor in selected_donors if donor is not None}),
            "fallback_occurrence_count": sum(bool(donor_robustness[target][str(seed)]["zero_transfer_fallback"]) for seed in seeds),
        }
    return {
        "experiment": "multiseed_formal_25pct_summary",
        "seeds": list(seeds),
        "methods": list(METHODS),
        "client_grid_names": list(TARGETS),
        "primary_metric": "four-scenario unweighted canonical-validation node-macro MAE",
        "metrics": stats,
        "per_seed_metrics": {method: {str(seed): _macro(normalized[method][seed]) for seed in seeds} for method in METHODS},
        "btd_vs_comparators": comparisons,
        "donor_selection_robustness": {"by_target": donor_robustness, "target_summary": target_summary, "positive_directed_benefit_count": positive, "non_positive_directed_benefit_count": non_positive, "fallback_occurrence_count": fallback_count, "directed_edge_observations": positive + non_positive},
        "guardrails": {"validation_used_for_selection": False, "audit_used_for_selection": False, "test_evaluated": False, "seed42_retrained": False},
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Frozen 25%-History Multi-Seed Robustness Summary", "", "Seeds 42, 123, and 2026 were fixed in advance. Seed 42 is historical and is read, not retrained. No hyperparameters or scientific definitions vary by seed.", "", "Canonical validation is robustness evaluation only; canonical test remains locked. Three seeds do not support significance claims.", "", "## Primary Metrics", "", "| Method | Validation node-MAE mean | std | Audit node-MAE mean | std |", "|---|---:|---:|---:|---:|"]
    for method in summary["methods"]:
        val = summary["metrics"][method]["validation"]["node_macro"]["mae"]; audit = summary["metrics"][method]["audit"]["node_macro"]["mae"]
        lines.append(f"| {method} | {val['mean']:.8g} | {val['std']:.8g} | {audit['mean']:.8g} | {audit['std']:.8g} |")
    lines += ["", "## Per-Seed Primary Node-MAE", "", "| Method | Seed 42 validation | Seed 123 validation | Seed 2026 validation | Seed 42 audit | Seed 123 audit | Seed 2026 audit |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in summary["methods"]:
        vals = [summary["metrics"][method][split]["node_macro"]["mae"]["per_seed"][str(seed)] for split in ("validation", "audit") for seed in summary["seeds"]]
        lines.append("| " + method + " | " + " | ".join(f"{value:.8g}" for value in vals) + " |")
    lines += ["", "## BTD-FL Direct Transfer Robustness", "", "| Comparator | Mean relative improvement | std | BTD wins by seed | Scarce-target validation wins / 12 |", "|---|---:|---:|---:|---:|"]
    for comparator, item in summary["btd_vs_comparators"].items():
        lines.append(f"| {comparator} | {item['mean_relative_improvement']:.8g} | {item['std_relative_improvement']:.8g} | {item['seeds_btd_wins']} | {item['scarce_target_validation_wins_out_of_12']} |")
    lines += ["", "## Donor Selection Robustness", "", "| Target | Seed 42 donor | Seed 123 donor | Seed 2026 donor |", "|---|---|---|---|"]
    for target in summary["client_grid_names"]:
        row = [summary["donor_selection_robustness"]["by_target"][target][str(seed)]["selected_donor"] or "none" for seed in summary["seeds"]]
        lines.append(f"| {target} | " + " | ".join(row) + " |")
    d = summary["donor_selection_robustness"]
    lines += ["", f"Directed benefit signs: {d['positive_directed_benefit_count']} positive, {d['non_positive_directed_benefit_count']} non-positive across {d['directed_edge_observations']} observations; fallbacks: {d['fallback_occurrence_count']}.", "", "## Guardrails", "validation_used_for_selection: false", "audit_used_for_selection: false", "test_evaluated: false", "seed42_retrained: false"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "federated" / "formal_25pct" / "multiseed")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    summary = summarize(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multiseed_25pct_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "multiseed_25pct_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {output_dir / 'multiseed_25pct_summary.json'} and {output_dir / 'multiseed_25pct_summary.md'}")


if __name__ == "__main__":
    main()
