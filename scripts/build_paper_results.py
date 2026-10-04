"""Build paper-ready tables, figures, manifests, and captions from frozen JSON."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_ready"
SEEDS = (42, 123, 2026)
METHODS = ("scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer")
LABELS = {
    "scarce_local": "Scarce Local", "industrial_scarce_local": "Industrial Local",
    "fedavg": "FedAvg", "fedprox": "FedProx", "fedper": "FedPer",
    "fedfomo_style": "FedFomo-style", "btd_fl_direct_transfer": "BTD-FL",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    def check(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite value in {path}")
    check(data)
    if data.get("test_evaluated", data.get("guardrails", {}).get("test_evaluated", False)) is not False:
        raise ValueError(f"TEST guardrail failed: {path}")
    if "industrial_external" in str(path):
        for field in ("industrial_test_evaluated", "reference_grid_tests_evaluated"):
            if data.get(field, data.get("guardrails", {}).get(field, False)) is not False:
                raise ValueError(f"industrial TEST guardrail failed: {path}")
    return data


def write_table(path: Path, headers: list[str], rows: list[list[Any]], title: str) -> None:
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(headers); writer.writerows(rows)
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    return f"{value:.8g}"


def metric(summary: dict[str, Any], method: str, fraction: float, split: str, scope: str, name: str) -> dict[str, Any]:
    return summary["metrics"][method][split][scope][name]


def validate_reference(path: Path, fraction: float) -> dict[str, Any]:
    data = read_json(path)
    if data.get("seeds") != list(SEEDS) or set(data.get("methods", [])) != set(METHODS):
        raise ValueError(f"reference summary contract failed: {path}")
    if abs(float(data.get("history_fraction", fraction)) - fraction) > 1e-12 and fraction == 0.5:
        raise ValueError(f"reference history contract failed: {path}")
    return data


def validate_industrial(path: Path, fraction: float) -> dict[str, Any]:
    data = read_json(path)
    if data.get("seeds") != list(SEEDS) or set(data.get("methods", {})) != {"industrial_scarce_local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer"}:
        raise ValueError(f"industrial summary contract failed: {path}")
    if abs(float(data.get("history_fraction", -1)) - fraction) > 1e-12:
        raise ValueError(f"industrial history contract failed: {path}")
    return data


def build_reference_tables(out: Path, ref25: dict[str, Any], ref50: dict[str, Any]) -> list[dict[str, Any]]:
    headers = ["Method", "25% validation MAE mean", "25% validation MAE std", "25% audit MAE mean", "25% audit MAE std", "50% validation MAE mean", "50% validation MAE std", "50% audit MAE mean", "50% audit MAE std"]
    rows = []
    for method in METHODS:
        rows.append([LABELS[method]] + [fmt(metric(summary, method, fraction, split, "node_macro", "mae")[key]) for summary, fraction in ((ref25, .25), (ref50, .5)) for split in ("validation", "audit") for key in ("mean", "std")])
    for comparator in METHODS[:-1]:
        rows.append([f"BTD improvement vs {LABELS[comparator]}", fmt(ref25["btd_vs_comparators"][comparator]["mean_relative_improvement"]), fmt(ref25["btd_vs_comparators"][comparator]["std_relative_improvement"]), fmt(ref50["btd_vs_comparators"][comparator]["mean_relative_improvement"]), fmt(ref50["btd_vs_comparators"][comparator]["std_relative_improvement"]), "", "", "", ""])
    write_table(out / "table_reference_federated_main", headers, rows, "Reference-grid scarce-history benchmark")
    return rows


def build_industrial_table(out: Path, i25: dict[str, Any], i50: dict[str, Any]) -> None:
    headers = ["Method", "25% val MAE mean", "25% val MAE std", "25% val RMSE mean", "25% val RMSE std", "25% val WAPE mean", "25% val WAPE std", "25% val sMAPE mean", "25% val sMAPE std", "25% val grid MAE mean", "25% val grid MAE std", "50% val MAE mean", "50% val MAE std", "50% val RMSE mean", "50% val RMSE std", "50% val WAPE mean", "50% val WAPE std", "50% val sMAPE mean", "50% val sMAPE std", "50% val grid MAE mean", "50% val grid MAE std"]
    rows = []
    for method in METHODS:
        im = "industrial_scarce_local" if method == "scarce_local" else method
        values = []
        for summary in (i25, i50):
            for scope, name in (("node_macro", "mae"), ("node_macro", "rmse"), ("node_macro", "wape_pct"), ("node_macro", "smape_pct"), ("grid_aggregate", "mae")):
                values += [fmt(summary["methods"][im]["validation"][scope][name][key]) for key in ("mean", "std")]
        rows.append([LABELS[method]] + values)
    for comparator in METHODS[:-1]:
        rows.append([f"BTD improvement vs {LABELS[comparator]}", fmt(i25["comparisons_btd_vs_comparators"]["industrial_" + comparator if comparator == "scarce_local" else comparator]["mean_relative_improvement"]), "", "", "", "", "", "", "", "", "", fmt(i50["comparisons_btd_vs_comparators"]["industrial_" + comparator if comparator == "scarce_local" else comparator]["mean_relative_improvement"]), "", "", "", "", "", "", "", "", ""])
    write_table(out / "table_industrial_external_main", headers, rows, "Industrial external validation benchmark")


def build_centralized(out: Path, source: Path) -> None:
    data = read_json(source)
    rows = []
    for method, label in (("shared_gru", "GRU anchor"), ("temporal_residual_only", "temporal_residual_only"), ("graph_wavenet", "Graph WaveNet"), ("full_puc_rstattn_v2", "conditional-utility V2")):
        item = data["macro_across_grids"]["node_macro"].get(method) or data["macro_across_grids"]["node_macro"].get("puc_rstattn_v2")
        if item is None: continue
        values = item["metrics"]
        source_method = "puc_rstattn_v2" if method == "full_puc_rstattn_v2" else method
        rows.append([label, "development validation", fmt(values["mae"]), fmt(values["rmse"]), fmt(values["wape_pct"]), fmt(values["smape_pct"]), str(source.relative_to(ROOT)), f"macro_across_grids.node_macro.{source_method}.metrics"])
    write_table(out / "table_centralized_main", ["Method", "Split", "Node MAE", "Node RMSE", "Node WAPE %", "Node sMAPE %", "Source", "JSON key path"], rows, "Centralized architecture evidence")


def build_ablations(out: Path) -> None:
    rows = []
    main = read_json(ROOT / "results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json")
    main_macro = main["four_target_unweighted_macro"]
    rows.extend([
        ["Final BTD direct temporal transfer", "positive-benefit donor", "no; raw donor temporal state", split, fmt(main_macro[split]["node_macro"]["mae"]), "results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json", "0"]
        for split in ("audit", "validation")
    ])
    for filename, label, transfer, adapted in (("btd_no_benefit_selection_25pct_seed42.json", "No benefit selection", "fixed donor", "no"), ("btd_full_model_transfer_25pct_seed42.json", "Full-model transfer", "full trainable state", "probe-specific"), ("btd_no_target_adaptation_25pct_seed42.json", "No target adaptation", "raw temporal state", "no")):
        data = read_json(ROOT / "results" / "federated" / "formal_25pct" / "ablations" / filename)
        for split in ("audit", "validation"):
            value = data.get("four_target_unweighted_macro", {}).get(split, {}).get("node_macro", {}).get("mae")
            if value is None: raise ValueError(f"missing ablation macro metric: {filename}:{split}")
            main_value = main_macro[split]["node_macro"]["mae"]
            relative_difference = (float(value) - float(main_value)) / float(main_value)
            rows.append([label, transfer, adapted, split, fmt(value), filename, fmt(relative_difference)])
    write_table(out / "table_btd_ablations", ["Ablation", "Who/what transferred", "Adapted probe reused", "Split", "Node MAE", "Source", "Relative difference vs final BTD"], rows, "BTD-FL mechanism ablations")


def build_donor_tables(out: Path, ref25: dict[str, Any], ref50: dict[str, Any], i25: dict[str, Any], i50: dict[str, Any]) -> None:
    rows = []
    for fraction, summary in ((.25, ref25), (.5, ref50)):
        for target, seeds in summary["donor_selection_robustness"]["by_target"].items():
            for seed, item in seeds.items():
                rows.append(["reference-grid", f"{int(fraction*100)}%", seed, target, item["selected_donor"], item["selected_calibration_benefit"], json.dumps(item["calibration_benefit_by_donor"], sort_keys=True), item["zero_transfer_fallback"]])
        rows.append(["reference-grid summary", f"{int(fraction*100)}%", "all", "all directed edges", "", "", json.dumps({"positive": summary["donor_selection_robustness"]["positive_directed_benefit_count"], "non_positive": summary["donor_selection_robustness"]["non_positive_directed_benefit_count"]}, sort_keys=True), summary["donor_selection_robustness"]["fallback_occurrence_count"]])
    for fraction, summary in ((.25, i25), (.5, i50)):
        for seed, item in summary["btd_donor_behavior"]["by_seed"].items():
            rows.append(["industrial", f"{int(fraction*100)}%", seed, "norway_industrial_mvlv", item["selected_donor"], item["selected_calibration_benefit"], json.dumps(item["benefits"], sort_keys=True), item["zero_transfer_fallback"]])
        rows.append(["industrial summary", f"{int(fraction*100)}%", "all", "all directed edges", "", "", json.dumps({"positive": summary["btd_donor_behavior"]["positive_benefit_count"], "non_positive": summary["btd_donor_behavior"]["non_positive_benefit_count"]}, sort_keys=True), summary["btd_donor_behavior"]["fallback_count"]])
    write_table(out / "table_btd_donor_behavior", ["Domain", "History", "Seed", "Target", "Selected donor", "Selected benefit", "All donor benefits", "Fallback"], rows, "BTD target-conditioned directed transfer benefits")


def build_figures(out: Path, ref25: dict[str, Any], ref50: dict[str, Any], i25: dict[str, Any], i50: dict[str, Any]) -> None:
    labels = [LABELS[m] for m in METHODS]
    x = np.arange(2); fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, title, s25, s50, prefix in ((axes[0], "Reference-grid development validation", ref25, ref50, ""), (axes[1], "External industrial validation", i25, i50, "industrial_")):
        for method, label in zip(METHODS, labels):
            key = ("industrial_scarce_local" if method == "scarce_local" else method) if prefix else method
            section = "metrics" if not prefix else "methods"
            vals = [s[section][key]["validation"]["node_macro"]["mae"]["mean"] for s in (s25, s50)]
            errs = [s[section][key]["validation"]["node_macro"]["mae"]["std"] for s in (s25, s50)]
            ax.errorbar(x, vals, yerr=errs, marker="o", capsize=3, label=label)
        ax.set_xticks(x, ["25%", "50%"]); ax.set_title(title); ax.set_xlabel("Target history fraction"); ax.set_ylabel("Validation node-macro MAE"); ax.grid(axis="y", alpha=.25)
    axes[1].legend(fontsize=8, loc="best"); fig.savefig(out / "fig_history_fraction_mae.png", dpi=220); fig.savefig(out / "fig_history_fraction_mae.pdf"); plt.close(fig)
    comps = ("scarce_local", "fedavg", "fedfomo_style"); fig, ax = plt.subplots(figsize=(8, 4.6)); positions = np.arange(4); width=.24
    for idx, comp in enumerate(comps):
        vals = [ref25["btd_vs_comparators"][comp]["mean_relative_improvement"], ref50["btd_vs_comparators"][comp]["mean_relative_improvement"], i25["comparisons_btd_vs_comparators"]["industrial_"+comp if comp == "scarce_local" else comp]["mean_relative_improvement"], i50["comparisons_btd_vs_comparators"]["industrial_"+comp if comp == "scarce_local" else comp]["mean_relative_improvement"]]
        ax.bar(positions + (idx-1)*width, vals, width, label=LABELS[comp])
    ax.set_xticks(positions, ["Ref 25%", "Ref 50%", "Ind 25%", "Ind 50%"]); ax.set_ylabel("BTD relative validation MAE improvement"); ax.axhline(0, color="black", linewidth=.8); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "fig_btd_relative_improvement.png", dpi=220); fig.savefig(out / "fig_btd_relative_improvement.pdf"); plt.close(fig)
    rows = []
    for fraction, summary in ((.25, i25), (.5, i50)):
        for seed, item in summary["btd_donor_behavior"]["by_seed"].items(): rows.append((f"{seed} / {int(fraction*100)}%", item["benefits"]))
    donors = list(rows[0][1]); matrix = np.array([[float(item[1][d]) for d in donors] for item in rows]); fig, ax = plt.subplots(figsize=(8, 4.8)); im=ax.imshow(matrix, cmap="coolwarm", aspect="auto"); ax.set_xticks(range(4), ["39", "50", "56", "80"]); ax.set_yticks(range(len(rows)), [item[0] for item in rows]); ax.set_xlabel("Reference donor"); ax.set_ylabel("Seed / history"); fig.colorbar(im, ax=ax, label="Directed calibration benefit"); fig.tight_layout(); fig.savefig(out / "fig_industrial_benefit_matrix.png", dpi=220); fig.savefig(out / "fig_industrial_benefit_matrix.pdf"); plt.close(fig)


def build_manifest(paths: list[Path]) -> None:
    entries = []
    for path in paths:
        data = read_json(path)
        guardrail = data.get("test_evaluated", data.get("guardrails", {}).get("test_evaluated", False))
        entries.append({"section": "frozen-results", "method": data.get("method_name", data.get("methods")), "dataset_domain": str(path.parent.relative_to(ROOT)), "history_fraction": data.get("history_fraction"), "seed_or_mean_std": data.get("seed", data.get("seeds")), "metric": "source artifact", "value": "see JSON", "source_artifact_path": str(path.relative_to(ROOT)), "json_key_path": "artifact root", "frozen_status": "accepted frozen", "test_evaluated": guardrail})
    for entry in entries:
        if not (ROOT / entry["source_artifact_path"]).exists():
            raise FileNotFoundError(entry["source_artifact_path"])
    manifest = {"description": "Audit trail for paper-ready frozen results; numeric tables are generated from exact JSON values.", "entries": entries}
    (OUT / "manifests" / "paper_results_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Paper Results Manifest", "", "All entries trace to frozen JSON artifacts. Canonical TEST metrics are excluded.", "", "| Source | Method | History | Seed(s) | TEST evaluated |", "|---|---|---:|---|---|"]
    lines += [f"| `{e['source_artifact_path']}` | {e['method']} | {e['history_fraction']} | {e['seed_or_mean_std']} | {e['test_evaluated']} |" for e in entries]
    (OUT / "manifests" / "paper_results_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_docs(ref25: dict[str, Any], ref50: dict[str, Any], i25: dict[str, Any], i50: dict[str, Any]) -> None:
    (OUT / "paper_captions.md").write_text("""# Paper Captions\n\n**Table A.** Centralized development-validation evidence across the four reference grids.\n\n**Table B.** Reference-grid three-seed robustness under 25% and 50% target history; validation is development validation.\n\n**Table C.** External industrial validation on hourly industrial load/energy measurements. Absolute MAE is not compared across domains.\n\n**Table D.** Frozen seed-42 BTD-FL mechanism ablations, distinguishing donor identity, transferred state, and probe reuse.\n\n**Table E.** Target-conditioned directed transfer benefits and selected donors; values are not causal effects.\n\n**Figure 1.** Mean and standard deviation of validation node-macro MAE across history fractions.\n\n**Figure 2.** BTD relative validation node-MAE improvement versus Local, FedAvg, and FedFomo-style.\n\n**Figure 3.** Industrial target-conditioned directed calibration benefits by seed and history fraction.\n""", encoding="utf-8")
    (OUT / "paper_results_summary.md").write_text("""# Frozen Results Digest\n\nAll statements below use frozen JSON artifacts and three-seed mean ± standard deviation; no statistical significance claim is made. Reference-grid canonical validation is development validation, while industrial canonical validation is external industrial validation.\n\n## Temporal dominance\n\nThe centralized macro evidence includes the GRU anchor, temporal-residual-only model, Graph WaveNet, and conditional-utility V2 evidence. The temporal-residual-only variant improves the GRU anchor in node-macro MAE, while physical-topology variants are not uniformly better.\n\n## Reference-grid robustness\n\nAt both 25% and 50% history, BTD-FL is compared using exact same-regime three-seed summaries. Relative improvements are reported in Table B and Figure 2; these are robustness summaries, not significance tests.\n\n## Industrial external validation\n\nBTD-FL's primary industrial node-macro MAE is compared within the industrial domain only. RMSE and grid-aggregate metrics provide supporting evidence, while node WAPE and sMAPE are not uniformly improved. Industrial values are hourly industrial load/energy measurements and are not instantaneous MW.\n\n## Mechanism and benefit behavior\n\nThe ablations distinguish donor selection, transfer scope, and target adaptation. Directed benefits are target-conditioned calibration quantities, not causal benefits. Industrial positive/non-positive directed benefit counts are reported directly; zero-transfer fallback was not empirically triggered in the accepted summaries and no fallback-performance claim is made.\n\nCanonical TEST remains locked and no TEST claim is made.\n""", encoding="utf-8")
    (OUT / "test_opening_protocol.md").write_text("""# Canonical TEST Opening Protocol\n\nFrozen commit: `1681741779df301dcb64587090ce262d2ad1d443`.\n\nOpening TEST is an evaluation event, not a development iteration. Before opening TEST, freeze code, seeds 42/123/2026, all hyperparameters, and the 25%/50% regimes. Evaluate the frozen Local, FedAvg, FedProx, FedPer, FedFomo-style, and BTD-FL methods on reference-grid TEST and industrial TEST according to the pre-registered data splits. Do not tune after TEST is opened.\n\nWrite one-shot artifacts with explicit `test_evaluated=true`, split provenance, seed, method, history fraction, and commit metadata. Reference-grid and industrial TEST outputs must be separate from development-validation artifacts. Negative TEST results are retained and reported without method changes or post-hoc seed selection.\n""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--no-figures", action="store_true"); args = parser.parse_args()
    tables, figures, manifests = OUT / "tables", OUT / "figures", OUT / "manifests"
    for directory in (tables, figures, manifests): directory.mkdir(parents=True, exist_ok=True)
    ref25 = validate_reference(ROOT / "results/federated/formal_25pct/multiseed/multiseed_25pct_summary.json", .25)
    ref50 = validate_reference(ROOT / "results/federated/formal_50pct/multiseed/multiseed_50pct_summary.json", .5)
    i25 = validate_industrial(ROOT / "results/federated/industrial_external/industrial_external_25pct_summary.json", .25)
    i50 = validate_industrial(ROOT / "results/federated/industrial_external/industrial_external_50pct_summary.json", .5)
    build_centralized(tables, ROOT / "results/centralized/centralized_validation_summary.json")
    build_reference_tables(tables, ref25, ref50); build_industrial_table(tables, i25, i50); build_ablations(tables); build_donor_tables(tables, ref25, ref50, i25, i50)
    if not args.no_figures: build_figures(figures, ref25, ref50, i25, i50)
    paths = [ROOT / "results/federated/formal_25pct/multiseed/multiseed_25pct_summary.json", ROOT / "results/federated/formal_50pct/multiseed/multiseed_50pct_summary.json", ROOT / "results/federated/industrial_external/industrial_external_25pct_summary.json", ROOT / "results/federated/industrial_external/industrial_external_50pct_summary.json", ROOT / "results/centralized/centralized_validation_summary.json", ROOT / "results/ablations/puc_rstattn_v2_ablation_summary.json"]
    build_manifest(paths); write_docs(ref25, ref50, i25, i50)
    print(OUT)


if __name__ == "__main__": main()
