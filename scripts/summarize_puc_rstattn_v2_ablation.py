"""Combine formal PUC-RSTAttn V2 ablation validation reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_puc_rstattn_v2_ablation import VARIANTS

GRIDS = ("39bus", "50bus", "56bus", "80bus")
FULL_FILE = "puc_rstattn_v2_{grid}_p_calendar.json"
ABLATION_FILE = "puc_rstattn_v2_{variant}_{grid}_p_calendar.json"
METRICS = ("mae", "rmse", "wape_pct", "smape_pct")
MODELS = (*VARIANTS, "full_puc_rstattn_v2")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_full(data: dict[str, Any], path: Path) -> dict[str, Any]:
    config = data.get("config", {})
    if config.get("model") != "PUC-RSTAttn-V2" or config.get("feature_mode") != "p_calendar" or config.get("test_evaluated") is not False or data.get("test_evaluated") is not False:
        raise ValueError(f"invalid frozen full-model result: {path.name}")
    return data["validation"]["final"]


def _validate_ablation(data: dict[str, Any], variant: str, path: Path) -> dict[str, Any]:
    config = data.get("config", {})
    required = {
        "experiment": "puc_rstattn_v2_formal_ablation",
        "variant": variant,
        "independent_training": True,
        "seed": 42,
        "feature_mode": "p_calendar",
        "history_length": 168,
        "forecast_horizon": 1,
        "utility_top_k": 3,
        "utility_positive_only": True,
        "test_evaluated": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"invalid {variant} metadata {key} in {path.name}")
    if config.get("validation_used_for_graph") is not False or data.get("test_evaluated") is not False:
        raise ValueError(f"invalid {variant} graph/test metadata in {path.name}")
    return data["validation"]["final"]


def load_reports(centralized_dir: Path, ablation_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    reports: dict[str, dict[str, dict[str, Any]]] = {grid: {} for grid in GRIDS}
    for grid in GRIDS:
        full_path = centralized_dir / FULL_FILE.format(grid=grid)
        if not full_path.exists():
            raise FileNotFoundError(full_path)
        reports[grid]["full_puc_rstattn_v2"] = _validate_full(_load_json(full_path), full_path)
        for variant in VARIANTS:
            path = ablation_dir / ABLATION_FILE.format(variant=variant, grid=grid)
            if not path.exists():
                raise FileNotFoundError(path)
            reports[grid][variant] = _validate_ablation(_load_json(path), variant, path)
    return reports


def _macro(reports: dict[str, dict[str, dict[str, Any]]], model: str, scope: str) -> dict[str, float]:
    return {metric: float(np.mean([reports[grid][model][scope][metric] for grid in GRIDS])) for metric in METRICS}


def summarize(centralized_dir: Path, ablation_dir: Path) -> dict[str, Any]:
    reports = load_reports(centralized_dir, ablation_dir)
    per_grid = {
        grid: {model: {scope: reports[grid][model][scope] for scope in ("node_macro", "grid_aggregate")} for model in MODELS}
        for grid in GRIDS
    }
    macro = {
        scope: {model: {"metrics": _macro(reports, model, scope), "available_grid_count": 4, "expected_grid_count": 4} for model in MODELS}
        for scope in ("node_macro", "grid_aggregate")
    }
    delta_pairs = (
        ("temporal_residual_only", "gru_anchor_only"),
        ("utility_uniform_spatial", "temporal_residual_only"),
        ("utility_prior_no_physics", "utility_uniform_spatial"),
        ("full_puc_rstattn_v2", "utility_prior_no_physics"),
    )
    deltas = {}
    for left, right in delta_pairs:
        deltas[f"{left}_vs_{right}"] = {
            "node_macro_mae_delta_by_grid": {grid: float(reports[grid][left]["node_macro"]["mae"] - reports[grid][right]["node_macro"]["mae"]) for grid in GRIDS},
            "node_macro_mae_delta_unweighted_macro": float(np.mean([reports[grid][left]["node_macro"]["mae"] - reports[grid][right]["node_macro"]["mae"] for grid in GRIDS])),
        }
        delta = deltas[f"{left}_vs_{right}"]["node_macro_mae_delta_unweighted_macro"]
        reference = float(np.mean([reports[grid][right]["node_macro"]["mae"] for grid in GRIDS]))
        deltas[f"{left}_vs_{right}"]["absolute_delta"] = delta
        deltas[f"{left}_vs_{right}"]["node_macro_mae_delta_unweighted_macro"] = delta
        deltas[f"{left}_vs_{right}"]["relative_pct"] = float(100.0 * delta / reference) if reference else None
    return {
        "metadata": {"experiment": "puc_rstattn_v2_formal_ablation", "cross_grid_aggregation": "unweighted_macro_mean", "test_evaluated": False},
        "models": list(MODELS),
        "per_grid": per_grid,
        "macro_across_grids": macro,
        "incremental_deltas": deltas,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# PUC-RSTAttn V2 Formal Ablation Summary", "", "Validation-only metrics; frozen full-model results are loaded from centralized outputs."]
    for scope, title in (("node_macro", "Per-grid node macro"), ("grid_aggregate", "Per-grid grid aggregate")):
        lines.extend(["", f"## {title}", "", "| Grid | Model | MAE | RMSE | WAPE (%) | sMAPE (%) |", "|---|---|---:|---:|---:|---:|"])
        for grid in GRIDS:
            for model in report["models"]:
                values = report["per_grid"][grid][model][scope]
                lines.append(f"| {grid} | {model} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
    for scope in ("node_macro", "grid_aggregate"):
        lines.extend(["", f"## Cross-grid {scope}", "", "| Model | MAE | RMSE | WAPE (%) | sMAPE (%) |", "|---|---:|---:|---:|---:|"])
        for model in report["models"]:
            values = report["macro_across_grids"][scope][model]["metrics"]
            lines.append(f"| {model} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
    lines.extend(["", "## Numeric node-macro MAE deltas", "", "| Comparison | Absolute delta | Relative (%) |", "|---|---:|---:|"])
    for name, item in report["incremental_deltas"].items():
        relative = item["relative_pct"]
        lines.append(f"| {name} | {item['absolute_delta']:.6g} | {'missing' if relative is None else f'{relative:.6g}'} |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centralized-dir", type=Path, default=Path("results/centralized"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("results/ablations"))
    parser.add_argument("--output-json", type=Path, default=Path("results/ablations/puc_rstattn_v2_ablation_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/ablations/puc_rstattn_v2_ablation_summary.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = summarize(args.centralized_dir, args.ablation_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
