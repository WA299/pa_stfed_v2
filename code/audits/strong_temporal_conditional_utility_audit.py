"""Train-only strong-temporal conditional predictive-utility audit.

The candidate graph is selected with a 173-dimensional self-temporal Ridge
model on the fit/selection segments.  The final audit segment is touched only
by the independent refit/evaluation stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.audits.predictive_utility_graph_audit import (
    AUDIT_SAMPLES,
    FIT_SAMPLES,
    FORMAL_TRAIN_VALID_TARGETS,
    SELECTION_SAMPLES,
    formal_train_target_indices,
    fit_scaled_ridge,
    predict_scaled_ridge,
    select_electrical_neighbors,
    signed_utility,
    split_train_targets as _split_train_targets,
)
from code.data.lv_grid_loader import LVGridData, LVGridLoader

SELF_HISTORY_LAGS = tuple(range(1, 169))
SOURCE_LAGS = (1, 24, 168)
CALENDAR_FEATURE_DIM = 5
SELF_FEATURE_DIM = len(SELF_HISTORY_LAGS) + CALENDAR_FEATURE_DIM
PAIR_FEATURE_DIM = SELF_FEATURE_DIM + len(SOURCE_LAGS)
RIDGE_ALPHA = 1.0
UTILITY_TOP_K = 3
MODEL_NAMES = (
    "strong_self",
    "conditional_utility_top3",
    "conditional_utility_uniform_mean_context",
)
METRIC_NAMES = ("mae", "rmse", "wape_pct", "smape_pct")


def chronological_split(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if len(indices) != FORMAL_TRAIN_VALID_TARGETS:
        raise ValueError(f"expected {FORMAL_TRAIN_VALID_TARGETS} formal train targets")
    if len(indices) > 1 and not np.all(np.diff(indices) > 0):
        raise ValueError("target indices must be strictly chronological")
    return _split_train_targets(indices)


split_train_targets = chronological_split


def calendar_features(timestamps: Any, target_indices: np.ndarray) -> np.ndarray:
    selected = timestamps[np.asarray(target_indices, dtype=int)]
    hour = np.asarray([value.hour for value in selected], dtype=float)
    weekday = np.asarray([value.dayofweek for value in selected], dtype=float)
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            (weekday >= 5).astype(float),
        ]
    )


def strong_self_features(
    p: np.ndarray, timestamps: Any, target_indices: np.ndarray, target_node: int
) -> np.ndarray:
    indices = np.asarray(target_indices, dtype=int)
    self_lags = [np.asarray(p[indices - lag, int(target_node)], dtype=float) for lag in SELF_HISTORY_LAGS]
    return np.column_stack(self_lags + [calendar_features(timestamps, indices)])


def pair_features(
    p: np.ndarray,
    timestamps: Any,
    target_indices: np.ndarray,
    target_node: int,
    source_node: int,
) -> np.ndarray:
    if int(source_node) == int(target_node):
        raise ValueError("source_node must differ from target_node")
    self_part = strong_self_features(p, timestamps, target_indices, target_node)
    indices = np.asarray(target_indices, dtype=int)
    source_part = [np.asarray(p[indices - lag, int(source_node)], dtype=float) for lag in SOURCE_LAGS]
    return np.column_stack([self_part] + source_part)


def selected_top3_features(
    p: np.ndarray,
    timestamps: Any,
    target_indices: np.ndarray,
    target_node: int,
    sources: Iterable[int],
) -> np.ndarray:
    sources = [int(source) for source in sources]
    if any(source == int(target_node) for source in sources):
        raise ValueError("selected sources must exclude target_node")
    x = strong_self_features(p, timestamps, target_indices, target_node)
    if sources:
        extra = [np.asarray(p[np.asarray(target_indices) - lag, source], dtype=float) for source in sources for lag in SOURCE_LAGS]
        x = np.column_stack([x] + extra)
    return np.asarray(x, dtype=float)


def uniform_mean_context_features(
    p: np.ndarray,
    timestamps: Any,
    target_indices: np.ndarray,
    target_node: int,
    sources: Iterable[int],
) -> np.ndarray:
    indices = np.asarray(target_indices, dtype=int)
    x = strong_self_features(p, timestamps, indices, target_node)
    sources = [int(source) for source in sources]
    if sources:
        context = [np.mean([p[indices - lag, source] for source in sources], axis=0) for lag in SOURCE_LAGS]
    else:
        context = [np.zeros(len(indices), dtype=float) for _ in SOURCE_LAGS]
    return np.column_stack([x] + context)


# Stable descriptive aliases used by audit notebooks and lightweight tests.
strong_self_feature_matrix = strong_self_features
conditional_pair_feature_matrix = pair_features
top3_feature_matrix = selected_top3_features
uniform_context_feature_matrix = uniform_mean_context_features
build_features = strong_self_features


def load_bus_indices(grid: LVGridData) -> np.ndarray:
    loads = np.flatnonzero(np.asarray(grid.load_bus_mask, dtype=bool)).astype(int)
    if len(loads) < 2:
        raise ValueError("strong-temporal audit requires at least two load buses")
    return loads


def compute_conditional_utilities(
    grid: LVGridData,
    fit_indices: np.ndarray,
    selection_indices: np.ndarray,
    loads: Iterable[int] | None = None,
) -> tuple[dict[tuple[int, int], float], list[dict[str, Any]]]:
    """Fit on fit only and score both models on selection only."""
    loads = load_bus_indices(grid) if loads is None else np.asarray(list(loads), dtype=int)
    valid_loads = set(load_bus_indices(grid).tolist())
    if any(int(node) not in valid_loads for node in loads):
        raise ValueError("conditional utility sources and targets must be load buses")
    utilities: dict[tuple[int, int], float] = {}
    records: list[dict[str, Any]] = []
    for target in loads:
        x_self_fit = strong_self_features(grid.p, grid.timestamps, fit_indices, int(target))
        x_self_selection = strong_self_features(grid.p, grid.timestamps, selection_indices, int(target))
        self_model = fit_scaled_ridge(x_self_fit, grid.p[fit_indices, target])
        mae_self = float(np.mean(np.abs(predict_scaled_ridge(self_model, x_self_selection) - grid.p[selection_indices, target])))
        for source in loads:
            if int(source) == int(target):
                continue
            x_pair_fit = pair_features(grid.p, grid.timestamps, fit_indices, int(target), int(source))
            x_pair_selection = pair_features(grid.p, grid.timestamps, selection_indices, int(target), int(source))
            pair_model = fit_scaled_ridge(x_pair_fit, grid.p[fit_indices, target])
            mae_pair = float(np.mean(np.abs(predict_scaled_ridge(pair_model, x_pair_selection) - grid.p[selection_indices, target])))
            utility = signed_utility(mae_self, mae_pair)
            key = (int(source), int(target))
            utilities[key] = utility
            records.append({
                "source_node_index": int(source),
                "source_bus_id": str(grid.node_ids[source]),
                "target_node_index": int(target),
                "target_bus_id": str(grid.node_ids[target]),
                "mae_strong_self_selection": mae_self,
                "mae_strong_self_plus_source_selection": mae_pair,
                "conditional_utility": utility,
            })
    return utilities, records


def select_conditional_neighbors(
    utilities: Mapping[tuple[int, int], float], loads: Iterable[int], top_k: int = UTILITY_TOP_K
) -> dict[int, list[tuple[int, float]]]:
    result: dict[int, list[tuple[int, float]]] = {}
    load_list = [int(node) for node in loads]
    for target in load_list:
        candidates = [(source, float(utilities[(source, target)])) for source in load_list if source != target and float(utilities[(source, target)]) > 0.0]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        result[target] = candidates[:top_k]
    return result


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    denominator = max(float(np.sum(np.abs(actual))), 1e-12)
    smape_denominator = np.maximum(np.abs(actual) + np.abs(prediction), 1e-12)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_pct": float(100.0 * np.sum(np.abs(error)) / denominator),
        "smape_pct": float(100.0 * np.mean(2.0 * np.abs(error) / smape_denominator)),
    }


def _neighbor_ids(neighbors: Mapping[int, list[tuple[int, float]]], target: int) -> list[int]:
    return [int(source) for source, _ in neighbors.get(int(target), [])]


def evaluate_selected_graph(grid: LVGridData, selection: Mapping[str, Any]) -> dict[str, Any]:
    refit = np.concatenate([selection["fit_indices"], selection["selection_indices"]])
    audit = np.asarray(selection["audit_indices"], dtype=int)
    if len(refit) != FIT_SAMPLES + SELECTION_SAMPLES:
        raise ValueError("conditional audit refit must contain exactly 4771 samples")
    if len(audit) != AUDIT_SAMPLES:
        raise ValueError("conditional audit segment must contain exactly 1193 samples")
    loads = np.asarray(selection["load_indices"], dtype=int)
    neighbors = selection["conditional_neighbors"]
    per_model: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        actual_columns: list[np.ndarray] = []
        prediction_columns: list[np.ndarray] = []
        per_target: dict[str, Any] = {}
        for target in loads:
            sources = _neighbor_ids(neighbors, int(target))
            if model_name == "strong_self":
                make_features = lambda indices: strong_self_features(grid.p, grid.timestamps, indices, int(target))
            elif model_name == "conditional_utility_top3":
                make_features = lambda indices: selected_top3_features(grid.p, grid.timestamps, indices, int(target), sources)
            else:
                make_features = lambda indices: uniform_mean_context_features(grid.p, grid.timestamps, indices, int(target), sources)
            x_refit = make_features(refit)
            x_audit = make_features(audit)
            fitted = fit_scaled_ridge(x_refit, grid.p[refit, target])
            prediction = predict_scaled_ridge(fitted, x_audit)
            actual = np.asarray(grid.p[audit, target], dtype=float)
            actual_columns.append(actual)
            prediction_columns.append(prediction)
            per_target[str(grid.node_ids[target])] = {
                "node_index": int(target),
                "source_node_indices": sources,
                "feature_dimension": int(x_refit.shape[1]),
                "fit_sample_count": int(fitted["fit_sample_count"]),
                "audit_sample_count": int(len(audit)),
                **_metrics(actual, prediction),
            }
        actual_matrix = np.column_stack(actual_columns)
        prediction_matrix = np.column_stack(prediction_columns)
        per_model[model_name] = {
            "node_macro": {metric: float(np.mean([value[metric] for value in per_target.values()])) for metric in METRIC_NAMES},
            "grid_aggregate": _metrics(actual_matrix.sum(axis=1), prediction_matrix.sum(axis=1)),
            "per_target": per_target,
        }
    return per_model


def _old_neighbors_from_grid_report(data: Mapping[str, Any], grid_name: str) -> dict[int, set[int]] | None:
    for grid in data.get("grids", []):
        if grid.get("grid_name") != grid_name:
            continue
        diagnostics = grid.get("graph_diagnostics", {}).get("per_target", {})
        if diagnostics:
            return {
                int(item["node_index"]): {int(entry["node_index"]) for entry in item.get("top3_utility_neighbors", [])}
                for item in diagnostics.values()
            }
        source = grid.get("utility_neighbors", {})
        return {
            int(key): {int(entry["node_index"]) for entry in values}
            for key, values in source.items()
            if str(key).isdigit()
        }
    return None


def graph_diagnostics(
    grid: LVGridData,
    selection: Mapping[str, Any],
    old_neighbors: Mapping[int, set[int]] | None = None,
) -> dict[str, Any]:
    utilities = selection["utilities"]
    loads = [int(node) for node in selection["load_indices"]]
    values = np.asarray(list(utilities.values()), dtype=float)
    positive = values[values > 0]
    neighbors = selection["conditional_neighbors"]
    electrical = selection["electrical_neighbors"]
    counts = {str(k): int(sum(len(neighbors[target]) == k for target in loads)) for k in range(UTILITY_TOP_K + 1)}
    asymmetry = [abs(utilities[(source, target)] - utilities[(target, source)]) for source, target in utilities if source < target]
    overlap_old = []
    overlap_electrical = []
    per_target = {}
    for target in loads:
        current = set(_neighbor_ids(neighbors, target))
        old = old_neighbors.get(target, set()) if old_neighbors is not None else set()
        elec = set(_neighbor_ids(electrical, target))
        overlap_old.append(len(current & old))
        overlap_electrical.append(len(current & elec))
        per_target[str(grid.node_ids[target])] = {
            "node_index": target,
            "top3_source_ids_and_utilities": [{"source_node_index": int(source), "source_bus_id": str(grid.node_ids[source]), "conditional_utility": float(value)} for source, value in neighbors[target]],
            "selected_neighbor_count": len(current),
            "old_utility_top3_overlap_count": len(current & old) if old_neighbors is not None else None,
            "electrical_top3_overlap_count": len(current & elec),
        }
    return {
        "load_bus_count": len(loads),
        "ordered_pair_count": len(utilities),
        "positive_conditional_utility_edge_count": int(len(positive)),
        "positive_conditional_utility_edge_fraction": float(len(positive) / len(values)) if len(values) else 0.0,
        "positive_fraction": float(len(positive) / len(values)) if len(values) else 0.0,
        "mean_signed_utility": float(np.mean(values)) if len(values) else 0.0,
        "median_signed_utility": float(np.median(values)) if len(values) else 0.0,
        "mean_positive_utility": float(np.mean(positive)) if len(positive) else 0.0,
        "median_positive_utility": float(np.median(positive)) if len(positive) else 0.0,
        "targets_with_selected_neighbors": counts,
        "top3_source_ids_and_utilities": per_target,
        "directional_asymmetry": {"mean_abs_utility_difference": float(np.mean(asymmetry)) if asymmetry else 0.0, "utility_graph_visibly_directed": bool(asymmetry and np.mean(asymmetry) > 1e-12)},
        "overlap_with_old_predictive_utility_top3": {"available": old_neighbors is not None, "mean_overlap": float(np.mean(overlap_old)) if old_neighbors is not None else None},
        "overlap_with_electrical_top3": {"mean_overlap": float(np.mean(overlap_electrical)) if overlap_electrical else 0.0},
    }


def audit_grid(grid: LVGridData, old_neighbors: Mapping[int, set[int]] | None = None) -> dict[str, Any]:
    indices = formal_train_target_indices(grid)
    fit_indices, selection_indices, audit_indices = chronological_split(indices)
    loads = load_bus_indices(grid)
    utilities, records = compute_conditional_utilities(grid, fit_indices, selection_indices, loads)
    neighbors = select_conditional_neighbors(utilities, loads)
    selection = {"fit_indices": fit_indices, "selection_indices": selection_indices, "audit_indices": audit_indices, "load_indices": loads, "utilities": utilities, "conditional_neighbors": neighbors, "electrical_neighbors": select_electrical_neighbors(grid.distance_matrices["impedance_abs_distance"], loads)}
    return {
        "grid_name": grid.grid_name,
        "node_count": int(grid.num_nodes),
        "split": {"fit_samples": len(fit_indices), "selection_samples": len(selection_indices), "audit_samples": len(audit_indices), "shuffle": False},
        "pairwise_utilities": records,
        "graph_diagnostics": graph_diagnostics(grid, selection, old_neighbors),
        "audit_models": evaluate_selected_graph(grid, selection),
    }


def report_metadata() -> dict[str, Any]:
    return {
        "time_scope": "formal_train_only",
        "fit_samples": FIT_SAMPLES,
        "selection_samples": SELECTION_SAMPLES,
        "audit_samples": AUDIT_SAMPLES,
        "self_history_lags": "1_to_168",
        "self_feature_dim": SELF_FEATURE_DIM,
        "source_lags": list(SOURCE_LAGS),
        "ridge_alpha": RIDGE_ALPHA,
        "utility_semantics": "strong_temporal_conditional_predictive_utility",
        "utility_direction": "source_to_target",
        "utility_top_k": UTILITY_TOP_K,
        "utility_positive_only": True,
        "node_scope": "load_buses_only",
        "validation_used": False,
        "test_evaluated": False,
        "causal_claim": False,
        "excluded_inputs": ["validation", "test", "q", "weather"],
    }


def build_report(data_root: Path, mapping_path: Path, old_audit_path: Path | None = None) -> dict[str, Any]:
    loader = LVGridLoader(data_root, mapping_path)
    old_data = json.loads(old_audit_path.read_text(encoding="utf-8")) if old_audit_path and old_audit_path.exists() else None
    grids = []
    old_comparisons = []
    for name in loader.available_grids():
        grid = loader.load(name)
        old_neighbors = _old_neighbors_from_grid_report(old_data, grid.grid_name) if old_data else None
        current = audit_grid(grid, old_neighbors)
        grids.append(current)
        if old_data:
            old_grid = next((item for item in old_data.get("grids", []) if item.get("grid_name") == grid.grid_name), None)
            if old_grid and "audit_models" in old_grid:
                old_models = old_grid["audit_models"]
                old_self = old_models.get("self_only", {}).get("node_macro", {}).get("mae")
                old_utility = old_models.get("utility_top3", {}).get("node_macro", {}).get("mae")
                new_self = current["audit_models"]["strong_self"]["node_macro"]["mae"]
                new_utility = current["audit_models"]["conditional_utility_top3"]["node_macro"]["mae"]
                if old_self is not None and old_utility is not None:
                    old_comparisons.append({"grid_name": grid.grid_name, "old_weak_self_utility_top3_improvement_pct": float(100.0 * (old_self - old_utility) / (old_self + 1e-12)), "new_strong_temporal_conditional_utility_top3_improvement_pct": float(100.0 * (new_self - new_utility) / (new_self + 1e-12))})
    macro = {model: {scope: {metric: float(np.mean([grid["audit_models"][model][scope][metric] for grid in grids])) for metric in METRIC_NAMES} for scope in ("node_macro", "grid_aggregate")} for model in MODEL_NAMES}
    return {"audit": "strong_temporal_conditional_utility", "generated_at_utc": datetime.now(timezone.utc).isoformat(), **report_metadata(), "grids": grids, "four_grid_unweighted_macro": macro, "old_audit_comparison": old_comparisons}


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Strong-Temporal Conditional Predictive-Utility Audit", "", "Train-only deterministic Ridge audit; validation and test labels are not accessed.", "", "| Grid | Model | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Aggregate MAE | Aggregate RMSE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for grid in report["grids"]:
        for model in MODEL_NAMES:
            node = grid["audit_models"][model]["node_macro"]
            aggregate = grid["audit_models"][model]["grid_aggregate"]
            lines.append(f"| {grid['grid_name']} | {model} | {node['mae']:.6g} | {node['rmse']:.6g} | {node['wape_pct']:.6g} | {node['smape_pct']:.6g} | {aggregate['mae']:.6g} | {aggregate['rmse']:.6g} |")
    lines.extend(["", "## Four-grid unweighted macro", "", "| Model | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |", "|---|---|---:|---:|---:|---:|"])
    for model in MODEL_NAMES:
        for scope in ("node_macro", "grid_aggregate"):
            value = report["four_grid_unweighted_macro"][model][scope]
            lines.append(f"| {model} | {scope} | {value['mae']:.6g} | {value['rmse']:.6g} | {value['wape_pct']:.6g} | {value['smape_pct']:.6g} |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--old-audit-json", type=Path, default=REPO_ROOT / "results" / "audits" / "predictive_utility_graph_audit.json")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "results" / "audits" / "strong_temporal_conditional_utility_audit.json")
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / "results" / "audits" / "strong_temporal_conditional_utility_audit.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args.data_root.resolve(), args.mapping_json.resolve(), args.old_audit_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"grids={len(report['grids'])}")


if __name__ == "__main__":
    main()
