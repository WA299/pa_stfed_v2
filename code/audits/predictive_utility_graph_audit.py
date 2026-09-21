"""Train-only predictive utility graph audit using fixed Ridge regressions.

The formal train targets are split chronologically into fit, selection, and
audit segments. Graph construction uses fit and selection only; the final audit
segment is reserved for independent evaluation of the selected graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data.lv_grid_loader import LVGridData, LVGridLoader


LAGS = (1, 24, 168)
FIT_SAMPLES = 3578
SELECTION_SAMPLES = 1193
AUDIT_SAMPLES = 1193
FORMAL_TRAIN_VALID_TARGETS = FIT_SAMPLES + SELECTION_SAMPLES + AUDIT_SAMPLES
RIDGE_ALPHA = 1.0
UTILITY_TOP_K = 3
ELECTRICAL_TOP_K = 3
MODEL_NAMES = ("self_only", "electrical_top3", "utility_top3", "union")
METRIC_NAMES = ("mae", "rmse", "wape_pct", "smape_pct")


def formal_train_target_indices(grid: LVGridData) -> np.ndarray:
    """Return only valid 168-lag targets inside the canonical formal train split."""
    train = grid.splits["train"]
    indices = np.arange(train.start_index + max(LAGS), train.end_index, dtype=int)
    if len(indices) != FORMAL_TRAIN_VALID_TARGETS:
        raise ValueError(
            f"expected {FORMAL_TRAIN_VALID_TARGETS} formal train targets, got {len(indices)}"
        )
    return indices


def split_train_targets(
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make the fixed chronological 3578/1193/1193 split without shuffling."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) != FORMAL_TRAIN_VALID_TARGETS:
        raise ValueError(
            f"expected {FORMAL_TRAIN_VALID_TARGETS} target indices, got {len(indices)}"
        )
    if len(indices) > 1 and not np.all(np.diff(indices) > 0):
        raise ValueError("formal train target indices must be strictly chronological")
    fit_end = FIT_SAMPLES
    selection_end = fit_end + SELECTION_SAMPLES
    return (
        indices[:fit_end].copy(),
        indices[fit_end:selection_end].copy(),
        indices[selection_end:].copy(),
    )


def load_bus_indices(grid: LVGridData) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(grid.load_bus_mask, dtype=bool)).astype(int)
    if len(indices) < 2:
        raise ValueError("predictive utility audit requires at least two load buses")
    return indices


def calendar_features(timestamps: Any, target_indices: np.ndarray) -> np.ndarray:
    selected = timestamps[np.asarray(target_indices, dtype=int)]
    hour = np.asarray([timestamp.hour for timestamp in selected], dtype=float)
    weekday = np.asarray([timestamp.dayofweek for timestamp in selected], dtype=float)
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            (weekday >= 5).astype(float),
        ]
    )


def feature_matrix(
    p: np.ndarray,
    timestamps: Any,
    target_indices: np.ndarray,
    target_node: int,
    source_nodes: Iterable[int] = (),
) -> np.ndarray:
    """Build 8 self features plus three lag features per unique source."""
    target_indices = np.asarray(target_indices, dtype=int)
    sources = []
    for source in source_nodes:
        source = int(source)
        if source == int(target_node):
            raise ValueError("source_nodes must exclude the target node")
        if source not in sources:
            sources.append(source)
    columns = [p[target_indices - lag, target_node] for lag in LAGS]
    matrix = np.column_stack(columns + [calendar_features(timestamps, target_indices)])
    extra = [p[target_indices - lag, source] for source in sources for lag in LAGS]
    if extra:
        matrix = np.column_stack([matrix] + extra)
    return np.asarray(matrix, dtype=float)


def fit_feature_scaler(x_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a safe standardizer from the provided fit matrix only."""
    x_fit = np.asarray(x_fit, dtype=float)
    if x_fit.ndim != 2 or len(x_fit) == 0:
        raise ValueError("x_fit must be a non-empty 2D matrix")
    mean = np.mean(x_fit, axis=0)
    raw_scale = np.std(x_fit, axis=0)
    zero_variance = (~np.isfinite(raw_scale)) | (raw_scale <= 0)
    scale = np.where(zero_variance, 1.0, raw_scale)
    return mean, scale, zero_variance


def fit_scaled_ridge(x_fit: np.ndarray, y_fit: np.ndarray) -> dict[str, Any]:
    mean, scale, zero_variance = fit_feature_scaler(x_fit)
    model = Ridge(alpha=RIDGE_ALPHA).fit((x_fit - mean) / scale, y_fit)
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "zero_variance": zero_variance,
        "fit_sample_count": int(len(x_fit)),
    }


def predict_scaled_ridge(fitted: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return np.asarray(
        fitted["model"].predict((x - fitted["mean"]) / fitted["scale"]),
        dtype=float,
    )


def signed_utility(mae_self: float, mae_self_plus_source: float) -> float:
    return float((mae_self - mae_self_plus_source) / (mae_self + 1e-12))


def select_utility_neighbors(
    utilities: dict[tuple[int, int], float],
    load_indices: Iterable[int],
    top_k: int = UTILITY_TOP_K,
) -> dict[int, list[tuple[int, float]]]:
    """Select positive source->target utilities with stable node-index ties."""
    loads = [int(node) for node in load_indices]
    result: dict[int, list[tuple[int, float]]] = {}
    for target in loads:
        candidates = [
            (source, float(utilities[(source, target)]))
            for source in loads
            if source != target and float(utilities[(source, target)]) > 0.0
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        result[target] = candidates[:top_k]
    return result


def select_electrical_neighbors(
    impedance_distance: np.ndarray,
    load_indices: Iterable[int],
    top_k: int = ELECTRICAL_TOP_K,
) -> dict[int, list[tuple[int, float]]]:
    """Select nearest other load buses from the canonical all-pairs distance."""
    distance = np.asarray(impedance_distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("impedance_abs_distance must be square")
    loads = [int(node) for node in load_indices]
    result: dict[int, list[tuple[int, float]]] = {}
    for target in loads:
        candidates = [
            (source, float(distance[target, source]))
            for source in loads
            if source != target
        ]
        if any(not np.isfinite(value) or value < 0 for _, value in candidates):
            raise ValueError("load-bus impedance distances must be finite and non-negative")
        candidates.sort(key=lambda item: (item[1], item[0]))
        result[target] = candidates[:top_k]
    return result


def choose_electrical_neighbors(
    grid: LVGridData,
    load_indices: Iterable[int] | None = None,
    max_neighbors: int = ELECTRICAL_TOP_K,
) -> dict[int, list[tuple[int, float]]]:
    """Compatibility wrapper using the loader's canonical distance matrix."""
    loads = load_bus_indices(grid) if load_indices is None else np.asarray(load_indices, dtype=int)
    return select_electrical_neighbors(
        grid.distance_matrices["impedance_abs_distance"], loads, max_neighbors
    )


def compute_pairwise_utilities(
    grid: LVGridData,
    fit_indices: np.ndarray,
    selection_indices: np.ndarray,
    loads: np.ndarray | None = None,
) -> tuple[dict[tuple[int, int], float], list[dict[str, Any]]]:
    """Fit pairwise models on fit and score utilities on selection only."""
    loads = load_bus_indices(grid) if loads is None else np.asarray(loads, dtype=int)
    utilities: dict[tuple[int, int], float] = {}
    records: list[dict[str, Any]] = []
    for target in loads:
        self_fit = feature_matrix(grid.p, grid.timestamps, fit_indices, int(target))
        self_selection = feature_matrix(
            grid.p, grid.timestamps, selection_indices, int(target)
        )
        self_model = fit_scaled_ridge(self_fit, grid.p[fit_indices, target])
        self_prediction = predict_scaled_ridge(self_model, self_selection)
        mae_self = float(
            np.mean(np.abs(self_prediction - grid.p[selection_indices, target]))
        )
        for source in loads:
            if source == target:
                continue
            pair_fit = feature_matrix(
                grid.p, grid.timestamps, fit_indices, int(target), [int(source)]
            )
            pair_selection = feature_matrix(
                grid.p, grid.timestamps, selection_indices, int(target), [int(source)]
            )
            pair_model = fit_scaled_ridge(pair_fit, grid.p[fit_indices, target])
            pair_prediction = predict_scaled_ridge(pair_model, pair_selection)
            mae_pair = float(
                np.mean(np.abs(pair_prediction - grid.p[selection_indices, target]))
            )
            utility = signed_utility(mae_self, mae_pair)
            utilities[(int(source), int(target))] = utility
            records.append(
                {
                    "source_node_index": int(source),
                    "source_bus_id": str(grid.node_ids[source]),
                    "target_node_index": int(target),
                    "target_bus_id": str(grid.node_ids[target]),
                    "mae_self_selection": mae_self,
                    "mae_self_plus_source_selection": mae_pair,
                    "signed_utility": utility,
                    "positive_utility": max(utility, 0.0),
                }
            )
    return utilities, records


def select_graph(grid: LVGridData) -> dict[str, Any]:
    """Construct both graphs without reading the audit target segment."""
    formal_indices = formal_train_target_indices(grid)
    fit_indices, selection_indices, audit_indices = split_train_targets(formal_indices)
    loads = load_bus_indices(grid)
    utilities, pair_records = compute_pairwise_utilities(
        grid, fit_indices, selection_indices, loads
    )
    utility_neighbors = select_utility_neighbors(utilities, loads)
    electrical_neighbors = select_electrical_neighbors(
        grid.distance_matrices["impedance_abs_distance"], loads
    )
    return {
        "load_indices": loads,
        "fit_indices": fit_indices,
        "selection_indices": selection_indices,
        # Returned for evaluation bookkeeping; never consumed by graph selection.
        "audit_indices": audit_indices,
        "utilities": utilities,
        "pair_records": pair_records,
        "utility_neighbors": utility_neighbors,
        "electrical_neighbors": electrical_neighbors,
    }


def _neighbor_indices(
    entries: list[tuple[int, float]],
) -> list[int]:
    return [int(node) for node, _ in entries]


def model_sources(selection: dict[str, Any], target: int, model_name: str) -> list[int]:
    electrical = _neighbor_indices(selection["electrical_neighbors"][target])
    utility = _neighbor_indices(selection["utility_neighbors"][target])
    if model_name == "self_only":
        return []
    if model_name == "electrical_top3":
        return electrical
    if model_name == "utility_top3":
        return utility
    if model_name == "union":
        return list(dict.fromkeys(electrical + utility))
    raise ValueError(f"unknown model: {model_name}")


# Descriptive aliases make the audit helpers convenient for lightweight tests.
chronological_split = split_train_targets
build_features = feature_matrix
choose_utility_neighbors = select_utility_neighbors


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    abs_denominator = max(float(np.sum(np.abs(actual))), 1e-12)
    smape_denominator = np.maximum(np.abs(actual) + np.abs(prediction), 1e-12)
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_pct": float(100.0 * np.sum(np.abs(error)) / abs_denominator),
        "smape_pct": float(
            100.0 * np.mean(2.0 * np.abs(error) / smape_denominator)
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("audit metrics must be finite")
    return metrics


def evaluate_selected_graph(grid: LVGridData, selection: dict[str, Any]) -> dict[str, Any]:
    """Refit on fit+selection and evaluate once on the held-out audit segment."""
    refit_indices = np.concatenate(
        [selection["fit_indices"], selection["selection_indices"]]
    )
    audit_indices = selection["audit_indices"]
    loads = selection["load_indices"]
    models: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        actual_columns = []
        prediction_columns = []
        per_target: dict[str, Any] = {}
        for target in loads:
            sources = model_sources(selection, int(target), model_name)
            x_refit = feature_matrix(
                grid.p, grid.timestamps, refit_indices, int(target), sources
            )
            x_audit = feature_matrix(
                grid.p, grid.timestamps, audit_indices, int(target), sources
            )
            fitted = fit_scaled_ridge(x_refit, grid.p[refit_indices, target])
            prediction = predict_scaled_ridge(fitted, x_audit)
            actual = np.asarray(grid.p[audit_indices, target], dtype=float)
            actual_columns.append(actual)
            prediction_columns.append(prediction)
            per_target[str(grid.node_ids[target])] = {
                "node_index": int(target),
                "source_node_indices": sources,
                "feature_dimension": int(x_refit.shape[1]),
                "scaler_fit_sample_count": int(fitted["fit_sample_count"]),
                "zero_variance_feature_count": int(fitted["zero_variance"].sum()),
                **_regression_metrics(actual, prediction),
            }
        actual_matrix = np.column_stack(actual_columns)
        prediction_matrix = np.column_stack(prediction_columns)
        node_macro = {
            metric: float(np.mean([item[metric] for item in per_target.values()]))
            for metric in METRIC_NAMES
        }
        models[model_name] = {
            "node_macro": node_macro,
            "grid_aggregate": _regression_metrics(
                actual_matrix.sum(axis=1), prediction_matrix.sum(axis=1)
            ),
            "per_target": per_target,
        }
    return models


def graph_diagnostics(grid: LVGridData, selection: dict[str, Any]) -> dict[str, Any]:
    utilities = selection["utilities"]
    signed_values = np.asarray(list(utilities.values()), dtype=float)
    positive_values = signed_values[signed_values > 0]
    per_target = {}
    overlaps = []
    for target in selection["load_indices"]:
        utility_entries = selection["utility_neighbors"][int(target)]
        electrical_entries = selection["electrical_neighbors"][int(target)]
        utility_sources = set(_neighbor_indices(utility_entries))
        electrical_sources = set(_neighbor_indices(electrical_entries))
        overlap = len(utility_sources & electrical_sources)
        overlaps.append(overlap)
        per_target[str(grid.node_ids[target])] = {
            "node_index": int(target),
            "positive_source_count": int(
                sum(
                    utilities[(int(source), int(target))] > 0
                    for source in selection["load_indices"]
                    if source != target
                )
            ),
            "top3_utility_neighbors": [
                {
                    "node_index": source,
                    "bus_id": str(grid.node_ids[source]),
                    "signed_utility": utility,
                }
                for source, utility in utility_entries
            ],
            "top3_electrical_neighbors": [
                {
                    "node_index": source,
                    "bus_id": str(grid.node_ids[source]),
                    "impedance_abs_distance": distance,
                }
                for source, distance in electrical_entries
            ],
            "overlap_count": overlap,
        }
    asymmetry = [
        abs(value - utilities[(target, source)])
        for (source, target), value in utilities.items()
        if source < target
    ]
    mean_asymmetry = float(np.mean(asymmetry)) if asymmetry else 0.0
    overlap_distribution = {
        str(value): int(overlaps.count(value)) for value in sorted(set(overlaps))
    }
    return {
        "load_bus_count": int(len(selection["load_indices"])),
        "ordered_pair_count": int(len(utilities)),
        "positive_utility_edge_count": int(len(positive_values)),
        "positive_utility_edge_fraction": float(
            len(positive_values) / len(signed_values)
        ),
        "global_utility": {
            "mean_signed_utility": float(np.mean(signed_values)),
            "median_signed_utility": float(np.median(signed_values)),
            "mean_positive_utility": float(np.mean(positive_values))
            if len(positive_values)
            else 0.0,
            "median_positive_utility": float(np.median(positive_values))
            if len(positive_values)
            else 0.0,
            "max_positive_utility": float(np.max(positive_values))
            if len(positive_values)
            else 0.0,
        },
        "electrical_top3_vs_utility_top3": {
            "mean_overlap": float(np.mean(overlaps)),
            "overlap_distribution": overlap_distribution,
        },
        "directionality": {
            "mean_abs_asymmetry": mean_asymmetry,
            "utility_graph_visibly_directed": bool(mean_asymmetry > 1e-12),
        },
        "per_target": per_target,
    }


def audit_grid(grid: LVGridData) -> dict[str, Any]:
    selection = select_graph(grid)
    utility_neighbors = {
        str(grid.node_ids[target]): [
            {
                "node_index": int(source),
                "bus_id": str(grid.node_ids[source]),
                "signed_utility": float(utility),
            }
            for source, utility in values
        ]
        for target, values in selection["utility_neighbors"].items()
    }
    electrical_neighbors = {
        str(grid.node_ids[target]): [
            {
                "node_index": int(source),
                "bus_id": str(grid.node_ids[source]),
                "impedance_abs_distance": float(distance),
            }
            for source, distance in values
        ]
        for target, values in selection["electrical_neighbors"].items()
    }
    return {
        "grid_name": grid.grid_name,
        "node_count": int(grid.num_nodes),
        "split": {
            "fit_target_start": int(selection["fit_indices"][0]),
            "fit_target_end": int(selection["fit_indices"][-1]),
            "selection_target_start": int(selection["selection_indices"][0]),
            "selection_target_end": int(selection["selection_indices"][-1]),
            "audit_target_start": int(selection["audit_indices"][0]),
            "audit_target_end": int(selection["audit_indices"][-1]),
            "shuffle": False,
        },
        "pairwise_utilities": selection["pair_records"],
        "utility_neighbors": utility_neighbors,
        "electrical_neighbors": electrical_neighbors,
        "graph_diagnostics": graph_diagnostics(grid, selection),
        "audit_models": evaluate_selected_graph(grid, selection),
    }


def report_metadata() -> dict[str, Any]:
    return {
        "time_scope": "formal_train_only",
        "fit_samples": FIT_SAMPLES,
        "selection_samples": SELECTION_SAMPLES,
        "audit_samples": AUDIT_SAMPLES,
        "utility_metric": "relative_mae_reduction",
        "utility_semantics": "out_of_sample_incremental_predictive_utility",
        "utility_terminology": "out_of_sample_incremental_predictive_utility",
        "utility_direction": "source_to_target",
        "ridge_alpha": RIDGE_ALPHA,
        "lags": list(LAGS),
        "utility_top_k": UTILITY_TOP_K,
        "electrical_top_k": ELECTRICAL_TOP_K,
        "node_scope": "load_buses_only",
        "electrical_distance_source": "canonical_impedance_abs_distance",
        "validation_used": False,
        "test_used": False,
        "validation_evaluated": False,
        "test_evaluated": False,
        "causal_claim": False,
        "excluded_inputs": ["validation", "test", "q", "weather"],
    }


def build_report(data_root: Path, mapping_path: Path) -> dict[str, Any]:
    loader = LVGridLoader(data_root, mapping_path)
    return {
        "audit": "v2_predictive_utility_graph_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **report_metadata(),
        "grids": [audit_grid(loader.load(name)) for name in loader.available_grids()],
    }


def _fmt(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Predictive Utility Graph Audit",
        "",
        "Train-only Ridge diagnostic. This audit makes no causal claim.",
        "",
        "| Grid | Load buses | Ordered pairs | Positive edges | Positive fraction | Mean asymmetry | Directed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for grid in report["grids"]:
        diagnostics = grid["graph_diagnostics"]
        directionality = diagnostics["directionality"]
        lines.append(
            f"| {grid['grid_name']} | {diagnostics['load_bus_count']} | "
            f"{diagnostics['ordered_pair_count']} | {diagnostics['positive_utility_edge_count']} | "
            f"{_fmt(diagnostics['positive_utility_edge_fraction'])} | "
            f"{_fmt(directionality['mean_abs_asymmetry'])} | "
            f"{directionality['utility_graph_visibly_directed']} |"
        )
        lines.extend(
            [
                "",
                f"## {grid['grid_name']}",
                "",
                "| Model | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Aggregate MAE | Aggregate RMSE |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name in MODEL_NAMES:
            model = grid["audit_models"][model_name]
            node = model["node_macro"]
            aggregate = model["grid_aggregate"]
            lines.append(
                f"| {model_name} | {_fmt(node['mae'])} | {_fmt(node['rmse'])} | "
                f"{_fmt(node['wape_pct'])} | {_fmt(node['smape_pct'])} | "
                f"{_fmt(aggregate['mae'])} | {_fmt(aggregate['rmse'])} |"
            )
    lines.extend(
        [
            "",
            "Graph selection uses fit and selection only; the audit segment is held out until final evaluation.",
            "Validation and test are not accessed.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT.parent / "pa_stfed_data_v2" / "raw"
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "audits" / "predictive_utility_graph_audit.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "results" / "audits" / "predictive_utility_graph_audit.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root.resolve(), args.mapping_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"grids={len(report['grids'])}")


if __name__ == "__main__":
    main()
