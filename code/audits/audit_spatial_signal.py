"""Train-only spatial signal audit for load buses across the four LV grids."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data.lv_grid_loader import LVGridLoader, LVGridData

LV_DATASET = "norway_4_lv_grids"
CORRELATION_TYPES = ("raw_standardized", "hour_of_week_residual", "one_hour_change")
HOP_GROUPS = ("hop=1", "hop=2", "hop=3-4", "hop>=5")


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def hop_group(hop: float | int) -> str:
    hop = int(hop)
    if hop == 1:
        return "hop=1"
    if hop == 2:
        return "hop=2"
    if 3 <= hop <= 4:
        return "hop=3-4"
    if hop >= 5:
        return "hop>=5"
    raise ValueError(f"load-bus pair hop distance must be positive, got {hop}")


def _standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(values, dtype=float)
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0)
    zero_variance = ~np.isfinite(scales) | (scales <= 0)
    safe_scales = np.where(zero_variance, 1.0, scales)
    return (values - means) / safe_scales, {"zero_variance_bus_count": int(zero_variance.sum())}


def _hour_of_week_residual(p_train: np.ndarray, timestamps: Any) -> np.ndarray:
    hours = np.asarray([int(ts.dayofweek) * 24 + int(ts.hour) for ts in timestamps], dtype=np.int64)
    residual = np.empty_like(np.asarray(p_train, dtype=float), dtype=float)
    for hour in range(168):
        mask = hours == hour
        if np.any(mask):
            residual[mask] = p_train[mask] - np.mean(p_train[mask], axis=0)
    return residual


def _pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 2:
        return None, "insufficient_finite_samples"
    if np.std(left[finite]) <= 0 or np.std(right[finite]) <= 0:
        return None, "zero_variance_pair"
    value = _finite_float(np.corrcoef(left[finite], right[finite])[0, 1])
    return value, None if value is not None else "non_finite_correlation"


def _summary(values: list[float], pair_count: int, excluded_count: int) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    return {
        "pair_count": int(pair_count),
        "finite_correlation_count": int(finite.size),
        "excluded_correlation_count": int(excluded_count),
        "mean_correlation": _finite_float(np.mean(finite)) if finite.size else None,
        "median_correlation": _finite_float(np.median(finite)) if finite.size else None,
        "q25": _finite_float(np.quantile(finite, 0.25)) if finite.size else None,
        "q75": _finite_float(np.quantile(finite, 0.75)) if finite.size else None,
    }


def _spearman(correlations: list[float], distances: list[float]) -> dict[str, Any]:
    if len(correlations) < 2:
        return {"rho": None, "pair_count": len(correlations), "reason": "insufficient_finite_pairs"}
    if np.std(correlations) <= 0 or np.std(distances) <= 0:
        return {"rho": None, "pair_count": len(correlations), "reason": "constant_input"}
    rho = spearmanr(np.asarray(correlations), -np.asarray(distances)).statistic
    return {"rho": _finite_float(rho), "pair_count": len(correlations), "reason": None}


def _pair_records(grid: LVGridData) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = grid.splits["train"]
    p = np.asarray(grid.p[train.start_index : train.end_index], dtype=float)
    timestamps = grid.timestamps[train.start_index : train.end_index]
    load_indices = np.flatnonzero(grid.load_bus_mask).astype(int)
    if len(load_indices) < 2:
        raise ValueError(f"{grid.grid_name} has fewer than two load buses")
    raw, raw_meta = _standardize(p[:, load_indices])
    residual = _hour_of_week_residual(p[:, load_indices], timestamps)
    delta_p = np.diff(p, axis=0)
    delta, delta_meta = _standardize(delta_p[:, load_indices])
    series = {
        "raw_standardized": raw,
        "hour_of_week_residual": residual,
        "one_hour_change": delta,
    }
    distances = grid.distance_matrices
    hop = np.asarray(distances["hop_distance"])
    impedance = np.asarray(distances["impedance_abs_distance"])
    if hop.shape != (grid.num_nodes, grid.num_nodes) or impedance.shape != hop.shape:
        raise ValueError("canonical distance matrices are not aligned to loader node order")
    if not np.allclose(hop, hop.T) or not np.allclose(impedance, impedance.T):
        raise ValueError("canonical distance matrices must be symmetric")
    records: list[dict[str, Any]] = []
    for local_i, node_i in enumerate(load_indices):
        for local_j in range(local_i + 1, len(load_indices)):
            node_j = int(load_indices[local_j])
            hop_value = int(hop[node_i, node_j])
            record: dict[str, Any] = {
                "node_i_index": int(node_i),
                "node_j_index": node_j,
                "node_i_id": str(grid.node_ids[node_i]),
                "node_j_id": str(grid.node_ids[node_j]),
                "hop_distance": hop_value,
                "hop_group": hop_group(hop_value),
                "impedance_abs_distance": float(impedance[node_i, node_j]),
                "correlations": {},
            }
            for name, values in series.items():
                corr, reason = _pearson(values[:, local_i], values[:, local_j])
                record["correlations"][name] = corr
                record.setdefault("correlation_exclusion_reasons", {})[name] = reason
            records.append(record)
    return records, {
        "load_bus_indices": load_indices.tolist(),
        "load_bus_ids": [str(grid.node_ids[i]) for i in load_indices],
        "series_metadata": {
            "raw_standardized": raw_meta,
            "hour_of_week_residual": {"mean_scope": "train_only_by_hour_of_week"},
            "one_hour_change": {**delta_meta, "valid_indices": [train.start_index + 1, train.end_index - 1]},
        },
    }


def audit_grid(grid: LVGridData) -> dict[str, Any]:
    records, selection = _pair_records(grid)
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    distance_trends: dict[str, dict[str, Any]] = {}
    adjacent_vs_nonadjacent: dict[str, dict[str, Any]] = {}
    for name in CORRELATION_TYPES:
        grouped[name] = {}
        for group in HOP_GROUPS:
            values = [r["correlations"][name] for r in records if r["hop_group"] == group]
            finite = [float(v) for v in values if v is not None and np.isfinite(v)]
            grouped[name][group] = _summary(finite, len(values), len(values) - len(finite))
        corr = [r["correlations"][name] for r in records if r["correlations"][name] is not None]
        hop_values = [r["hop_distance"] for r in records if r["correlations"][name] is not None]
        imp_values = [r["impedance_abs_distance"] for r in records if r["correlations"][name] is not None]
        distance_trends[name] = {
            "spearman_correlation_vs_negative_hop_distance": _spearman(corr, hop_values),
            "spearman_correlation_vs_negative_impedance_abs_distance": _spearman(corr, imp_values),
        }
        adjacent = [r["correlations"][name] for r in records if r["hop_distance"] == 1 and r["correlations"][name] is not None]
        nonadjacent = [r["correlations"][name] for r in records if r["hop_distance"] > 1 and r["correlations"][name] is not None]
        adjacent_vs_nonadjacent[name] = {
            "adjacent_pairs_hop_eq_1": _summary(adjacent, sum(r["hop_distance"] == 1 for r in records), sum(r["hop_distance"] == 1 for r in records) - len(adjacent)),
            "non_adjacent_pairs_hop_gt_1": _summary(nonadjacent, sum(r["hop_distance"] > 1 for r in records), sum(r["hop_distance"] > 1 for r in records) - len(nonadjacent)),
        }
    return {
        "grid_name": grid.grid_name,
        "node_count": grid.num_nodes,
        "load_bus_count": int(grid.load_bus_mask.sum()),
        "pair_count": len(records),
        "expected_pair_count": int(grid.load_bus_mask.sum()) * (int(grid.load_bus_mask.sum()) - 1) // 2,
        "train_split": {"start_index": grid.splits["train"].start_index, "end_index": grid.splits["train"].end_index, "sample_count": grid.splits["train"].sample_count},
        "selection": selection,
        "by_hop_group": grouped,
        "distance_trends": distance_trends,
        "adjacent_vs_non_adjacent": adjacent_vs_nonadjacent,
        "zero_variance_or_invalid_pair_policy": "correlation=None; excluded from finite summaries and Spearman, with exclusion counts/reasons recorded",
        "pair_records": records,
    }


def build_report(data_root: Path, mapping_path: Path) -> dict[str, Any]:
    loader = LVGridLoader(data_root, mapping_path)
    grids = [audit_grid(loader.load(name)) for name in loader.available_grids()]
    return {
        "audit": "v2_spatial_signal_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "time_scope": "train_only",
        "node_scope": "load_buses_only",
        "distance_sources": "canonical_loader",
        "correlation_types": list(CORRELATION_TYPES),
        "excluded_inputs": ["validation", "test", "q", "weather", "static_load_metadata", "model_predictions", "neural_network_residuals"],
        "grids": grids,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Train-only Spatial Signal Audit", "",
        "This is a structural diagnostic, not a significance test or a model result.", "",
        f"time_scope: `{report['time_scope']}`; node_scope: `{report['node_scope']}`; distance_sources: `{report['distance_sources']}`.",
        f"correlation_types: `{', '.join(report['correlation_types'])}`.", "",
        "| Grid | Load buses | Load-load pairs | Expected pairs | Train samples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for grid in report["grids"]:
        lines.append(f"| {grid['grid_name']} | {grid['load_bus_count']} | {grid['pair_count']} | {grid['expected_pair_count']} | {grid['train_split']['sample_count']} |")
        lines.extend(["", f"## {grid['grid_name']}", "", "| Sequence | Hop group | Pairs | Finite | Excluded | Mean | Median | Q25 | Q75 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for name in CORRELATION_TYPES:
            for group in HOP_GROUPS:
                item = grid["by_hop_group"][name][group]
                lines.append(f"| {name} | {group} | {item['pair_count']} | {item['finite_correlation_count']} | {item['excluded_correlation_count']} | {_fmt(item['mean_correlation'])} | {_fmt(item['median_correlation'])} | {_fmt(item['q25'])} | {_fmt(item['q75'])} |")
        lines.extend(["", "| Sequence | Spearman(corr, -hop) | Spearman(corr, -impedance) | Adjacent mean/median | Non-adjacent mean/median |", "| --- | ---: | ---: | --- | --- |"])
        for name in CORRELATION_TYPES:
            trend = grid["distance_trends"][name]
            adj = grid["adjacent_vs_non_adjacent"][name]["adjacent_pairs_hop_eq_1"]
            non = grid["adjacent_vs_non_adjacent"][name]["non_adjacent_pairs_hop_gt_1"]
            lines.append(f"| {name} | {_fmt(trend['spearman_correlation_vs_negative_hop_distance']['rho'])} | {_fmt(trend['spearman_correlation_vs_negative_impedance_abs_distance']['rho'])} | {_fmt(adj['mean_correlation'])} / {_fmt(adj['median_correlation'])} | {_fmt(non['mean_correlation'])} / {_fmt(non['median_correlation'])} |")
        lines.extend(["", f"Zero-variance/invalid policy: {grid['zero_variance_or_invalid_pair_policy']}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=repo_root / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-json", type=Path, default=repo_root / "results" / "audits" / "spatial_signal_audit.json")
    parser.add_argument("--output-md", type=Path, default=repo_root / "results" / "audits" / "spatial_signal_audit.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root.resolve(), args.mapping_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"grids={len(report['grids'])}")


if __name__ == "__main__":
    main()
