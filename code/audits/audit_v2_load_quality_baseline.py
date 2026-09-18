"""Audit LV load quality and deterministic persistence baselines.

This is an LV-only audit. It uses the previously verified positional
profile-to-bus mapping, aggregates multiple consumers on the same physical
bus, and evaluates persistence baselines on validation data only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LV_DATASET = "norway_4_lv_grids"
PERSISTENCE_LAGS = {"persistence_1h": 1, "persistence_24h": 24, "persistence_168h": 168}
QUALITY_NEAR_CONSTANT_CV = 1e-3
ZERO_RATIO_HIGH_THRESHOLD = 0.05
WAPE_NEAR_ZERO_ABS_SUM = 1e-12
AUTOCORRELATION_LAGS = (1, 24, 168)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root.parent / "pa_stfed_data_v2" / "raw",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=repo_root / "results" / "audits" / "v2_schema_mapping.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "results" / "audits" / "v2_load_quality_baseline.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=repo_root / "results" / "audits" / "v2_load_quality_baseline.md",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def iso_timestamp(value: Any) -> str | None:
    parsed = pd.Timestamp(value)
    return parsed.isoformat(sep=" ") if not pd.isna(parsed) else None


def read_profile_values(path: Path) -> tuple[pd.Series, np.ndarray]:
    """Read data by position so duplicate CSV headers cannot alter mapping."""
    frame = pd.read_csv(
        path,
        header=None,
        skiprows=1,
        dtype=str,
        encoding="utf-8-sig",
    )
    dates = pd.to_datetime(frame.iloc[:, 0], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return dates, values


def aggregate_by_bus(
    values: np.ndarray,
    mappings: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    profiles: dict[str, list[np.ndarray]] = {}
    positions: dict[str, list[int]] = {}
    for item in mappings:
        bus = item.get("mapped_bus_i")
        position = int(item["profile_position"]) - 1
        if bus is None or position < 0 or position >= values.shape[1]:
            raise ValueError(f"Invalid verified mapping at profile position {item.get('profile_position')}")
        profiles.setdefault(str(bus), []).append(values[:, position])
        positions.setdefault(str(bus), []).append(position + 1)

    aggregated: dict[str, np.ndarray] = {}
    for bus, series_list in profiles.items():
        with np.errstate(invalid="ignore", over="ignore"):
            aggregated[bus] = np.sum(np.vstack(series_list), axis=0)
    return aggregated, positions


def quality_stats(series: np.ndarray) -> dict[str, Any]:
    array = np.asarray(series, dtype=float)
    count = int(array.size)
    nan_mask = np.isnan(array)
    inf_mask = np.isinf(array)
    finite = array[np.isfinite(array)]
    finite_count = int(finite.size)
    mean = float(np.mean(finite)) if finite_count else None
    std = float(np.std(finite, ddof=0)) if finite_count else None
    minimum = float(np.min(finite)) if finite_count else None
    maximum = float(np.max(finite)) if finite_count else None
    cv = None
    if mean is not None and std is not None:
        cv = 0.0 if mean == 0 and std == 0 else (std / abs(mean) if mean != 0 else None)
    constant = bool(finite_count > 0 and finite_count == count and np.all(finite == finite[0]))
    near_constant = bool(
        finite_count > 0
        and finite_count == count
        and (constant or (cv is not None and cv <= QUALITY_NEAR_CONSTANT_CV))
    )
    return {
        "sample_count": count,
        "nan_count": int(nan_mask.sum()),
        "nan_ratio": float(nan_mask.mean()) if count else None,
        "inf_count": int(inf_mask.sum()),
        "inf_ratio": float(inf_mask.mean()) if count else None,
        "negative_count": int((array < 0).sum()),
        "negative_ratio": float((array < 0).mean()) if count else None,
        "zero_count": int((array == 0).sum()),
        "zero_ratio": float((array == 0).mean()) if count else None,
        "finite_count": finite_count,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "cv": safe_float(cv),
        "p95": safe_float(np.percentile(finite, 95)) if finite_count else None,
        "p99": safe_float(np.percentile(finite, 99)) if finite_count else None,
        "constant": constant,
        "near_constant": near_constant,
    }


def autocorrelation(series: np.ndarray, lag: int) -> float | None:
    if len(series) <= lag:
        return None
    left = np.asarray(series[lag:], dtype=float)
    right = np.asarray(series[:-lag], dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    left = left[mask]
    right = right[mask]
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return safe_float(np.corrcoef(left, right)[0, 1])


def autocorrelation_stats(series: np.ndarray) -> dict[str, float | None]:
    return {f"lag_{lag}": autocorrelation(series, lag) for lag in AUTOCORRELATION_LAGS}


def split_indices(sample_count: int) -> dict[str, dict[str, int]]:
    train_end = int(sample_count * 0.70)
    validation_end = train_end + int(sample_count * 0.15)
    return {
        "train": {"start_index": 0, "end_index": train_end, "sample_count": train_end},
        "validation": {
            "start_index": train_end,
            "end_index": validation_end,
            "sample_count": validation_end - train_end,
        },
        "test": {
            "start_index": validation_end,
            "end_index": sample_count,
            "sample_count": sample_count - validation_end,
        },
    }


def add_split_ranges(splits: dict[str, dict[str, int]], dates: pd.Series) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, split in splits.items():
        start = split["start_index"]
        end = split["end_index"]
        item = dict(split)
        item["start_time"] = iso_timestamp(dates.iloc[start]) if end > start else None
        item["end_time"] = iso_timestamp(dates.iloc[end - 1]) if end > start else None
        result[name] = item
    return result


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite_pair = np.isfinite(actual) & np.isfinite(predicted)
    valid_actual = actual[finite_pair]
    valid_predicted = predicted[finite_pair]
    valid_count = int(finite_pair.sum())
    invalid_count = int(len(actual) - valid_count)
    if valid_count == 0:
        return {
            "valid_sample_count": 0,
            "invalid_pair_count": invalid_count,
            "mae": None,
            "rmse": None,
            "wape": None,
            "smape": None,
            "wape_denominator_abs_sum": None,
            "wape_denominator_mean_abs": None,
            "wape_denominator_near_zero": None,
            "wape_status": "no_finite_pairs",
            "smape_valid_sample_count": 0,
        }

    error = valid_predicted - valid_actual
    absolute_error = np.abs(error)
    denominator = float(np.sum(np.abs(valid_actual)))
    denominator_mean = float(np.mean(np.abs(valid_actual)))
    wape_near_zero = denominator <= WAPE_NEAR_ZERO_ABS_SUM
    if denominator == 0:
        wape = None
        wape_status = "undefined_zero_denominator"
    else:
        wape = float(np.sum(absolute_error) / denominator * 100.0)
        wape_status = "near_zero_denominator" if wape_near_zero else "ok"

    smape_denominator = np.abs(valid_actual) + np.abs(valid_predicted)
    smape_mask = smape_denominator > 0
    smape = (
        float(np.mean(2.0 * absolute_error[smape_mask] / smape_denominator[smape_mask]) * 100.0)
        if smape_mask.any()
        else None
    )
    return {
        "valid_sample_count": valid_count,
        "invalid_pair_count": invalid_count,
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": wape,
        "smape": smape,
        "wape_denominator_abs_sum": denominator,
        "wape_denominator_mean_abs": denominator_mean,
        "wape_denominator_near_zero": wape_near_zero,
        "wape_status": wape_status,
        "smape_valid_sample_count": int(smape_mask.sum()),
    }


def persistence_metrics(series: np.ndarray, validation: dict[str, int]) -> dict[str, dict[str, Any]]:
    start = validation["start_index"]
    end = validation["end_index"]
    actual = series[start:end]
    result: dict[str, dict[str, Any]] = {}
    for name, lag in PERSISTENCE_LAGS.items():
        predicted = series[start - lag : end - lag]
        result[name] = metric_summary(actual, predicted)
    return result


def macro_average(
    per_bus: dict[str, dict[str, dict[str, Any]]], target: str, baseline: str
) -> dict[str, Any]:
    metric_names = ("mae", "rmse", "wape", "smape")
    result: dict[str, Any] = {}
    for metric in metric_names:
        values = [
            item[target][baseline][metric]
            for item in per_bus.values()
            if item[target][baseline][metric] is not None
        ]
        result[metric] = float(np.mean(values)) if values else None
        result[f"{metric}_bus_count"] = len(values)
    return result


def best_persistence(metric_block: dict[str, dict[str, Any]], metric: str = "mae") -> str | None:
    available = {
        name: values[metric]
        for name, values in metric_block.items()
        if values.get(metric) is not None
    }
    return min(available, key=available.get) if available else None


def build_grid_report(grid_root: Path, mapping_grid: dict[str, Any]) -> dict[str, Any]:
    if not mapping_grid.get("profile_to_bus_mapping_verified"):
        raise ValueError(f"Mapping is not verified for {mapping_grid.get('name')}")

    p_dates, p_values = read_profile_values(grid_root / "p_load.csv")
    q_dates, q_values = read_profile_values(grid_root / "q_load.csv")
    if len(p_dates) != len(q_dates) or not p_dates.equals(q_dates):
        raise ValueError(f"P/Q timestamps are not aligned for {mapping_grid['name']}")
    if p_values.shape != q_values.shape:
        raise ValueError(f"P/Q profile matrices are not aligned for {mapping_grid['name']}")

    mappings = mapping_grid["profile_mapping"]
    p_by_bus, positions = aggregate_by_bus(p_values, mappings)
    q_by_bus, q_positions = aggregate_by_bus(q_values, mappings)
    if positions != q_positions:
        raise ValueError(f"P/Q bus aggregation positions differ for {mapping_grid['name']}")

    bus_quality: dict[str, Any] = {}
    for bus in sorted(p_by_bus, key=lambda value: int(value) if value.isdigit() else value):
        bus_quality[bus] = {
            "profile_count": len(positions[bus]),
            "profile_positions": positions[bus],
            "p": quality_stats(p_by_bus[bus]),
            "q": quality_stats(q_by_bus[bus]),
            "p_autocorrelation": autocorrelation_stats(p_by_bus[bus]),
            "q_autocorrelation": autocorrelation_stats(q_by_bus[bus]),
        }

    bus_names = list(p_by_bus)
    with np.errstate(invalid="ignore", over="ignore"):
        aggregate_p = np.sum(np.vstack([p_by_bus[bus] for bus in bus_names]), axis=0)
        aggregate_q = np.sum(np.vstack([q_by_bus[bus] for bus in bus_names]), axis=0)
    aggregate_quality = {
        "p": quality_stats(aggregate_p),
        "q": quality_stats(aggregate_q),
        "p_autocorrelation": autocorrelation_stats(aggregate_p),
        "q_autocorrelation": autocorrelation_stats(aggregate_q),
    }

    splits = add_split_ranges(split_indices(len(p_dates)), p_dates)
    validation = splits["validation"]
    per_bus_baselines: dict[str, Any] = {}
    for bus in sorted(p_by_bus, key=lambda value: int(value) if value.isdigit() else value):
        per_bus_baselines[bus] = {
            "p": persistence_metrics(p_by_bus[bus], validation),
            "q": persistence_metrics(q_by_bus[bus], validation),
        }
    macro = {
        target: {
            baseline: macro_average(per_bus_baselines, target, baseline)
            for baseline in PERSISTENCE_LAGS
        }
        for target in ("p", "q")
    }
    aggregate_baselines = {
        "p": persistence_metrics(aggregate_p, validation),
        "q": persistence_metrics(aggregate_q, validation),
    }
    best = {
        "selection_metric": "validation_mae_lowest",
        "node_macro_average": {
            target: best_persistence(macro[target])
            for target in ("p", "q")
        },
        "aggregate_load": {
            target: best_persistence(aggregate_baselines[target]) for target in ("p", "q")
        },
    }

    constant_buses = {
        target: [bus for bus, item in bus_quality.items() if item[target]["constant"]]
        for target in ("p", "q")
    }
    near_constant_buses = {
        target: [bus for bus, item in bus_quality.items() if item[target]["near_constant"]]
        for target in ("p", "q")
    }
    zero_ratio_high_buses = {
        target: [
            bus
            for bus, item in bus_quality.items()
            if item[target]["zero_ratio"] is not None and item[target]["zero_ratio"] > ZERO_RATIO_HIGH_THRESHOLD
        ]
        for target in ("p", "q")
    }
    all_baseline_metrics: list[dict[str, Any]] = []
    for bus_metrics in per_bus_baselines.values():
        for target in ("p", "q"):
            all_baseline_metrics.extend(bus_metrics[target].values())
    for target in ("p", "q"):
        all_baseline_metrics.extend(aggregate_baselines[target].values())
    wape_warning_count = sum(
        metrics["wape_status"] != "ok" for metrics in all_baseline_metrics
    )
    invalid_quality_target_count = sum(
        item[target]["nan_count"] > 0 or item[target]["inf_count"] > 0
        for item in bus_quality.values()
        for target in ("p", "q")
    )
    node_macro_mae_order = {
        target: sorted(
            PERSISTENCE_LAGS,
            key=lambda baseline: macro[target][baseline]["mae"]
            if macro[target][baseline]["mae"] is not None
            else float("inf"),
        )
        for target in ("p", "q")
    }
    aggregate_mae_order = {
        target: sorted(
            PERSISTENCE_LAGS,
            key=lambda baseline: aggregate_baselines[target][baseline]["mae"]
            if aggregate_baselines[target][baseline]["mae"] is not None
            else float("inf"),
        )
        for target in ("p", "q")
    }
    difficulty_pathology_free = (
        invalid_quality_target_count == 0
        and wape_warning_count == 0
        and all(
            metrics[metric] is not None
            for metrics in all_baseline_metrics
            for metric in ("mae", "rmse", "wape", "smape")
        )
    )
    return {
        "name": mapping_grid["name"],
        "relative_root": mapping_grid["name"],
        "profile_to_bus_mapping_verified": True,
        "physical_load_bus_count": len(bus_names),
        "consumer_profile_count": len(mappings),
        "mapping_source": "results/audits/v2_schema_mapping.json",
        "split": {
            "method": "chronological; no shuffle",
            "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "ranges": splits,
        },
        "quality_definition": {
            "series": "sum of P profiles and sum of Q profiles per mapped physical bus",
            "stats_use": "finite values only; invalid values are counted and not imputed",
            "near_constant_cv_threshold": QUALITY_NEAR_CONSTANT_CV,
            "zero_ratio_high_threshold": ZERO_RATIO_HIGH_THRESHOLD,
        },
        "per_bus_quality": bus_quality,
        "constant_buses": constant_buses,
        "near_constant_buses": near_constant_buses,
        "zero_ratio_high_buses": zero_ratio_high_buses,
        "aggregate_quality": aggregate_quality,
        "validation_baselines": {
            "metric_units": {
                "mae": "load units",
                "rmse": "load units",
                "wape": "percent",
                "smape": "percent",
            },
            "wape_near_zero_abs_sum_threshold": WAPE_NEAR_ZERO_ABS_SUM,
            "per_bus": per_bus_baselines,
            "node_macro_average": macro,
            "aggregate_load": aggregate_baselines,
            "best_persistence": best,
        },
        "prediction_difficulty_checks": {
            "invalid_quality_target_count": invalid_quality_target_count,
            "wape_warning_count": wape_warning_count,
            "node_macro_mae_order": node_macro_mae_order,
            "aggregate_mae_order": aggregate_mae_order,
            "assessment": (
                "no_pathological_quality_or_wape_condition_observed"
                if difficulty_pathology_free
                else "review_required"
            ),
            "assessment_basis": [
                "finite baseline metrics on validation pairs",
                "no NaN/Inf quality values on physical buses",
                "no undefined or near-zero WAPE denominator",
            ],
        },
    }


def build_report(data_root: Path, mapping_path: Path) -> dict[str, Any]:
    mapping_report = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_by_name = {grid["name"]: grid for grid in mapping_report.get("lv_grids", [])}
    lv_root = data_root / LV_DATASET
    grid_reports = []
    for grid_root in sorted(path for path in lv_root.iterdir() if path.is_dir()):
        if grid_root.name not in mapping_by_name:
            raise ValueError(f"Missing verified mapping for {grid_root.name}")
        grid_reports.append(build_grid_report(grid_root, mapping_by_name[grid_root.name]))
    return {
        "audit": "v2_load_quality_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "Four norway_4_lv_grids only; industrial data is not processed.",
            "Multiple consumers on one physical bus are summed separately for P and Q.",
            "No source data is modified, no neural network is trained, and no test metric is calculated.",
        ],
        "metric_definitions": {
            "mae": "mean(abs(predicted - actual))",
            "rmse": "sqrt(mean((predicted - actual)^2))",
            "wape": "sum(abs(error)) / sum(abs(actual)) * 100; denominator is reported",
            "smape": "mean(2 * abs(error) / (abs(actual) + abs(predicted))) * 100 over nonzero denominators",
            "autocorrelation": "Pearson correlation between x[t] and x[t-lag] on finite pairs",
        },
        "persistence_definitions": {
            "persistence_1h": "prediction equals value 1 hour earlier",
            "persistence_24h": "prediction equals value 24 hours earlier",
            "persistence_168h": "prediction equals value 168 hours earlier",
        },
        "lv_grids": grid_reports,
    }


def md_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 LV Load Quality and Persistence Baseline Audit",
        "",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        "",
        "Only the four LV grids are processed. Test data has ranges and sample counts only; all baseline metrics are validation-only.",
        "",
    ]
    for grid in report["lv_grids"]:
        lines.extend(
            [
                f"## {grid['name']}",
                "",
                f"Physical load buses: **{grid['physical_load_bus_count']}**; consumer profiles: **{grid['consumer_profile_count']}**.",
                f"profile_to_bus_mapping_verified: **{md_value(grid['profile_to_bus_mapping_verified'])}**.",
                "",
                "### Chronological split",
                "",
                "| Split | Samples | Start | End |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for name in ("train", "validation", "test"):
            split = grid["split"]["ranges"][name]
            lines.append(
                f"| {name} | {split['sample_count']} | {md_value(split['start_time'])} | {md_value(split['end_time'])} |"
            )

        lines.extend(
            [
                "",
                "### Quality by physical bus",
                "",
                "NaN/Inf/negative/zero ratios use all hourly samples; descriptive statistics use finite samples only.",
                "",
                "| Bus | Profiles | Target | NaN % | Inf % | Negative % | Zero % | Mean | Std | Min | Max | CV | P95 | P99 | Constant | Near-constant |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for bus, item in grid["per_bus_quality"].items():
            for target in ("p", "q"):
                stats = item[target]
                lines.append(
                    f"| `{bus}` | {item['profile_count']} | {target} | "
                    f"{md_value(stats['nan_ratio'] * 100 if stats['nan_ratio'] is not None else None)} | "
                    f"{md_value(stats['inf_ratio'] * 100 if stats['inf_ratio'] is not None else None)} | "
                    f"{md_value(stats['negative_ratio'] * 100 if stats['negative_ratio'] is not None else None)} | "
                    f"{md_value(stats['zero_ratio'] * 100 if stats['zero_ratio'] is not None else None)} | "
                    f"{md_value(stats['mean'])} | {md_value(stats['std'])} | {md_value(stats['min'])} | "
                    f"{md_value(stats['max'])} | {md_value(stats['cv'])} | {md_value(stats['p95'])} | "
                    f"{md_value(stats['p99'])} | {md_value(stats['constant'])} | {md_value(stats['near_constant'])} |"
                )
        lines.extend(
            [
                "",
                f"Constant buses: P={md_value(grid['constant_buses']['p'])}; Q={md_value(grid['constant_buses']['q'])}.",
                f"Near-constant buses: P={md_value(grid['near_constant_buses']['p'])}; Q={md_value(grid['near_constant_buses']['q'])}.",
                f"Zero-ratio-high buses (>5%): P={md_value(grid['zero_ratio_high_buses']['p'])}; Q={md_value(grid['zero_ratio_high_buses']['q'])}.",
                "",
                "### Autocorrelation by physical bus",
                "",
                "| Bus | P lag-1 | P lag-24 | P lag-168 | Q lag-1 | Q lag-24 | Q lag-168 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for bus, item in grid["per_bus_quality"].items():
            p_auto = item["p_autocorrelation"]
            q_auto = item["q_autocorrelation"]
            lines.append(
                f"| `{bus}` | {md_value(p_auto['lag_1'])} | {md_value(p_auto['lag_24'])} | "
                f"{md_value(p_auto['lag_168'])} | {md_value(q_auto['lag_1'])} | "
                f"{md_value(q_auto['lag_24'])} | {md_value(q_auto['lag_168'])} |"
            )

        lines.extend(
            [
                "",
                "### Validation persistence baselines",
                "",
                "WAPE is percent; its actual-load denominator and near-zero status are retained in JSON.",
                "",
                "| Bus | Target | Baseline | Valid n | MAE | RMSE | WAPE % | sMAPE % | WAPE status |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for bus, targets in grid["validation_baselines"]["per_bus"].items():
            for target in ("p", "q"):
                for baseline in PERSISTENCE_LAGS:
                    metrics = targets[target][baseline]
                    lines.append(
                        f"| `{bus}` | {target} | {baseline} | {metrics['valid_sample_count']} | "
                        f"{md_value(metrics['mae'])} | {md_value(metrics['rmse'])} | {md_value(metrics['wape'])} | "
                        f"{md_value(metrics['smape'])} | {metrics['wape_status']} |"
                    )
        lines.extend(
            [
                "",
                "#### Node macro average",
                "",
                "| Target | Baseline | Bus count | MAE | RMSE | WAPE % | sMAPE % |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        macro = grid["validation_baselines"]["node_macro_average"]
        for target in ("p", "q"):
            for baseline in PERSISTENCE_LAGS:
                metrics = macro[target][baseline]
                lines.append(
                    f"| {target} | {baseline} | {metrics['mae_bus_count']} | {md_value(metrics['mae'])} | "
                    f"{md_value(metrics['rmse'])} | {md_value(metrics['wape'])} | {md_value(metrics['smape'])} |"
                )
        lines.extend(
            [
                "",
                "#### Grid aggregate load",
                "",
                "| Target | Baseline | Valid n | MAE | RMSE | WAPE % | sMAPE % | WAPE status |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        aggregate = grid["validation_baselines"]["aggregate_load"]
        for target in ("p", "q"):
            for baseline in PERSISTENCE_LAGS:
                metrics = aggregate[target][baseline]
                lines.append(
                    f"| {target} | {baseline} | {metrics['valid_sample_count']} | {md_value(metrics['mae'])} | "
                    f"{md_value(metrics['rmse'])} | {md_value(metrics['wape'])} | {md_value(metrics['smape'])} | "
                    f"{metrics['wape_status']} |"
                )
        best = grid["validation_baselines"]["best_persistence"]
        lines.extend(
            [
                "",
                f"Strongest by lowest validation MAE (node macro): P={best['node_macro_average']['p']}; Q={best['node_macro_average']['q']}; "
                f"aggregate: P={best['aggregate_load']['p']}; Q={best['aggregate_load']['q']}.",
                f"Difficulty check: **{grid['prediction_difficulty_checks']['assessment']}**; "
                f"WAPE warnings={grid['prediction_difficulty_checks']['wape_warning_count']}; "
                f"invalid quality targets={grid['prediction_difficulty_checks']['invalid_quality_target_count']}.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root.resolve(), args.mapping_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"lv_grids={len(report['lv_grids'])} "
        f"physical_buses={sum(grid['physical_load_bus_count'] for grid in report['lv_grids'])}"
    )


if __name__ == "__main__":
    main()
