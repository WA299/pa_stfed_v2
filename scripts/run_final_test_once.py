"""One-shot, inference-only canonical TEST evaluator.

This entry point is deliberately fail-closed.  It reads no TEST target until
all 180 tensor checkpoints, sidecars, SHA256 values, the complete pre-TEST
parity report, and a separate protocol-lock file have passed validation.  It
never trains, tunes, selects, or rewrites accepted result artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.federated.accepted_results import resolve_all_accepted_results
from code.federated.checkpointing import (
    FROZEN_COMMITS,
    INDUSTRIAL_TARGET,
    REFERENCE_TARGETS,
    checkpoint_path,
    matrix_cells,
    replay_checkpoint,
    sha256_file,
)
from scripts.validate_frozen_checkpoints import validate_frozen_checkpoints

DEFAULT_CHECKPOINT_ROOT = ROOT / "artifacts" / "frozen_pretest_checkpoints"
DEFAULT_PARITY = DEFAULT_CHECKPOINT_ROOT / "parity_report.json"
DEFAULT_OUTPUT = ROOT / "results" / "final_test_opening"
DEFAULT_PROTOCOL_LOCK = ROOT / "paper" / "preflight" / "test_protocol_lock.json"
EXPECTED_REPORTING_COMMIT = "58dadfa088b78406d8de208faa2bff43a509f49c"
METHODS = ("local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer")
SEEDS = (42, 123, 2026)
FRACTIONS = (0.25, 0.50)
METRICS = ("mae", "rmse", "wape_pct", "smape_pct")
SCOPES = ("node_macro", "grid_aggregate")
COMPARATORS = ("local", "fedavg", "fedprox", "fedper", "fedfomo_style")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_log(output: Path, message: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "final_test.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{_utc_now()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _assert_output_isolated(output: Path) -> None:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"final TEST output must be exactly {DEFAULT_OUTPUT}")
    if output.resolve() in {ROOT / "results", ROOT / "paper"}:
        raise ValueError("refusing a broad output directory")


def _read_protocol_lock(
    path: Path, *, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    parity_report: Path = DEFAULT_PARITY,
) -> dict[str, Any]:
    """Return a lock only after validating its committed Git and artifact identity."""
    from scripts.manage_test_protocol_lock import check_protocol_lock

    return check_protocol_lock(
        path, checkpoint_root=checkpoint_root, parity_report=parity_report,
    )


def preflight(
    checkpoint_root: Path, parity_report: Path, protocol_lock: Path,
    *, authorize_test: bool,
) -> dict[str, Any]:
    """Validate every pre-TEST identity without loading any grid data."""
    if not authorize_test:
        raise PermissionError("final TEST requires the explicit --authorize-test flag")
    if not protocol_lock.is_file():
        raise RuntimeError(f"protocol-lock file is missing: {protocol_lock}")
    validation = validate_frozen_checkpoints(checkpoint_root, parity_report)
    if validation.get("test_unlock_ready") is not True:
        raise RuntimeError("checkpoint/parity preflight failed; TEST remains locked: " + json.dumps(validation, ensure_ascii=False))
    mapping = resolve_all_accepted_results()
    if len(mapping) != 180:
        raise RuntimeError(f"accepted result mapping coverage is {len(mapping)}, expected 180")
    protocol = _read_protocol_lock(protocol_lock, checkpoint_root=checkpoint_root, parity_report=parity_report)
    for cell in matrix_cells():
        path = checkpoint_path(checkpoint_root, cell)
        if sha256_file(path) != json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))["sha256"]:
            raise RuntimeError(f"checkpoint hash changed during preflight: {cell['cell_id']}")
    return {
        "mapping_coverage": len(mapping),
        "checkpoint_validation": validation,
        "protocol_lock": protocol["lock"],
        "protocol_lock_identity": protocol["git_identity"],
        "test_accessed": False,
        "test_evaluated": False,
    }


def _result_path(output: Path, cell: Mapping[str, Any]) -> Path:
    fraction = int(round(float(cell["history_fraction"]) * 100))
    return output / str(cell["domain"]) / str(cell["target"]) / f"seed{int(cell['seed'])}" / f"{fraction}pct" / f"{cell['method']}.json"


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a cell result without ever replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link creation fails atomically if another process already
        # created this cell path; it cannot overwrite a successful result.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required input source file is missing: {path}")
    return {"path": str(path.resolve()), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def _grid_data_identity(data_root: Path, target: str, grid: Any) -> dict[str, Any]:
    if target == INDUSTRIAL_TARGET:
        candidate = data_root / target
        directory = candidate if candidate.is_dir() else data_root.parent / target
    else:
        directory = data_root / "norway_4_lv_grids" / target
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset source directory is missing for {target}: {directory}")
    files = [_file_identity(path) for path in sorted(directory.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError(f"dataset source directory has no files: {directory}")
    metadata = dict(getattr(grid, "metadata", {}) or {})
    metadata_bytes = json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return {
        "dataset": target,
        "directory": str(directory.resolve()),
        "files": files,
        "grid_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "node_order_sha256": hashlib.sha256("\n".join(str(node) for node in grid.node_ids).encode("utf-8")).hexdigest(),
        "load_bus_mask_sha256": hashlib.sha256(__import__("numpy").asarray(grid.load_bus_mask, dtype="uint8").tobytes()).hexdigest(),
    }


def _test_split_identity(cell: Mapping[str, Any], grid: Any) -> dict[str, Any]:
    import numpy as np

    split = grid.splits["test"]
    start, end = int(split.start_index), int(split.end_index)
    if end <= start:
        raise ValueError(f"canonical TEST split is empty or invalid for {cell['target']}")
    indices = np.arange(start, end, dtype=np.int64)
    return {
        "split_name": "test",
        "dataset": str(cell["target"]),
        "start_index": start,
        "end_index": end,
        "target_count": int(len(indices)),
        "start_time": str(getattr(split, "start_time", "")),
        "end_time": str(getattr(split, "end_time", "")),
        "target_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def _source_provenance(
    cell: Mapping[str, Any], checkpoint: Path, metadata: Mapping[str, Any],
    protocol_lock_identity: Mapping[str, Any], data_identity: Mapping[str, Any],
    mapping_identity: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_identity = metadata.get("source_artifact_identity", {})
    accepted_path = Path(str(metadata.get("accepted_result_artifact", "")))
    accepted_file = _file_identity(accepted_path)
    if accepted_file["sha256"] != accepted_identity.get("sha256"):
        raise ValueError(f"accepted result source hash changed for {cell['cell_id']}")
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar_sha256": sha256_file(checkpoint.with_suffix(checkpoint.suffix + ".json")),
        "model_class": metadata.get("model_class"),
        "constructor_identity": metadata.get("constructor_identity"),
        "selected_training_state_identity": metadata.get("selected_training_state_identity"),
        "accepted_result_artifact": metadata.get("accepted_result_artifact"),
        "accepted_result_json_keys": metadata.get("accepted_result_json_keys"),
        "accepted_result_source_identity": accepted_file,
        "frozen_commits": metadata.get("frozen_commits"),
        "target_scaler": metadata.get("target_scaler"),
        "target_node_order": metadata.get("target_node_order"),
        "target_load_bus_mask": metadata.get("target_load_bus_mask"),
        "protocol_lock_identity": dict(protocol_lock_identity),
        "data_source_identity": dict(data_identity),
        "mapping_source_identity": dict(mapping_identity),
        "matrix_identity": {key: cell[key] for key in ("cell_id", "domain", "target", "seed", "history_fraction", "method", "canonical_method")},
    }


def _validate_existing_result(
    path: Path, cell: Mapping[str, Any], checkpoint: Path, *,
    protocol_lock_identity: Mapping[str, Any], expected_test_split: Mapping[str, Any] | None = None,
    expected_source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load a prior result or fail loudly; invalid records are never rerun over."""
    if not path.exists():
        return None
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(item, Mapping):
            raise ValueError("result root is not an object")
        if item.get("result_schema_version") != 2:
            raise ValueError("result schema version is missing/unsupported")
        for key in ("cell_id", "domain", "target", "seed", "history_fraction", "method"):
            if item.get(key) != cell[key]:
                raise ValueError(f"cell identity mismatch: {key}")
        if item.get("test_accessed") is not True or item.get("test_evaluated") is not True:
            raise ValueError("TEST evaluation flags are missing or false")
        if item.get("industrial_test_evaluated") is not (cell["domain"] == "industrial_external"):
            raise ValueError("industrial TEST flag does not match cell domain")
        if item.get("reference_grid_tests_evaluated") is not (cell["domain"] == "reference_grid"):
            raise ValueError("reference TEST flag does not match cell domain")
        if item.get("selection_used_test") is not False or item.get("training_used_test") is not False:
            raise ValueError("result indicates TEST affected model selection/training")
        if item.get("checkpoint_path") != str(checkpoint.resolve()):
            raise ValueError("checkpoint path identity mismatch")
        if item.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError("checkpoint SHA256 mismatch")
        if item.get("protocol_lock_identity") != dict(protocol_lock_identity):
            raise ValueError("protocol-lock Git identity mismatch")
        evaluated_at = item.get("evaluation_started_at_utc")
        completed_at = item.get("evaluation_completed_at_utc")
        if not isinstance(evaluated_at, str) or not isinstance(completed_at, str):
            raise ValueError("original TEST evaluation timestamps are missing")
        datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
        datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        elapsed = item.get("evaluation_elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(float(elapsed)) or elapsed < 0:
            raise ValueError("original TEST evaluation duration is missing or invalid")
        split_identity = item.get("test_split_identity")
        if not isinstance(split_identity, Mapping) or split_identity.get("split_name") != "test":
            raise ValueError("canonical TEST split identity is missing")
        if not isinstance(split_identity.get("target_index_sha256"), str) or len(split_identity["target_index_sha256"]) != 64:
            raise ValueError("canonical TEST target-index digest is missing")
        if expected_test_split is not None and dict(split_identity) != dict(expected_test_split):
            raise ValueError("TEST split identity/boundaries do not match canonical target split")
        if item.get("test_index_start") != split_identity.get("start_index") or item.get("test_index_end") != split_identity.get("end_index"):
            raise ValueError("TEST index range does not match recorded split identity")
        provenance = item.get("source_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("complete source provenance is missing")
        if provenance.get("protocol_lock_identity") != dict(protocol_lock_identity):
            raise ValueError("source provenance is not bound to this protocol-lock commit")
        if expected_source_provenance is not None and dict(provenance) != dict(expected_source_provenance):
            raise ValueError("complete source provenance does not match current frozen inputs")
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping) or not _finite(metrics):
            raise ValueError("metrics are missing or non-finite")
        split_metrics = metrics.get("test")
        if not isinstance(split_metrics, Mapping):
            raise ValueError("TEST metrics are missing")
        for scope in SCOPES:
            scope_metrics = split_metrics.get(scope)
            if not isinstance(scope_metrics, Mapping):
                raise ValueError(f"TEST metric scope is missing: {scope}")
            for metric in METRICS:
                value = scope_metrics.get(metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"TEST metric is missing/non-finite: {scope}.{metric}")
        if int(item.get("test_target_count", -1)) != int(split_identity.get("target_count", -2)):
            raise ValueError("TEST target count does not match split identity")
        return dict(item)
    except Exception as exc:
        raise RuntimeError(
            f"existing TEST result is invalid; refusing reevaluation or overwrite: {path}: {exc}"
        ) from exc


def _test_metrics(model: Any, grid: Any, scaler: Any, indices: Any, device: str, batch_size: int) -> dict[str, Any]:
    from code.audits.federated_transfer_benefit import _evaluate

    return {"test": _evaluate(model, grid, indices, scaler, device, None, batch_size)}


def _evaluate_cell(
    cell: Mapping[str, Any], grids: Mapping[str, Any], checkpoint_root: Path,
    device: str, batch_size: int, *, protocol_lock_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any], mapping_identity: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model
    from code.audits.federated_transfer_benefit import fit_fit_only_scaler, scarce_split

    evaluation_started_at = _utc_now()
    timer_start = time.perf_counter()
    grid = grids[str(cell["target"])]
    split = scarce_split(grid, float(cell["history_fraction"]))
    scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
    graph = build_scarce_target_graph(grid, split)
    model = full_model(grid, graph, device, seed=int(cell["seed"]))
    checkpoint = checkpoint_path(checkpoint_root, cell)
    metadata = replay_checkpoint(checkpoint, model, cell=cell, grid=grid, scaler=scaler)
    test_split = grid.splits["test"]
    indices = np.arange(test_split.start_index, test_split.end_index, dtype=np.int64)
    metrics = _test_metrics(model, grid, scaler, indices, device, batch_size)
    evaluation_completed_at = _utc_now()
    source_provenance = _source_provenance(
        cell, checkpoint, metadata, protocol_lock_identity, data_identity, mapping_identity,
    )
    return {
        "result_schema_version": 2,
        **dict(cell),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_training_state_identity": metadata["selected_training_state_identity"],
        "protocol_lock_identity": dict(protocol_lock_identity),
        "source_provenance": source_provenance,
        "evaluation_started_at_utc": evaluation_started_at,
        "evaluation_completed_at_utc": evaluation_completed_at,
        "evaluation_elapsed_seconds": float(time.perf_counter() - timer_start),
        "test_split_identity": _test_split_identity(cell, grid),
        "test_index_start": int(test_split.start_index),
        "test_index_end": int(test_split.end_index),
        "test_target_count": int(len(indices)),
        "metrics": metrics,
        "metric_units": "industrial Load_kWh per hourly interval" if cell["domain"] == "industrial_external" else "reference-grid P; provisionally treated as MW",
        "reporting_units": {"node_macro_mae": "kWh per hourly interval" if cell["domain"] == "industrial_external" else "MW (provisional); multiply by 1000 for kW display"},
        "test_accessed": True,
        "test_evaluated": True,
        "industrial_test_evaluated": cell["domain"] == "industrial_external",
        "reference_grid_tests_evaluated": cell["domain"] == "reference_grid",
        "selection_used_test": False,
        "training_used_test": False,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "std": float(np.std(array, ddof=0))}


def summarize(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    expected_cells = {cell["cell_id"]: cell for cell in matrix_cells()}
    actual_ids = [str(item.get("cell_id", "")) for item in results]
    if len(results) != 180 or len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_cells):
        raise ValueError("TEST summary requires exactly 180 unique complete matrix results")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in results:
        cell_id = str(item["cell_id"])
        expected = expected_cells[cell_id]
        if any(item.get(key) != expected[key] for key in ("domain", "target", "seed", "history_fraction", "method")):
            raise ValueError(f"TEST summary cell identity mismatch: {cell_id}")
        if item.get("test_evaluated") is not True or item.get("test_accessed") is not True:
            raise ValueError(f"TEST summary received non-TEST result: {cell_id}")
        metrics = item.get("metrics", {}).get("test", {})
        for scope in SCOPES:
            for metric in METRICS:
                value = metrics.get(scope, {}).get(metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"TEST summary metric missing/non-finite: {cell_id} {scope}/{metric}")
        by_id[cell_id] = item

    # First reduce each seed to an unweighted four-grid macro for the
    # reference domain.  Industrial has one target, so its seed value is the
    # target's value.  Only then calculate the three-seed mean/std.
    per_seed: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for domain in ("reference_grid", "industrial_external"):
        per_seed[domain] = {}
        summaries[domain] = {}
        for fraction in FRACTIONS:
            fraction_key = str(fraction)
            per_seed[domain][fraction_key] = {}
            summaries[domain][fraction_key] = {}
            for method in METHODS:
                per_seed[domain][fraction_key][method] = {}
                summaries[domain][fraction_key][method] = {}
                for seed in SEEDS:
                    seed_items = [
                        by_id[cell["cell_id"]] for cell in expected_cells.values()
                        if cell["domain"] == domain and cell["history_fraction"] == fraction
                        and cell["seed"] == seed and cell["method"] == method
                    ]
                    expected_targets = len(REFERENCE_TARGETS) if domain == "reference_grid" else 1
                    if len(seed_items) != expected_targets:
                        raise ValueError(f"incomplete per-seed TEST group: {domain}/{fraction}/{seed}/{method}")
                    seed_metrics: dict[str, Any] = {}
                    for scope in SCOPES:
                        seed_metrics[scope] = {}
                        for metric in METRICS:
                            seed_metrics[scope][metric] = float(np.mean([
                                float(item["metrics"]["test"][scope][metric]) for item in seed_items
                            ]))
                    per_seed[domain][fraction_key][method][str(seed)] = seed_metrics
                for scope in SCOPES:
                    summaries[domain][fraction_key][method][scope] = {}
                    for metric in METRICS:
                        summaries[domain][fraction_key][method][scope][metric] = _mean_std([
                            per_seed[domain][fraction_key][method][str(seed)][scope][metric]
                            for seed in SEEDS
                        ])
    comparisons: dict[str, Any] = {}
    for domain in ("reference_grid", "industrial_external"):
        comparisons[domain] = {}
        for fraction in FRACTIONS:
            comparisons[domain][str(fraction)] = {}
            for method in COMPARATORS:
                gains: list[float] = []
                wins = 0
                for seed in SEEDS:
                    base = per_seed[domain][str(fraction)][method][str(seed)]["node_macro"]["mae"]
                    value = per_seed[domain][str(fraction)]["btd_fl_direct_transfer"][str(seed)]["node_macro"]["mae"]
                    gains.append((base - value) / (base + 1e-12))
                    wins += int(value < base)
                comparisons[domain][str(fraction)][method] = {
                    "relative_node_macro_mae": _mean_std(gains) if gains else None,
                    "seed_wins": wins,
                    "comparison_cells": len(gains),
                }
    return {
        "schema_version": 2,
        "cell_count": len(results),
        "metrics": summaries,
        "per_seed_metrics": per_seed,
        "btd_comparisons": comparisons,
        "comparison_methods": list(COMPARATORS),
        "statistical_contract": {
            "reference_grid": "unweighted mean across the four targets within each seed, then mean/std across seeds 42/123/2026",
            "industrial_external": "one target value per seed, then mean/std across seeds 42/123/2026",
            "standard_deviation": "population standard deviation (ddof=0)",
            "raw_reference_p_unit": "MW provisional; raw values unchanged",
            "industrial_load_unit": "Load_kWh per hourly interval",
        },
        "test_evaluated": True,
    }


def _write_summary_tables(output: Path, summary: Mapping[str, Any]) -> None:
    """Write explicit domain/fraction summaries without mixing physical units."""
    rows: list[dict[str, Any]] = []
    for domain, fractions in summary["metrics"].items():
        for fraction, methods in fractions.items():
            for method, scopes in methods.items():
                node = scopes["node_macro"]
                aggregate = scopes["grid_aggregate"]
                rows.append({
                    "domain": domain, "history_fraction": fraction, "method": method,
                    "raw_error_unit": "MW provisional" if domain == "reference_grid" else "Load_kWh per hourly interval",
                    "reference_display_conversion_factor_to_kW": 1000.0 if domain == "reference_grid" else None,
                    "node_macro_mae_raw_mean": node["mae"]["mean"], "node_macro_mae_raw_std": node["mae"]["std"],
                    "node_macro_mae_display_kW_mean": node["mae"]["mean"] * 1000.0 if domain == "reference_grid" else None,
                    "node_macro_mae_display_kW_std": node["mae"]["std"] * 1000.0 if domain == "reference_grid" else None,
                    "grid_aggregate_mae_display_kW_mean": aggregate["mae"]["mean"] * 1000.0 if domain == "reference_grid" else None,
                    "grid_aggregate_mae_display_kW_std": aggregate["mae"]["std"] * 1000.0 if domain == "reference_grid" else None,
                    "node_macro_mae_industrial_kWh_mean": node["mae"]["mean"] if domain == "industrial_external" else None,
                    "node_macro_mae_industrial_kWh_std": node["mae"]["std"] if domain == "industrial_external" else None,
                    "node_macro_rmse_mean": node["rmse"]["mean"], "node_macro_rmse_std": node["rmse"]["std"],
                    "node_macro_wape_pct_mean": node["wape_pct"]["mean"], "node_macro_wape_pct_std": node["wape_pct"]["std"],
                    "node_macro_smape_pct_mean": node["smape_pct"]["mean"], "node_macro_smape_pct_std": node["smape_pct"]["std"],
                    "grid_aggregate_mae_mean": aggregate["mae"]["mean"], "grid_aggregate_mae_std": aggregate["mae"]["std"],
                })
    fieldnames = list(rows[0]) if rows else ["domain", "history_fraction", "method"]
    with (output / "test_summary_by_domain_fraction.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    for domain, filename in (("reference_grid", "reference_grid_test_summary.json"), ("industrial_external", "industrial_test_summary.json")):
        _atomic_json(output / filename, {
            "schema_version": 1, "domain": domain,
            "metric_units": "reference-grid P; provisionally MW (raw metrics unchanged)" if domain == "reference_grid" else "industrial Load_kWh per hourly interval",
            "metrics": summary["metrics"].get(domain, {}),
            "btd_comparisons": summary["btd_comparisons"].get(domain, {}),
            "test_evaluated": True,
        })
    comparison_rows = []
    for domain, fractions in summary["btd_comparisons"].items():
        for fraction, comparators in fractions.items():
            for comparator, values in comparators.items():
                gain = values["relative_node_macro_mae"]
                comparison_rows.append({
                    "domain": domain, "history_fraction": fraction, "comparator": comparator,
                    "btd_relative_node_macro_mae_mean": None if gain is None else gain["mean"],
                    "btd_relative_node_macro_mae_std": None if gain is None else gain["std"],
                    "btd_seed_wins": values["seed_wins"],
                    "comparison_cells": values["comparison_cells"],
                })
    comparison_fields = list(comparison_rows[0]) if comparison_rows else ["domain", "history_fraction", "comparator"]
    with (output / "btd_test_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader(); writer.writerows(comparison_rows)
    _atomic_json(output / "history_fraction_comparison.json", {
        "schema_version": 1, "metrics": summary["metrics"], "btd_comparisons": summary["btd_comparisons"],
        "test_evaluated": True,
    })


def _write_report_figure(output: Path, summary: Mapping[str, Any]) -> None:
    """Best-effort report figure; TEST execution never depends on plotting."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _atomic_json(output / "figure_generation_status.json", {"generated": False, "reason": "matplotlib is not installed", "test_evaluated": True})
        return
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, domain, title, ylabel in (
        (axes[0], "reference_grid", "参考配电网 TEST", "节点宏平均 MAE（MW 暂定口径；展示 kW=×1000）"),
        (axes[1], "industrial_external", "工业配电网 TEST", "节点宏平均 MAE（Load_kWh/小时）"),
    ):
        domain_metrics = summary["metrics"].get(domain, {})
        for method in METHODS:
            means = [domain_metrics.get(str(fraction), {}).get(method, {}).get("node_macro", {}).get("mae", {}).get("mean", float("nan")) for fraction in FRACTIONS]
            stds = [domain_metrics.get(str(fraction), {}).get(method, {}).get("node_macro", {}).get("mae", {}).get("std", 0.0) for fraction in FRACTIONS]
            axis.errorbar([25, 50], means, yerr=stds, marker="o", capsize=3, label=method)
        axis.set_title(title); axis.set_xlabel("历史数据比例（%）"); axis.set_ylabel(ylabel); axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.savefig(figure_dir / "test_history_fraction_node_macro_mae.png", dpi=220)
    fig.savefig(figure_dir / "test_history_fraction_node_macro_mae.pdf")
    plt.close(fig)
    _atomic_json(output / "figure_generation_status.json", {"generated": True, "files": ["figures/test_history_fraction_node_macro_mae.png", "figures/test_history_fraction_node_macro_mae.pdf"], "test_evaluated": True})


def write_outputs(output: Path, results: list[Mapping[str, Any]], summary: Mapping[str, Any], preflight_report: Mapping[str, Any]) -> None:
    rows = []
    for item in results:
        metric = item["metrics"]["test"]
        rows.append({
            "cell_id": item["cell_id"], "domain": item["domain"], "target": item["target"],
            "seed": item["seed"], "history_fraction": item["history_fraction"], "method": item["method"],
            "node_macro_mae_raw": metric["node_macro"]["mae"],
            "node_macro_mae_display_kW": metric["node_macro"]["mae"] * 1000.0 if item["domain"] == "reference_grid" else None,
            "node_macro_mae_industrial_kWh": metric["node_macro"]["mae"] if item["domain"] == "industrial_external" else None,
            "node_macro_rmse": metric["node_macro"]["rmse"],
            "node_macro_wape_pct": metric["node_macro"]["wape_pct"], "node_macro_smape_pct": metric["node_macro"]["smape_pct"],
            "grid_aggregate_mae": metric["grid_aggregate"]["mae"], "grid_aggregate_rmse": metric["grid_aggregate"]["rmse"],
            "metric_units": item["metric_units"],
        })
    csv_path = output / "all_test_cell_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _atomic_json(output / "all_test_cell_results.json", {"test_evaluated": True, "cell_count": len(results), "cells": results})
    _atomic_json(output / "test_summary.json", summary)
    _write_summary_tables(output, summary)
    _write_report_figure(output, summary)
    _atomic_json(output / "final_test_opening_manifest.json", {
        "schema_version": 1, "cell_count": len(results), "expected_cell_count": 180,
        "preflight": preflight_report, "test_evaluated": True,
        "industrial_test_evaluated": True, "reference_grid_tests_evaluated": True,
        "historical_artifacts_modified": False,
    })
    lines = ["# 最终独立 TEST 结果摘要", "", f"完成单元：**{len(results)}/180**", "", "所有数值均来自 checkpoint 加载后的 TEST 推理；不同域单位不混合。参考配电网原始 P 误差按暂定 MW 口径保留，kW 仅作 ×1000 展示换算；工业误差保留 Load_kWh/小时。", ""]
    for domain, fractions in summary["metrics"].items():
        lines.append(f"## {domain}")
        lines.append("")
        for fraction, methods in fractions.items():
            lines.append(f"### history_fraction={fraction}")
            lines.append("")
            lines.append("| 方法 | node-macro MAE raw mean | std | 展示单位/口径 |")
            lines.append("| --- | ---: | ---: | --- |")
            for method, metrics in methods.items():
                unit = "Load_kWh/小时" if domain == "industrial_external" else "MW 暂定口径（可换算 kW）"
                lines.append(f"| {method} | {metrics['node_macro']['mae']['mean']:.12g} | {metrics['node_macro']['mae']['std']:.12g} | {unit} |")
            lines.append("")
    (output / "results_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_once(
    *, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT, parity_report: Path = DEFAULT_PARITY,
    protocol_lock: Path = DEFAULT_PROTOCOL_LOCK, output: Path = DEFAULT_OUTPUT,
    data_root: Path, mapping_json: Path, device: str = "cuda", batch_size: int = 32,
    authorize_test: bool = False, resume: bool = True,
) -> dict[str, Any]:
    _assert_output_isolated(output)
    preflight_report = preflight(checkpoint_root, parity_report, protocol_lock, authorize_test=authorize_test)
    protocol_identity = preflight_report["protocol_lock_identity"]
    output.mkdir(parents=True, exist_ok=True)
    _append_log(output, "protocol/checkpoint/parity gates passed; TEST inputs may now be loaded under explicit authorization")
    # Refuse malformed, incomplete, or stale prior records before loading data.
    prior_raw: dict[str, dict[str, Any]] = {}
    for cell in matrix_cells():
        result_path = _result_path(output, cell)
        if result_path.exists():
            item = _validate_existing_result(
                result_path, cell, checkpoint_path(checkpoint_root, cell),
                protocol_lock_identity=protocol_identity,
            )
            if item is not None:
                prior_raw[cell["cell_id"]] = item
    if not resume and prior_raw:
        raise RuntimeError("existing validated TEST cells are present; rerun only with resume enabled")

    # Dataset loading occurs only after explicit TEST authorization and all
    # static protocol/checkpoint/result identities have passed.
    from code.data.lv_grid_loader import LVGridLoader
    from code.federated.industrial_external import load_external_grids

    loader = LVGridLoader(data_root, mapping_json)
    grids = loader.load_all()
    grids.update(load_external_grids(data_root, mapping_json))
    mapping_identity = _file_identity(mapping_json)
    data_identities = {
        target: _grid_data_identity(data_root, target, grids[target])
        for target in sorted({cell["target"] for cell in matrix_cells()})
    }
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    split_identities: dict[str, dict[str, Any]] = {}
    source_provenance: dict[str, dict[str, Any]] = {}
    # Complete validation of every persisted result—including canonical split
    # indices/timestamps—finishes before any cell inference starts.
    for cell in matrix_cells():
        checkpoint = checkpoint_path(checkpoint_root, cell)
        sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
        checkpoint_metadata[cell["cell_id"]] = json.loads(sidecar_path.read_text(encoding="utf-8"))
        split_identities[cell["cell_id"]] = _test_split_identity(cell, grids[cell["target"]])
        source_provenance[cell["cell_id"]] = _source_provenance(
            cell, checkpoint, checkpoint_metadata[cell["cell_id"]], protocol_identity,
            data_identities[cell["target"]], mapping_identity,
        )
        if cell["cell_id"] in prior_raw:
            validated = _validate_existing_result(
                _result_path(output, cell), cell, checkpoint,
                protocol_lock_identity=protocol_identity,
                expected_test_split=split_identities[cell["cell_id"]],
                expected_source_provenance=source_provenance[cell["cell_id"]],
            )
            prior_raw[cell["cell_id"]] = validated

    results_by_id: dict[str, Mapping[str, Any]] = dict(prior_raw)
    completed: set[str] = set(results_by_id)
    _atomic_json(output / "test_progress.json", {
        "completed_cells": len(completed), "total_cells": 180,
        "completed_cell_ids": sorted(completed), "current": None,
        "test_accessed": bool(completed), "test_evaluated": bool(completed),
        "resume_validated_existing_results": len(completed),
        "protocol_lock_identity": protocol_identity,
    })
    for cell in matrix_cells():
        cell_id = cell["cell_id"]
        if cell_id in results_by_id:
            _append_log(output, f"resume skip validated TEST result {cell_id}")
            continue
        from scripts.manage_test_protocol_lock import git_protocol_identity
        if git_protocol_identity(protocol_lock) != protocol_identity:
            raise RuntimeError("protocol-lock Git identity changed during final TEST execution")
        _append_log(output, f"evaluating TEST cell {cell_id}")
        item = _evaluate_cell(
            cell, grids, checkpoint_root, device, batch_size,
            protocol_lock_identity=protocol_identity,
            data_identity=data_identities[cell["target"]], mapping_identity=mapping_identity,
        )
        if item["test_split_identity"] != split_identities[cell_id]:
            raise RuntimeError(f"TEST split identity changed during evaluation: {cell_id}")
        if item["source_provenance"] != source_provenance[cell_id]:
            raise RuntimeError(f"checkpoint/input provenance changed during evaluation: {cell_id}")
        if git_protocol_identity(protocol_lock) != protocol_identity:
            raise RuntimeError("protocol-lock Git identity changed while a TEST cell was being evaluated")
        path = _result_path(output, cell)
        try:
            _atomic_create_json(path, item)
        except FileExistsError as exc:
            raise RuntimeError(f"TEST result appeared concurrently; refusing overwrite: {path}") from exc
        results_by_id[cell_id] = item
        completed.add(cell_id)
        _atomic_json(output / "test_progress.json", {
            "completed_cells": len(completed), "total_cells": 180,
            "completed_cell_ids": sorted(completed), "current": cell,
            "test_accessed": True, "test_evaluated": True,
            "protocol_lock_identity": protocol_identity,
        })
    results = [results_by_id[cell["cell_id"]] for cell in matrix_cells() if cell["cell_id"] in results_by_id]
    if len(results) != 180 or len({item["cell_id"] for item in results}) != 180:
        raise RuntimeError("final TEST matrix is incomplete or duplicated")
    summary = summarize(results)
    write_outputs(output, results, summary, preflight_report)
    _append_log(output, "all 180 TEST cells complete")
    return {"cell_count": 180, "output": str(output), "test_evaluated": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-test", action="store_true", help="explicit one-shot TEST authorization")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_once(
        checkpoint_root=args.checkpoint_root.resolve(), parity_report=args.parity_report.resolve(),
        protocol_lock=args.protocol_lock.resolve(), output=args.output.resolve(),
        data_root=args.data_root.resolve(), mapping_json=args.mapping_json.resolve(),
        device=args.device, authorize_test=args.authorize_test, resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
