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
from collections import defaultdict
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


def _read_protocol_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"protocol-lock file is missing: {path}")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"protocol-lock file is malformed: {exc}") from exc
    if not isinstance(lock, Mapping):
        raise RuntimeError("protocol-lock must be a JSON object")
    required = {
        "protocol_lock_commit", "reporting_commit", "placeholder_resolved",
        "matrix_cells", "methods", "seeds", "history_fractions",
        "test_opening_authorized",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise RuntimeError(f"protocol-lock is incomplete: {missing}")
    if lock.get("placeholder_resolved") is not True or lock.get("test_opening_authorized") is not True:
        raise RuntimeError("protocol-lock does not explicitly authorize the one-shot TEST event")
    if lock.get("reporting_commit") != EXPECTED_REPORTING_COMMIT:
        raise RuntimeError("protocol-lock does not identify the accepted paper-ready reporting commit")
    commit = str(lock.get("protocol_lock_commit"))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise RuntimeError("protocol-lock commit must be a complete 40-character SHA")
    if lock.get("matrix_cells") != 180 or tuple(lock.get("methods", ())) != METHODS:
        raise RuntimeError("protocol-lock matrix method contract is not the frozen 180-cell matrix")
    if tuple(lock.get("seeds", ())) != SEEDS or tuple(lock.get("history_fractions", ())) != FRACTIONS:
        raise RuntimeError("protocol-lock seed/fraction contract is not frozen")
    return dict(lock)


def preflight(
    checkpoint_root: Path, parity_report: Path, protocol_lock: Path,
    *, authorize_test: bool,
) -> dict[str, Any]:
    """Validate every pre-TEST identity without loading any grid data."""
    if not authorize_test:
        raise PermissionError("final TEST requires the explicit --authorize-test flag")
    mapping = resolve_all_accepted_results()
    if len(mapping) != 180:
        raise RuntimeError(f"accepted result mapping coverage is {len(mapping)}, expected 180")
    lock = _read_protocol_lock(protocol_lock)
    validation = validate_frozen_checkpoints(checkpoint_root, parity_report)
    if validation.get("test_unlock_ready") is not True:
        raise RuntimeError("checkpoint/parity preflight failed; TEST remains locked: " + json.dumps(validation, ensure_ascii=False))
    for cell in matrix_cells():
        path = checkpoint_path(checkpoint_root, cell)
        if sha256_file(path) != json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))["sha256"]:
            raise RuntimeError(f"checkpoint hash changed during preflight: {cell['cell_id']}")
    return {
        "mapping_coverage": len(mapping),
        "checkpoint_validation": validation,
        "protocol_lock": lock,
        "test_accessed": False,
        "test_evaluated": False,
    }


def _result_path(output: Path, cell: Mapping[str, Any]) -> Path:
    fraction = int(round(float(cell["history_fraction"]) * 100))
    return output / str(cell["domain"]) / str(cell["target"]) / f"seed{int(cell['seed'])}" / f"{fraction}pct" / f"{cell['method']}.json"


def _valid_existing_result(path: Path, cell: Mapping[str, Any], checkpoint: Path) -> bool:
    if not path.is_file():
        return False
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(item, Mapping):
            return False
        for key in ("cell_id", "domain", "target", "seed", "history_fraction", "method"):
            if item.get(key) != cell[key]:
                return False
        if item.get("test_accessed") is not True or item.get("test_evaluated") is not True:
            return False
        if item.get("industrial_test_evaluated") is not (cell["domain"] == "industrial_external"):
            return False
        if item.get("reference_grid_tests_evaluated") is not (cell["domain"] == "reference_grid"):
            return False
        if item.get("selection_used_test") is not False or item.get("training_used_test") is not False:
            return False
        if item.get("checkpoint_sha256") != sha256_file(checkpoint):
            return False
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping) or not _finite(metrics):
            return False
        for split in ("test",):
            split_metrics = metrics.get(split)
            if not isinstance(split_metrics, Mapping):
                return False
            for scope in SCOPES:
                scope_metrics = split_metrics.get(scope)
                if not isinstance(scope_metrics, Mapping):
                    return False
                for metric in METRICS:
                    value = scope_metrics.get(metric)
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                        return False
        return True
    except Exception:
        return False


def _test_metrics(model: Any, grid: Any, scaler: Any, indices: Any, device: str, batch_size: int) -> dict[str, Any]:
    from code.audits.federated_transfer_benefit import _evaluate

    return {"test": _evaluate(model, grid, indices, scaler, device, None, batch_size)}


def _evaluate_cell(cell: Mapping[str, Any], grids: Mapping[str, Any], checkpoint_root: Path, device: str, batch_size: int) -> dict[str, Any]:
    import numpy as np
    from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model
    from code.audits.federated_transfer_benefit import fit_fit_only_scaler, scarce_split

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
    return {
        **dict(cell),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_training_state_identity": metadata["selected_training_state_identity"],
        "test_index_start": int(test_split.start_index),
        "test_index_end": int(test_split.end_index),
        "test_target_count": int(len(indices)),
        "metrics": metrics,
        "metric_units": "industrial Load_kWh per hourly interval" if cell["domain"] == "industrial_external" else "reference-grid active-power P units",
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
    grouped: dict[tuple[str, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[(str(item["domain"]), float(item["history_fraction"]), str(item["method"]))].append(item)
    summaries: dict[str, Any] = {}
    for (domain, fraction, method), items in sorted(grouped.items()):
        metric_summary: dict[str, Any] = {}
        for scope in SCOPES:
            metric_summary[scope] = {}
            for metric in METRICS:
                metric_summary[scope][metric] = _mean_std([float(item["metrics"]["test"][scope][metric]) for item in items])
        summaries.setdefault(domain, {}).setdefault(str(fraction), {})[method] = metric_summary
    comparisons: dict[str, Any] = {}
    for domain in ("reference_grid", "industrial_external"):
        comparisons[domain] = {}
        for fraction in FRACTIONS:
            local = {(int(item["seed"]), str(item["target"])): item for item in results if item["domain"] == domain and float(item["history_fraction"]) == fraction and item["method"] == "local"}
            comparisons[domain][str(fraction)] = {}
            for method in COMPARATORS:
                btd = {(int(item["seed"]), str(item["target"])): item for item in results if item["domain"] == domain and float(item["history_fraction"]) == fraction and item["method"] == "btd_fl_direct_transfer"}
                comparator = {(int(item["seed"]), str(item["target"])): item for item in results if item["domain"] == domain and float(item["history_fraction"]) == fraction and item["method"] == method}
                gains = []
                wins = 0
                for key, btd_item in btd.items():
                    if key not in comparator:
                        continue
                    base = float(comparator[key]["metrics"]["test"]["node_macro"]["mae"])
                    value = float(btd_item["metrics"]["test"]["node_macro"]["mae"])
                    gains.append((base - value) / (base + 1e-12))
                    wins += int(value < base)
                comparisons[domain][str(fraction)][method] = {
                    "relative_node_macro_mae": _mean_std(gains) if gains else None,
                    "seed_target_wins": wins,
                    "comparison_cells": len(gains),
                }
    return {
        "schema_version": 1,
        "cell_count": len(results),
        "metrics": summaries,
        "btd_comparisons": comparisons,
        "comparison_methods": list(COMPARATORS),
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
                    "node_macro_mae_mean": node["mae"]["mean"], "node_macro_mae_std": node["mae"]["std"],
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
            "metric_units": "reference-grid active-power P units" if domain == "reference_grid" else "industrial Load_kWh per hourly interval",
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
                    "btd_seed_target_wins": values["seed_target_wins"],
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
        (axes[0], "reference_grid", "参考配电网 TEST", "节点宏平均 MAE（参考网 P 单位）"),
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
            "node_macro_mae": metric["node_macro"]["mae"], "node_macro_rmse": metric["node_macro"]["rmse"],
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
    lines = ["# 最终独立 TEST 结果摘要", "", f"完成单元：**{len(results)}/180**", "", "所有数值均来自 checkpoint 加载后的 TEST 推理；不同域单位不混合。", ""]
    for domain, fractions in summary["metrics"].items():
        lines.append(f"## {domain}")
        lines.append("")
        for fraction, methods in fractions.items():
            lines.append(f"### history_fraction={fraction}")
            lines.append("")
            lines.append("| 方法 | node-macro MAE mean | std | 单位 |")
            lines.append("| --- | ---: | ---: | --- |")
            for method, metrics in methods.items():
                unit = "Load_kWh/小时" if domain == "industrial_external" else "参考网 P 单位"
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
    output.mkdir(parents=True, exist_ok=True)
    _append_log(output, "all preflight gates passed; TEST target access begins")
    # Imports and grid loading occur only after fail-closed preflight.
    from code.data.lv_grid_loader import LVGridLoader
    from code.federated.industrial_external import load_external_grids

    loader = LVGridLoader(data_root, mapping_json)
    grids = loader.load_all()
    grids.update(load_external_grids(data_root, mapping_json))
    results: list[Mapping[str, Any]] = []
    completed: set[str] = set()
    for cell in matrix_cells():
        path = _result_path(output, cell)
        checkpoint = checkpoint_path(checkpoint_root, cell)
        if resume and _valid_existing_result(path, cell, checkpoint):
            item = json.loads(path.read_text(encoding="utf-8"))
            results.append(item); completed.add(cell["cell_id"])
            continue
        _append_log(output, f"evaluating {cell['cell_id']}")
        item = _evaluate_cell(cell, grids, checkpoint_root, device, batch_size)
        _atomic_json(path, item)
        results.append(item); completed.add(cell["cell_id"])
        _atomic_json(output / "test_progress.json", {
            "completed_cells": len(completed), "total_cells": 180,
            "current": cell, "test_accessed": True, "test_evaluated": True,
        })
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
