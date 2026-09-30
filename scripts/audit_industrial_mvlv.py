"""Run the Stage-1 industrial MV/LV schema and compatibility audit.

Only parsing, quality checks, scarce split arithmetic, and FIT-only graph
construction are performed.  No neural model or forecasting metric is run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.audits.btd_full_backbone_bridge import build_scarce_target_graph
from code.audits.federated_transfer_benefit import scarce_split
from code.data.industrial_mvlv_loader import DEFAULT_DATA_ROOT, load_industrial_mvlv


def _split_record(split: Any) -> dict[str, Any]:
    return {
        "available_raw_hours": int(split.available_raw_hours),
        "eligible_targets": int(len(split.eligible_indices)),
        "fit": int(len(split.fit_indices)),
        "calibration": int(len(split.calibration_indices)),
        "audit": int(len(split.audit_indices)),
        "available_start": int(split.available_start),
        "available_end": int(split.available_end),
        "history_length": 168,
        "all_histories_within_available_window": bool(
            len(split.eligible_indices) == 0
            or int(split.eligible_indices.min()) - 168 >= int(split.available_start)
        ),
    }


def _graph_audit(grid: Any, fraction: float) -> dict[str, Any]:
    split = scarce_split(grid, fraction)
    graph = build_scarce_target_graph(grid, split)
    relations = np.asarray(graph.relation_features, dtype=float)
    return {
        "load_index_count": int(len(graph.load_indices)),
        "selected_edge_count": int(graph.edge_index.shape[1]),
        "relation_shape": list(relations.shape),
        "relation_features_finite": bool(np.isfinite(relations).all()),
        "graph_uses_target_fit_only": bool(graph.diagnostics["graph_uses_target_fit_only"]),
        "graph_uses_calibration": bool(graph.diagnostics["graph_uses_calibration"]),
        "graph_uses_audit": bool(graph.diagnostics["graph_uses_audit"]),
        "graph_uses_validation": bool(graph.diagnostics["graph_uses_validation"]),
        "graph_uses_test": bool(graph.diagnostics["graph_uses_test"]),
        "diagnostics": graph.diagnostics,
    }


def build_audit(data_root: str | Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    grid = load_industrial_mvlv(data_root)
    splits = {str(fraction): _split_record(scarce_split(grid, fraction)) for fraction in (0.25, 0.50)}
    graph_checks = {str(fraction): _graph_audit(grid, fraction) for fraction in (0.25, 0.50)}
    file_meta = grid.metadata["load_file_metadata"]
    late_starting = {
        bus_id: details["start_time"]
        for bus_id, details in file_meta.items()
        if details["start_time"] != grid.metadata["common_interval"]["start_time"]
    }
    return {
        "audit": "industrial_mvlv_schema_audit_stage_1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "Schema and data-quality audit only; no neural training.",
            "No validation or test forecasting metric was calculated.",
            "Industrial values are hourly Load_kWh per interval, not MW.",
        ],
        "grid_name": grid.grid_name,
        "topology": {
            "node_count": grid.num_nodes,
            "edge_count": grid.num_edges,
            "connected": bool(grid.metadata["connected"]),
            "tree": bool(grid.metadata["tree"]),
            "source_slack_buses": grid.metadata["source_slack_bus_ids"],
            "base_kv_counts": grid.metadata["base_kv_counts"],
            "load_bus_count": int(grid.load_bus_mask.sum()),
            "load_count_by_region": grid.metadata["load_count_by_region"],
            "load_count_by_voltage": grid.metadata["load_count_by_voltage"],
            "root_depth_definition": grid.metadata["root_depth_definition"],
        },
        "time_series": {
            "file_count": grid.metadata["load_file_count"],
            "start_end_by_file": {
                bus_id: {"start": item["start_time"], "end": item["end_time"]}
                for bus_id, item in file_meta.items()
            },
            "late_starting_files": late_starting,
            "duplicate_timestamps": {
                bus_id: item["duplicate_timestamps"] for bus_id, item in file_meta.items()
            },
            "duplicate_equality_check": all(item["duplicate_values_identical"] for item in file_meta.values()),
            "nan_count": int(sum(item["nan_count"] for item in file_meta.values())),
            "negative_count": int(sum(item["negative_count"] for item in file_meta.values())),
            "zero_load_count_by_bus": grid.metadata["zero_load_count_by_bus"],
            "zero_load_fraction_by_bus": grid.metadata["zero_load_fraction_by_bus"],
            "common_interval": grid.metadata["common_interval"],
            "common_row_count": grid.num_timesteps,
            "missing_hour_count": grid.metadata["common_interval"]["missing_hour_count"],
        },
        "electrical": grid.metadata["electrical_diagnostics"] | {
            "nonzero_r_count": grid.metadata["r_nonzero_count"],
            "zero_r_count": grid.metadata["r_zero_count"],
            "nonzero_x_count": grid.metadata["x_nonzero_count"],
            "zero_x_count": grid.metadata["x_zero_count"],
        },
        "splits": {
            name: {
                "start_index": split.start_index,
                "end_index": split.end_index,
                "sample_count": split.sample_count,
                "start_time": split.start_time,
                "end_time": split.end_time,
            }
            for name, split in grid.splits.items()
        },
        "scarce_history": splits,
        "fit_only_graph_compatibility": graph_checks,
        "guardrails": {"test_evaluated": False, "test_locked": True, "training_executed": False},
    }


def _markdown(report: dict[str, Any]) -> str:
    topology = report["topology"]
    common = report["time_series"]["common_interval"]
    lines = [
        "# Industrial MV/LV schema audit",
        "",
        "Stage 1 schema/quality audit only. No neural training or forecasting metric was evaluated.",
        "",
        "## Topology",
        "",
        f"- Nodes/branches: {topology['node_count']} / {topology['edge_count']}",
        f"- Connected/tree: {topology['connected']} / {topology['tree']}",
        f"- Source/slack buses: {', '.join(topology['source_slack_buses'])}",
        f"- Load buses: {topology['load_bus_count']}",
        f"- BASE_KV counts: `{json.dumps(topology['base_kv_counts'], sort_keys=True)}`",
        "",
        "## Time series",
        "",
        f"- Common interval: `{common['start_time']}` through `{common['end_time']}`",
        f"- Common rows: {common['row_count']}; missing hours: {common['missing_hour_count']}",
        f"- Duplicate groups identical and deduplicated: `{report['time_series']['duplicate_equality_check']}`",
        "",
        "## Splits",
        "",
    ]
    for name, split in report["splits"].items():
        lines.append(f"- {name}: [{split['start_index']}, {split['end_index']}) = {split['sample_count']} rows")
    for fraction, split in report["scarce_history"].items():
        lines.append(
            f"- {int(float(fraction) * 100)}% scarce: available={split['available_raw_hours']}, "
            f"eligible={split['eligible_targets']}, fit/calibration/audit="
            f"{split['fit']}/{split['calibration']}/{split['audit']}"
        )
    lines.extend(["", "## Guardrails", "", "- `test_evaluated = false`", "- `test_locked = true`", "- `training_executed = false`", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "audits")
    args = parser.parse_args()
    report = build_audit(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "industrial_mvlv_schema_audit.json"
    md_path = args.output_dir / "industrial_mvlv_schema_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
