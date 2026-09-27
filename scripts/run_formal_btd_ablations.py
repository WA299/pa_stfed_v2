"""Run efficient mechanism-only ablations for formal 25%-history BTD-FL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data.lv_grid_loader import LVGridLoader  # noqa: E402
from code.federated.formal_btd_ablations import ABLATIONS, load_frozen_reference, run_ablation  # noqa: E402
from scripts.run_federated import build_synthetic_clients  # noqa: E402


def _formal_grid_from_client(client: Any) -> Any:
    from types import SimpleNamespace

    n, nodes = 6400, client.model.num_nodes
    rng = np.random.default_rng(700 + nodes)
    dynamic = rng.normal(size=(n, nodes, 7)).astype(np.float32)
    edge_count = max(nodes - 1, 0)
    edge_index = np.asarray(
        [np.arange(1, nodes), np.zeros(edge_count, dtype=np.int64)], dtype=np.int64
    )
    distance = np.abs(np.arange(nodes)[:, None] - np.arange(nodes)[None, :]).astype(np.float32)
    return SimpleNamespace(
        grid_name=client.grid_name,
        num_nodes=nodes,
        node_ids=np.arange(nodes),
        load_bus_mask=client.load_bus_mask,
        dynamic_features=dynamic,
        p=dynamic[:, :, 0].copy(),
        timestamps=pd.date_range("2020-01-01", periods=n, freq="h"),
        edge_index=edge_index,
        distance_matrices={"hop_distance": distance, "impedance_abs_distance": distance},
        splits={
            "train": SimpleNamespace(start_index=0, end_index=6132),
            "validation": SimpleNamespace(start_index=6132, end_index=6250),
            "test": SimpleNamespace(start_index=6250, end_index=n),
        },
    )


def _fmt(value: Any) -> str:
    return f"{float(value):.6g}"


def render_markdown(report: dict[str, Any]) -> str:
    names = report["client_grid_names"]
    lines = [
        f"# Formal BTD-FL Ablation: {report['ablation']}",
        "",
        f"Mechanism changed: {report['mechanism_changed']}.",
        "The accepted main BTD-FL JSON is a frozen reference only; no baseline is retrained.",
        "Canonical validation is evaluation-only. Canonical test is never evaluated.",
        "",
        "## Per-target audit node/grid metrics",
        "",
        "| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in names:
        item = report["scenarios"][target]
        lines.append(
            f"| {target} | {_fmt(item['metrics']['audit']['node_macro']['mae'])} | "
            f"{_fmt(item['main_btd_fl_metrics']['audit']['node_macro']['mae'])} | "
            f"{_fmt(item['scarce_local_metrics']['audit']['node_macro']['mae'])} | "
            f"{_fmt(item['metrics']['audit']['grid_aggregate']['mae'])} |"
        )
    lines += [
        "",
        "## Per-target canonical-validation node/grid metrics",
        "",
        "| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in names:
        item = report["scenarios"][target]
        lines.append(
            f"| {target} | {_fmt(item['metrics']['validation']['node_macro']['mae'])} | "
            f"{_fmt(item['main_btd_fl_metrics']['validation']['node_macro']['mae'])} | "
            f"{_fmt(item['scarce_local_metrics']['validation']['node_macro']['mae'])} | "
            f"{_fmt(item['metrics']['validation']['grid_aggregate']['mae'])} |"
        )
    lines += [
        "",
        "## Donors and relative change",
        "",
        "| Target | Donor | Rule | Audit vs BTD-FL | Validation vs BTD-FL | Audit vs local | Validation vs local |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for target in names:
        item = report["scenarios"][target]
        lines.append(
            f"| {target} | {item['donor_used']} | {item['selection_rule']} | "
            f"{_fmt(item['relative_change_vs_main_btd_fl']['audit'])} | "
            f"{_fmt(item['relative_change_vs_main_btd_fl']['validation'])} | "
            f"{_fmt(item['relative_change_vs_scarce_local']['audit'])} | "
            f"{_fmt(item['relative_change_vs_scarce_local']['validation'])} |"
        )
    lines += [
        "",
        "## Four-target macro",
        "",
        "| Split | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("audit", "validation"):
        lines.append(
            f"| {split} | {_fmt(report['four_target_unweighted_macro'][split]['node_macro']['mae'])} | "
            f"{_fmt(report['reference_macro']['main_btd_fl'][split]['node_macro']['mae'])} | "
            f"{_fmt(report['reference_macro']['scarce_local'][split]['node_macro']['mae'])} | "
            f"{_fmt(report['four_target_unweighted_macro'][split]['grid_aggregate']['mae'])} |"
        )
    lines += [
        "",
        f"Win counts: versus main BTD-FL = {report['win_counts']['main_btd_fl']}; "
        f"versus scarce-local = {report['win_counts']['scarce_local']}.",
        "",
        "## Guardrails",
        "",
        "validation_used_for_selection: false",
        "audit_used_for_selection: false",
        "test_evaluated: false",
    ]
    return "\n".join(lines)


def render_combined_markdown(reports: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Formal BTD-FL Mechanism Ablations: 25% History",
        "",
        "The accepted main BTD-FL JSON is the frozen reference. Invariant FedAvg, FedProx, FedPer, and scarce-local results are not retrained.",
        "",
        "| Ablation | Audit node MAE | Validation node MAE | Wins vs main BTD-FL | Wins vs scarce-local |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in sorted(reports.items()):
        lines.append(
            f"| {name} | {_fmt(report['four_target_unweighted_macro']['audit']['node_macro']['mae'])} | "
            f"{_fmt(report['four_target_unweighted_macro']['validation']['node_macro']['mae'])} | "
            f"{report['win_counts']['main_btd_fl']} | {report['win_counts']['scarce_local']} |"
        )
    lines += ["", "test_evaluated: false", "validation_used_for_selection: false", "audit_used_for_selection: false"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ABLATIONS, required=True)
    parser.add_argument("--reference-json", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_25pct_seed42.json")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "ablations")
    args = parser.parse_args()
    reference = load_frozen_reference(args.reference_json)
    if args.synthetic_smoke:
        grids = {client.grid_name: _formal_grid_from_client(client) for client in build_synthetic_clients(42)}
    else:
        grids = {name: LVGridLoader(args.data_root, args.mapping_json).load(name) for name in reference["client_grid_names"]}
    report = run_ablation(
        grids, reference, args.variant, device=args.device,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_25pct_seed42"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")
    combined = {args.variant: report}
    for other in ABLATIONS:
        other_path = args.output_dir / f"{other}_25pct_seed42.json"
        if other_path.exists():
            combined[other] = json.loads(other_path.read_text(encoding="utf-8"))
    (args.output_dir / "btd_mechanism_ablations_25pct_seed42.md").write_text(
        render_combined_markdown(combined) + "\n", encoding="utf-8"
    )
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
