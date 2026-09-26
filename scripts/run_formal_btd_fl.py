"""Run the formal 25%-history BTD-FL benchmark."""
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

from code.federated.formal_btd import (  # noqa: E402
    CLIENT_NAMES,
    METHODS,
    load_selected_donors,
    load_selection_metadata,
    run_formal_benchmark,
)
from code.data.lv_grid_loader import LVGridLoader  # noqa: E402
from scripts.run_btd_full_backbone_bridge import _grid_from_client  # noqa: E402
from scripts.run_federated import build_synthetic_clients  # noqa: E402


def _formal_grid_from_client(client: Any) -> Any:
    from types import SimpleNamespace
    n, nodes = 6400, client.model.num_nodes
    rng = np.random.default_rng(700 + nodes)
    dynamic = rng.normal(size=(n, nodes, 7)).astype(np.float32)
    edge_count = max(nodes - 1, 0)
    edge_index = np.asarray([np.arange(1, nodes), np.zeros(edge_count, dtype=np.int64)], dtype=np.int64)
    distance = np.abs(np.arange(nodes)[:, None] - np.arange(nodes)[None, :]).astype(np.float32)
    return SimpleNamespace(
        grid_name=client.grid_name, num_nodes=nodes, node_ids=np.arange(nodes), load_bus_mask=client.load_bus_mask,
        dynamic_features=dynamic, p=dynamic[:, :, 0].copy(), timestamps=pd.date_range("2020-01-01", periods=n, freq="h"), edge_index=edge_index,
        distance_matrices={"hop_distance": distance, "impedance_abs_distance": distance},
        splits={"train": SimpleNamespace(start_index=0, end_index=6132), "validation": SimpleNamespace(start_index=6132, end_index=6250), "test": SimpleNamespace(start_index=6250, end_index=n)},
    )


def render_markdown(report: dict[str, Any]) -> str:
    names = report["client_grid_names"]
    lines = [
        "# Formal BTD-FL Benchmark: 25% History",
        "",
        "BTD-FL: Benefit-Triggered Topology-Decoupled Federated Learning.",
        "Canonical validation is evaluation-only; canonical test is never evaluated.",
        "Donor selection uses train-internal calibration only. No causal claim is made.",
        "",
        "## Scarce-Target Validation Node-MAE",
        "",
        "| Scarce target | scarce_local | FedAvg | FedProx | FedPer | BTD-FL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        methods = report["scenarios"][name]["methods"]
        lines.append("| " + name + " | " + " | ".join(f"{methods[m]['validation']['node_macro']['mae']:.6g}" for m in METHODS) + " |")
    lines += ["", "## Four-Scenario Macro Node-MAE", "", "| Split | scarce_local | FedAvg | FedProx | FedPer | BTD-FL |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("audit", "validation"):
        lines.append("| " + split + " | " + " | ".join(f"{report['four_scenario_macro'][split][m]['node_macro']['mae']:.6g}" for m in METHODS) + " |")
    lines += ["", "## BTD-FL Relative Improvements vs Comparators", "", "| Comparator | Audit macro improvement | Validation macro improvement | Validation wins |", "|---|---:|---:|---:|"]
    for method in ("scarce_local", "fedavg", "fedprox", "fedper"):
        audit_local = report["four_scenario_macro"]["audit"][method]["node_macro"]["mae"]; audit_btd = report["four_scenario_macro"]["audit"]["btd_fl"]["node_macro"]["mae"]
        val_local = report["four_scenario_macro"]["validation"][method]["node_macro"]["mae"]; val_btd = report["four_scenario_macro"]["validation"]["btd_fl"]["node_macro"]["mae"]
        lines.append(f"| {method} | {(audit_local-audit_btd)/(audit_local+1e-12):.6g} | {(val_local-val_btd)/(val_local+1e-12):.6g} | {report['btd_fl_win_counts'][method]} |")
    lines += ["", "## Selected Donors and Graph Metadata", "", "| Target | Candidate donors | Selected donor | Graph fit | Graph selection | Selected edge count |", "|---|---|---|---:|---:|---:|"]
    for name in names:
        item = report["scenarios"][name]["metadata"]; graph = item["target_graph_metadata"]
        lines.append(f"| {name} | {', '.join(item['candidate_donors'])} | {item['selected_donor'] or 'none'} | {graph['graph_fit_target_count']} | {graph['graph_selection_target_count']} | {graph['selected_edge_count']} |")
    lines += ["", "## Guardrails", "", "validation_used_for_selection: false", "audit_used_for_selection: false", "test_evaluated: false"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--audit-json", type=Path, default=ROOT / "results" / "audits" / "federated_transfer_benefit_25pct.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--btd-variant", choices=("btd_fl", "btd_no_benefit_selection", "btd_no_zero_transfer", "btd_full_model_transfer"), default="btd_fl")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "formal_25pct")
    args = parser.parse_args()
    grids = ({client.grid_name: _formal_grid_from_client(client) for client in build_synthetic_clients(42)}
             if args.synthetic_smoke else {name: LVGridLoader(args.data_root, args.mapping_json).load(name) for name in CLIENT_NAMES})
    donors = load_selected_donors(args.audit_json)
    selection_metadata = load_selection_metadata(args.audit_json)
    smoke_batch_size = 256 if args.synthetic_smoke else 32
    report = run_formal_benchmark(grids, donors, args.device, args.rounds, args.local_epochs, args.max_epochs, args.btd_variant, smoke_batch_size, selection_metadata)
    output = args.output_dir / "btd_fl_25pct_seed42.json"; markdown = args.output_dir / "btd_fl_25pct_seed42.md"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {output} and {markdown}")


if __name__ == "__main__":
    main()
