"""Run the FedFomo-style frozen-protocol comparator."""
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
from code.federated.fedfomo_style import run_fedfomo  # noqa: E402
from code.federated.formal_btd import CLIENT_NAMES  # noqa: E402
from code.federated.formal_btd_ablations import load_frozen_reference  # noqa: E402
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
        grid_name=client.grid_name, num_nodes=nodes, node_ids=np.arange(nodes),
        load_bus_mask=client.load_bus_mask, dynamic_features=dynamic, p=dynamic[:, :, 0].copy(),
        timestamps=pd.date_range("2020-01-01", periods=n, freq="h"), edge_index=edge_index,
        distance_matrices={"hop_distance": distance, "impedance_abs_distance": distance},
        splits={"train": SimpleNamespace(start_index=0, end_index=6132), "validation": SimpleNamespace(start_index=6132, end_index=6250), "test": SimpleNamespace(start_index=6250, end_index=n)},
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FedFomo-style Heterogeneous-Grid Comparator: 25% History",
        "",
        report["heterogeneous_grid_adaptation"],
        "No claim of exact paper FedFomo is made. Validation is evaluation-only; test is never evaluated.",
        "",
        "## Four-scenario macro node-MAE",
        "",
        "| Method | Audit node MAE | Validation node MAE |",
        "|---|---:|---:|",
        f"| FedFomo-style | {report['four_scenario_unweighted_macro']['audit']['node_macro']['mae']:.6g} | {report['four_scenario_unweighted_macro']['validation']['node_macro']['mae']:.6g} |",
    ]
    lines += ["", "## Frozen comparator improvements", "", "| Comparator | Audit relative improvement | Validation relative improvement | Validation wins |", "|---|---:|---:|---:|"]
    for name, values in report["relative_improvement_vs_frozen_comparators"].items():
        lines.append(f"| {name} | {values['audit']:.6g} | {values['validation']:.6g} | {report['win_counts_vs_frozen_comparators'][name]} |")
    lines += ["", "## Scenario metrics", "", "| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |", "|---|---:|---:|---:|---:|"]
    for target in report["client_grid_names"]:
        item = report["scenarios"][target]
        lines.append(f"| {target} | {item['audit']['node_macro']['mae']:.6g} | {item['audit']['grid_aggregate']['mae']:.6g} | {item['validation']['node_macro']['mae']:.6g} | {item['validation']['grid_aggregate']['mae']:.6g} |")
    lines += ["", "## Scarce-target round peer weights", "", "| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |", "|---:|---:|---:|---|---|"]
    for round_item in report["round_history"]:
        for target in report["client_grid_names"]:
            diag = round_item["diagnostics"][target]
            if diag.get("is_scarce_target"):
                lines.append(f"| {round_item['round']} | {diag['baseline_self_calibration_loss']:.6g} | {diag['positive_peer_count']} | {diag['no_positive_peer_fallback']} | {json.dumps(diag['normalized_weight_by_donor'], sort_keys=True)} |")
    lines += ["", "validation_used_for_peer_weighting: false", "audit_used_for_peer_weighting: false", "test_evaluated: false", 'communication_round_selection: "fixed_final_round"']
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_25pct_seed42.json")
    parser.add_argument("--direct-transfer-json", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_direct_transfer_25pct_seed42.json")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--scarce-target", choices=CLIENT_NAMES, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "formal_25pct")
    args = parser.parse_args()
    reference = load_frozen_reference(args.reference_json)
    direct = json.loads(args.direct_transfer_json.read_text(encoding="utf-8"))
    if direct.get("test_evaluated") is not False or direct.get("btd_variant") != "btd_fl_direct_transfer":
        raise ValueError("direct-transfer frozen result violates guardrails")
    if args.synthetic_smoke:
        grids = {client.grid_name: _formal_grid_from_client(client) for client in build_synthetic_clients(42)}
        batch_size = 256
    else:
        grids = {name: LVGridLoader(args.data_root, args.mapping_json).load(name) for name in CLIENT_NAMES}
        batch_size = args.batch_size
    report = run_fedfomo(grids, reference, args.scarce_target, args.rounds, args.local_epochs, batch_size, args.device)
    # Direct-transfer metrics are added as a frozen comparator without training.
    report["frozen_direct_transfer_metrics"] = {
        target: direct["scenarios"][target]["metrics"] for target in CLIENT_NAMES
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fedfomo_style_{args.scarce_target}_25pct_seed42"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n", encoding="utf-8")
    (args.output_dir / f"{stem}.md").write_text(render_markdown(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
