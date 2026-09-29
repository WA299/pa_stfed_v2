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
from code.federated.result_validation import load_valid_result, resumable_result  # noqa: E402
from scripts.run_federated import build_synthetic_clients  # noqa: E402
from code.federated.seeding import set_global_seed  # noqa: E402


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
    comparison_key = "relative_improvement_vs_same_seed_comparators" if "relative_improvement_vs_same_seed_comparators" in report else "relative_improvement_vs_frozen_comparators"
    wins_key = "win_counts_vs_same_seed_comparators" if "win_counts_vs_same_seed_comparators" in report else "win_counts_vs_frozen_comparators"
    for name, values in report.get(comparison_key, {}).items():
        lines.append(f"| {name} | {values['audit']:.6g} | {values['validation']:.6g} | {report.get(wins_key, {}).get(name, 0)} |")
    if "relative_improvement_vs_final_btd_fl_direct_transfer" in report:
        values = report["relative_improvement_vs_final_btd_fl_direct_transfer"]
        lines.append(f"| BTD-FL Direct Temporal Transfer (final) | {values['audit']:.6g} | {values['validation']:.6g} | {report['win_counts_vs_final_btd_fl_direct_transfer']['validation']} |")
    lines += ["", "## Scenario metrics", "", "| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |", "|---|---:|---:|---:|---:|"]
    for target in report["client_grid_names"]:
        item = report["scenarios"][target]["scarce_target_metrics"]
        lines.append(f"| {target} | {item['audit']['node_macro']['mae']:.6g} | {item['audit']['grid_aggregate']['mae']:.6g} | {item['validation']['node_macro']['mae']:.6g} | {item['validation']['grid_aggregate']['mae']:.6g} |")
    lines += ["", "## Scarce-target round peer weights", "", "| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |", "|---:|---:|---:|---|---|"]
    target = report["client_grid_names"][0]
    for round_item in report["scenarios"][target]["round_history"]:
        diag = round_item["diagnostics"][target]
        lines.append(f"| {round_item['round']} | {diag['baseline_self_calibration_loss']:.6g} | {diag['positive_candidate_count']} | {diag['no_positive_candidate_fallback']} | {json.dumps(diag['normalized_weight_by_client'], sort_keys=True)} |")
    lines += ["", "validation_used_for_peer_weighting: false", "audit_used_for_peer_weighting: false", "test_evaluated: false", 'communication_round_selection: "fixed_final_round"']
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_25pct_seed42.json")
    parser.add_argument("--direct-transfer-json", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_direct_transfer_25pct_seed42.json")
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "formal_25pct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-fraction", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_mode = "synthetic_smoke" if args.synthetic_smoke else "real"
    output_dir = args.output_dir / "smoke" if args.synthetic_smoke else args.output_dir
    percent = int(round(args.history_fraction * 100))
    if percent not in (25, 50):
        raise ValueError("history_fraction must be 0.25 or 0.50")
    stem = f"fedfomo_style_{percent}pct_seed{args.seed}" + ("_synthetic_smoke" if args.synthetic_smoke else "")
    json_path, md_path = output_dir / f"{stem}.json", output_dir / f"{stem}.md"
    if args.seed == 42 and run_mode == "real" and json_path.exists():
        print(f"historical seed-42 result is frozen: {json_path}"); return
    if json_path.exists():
        if resumable_result(json_path, seed=args.seed, artifact_type="fedfomo_style", run_mode=run_mode, methods=("fedfomo_style",), rounds=args.rounds, local_epochs=args.local_epochs, batch_size=256 if args.synthetic_smoke else 32, learning_rate=1e-3, max_epochs=50, patience=8, history_fraction=args.history_fraction):
            if args.seed == 42 or not args.force:
                print(f"resume: keeping frozen/completed result {json_path}"); return
        elif not args.force:
            raise ValueError(f"existing result is incomplete or incompatible; use --force to rerun: {json_path}")
    reference = load_frozen_reference(args.reference_json)
    direct = json.loads(args.direct_transfer_json.read_text(encoding="utf-8"))
    baseline_reference = None
    if args.seed != 42:
        if args.baseline_json is None:
            raise ValueError("new-seed FedFomo requires a same-seed baseline reference")
        baseline_reference = load_valid_result(
            args.baseline_json, seed=args.seed, artifact_type="formal_baselines",
            run_mode=run_mode, methods=("scarce_local", "fedavg", "fedprox", "fedper"),
            rounds=args.rounds, local_epochs=args.local_epochs,
            batch_size=256 if args.synthetic_smoke else args.batch_size,
            learning_rate=1e-3, max_epochs=1 if args.synthetic_smoke else 50, patience=8, history_fraction=args.history_fraction,
        )
        direct = load_valid_result(
            args.direct_transfer_json, seed=args.seed, artifact_type="btd_fl_direct_transfer",
            run_mode=run_mode, methods=("btd_fl_direct_transfer",),
            batch_size=256 if args.synthetic_smoke else args.batch_size,
            learning_rate=1e-3, max_epochs=1 if args.synthetic_smoke else 50, patience=8, history_fraction=args.history_fraction,
        )
        if direct.get("btd_variant") != "btd_fl_direct_transfer":
            raise ValueError("same-seed direct-transfer reference has the wrong variant")
    elif direct.get("test_evaluated") is not False or direct.get("btd_variant") != "btd_fl_direct_transfer":
        raise ValueError("direct-transfer frozen result violates guardrails")
    if args.synthetic_smoke:
        grids = {client.grid_name: _formal_grid_from_client(client) for client in build_synthetic_clients(args.seed)}
        batch_size = 256
    else:
        grids = {name: LVGridLoader(args.data_root, args.mapping_json).load(name) for name in CLIENT_NAMES}
        batch_size = args.batch_size
    set_global_seed(args.seed)
    if args.seed == 42 and args.baseline_json is not None:
        baseline_reference = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    report = run_fedfomo(grids, reference, args.rounds, args.local_epochs, batch_size, args.device, direct_transfer_reference=direct, baseline_reference=baseline_reference, seed=args.seed, history_fraction=args.history_fraction)
    report["frozen_direct_transfer_metrics"] = {target: direct["scenarios"][target]["metrics"] for target in CLIENT_NAMES}
    report["run_mode"] = run_mode
    report["artifact_type"] = "fedfomo_style"
    report["methods"] = ["fedfomo_style"]
    if args.seed != 42 and baseline_reference is not None:
        report["relative_improvement_vs_same_seed_comparators"] = report.pop("relative_improvement_vs_frozen_comparators", {})
        report["win_counts_vs_same_seed_comparators"] = report.pop("win_counts_vs_frozen_comparators", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
