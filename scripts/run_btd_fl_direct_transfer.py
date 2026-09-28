"""Run the self-contained direct-temporal-transfer BTD-FL candidate."""
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
from code.federated.direct_transfer_btd import run_direct_transfer  # noqa: E402
from code.federated.formal_btd_ablations import load_frozen_reference  # noqa: E402
from code.federated.formal_btd import CLIENT_NAMES  # noqa: E402
from code.federated.result_validation import resumable_result  # noqa: E402
from scripts.run_federated import build_synthetic_clients  # noqa: E402
from code.federated.seeding import set_global_seed  # noqa: E402


def _formal_grid_from_client(client: Any) -> Any:
    from types import SimpleNamespace

    n, nodes = 6400, client.model.num_nodes
    rng = np.random.default_rng(700 + nodes)
    dynamic = rng.normal(size=(n, nodes, 7)).astype(np.float32)
    edge_count = max(nodes - 1, 0)
    edges = np.asarray([np.arange(1, nodes), np.zeros(edge_count, dtype=np.int64)], dtype=np.int64)
    distance = np.abs(np.arange(nodes)[:, None] - np.arange(nodes)[None, :]).astype(np.float32)
    return SimpleNamespace(
        grid_name=client.grid_name, num_nodes=nodes, node_ids=np.arange(nodes),
        load_bus_mask=client.load_bus_mask, dynamic_features=dynamic,
        p=dynamic[:, :, 0].copy(), timestamps=pd.date_range("2020-01-01", periods=n, freq="h"),
        edge_index=edges,
        distance_matrices={"hop_distance": distance, "impedance_abs_distance": distance},
        splits={
            "train": SimpleNamespace(start_index=0, end_index=6132),
            "validation": SimpleNamespace(start_index=6132, end_index=6250),
            "test": SimpleNamespace(start_index=6250, end_index=n),
        },
    )


def render_markdown(report: dict[str, Any]) -> str:
    names = report["client_grid_names"]
    lines = [
        "# BTD-FL Direct Temporal Transfer: 25% History",
        "",
        "Target-conditioned donor adaptations are calibration-only benefit probes. The final transferred initialization is the raw selected donor temporal state.",
        "Canonical validation is evaluation-only. Canonical test is never evaluated.",
        "",
        "## Selected donors and benefits",
        "",
        "| Target | Selected donor | Benefit | Fallback | Matches frozen formal run |",
        "|---|---|---:|---|---|",
    ]
    for target in names:
        meta = report["scenarios"][target]["metadata"]
        lines.append(f"| {target} | {meta['selected_donor'] or 'none'} | {meta['selected_calibration_benefit']:.6g} | {meta['zero_transfer_fallback']} | {meta['selection_matches_previous_formal_run']} |")
    lines += ["", "## Current-run calibration benefit matrix", "", "| Target \\ Donor | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for target in names:
        benefits = report["scenarios"][target]["metadata"]["calibration_benefit_by_donor"]
        lines.append("| " + target + " | " + " | ".join("-" if donor == target else f"{benefits[donor]:.6g}" for donor in names) + " |")
    for split in ("audit", "validation"):
        label = "Train audit" if split == "audit" else "Canonical validation"
        lines += ["", f"## {label} metrics", "", "| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for target in names:
            metric = report["scenarios"][target]["metrics"][split]
            vals = [metric[scope][key] for scope in ("node_macro", "grid_aggregate") for key in ("mae", "rmse", "wape_pct", "smape_pct")]
            lines.append("| " + target + " | " + " | ".join(f"{value:.6g}" for value in vals) + " |")
        macro = report["four_target_unweighted_macro"][split]
        vals = [macro[scope][key] for scope in ("node_macro", "grid_aggregate") for key in ("mae", "rmse", "wape_pct", "smape_pct")]
        lines.append("| Four-target macro | " + " | ".join(f"{value:.6g}" for value in vals) + " |")
    lines += ["", "## Relative improvement vs frozen comparators", "", "| Comparator | Audit | Validation | Validation wins |", "|---|---:|---:|---:|"]
    for comparator, values in report.get("relative_improvement_vs_frozen_comparators", {}).items():
        lines.append(f"| {comparator} | {values['audit']:.6g} | {values['validation']:.6g} | {report.get('win_counts_vs_frozen_comparators', {}).get(comparator, 0)} |")
    lines += ["", "## Guardrails", "", "validation_used_for_selection: false", "audit_used_for_selection: false", "test_evaluated: false", "adapted_probe_states_used_for_final_initialization: false"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_25pct_seed42.json")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "formal_25pct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_mode = "synthetic_smoke" if args.synthetic_smoke else "real"
    output_dir = args.output_dir / "smoke" if args.synthetic_smoke else args.output_dir
    stem = f"btd_fl_direct_transfer_25pct_seed{args.seed}" + ("_synthetic_smoke" if args.synthetic_smoke else "")
    json_path, md_path = output_dir / f"{stem}.json", output_dir / f"{stem}.md"
    if args.seed == 42 and run_mode == "real" and json_path.exists():
        print(f"historical seed-42 result is frozen: {json_path}"); return
    if json_path.exists():
        if resumable_result(json_path, seed=args.seed, artifact_type="btd_fl_direct_transfer", run_mode=run_mode, methods=("btd_fl_direct_transfer",), batch_size=256 if args.synthetic_smoke else 32, learning_rate=1e-3, max_epochs=args.max_epochs, patience=8):
            if args.seed == 42 or not args.force:
                print(f"resume: keeping frozen/completed result {json_path}"); return
        elif not args.force:
            raise ValueError(f"existing result is incomplete or incompatible; use --force to rerun: {json_path}")
    frozen = load_frozen_reference(args.reference_json)
    if args.synthetic_smoke:
        grids = {client.grid_name: _formal_grid_from_client(client) for client in build_synthetic_clients(args.seed)}
        batch_size = 256
    else:
        grids = {name: LVGridLoader(args.data_root, args.mapping_json).load(name) for name in CLIENT_NAMES}
        batch_size = args.batch_size
    set_global_seed(args.seed)
    report = run_direct_transfer(grids, frozen, args.device, args.max_epochs, batch_size, seed=args.seed)
    report["run_mode"] = run_mode
    report["artifact_type"] = "btd_fl_direct_transfer"
    report["methods"] = ["btd_fl_direct_transfer"]
    if args.seed != 42:
        report.pop("relative_improvement_vs_frozen_comparators", None)
        report.pop("win_counts_vs_frozen_comparators", None)
        for item in report["scenarios"].values():
            item.pop("frozen_comparator_metrics", None)
    output_dir.mkdir(parents=True, exist_ok=True)
    if json_path.resolve() == args.reference_json.resolve() or md_path.resolve() == args.reference_json.with_suffix(".md").resolve():
        raise ValueError("direct-transfer output must not overwrite frozen main result")
    json_path.write_text(json.dumps(report, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
