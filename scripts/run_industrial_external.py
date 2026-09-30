"""Run the frozen five-client industrial external-domain benchmark.

Real execution is intentionally separate from Stage-1 audit and is not
invoked by repository tests.  ``--synthetic-smoke`` uses tiny synthetic grids
and writes only to an isolated smoke directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.federated.industrial_external import (
    BASELINE_METHODS,
    INDUSTRIAL_TARGET,
    load_external_grids,
    run_external_baselines,
    run_external_btd,
    run_external_fedfomo,
)
from code.federated.industrial_result_validation import resumable_external_result
from scripts.run_federated import build_synthetic_clients


def _synthetic_external(seed: int) -> dict[str, Any]:
    clients = build_synthetic_clients(seed)
    from scripts.run_fedfomo_style import _formal_grid_from_client
    grids = {_client.grid_name: _formal_grid_from_client(_client) for _client in clients}
    # The fifth client is deliberately a distinct synthetic target with the
    # same interface; no production data or result path is touched.
    target = _formal_grid_from_client(clients[-1])
    target.grid_name = INDUSTRIAL_TARGET
    grids[INDUSTRIAL_TARGET] = target
    return grids


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Industrial External Benchmark: {int(report['history_fraction'] * 100)}%",
        "",
        "Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.",
        "",
        "## Industrial target metrics",
        "",
        "| Method | Audit node MAE | Validation node MAE |",
        "|---|---:|---:|",
    ]
    for method, metrics in report.get("industrial_target_macro", {}).items():
        lines.append(
            f"| {method} | {metrics['audit']['node_macro']['mae']:.6g} | {metrics['validation']['node_macro']['mae']:.6g} |"
        )
    lines.extend(["", "Guardrails:", "", "- industrial_test_locked = true", "- industrial_test_evaluated = false", "- reference_grid_tests_evaluated = false"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-fraction", type=float, choices=(0.25, 0.50), default=0.25)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "industrial_external")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--method", choices=("baselines", "btd", "fedfomo", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        output_dir = args.output_dir / "smoke"
        grids = _synthetic_external(args.seed)
        batch_size = max(args.batch_size, 256)
        max_epochs = min(args.max_epochs, 1)
        rounds = min(args.rounds, 1)
        local_epochs = min(args.local_epochs, 1)
        run_mode = "synthetic_smoke"
    else:
        output_dir = args.output_dir
        grids = load_external_grids(args.data_root, args.mapping_json)
        batch_size = args.batch_size
        max_epochs = args.max_epochs
        rounds = args.rounds
        local_epochs = args.local_epochs
        run_mode = "real"
    from code.federated.seeding import set_global_seed
    set_global_seed(args.seed)
    percent = int(args.history_fraction * 100)
    reports = []
    def output_path(stem: str) -> Path:
        return output_dir / f"{stem}_{percent}pct_seed{args.seed}{'_synthetic_smoke' if args.synthetic_smoke else ''}.json"

    def should_run(stem: str, artifact_type: str, expected_rounds: int, expected_local_epochs: int) -> bool:
        path = output_path(stem)
        if not path.exists() or args.force:
            return True
        if resumable_external_result(path, artifact_type=artifact_type, seed=args.seed,
                                     fraction=args.history_fraction, run_mode=run_mode,
                                     rounds=expected_rounds, local_epochs=expected_local_epochs,
                                     max_epochs=max_epochs, batch_size=batch_size):
            print(f"resume: {path}")
            return False
        raise ValueError(f"existing result is incomplete or incompatible; use --force to rerun: {path}")

    if args.method in ("baselines", "all"):
        if should_run("industrial_baselines", "industrial_baselines", rounds, local_epochs):
            reports.append(("industrial_baselines", run_external_baselines(grids, args.history_fraction, rounds, local_epochs, max_epochs, batch_size, args.device, args.seed)))
    if args.method in ("btd", "all"):
        if should_run("industrial_btd_direct_transfer", "industrial_btd_direct_transfer", 0, 0):
            reports.append(("industrial_btd_direct_transfer", run_external_btd(grids, args.history_fraction, max_epochs, batch_size, args.device, args.seed)))
    if args.method in ("fedfomo", "all"):
        if should_run("industrial_fedfomo_style", "industrial_fedfomo_style", rounds, local_epochs):
            reports.append(("industrial_fedfomo_style", run_external_fedfomo(grids, args.history_fraction, rounds, local_epochs, batch_size, args.device, args.seed)))
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, report in reports:
        report["run_mode"] = run_mode
        report["seed"] = args.seed
        report["history_fraction"] = args.history_fraction
        report["industrial_test_evaluated"] = False
        report["reference_grid_tests_evaluated"] = False
        report["test_evaluated"] = False
        path = output_dir / f"{stem}_{percent}pct_seed{args.seed}{'_synthetic_smoke' if args.synthetic_smoke else ''}.json"
        md = path.with_suffix(".md")
        path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8")
        md.write_text(_markdown(report) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
