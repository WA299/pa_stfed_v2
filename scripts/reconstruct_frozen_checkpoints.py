"""Plan or (only when explicitly requested) export frozen pre-TEST checkpoints.

The default command is plan-only.  It never loads TEST targets and never
starts a real reconstruction.  Real execution is intentionally guarded by a
separate explicit flag for the future protocol lock.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from code.federated.accepted_results import resolve_accepted_result, resolve_all_accepted_results
from code.federated.checkpointing import checkpoint_path, matrix_cells
DEFAULT_OUTPUT = ROOT / "artifacts" / "frozen_pretest_checkpoints"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_reconstruction_plan(output: Path = DEFAULT_OUTPUT) -> dict:
    cells = matrix_cells()
    return {
        "schema_version": 1,
        "output_root": str(output),
        "cell_count": len(cells),
        "cells": cells,
        "source_data_splits": {"reference": "canonical TRAIN only", "industrial": "FIT/CALIBRATION only"},
        "test_accessed": False,
        "reconstruction_executed": False,
        "checkpoint_selection": {
            "local": "best calibration-selected full target model",
            "btd_fl_direct_transfer": "final best full target model after frozen raw temporal transfer/fallback",
            "fedavg": "target model after fixed final round 10",
            "fedprox": "target model after fixed final round 10",
            "fedper": "target model after fixed final round 10",
            "fedfomo_style": "target model after fixed final round 10",
        },
        "parity_required_before_test_unlock": True,
        "historical_artifacts_modified": False,
        "reconstruction_authorization_required": True,
        "test_authorization_separate": True,
        "accepted_result_mapping_coverage": 180,
    }


def _assert_isolated_output(output: Path) -> None:
    resolved = output.resolve()
    required = DEFAULT_OUTPUT.resolve()
    if required not in resolved.parents and resolved != required:
        raise ValueError(f"reconstruction output must be isolated below {required}")
    if any(part in {"results", "paper", "data"} for part in resolved.parts[-3:]):
        raise ValueError("reconstruction output cannot be a historical result directory")


def build_parity_cell(cell: Mapping[str, Any], observed_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Build one explicit pre-TEST parity record from observed AUDIT/VALIDATION."""
    from code.federated.checkpointing import compare_metric_trees

    resolved = resolve_accepted_result(cell)
    document = json.loads(resolved["source_artifact"].read_text(encoding="utf-8"))
    expected: dict[str, Any] = {}
    for split in ("audit", "validation"):
        current: Any = document
        for part in resolved["source_json_key"][split].split("."):
            current = current[part]
        expected[split] = current
    discrepancies = compare_metric_trees(observed_metrics, expected)
    comparisons = []
    for split in ("audit", "validation"):
        for scope in ("node_macro", "grid_aggregate"):
            for metric in ("mae", "rmse", "wape_pct", "smape_pct"):
                exp = float(expected[split][scope][metric]); obs = float(observed_metrics[split][scope][metric])
                absolute = abs(obs - exp); relative = absolute / max(abs(exp), 1e-12)
                key = f"{resolved['source_json_key'][split]}.{scope}.{metric}"
                comparisons.append({
                    "split": split, "scope": scope, "metric": metric,
                    "json_key_path": key,
                    "accepted_artifact": str(resolved["source_artifact"]),
                    "expected": exp, "observed": obs,
                    "absolute_discrepancy": absolute,
                    "relative_discrepancy": relative,
                    "parity_passed": math.isfinite(obs) and math.isfinite(exp) and absolute <= 1e-10 + 1e-6 * abs(exp),
                })
    return {
        **dict(cell), "test_accessed": False, "test_evaluated": False,
        "parity_passed": not discrepancies,
        "metric_comparisons": comparisons,
        "accepted_artifact": str(resolved["source_artifact"]),
        "accepted_json_keys": resolved["source_json_key"],
    }


def execute_reconstruction(
    output: Path = DEFAULT_OUTPUT, *, authorize_reconstruction: bool = False,
    authorize_test: bool = False, cell_runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an explicitly authorized pre-TEST reconstruction callback.

    The callback is supplied by the frozen data/model runner.  This layer
    enforces the cell matrix, output isolation, accepted-result mapping, and
    TEST lock; it never grants TEST authorization and never writes historical
    result directories.
    """
    if not authorize_reconstruction:
        raise PermissionError("reconstruction requires --authorize-reconstruction")
    if authorize_test:
        raise PermissionError("TEST authorization is a separate later protocol-lock event")
    if cell_runner is None:
        raise ValueError("an explicit frozen cell runner is required for reconstruction")
    _assert_isolated_output(output)
    mapping = resolve_all_accepted_results()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for cell in matrix_cells():
        observed = dict(cell_runner(cell, output))
        record = build_parity_cell(cell, observed)
        record["checkpoint_path"] = str(checkpoint_path(output, cell))
        records.append(record)
    parity = {"schema_version": 2, "test_accessed": False, "test_evaluated": False,
              "parity_passed": bool(len(records) == 180 and all(item["parity_passed"] for item in records)),
              "cells": records, "accepted_result_mapping_coverage": len(mapping),
              "reconstruction_executed": True, "historical_artifacts_modified": False}
    parity_path = output / "parity_report.json"
    _atomic_json(parity_path, parity)
    return parity


def _record_report_metrics(
    observed: dict[str, Mapping[str, Any]], report: Mapping[str, Any], *, domain: str,
    fraction: float, seed: int, family: str,
) -> None:
    """Collect only AUDIT/VALIDATION metrics emitted by frozen runners."""
    scenarios = report.get("scenarios", {})
    for target, scenario in scenarios.items():
        if family == "baseline":
            methods = scenario.get("methods", {})
            for public, canonical in (("local", "scarce_local" if domain == "reference_grid" else "industrial_scarce_local"), ("fedavg", "fedavg"), ("fedprox", "fedprox"), ("fedper", "fedper")):
                if canonical not in methods:
                    continue
                cell_id = f"{domain}|{target}|seed{seed}|{int(fraction * 100)}pct|{public}"
                observed[cell_id] = methods[canonical]
        elif family == "btd":
            cell_id = f"{domain}|{target}|seed{seed}|{int(fraction * 100)}pct|btd_fl_direct_transfer"
            observed[cell_id] = scenario.get("metrics", {})
        elif family == "fedfomo":
            cell_id = f"{domain}|{target}|seed{seed}|{int(fraction * 100)}pct|fedfomo_style"
            observed[cell_id] = scenario.get("scarce_target_metrics", {})


def run_frozen_reconstruction(
    output: Path = DEFAULT_OUTPUT, *, data_root: Path, mapping_json: Path,
    device: str = "cuda", authorize_reconstruction: bool = False,
) -> dict[str, Any]:
    """Reconstruct all 180 cells with the existing frozen runners.

    This is intentionally a separate, explicit pre-TEST operation.  It is
    never called by tests or by the default CLI plan path in this commit.
    Every runner receives the isolated checkpoint root and reports only
    AUDIT/VALIDATION metrics; no TEST indices are constructed here.
    """
    if not authorize_reconstruction:
        raise PermissionError("real reconstruction requires --authorize-reconstruction")
    _assert_isolated_output(output)
    # Resolve before loading data so missing/ambiguous accepted identities fail
    # without starting any training process.
    mapping = resolve_all_accepted_results()
    from code.data.lv_grid_loader import LVGridLoader
    from code.federated.direct_transfer_btd import run_direct_transfer
    from code.federated.formal_btd import CLIENT_NAMES, run_formal_baselines
    from code.federated.fedfomo_style import run_fedfomo_scenario
    from code.federated.industrial_external import (
        load_external_grids,
        run_external_baselines, run_external_btd, run_external_fedfomo,
    )
    from code.federated.checkpointing import FROZEN_COMMITS

    output.mkdir(parents=True, exist_ok=True)
    observed: dict[str, Mapping[str, Any]] = {}
    loader = LVGridLoader(data_root, mapping_json)
    for fraction in (0.25, 0.50):
        for seed in (42, 123, 2026):
            references = {name: loader.load(name) for name in CLIENT_NAMES}
            baseline = run_formal_baselines(
                references, device=device, rounds=10, local_epochs=5, max_epochs=50,
                batch_size=32, seed=seed, history_fraction=fraction,
                checkpoint_root=output, frozen_commits=FROZEN_COMMITS,
                source_artifact=str(mapping[next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "local")]["source_artifact"]),
            )
            _record_report_metrics(observed, baseline, domain="reference_grid", fraction=fraction, seed=seed, family="baseline")
            reference_source = mapping[next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "btd_fl_direct_transfer")]["source_artifact"]
            reference_fomo_source = mapping[next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "fedfomo_style")]["source_artifact"]
            direct = run_direct_transfer(
                references, reference_source if fraction == 0.25 and seed == 42 else None, device=device, max_epochs=50, batch_size=32,
                seed=seed, history_fraction=fraction, checkpoint_root=output,
                source_artifact=str(reference_source), frozen_commits=FROZEN_COMMITS,
            )
            _record_report_metrics(observed, direct, domain="reference_grid", fraction=fraction, seed=seed, family="btd")
            fomo_scenarios = {
                target: run_fedfomo_scenario(
                    references, target, 10, 5, 32, device, 1e-12, seed, fraction,
                    CLIENT_NAMES, checkpoint_root=output, checkpoint_domain="reference_grid",
                    source_artifact=str(reference_fomo_source), frozen_commits=FROZEN_COMMITS,
                ) for target in CLIENT_NAMES
            }
            fomo = {"scenarios": fomo_scenarios}
            _record_report_metrics(observed, fomo, domain="reference_grid", fraction=fraction, seed=seed, family="fedfomo")

            grids = load_external_grids(data_root, mapping_json)
            industrial_source = mapping[next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "local")]["source_artifact"]
            ext_baseline = run_external_baselines(
                grids, history_fraction=fraction, rounds=10, local_epochs=5, max_epochs=50,
                batch_size=32, device=device, seed=seed, checkpoint_root=output,
                source_artifact=str(industrial_source), frozen_commits=FROZEN_COMMITS,
            )
            _record_report_metrics(observed, ext_baseline, domain="industrial_external", fraction=fraction, seed=seed, family="baseline")
            ext_btd = run_external_btd(
                grids, history_fraction=fraction, max_epochs=50, batch_size=32,
                device=device, seed=seed, checkpoint_root=output,
                source_artifact=str(mapping[next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "btd_fl_direct_transfer")]["source_artifact"]),
                frozen_commits=FROZEN_COMMITS,
            )
            _record_report_metrics(observed, ext_btd, domain="industrial_external", fraction=fraction, seed=seed, family="btd")
            industrial_fomo_source = mapping[next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "fedfomo_style")]["source_artifact"]
            ext_fomo = run_external_fedfomo(
                grids, history_fraction=fraction, rounds=10, local_epochs=5, batch_size=32,
                device=device, seed=seed, max_epochs_metadata=50, checkpoint_root=output,
                checkpoint_domain="industrial_external", source_artifact=str(industrial_fomo_source),
                frozen_commits=FROZEN_COMMITS,
            )
            _record_report_metrics(observed, ext_fomo, domain="industrial_external", fraction=fraction, seed=seed, family="fedfomo")

    expected = {item["cell_id"] for item in matrix_cells()}
    if set(observed) != expected:
        raise RuntimeError(f"frozen runners did not produce all 180 metric cells: missing={sorted(expected - set(observed))}")
    records = [build_parity_cell(cell, observed[cell["cell_id"]]) for cell in matrix_cells()]
    parity = {"schema_version": 2, "test_accessed": False, "test_evaluated": False,
              "parity_passed": bool(all(item["parity_passed"] for item in records)),
              "cells": records, "accepted_result_mapping_coverage": len(mapping),
              "reconstruction_executed": True, "historical_artifacts_modified": False}
    _atomic_json(output / "parity_report.json", parity)
    return parity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true", default=True)
    parser.add_argument("--execute-reconstruction", action="store_true", help="Run pre-TEST reconstruction only with explicit authorization")
    parser.add_argument("--execute-real", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--authorize-reconstruction", action="store_true")
    parser.add_argument("--authorize-test", action="store_true", help="Never accepted by this pre-TEST command")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    _assert_isolated_output(args.output_root.resolve())
    if args.authorize_test:
        raise SystemExit("TEST authorization is unavailable during pre-TEST reconstruction")
    if args.execute_reconstruction or args.execute_real:
        report = run_frozen_reconstruction(
            args.output_root.resolve(), data_root=args.data_root.resolve(),
            mapping_json=args.mapping_json.resolve(), device=args.device,
            authorize_reconstruction=args.authorize_reconstruction,
        )
        print(json.dumps({"parity_report": str(args.output_root.resolve() / "parity_report.json"), "cells": len(report["cells"]), "test_accessed": False}, indent=2))
        return
    plan = build_reconstruction_plan(args.output_root.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "reconstruction_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(path), "cells": len(plan["cells"]), "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
