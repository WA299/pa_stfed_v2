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
from code.federated.accepted_results import (
    direct_transfer_reference_input,
    resolve_accepted_result,
    resolve_all_accepted_results,
)
from code.federated.checkpointing import checkpoint_path, load_checkpoint, matrix_cells, sha256_file, validate_matrix_identity
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
        "test_evaluated": False,
        "industrial_test_evaluated": False,
        "reference_grid_tests_evaluated": False,
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
        "direct_transfer_reference_contract": {
            "load_frozen_reference_input": "results/federated/formal_25pct/btd_fl_25pct_seed42.json",
            "final_checkpoint_parity_source": "results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json",
            "input_and_parity_sources_are_distinct": True,
        },
    }


def _assert_isolated_output(output: Path) -> None:
    resolved = output.resolve()
    required = DEFAULT_OUTPUT.resolve()
    if required not in resolved.parents and resolved != required:
        raise ValueError(f"reconstruction output must be isolated below {required}")
    if any(part in {"results", "paper", "data"} for part in resolved.parts[-3:]):
        raise ValueError("reconstruction output cannot be a historical result directory")


def build_parity_cell(
    cell: Mapping[str, Any], observed_metrics: Mapping[str, Any],
    checkpoint: Path | None = None,
) -> dict[str, Any]:
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
    checkpoint_path_value = str(Path(checkpoint).resolve()) if checkpoint is not None else None
    checkpoint_sha256: str | None = None
    checkpoint_error: str | None = None
    if checkpoint is None:
        checkpoint_error = "checkpoint path was not supplied"
    else:
        try:
            path = Path(checkpoint)
            # This verifies the exact sidecar hash and performs a safe
            # weights_only production load before parity is reported.
            _state, metadata = load_checkpoint(
                path,
                expected={
                    "test_accessed": False,
                    "test_evaluated": False,
                    "industrial_test_evaluated": False,
                    "reference_grid_tests_evaluated": False,
                },
                production=True,
            )
            validate_matrix_identity(metadata, cell)
            checkpoint_sha256 = sha256_file(path)
        except Exception as exc:
            checkpoint_error = str(exc)
    comparisons = []
    for split in ("audit", "validation"):
        for scope in ("node_macro", "grid_aggregate"):
            for metric in ("mae", "rmse", "wape_pct", "smape_pct"):
                exp = float(expected[split][scope][metric])
                try:
                    obs = float(observed_metrics[split][scope][metric])
                except (KeyError, TypeError, ValueError):
                    obs = float("nan")
                absolute = abs(obs - exp) if math.isfinite(obs) else float("nan")
                relative = absolute / max(abs(exp), 1e-12) if math.isfinite(absolute) else float("nan")
                key = f"{resolved['source_json_key'][split]}.{scope}.{metric}"
                comparisons.append({
                    "split": split, "scope": scope, "metric": metric,
                    "json_key_path": key,
                    "accepted_artifact": str(resolved["source_artifact"]),
                    "expected": exp, "observed": obs if math.isfinite(obs) else None,
                    "absolute_discrepancy": absolute if math.isfinite(absolute) else None,
                    "relative_discrepancy": relative if math.isfinite(relative) else None,
                    "parity_passed": math.isfinite(obs) and math.isfinite(exp) and math.isfinite(absolute) and absolute <= 1e-10 + 1e-6 * abs(exp),
                })
    return {
        **dict(cell), "test_accessed": False, "test_evaluated": False,
        "parity_passed": not discrepancies and checkpoint_error is None,
        "observed_metrics": dict(observed_metrics),
        "metric_comparisons": comparisons,
        "accepted_artifact": str(resolved["source_artifact"]),
        "accepted_json_keys": resolved["source_json_key"],
        "checkpoint_path": checkpoint_path_value,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_replay_error": checkpoint_error,
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
        record = build_parity_cell(cell, observed, checkpoint_path(output, cell))
        records.append(record)
    parity = {"schema_version": 2, "test_accessed": False, "test_evaluated": False,
              "industrial_test_evaluated": False, "reference_grid_tests_evaluated": False,
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


def replay_target_checkpoint_metrics(
    cell: Mapping[str, Any], grids: Mapping[str, Any], checkpoint_root: Path,
    *, device: str = "cpu", batch_size: int = 32,
) -> Mapping[str, Any]:
    """Reload one exported target checkpoint and run only AUDIT/VALIDATION.

    The graph and FIT-only scaler are rebuilt from the frozen permitted data;
    the serialized tensor file is then loaded strictly into the exact frozen
    model class before either metric is computed.  No TEST index is created.
    """
    from code.audits.btd_full_backbone_bridge import build_scarce_target_graph, full_model
    from code.audits.federated_transfer_benefit import fit_fit_only_scaler, scarce_split
    from code.federated.checkpointing import replay_checkpoint
    from code.federated.formal_btd import _method_metrics
    from code.federated.industrial_external import _target_metrics

    target = str(cell["target"])
    grid = grids[target]
    split = scarce_split(grid, float(cell["history_fraction"]))
    scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
    graph = build_scarce_target_graph(grid, split)
    model = full_model(grid, graph, device, seed=int(cell["seed"]))
    path = checkpoint_path(checkpoint_root, cell)
    replay_checkpoint(path, model, cell=cell, grid=grid, scaler=scaler)
    if cell["domain"] == "industrial_external":
        return _target_metrics(model, grid, split, scaler, device, batch_size)
    return _method_metrics(model, grid, split, scaler, device, batch_size)


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
    # Kept separate by role: this is the historical BTD input required by
    # load_frozen_reference, while final parity maps to the direct-transfer
    # artifact from accepted_results.py.
    replay_reference_grids: dict[str, Any] = {}
    replay_industrial_grids: dict[str, Any] = {}
    loader = LVGridLoader(data_root, mapping_json)
    for fraction in (0.25, 0.50):
        for seed in (42, 123, 2026):
            references = {name: loader.load(name) for name in CLIENT_NAMES}
            replay_reference_grids = references
            baseline = run_formal_baselines(
                references, device=device, rounds=10, local_epochs=5, max_epochs=50,
                batch_size=32, seed=seed, history_fraction=fraction,
                checkpoint_root=output, frozen_commits=FROZEN_COMMITS,
                source_artifact=str(mapping[next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "local")]["source_artifact"]),
            )
            _record_report_metrics(observed, baseline, domain="reference_grid", fraction=fraction, seed=seed, family="baseline")
            # ``reference`` is the historical formal BTD input consumed by
            # load_frozen_reference; ``source_artifact`` remains the exact
            # direct-transfer artifact used for final per-cell parity.
            reference_source = mapping[next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "btd_fl_direct_transfer")]["source_artifact"]
            btd_reference_input = direct_transfer_reference_input(history_fraction=fraction, seed=seed)
            if btd_reference_input is not None:
                # Fail before training if the historical input is missing or
                # no longer satisfies the frozen load_frozen_reference schema.
                from code.federated.formal_btd_ablations import load_frozen_reference
                load_frozen_reference(btd_reference_input)
            reference_fomo_source = mapping[next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "fedfomo_style")]["source_artifact"]
            direct = run_direct_transfer(
                references, btd_reference_input, device=device, max_epochs=50, batch_size=32,
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
            replay_industrial_grids = grids
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

    # Discard all in-memory runner metrics as parity evidence.  Reconstruct
    # the exact target graph/scaler from permitted data and infer from every
    # serialized checkpoint after a verified weights_only reload.
    industrial_grids = replay_industrial_grids
    if not industrial_grids:
        raise RuntimeError("industrial grids were not loaded for checkpoint replay")
    replay_grids = {"reference_grid": replay_reference_grids, "industrial_external": industrial_grids}
    runner_observed = dict(observed)
    replay_observed: dict[str, Mapping[str, Any]] = {}
    replay_errors: dict[str, str] = {}
    for cell in matrix_cells():
        try:
            replay_observed[cell["cell_id"]] = replay_target_checkpoint_metrics(
                cell, replay_grids[cell["domain"]], output, device=device, batch_size=32,
            )
        except Exception as exc:
            # Keep a complete per-cell report even when replay fails.  The
            # runner metric is diagnostic only and can never make the cell
            # pass; checkpoint_replay_error is recorded below.
            replay_errors[cell["cell_id"]] = str(exc)
            replay_observed[cell["cell_id"]] = runner_observed.get(cell["cell_id"], {})
    observed = replay_observed
    expected = {item["cell_id"] for item in matrix_cells()}
    if set(observed) != expected:
        raise RuntimeError(f"frozen runners did not produce all 180 metric cells: missing={sorted(expected - set(observed))}")
    records = []
    for cell in matrix_cells():
        record = build_parity_cell(cell, observed[cell["cell_id"]], checkpoint_path(output, cell))
        if cell["cell_id"] in replay_errors:
            record["parity_passed"] = False
            record["checkpoint_replay_error"] = replay_errors[cell["cell_id"]]
        records.append(record)
    parity = {"schema_version": 2, "test_accessed": False, "test_evaluated": False,
              "industrial_test_evaluated": False, "reference_grid_tests_evaluated": False,
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
