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
import shutil
import sys
import tempfile
from datetime import datetime, timezone
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
PROGRESS_NAME = "reconstruction_progress.json"
LOG_NAME = "reconstruction.log"
MAPPING_NAME = "accepted_result_mapping_manifest.json"


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(output: Path, message: str) -> None:
    """Persist a human-readable progress line immediately and atomically enough.

    The log is diagnostic only; all resumability decisions use validated
    tensor/sidecar pairs and never rely on log text.
    """
    output.mkdir(parents=True, exist_ok=True)
    with (output / LOG_NAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{_utc_now()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_valid_for_cell(output: Path, cell: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Validate one complete checkpoint before allowing resume to skip it."""
    path = checkpoint_path(output, cell)
    try:
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
        accepted = resolve_accepted_result(cell)
        accepted_path = str(accepted["source_artifact"].resolve())
        if metadata.get("source_artifact") != accepted_path:
            raise ValueError("checkpoint source artifact is not the accepted per-cell artifact")
        if metadata.get("accepted_result_artifact") != accepted_path:
            raise ValueError("checkpoint accepted-result identity is inconsistent")
        if dict(metadata.get("accepted_result_json_keys", {})) != dict(accepted["source_json_key"]):
            raise ValueError("checkpoint accepted-result JSON keys are inconsistent")
        return True, None
    except Exception as exc:
        return False, str(exc)


def _block_cells(*, domain: str, seed: int, fraction: float, methods: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        cell for cell in matrix_cells()
        if cell["domain"] == domain
        and int(cell["seed"]) == int(seed)
        and float(cell["history_fraction"]) == float(fraction)
        and cell["method"] in methods
    ]


def _block_ready(output: Path, cells: list[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for cell in cells:
        valid, error = _checkpoint_valid_for_cell(output, cell)
        if not valid:
            failures.append(f"{cell['cell_id']}: {error}")
    return not failures, failures


def _atomic_copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with source.open("rb") as input_handle, os.fdopen(fd, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _promote_staged_block(staging: Path, output: Path, cells: list[Mapping[str, Any]]) -> list[str]:
    """Promote only invalid/missing cells; never replace a validated checkpoint."""
    promote: list[Mapping[str, Any]] = []
    for cell in cells:
        current_valid, _ = _checkpoint_valid_for_cell(output, cell)
        if current_valid:
            continue
        staged_valid, staged_error = _checkpoint_valid_for_cell(staging, cell)
        if not staged_valid:
            raise RuntimeError(f"staged checkpoint failed provenance validation for {cell['cell_id']}: {staged_error}")
        promote.append(cell)
    promoted: list[str] = []
    for cell in promote:
        source_path = checkpoint_path(staging, cell)
        destination_path = checkpoint_path(output, cell)
        _atomic_copy_checkpoint(source_path, destination_path)
        sidecar = json.loads(source_path.with_suffix(source_path.suffix + ".json").read_text(encoding="utf-8"))
        sidecar["checkpoint_path"] = destination_path.as_posix()
        sidecar["sha256"] = sha256_file(destination_path)
        sidecar["file_size_bytes"] = int(destination_path.stat().st_size)
        _atomic_json(destination_path.with_suffix(destination_path.suffix + ".json"), sidecar)
        promoted.append(str(cell["cell_id"]))
    ready, failures = _block_ready(output, cells)
    if not ready:
        raise RuntimeError("promoted checkpoint block is incomplete: " + "; ".join(failures))
    return promoted


def _run_protected_block(
    output: Path, cells: list[Mapping[str, Any]], run: Callable[[Path], Any], label: str,
) -> Any | None:
    """Run one frozen group in staging, preserving every already-valid weight."""
    ready, failures = _block_ready(output, cells)
    if ready:
        _append_log(output, f"skip validated checkpoint block {label}")
        return None
    _append_log(output, f"rebuild checkpoint block {label}; valid existing cells will be protected; first issue={failures[0] if failures else 'unknown'}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-stage-", dir=str(output)) as staging_name:
        staging = Path(staging_name)
        report = run(staging)
        promoted = _promote_staged_block(staging, output, cells)
    _append_log(output, f"checkpoint block {label} complete; promoted={len(promoted)}; preserved={len(cells) - len(promoted)}")
    return report


def _write_progress(
    output: Path, *, mapping_coverage: int, started_at: str,
    current: Mapping[str, Any] | None = None, status: str = "running",
) -> dict[str, Any]:
    cells = matrix_cells()
    completed: list[str] = []
    invalid: dict[str, str] = {}
    for cell in cells:
        valid, error = _checkpoint_valid_for_cell(output, cell)
        if valid:
            completed.append(cell["cell_id"])
        elif error:
            invalid[cell["cell_id"]] = error
    payload = {
        "schema_version": 1,
        "status": status,
        "started_at_utc": started_at,
        "updated_at_utc": _utc_now(),
        "total_cells": len(cells),
        "completed_cells": len(completed),
        "completed_cell_ids": completed,
        "invalid_or_incomplete_cells": invalid,
        "current": dict(current) if current is not None else None,
        "accepted_result_mapping_coverage": int(mapping_coverage),
        "test_accessed": False,
        "test_evaluated": False,
        "industrial_test_evaluated": False,
        "reference_grid_tests_evaluated": False,
        "historical_artifacts_modified": False,
    }
    _atomic_json(output / PROGRESS_NAME, payload)
    return payload


def _write_mapping_manifest(output: Path, mapping: Mapping[str, Mapping[str, Any]]) -> None:
    entries = []
    for cell in matrix_cells():
        resolved = mapping[cell["cell_id"]]
        source = Path(resolved["source_artifact"]).resolve()
        entries.append({
            "cell_id": cell["cell_id"],
            "source_artifact": str(source),
            "source_sha256": sha256_file(source),
            "source_json_key": dict(resolved["source_json_key"]),
            "test_evaluated": False,
        })
    _atomic_json(output / MAPPING_NAME, {
        "schema_version": 1,
        "cell_count": len(entries),
        "entries": entries,
        "test_accessed": False,
        "test_evaluated": False,
    })


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


def _accepted_source_path(mapping: Mapping[str, Mapping[str, Any]], cell: Mapping[str, Any]) -> Path:
    """Return the exact accepted artifact for one matrix cell.

    ``resolve_all_accepted_results`` is keyed by ``cell_id``.  Keep this
    lookup explicit so the reconstruction runner never confuses a cell
    mapping (a dict) with its string identity.
    """
    cell_id = str(cell["cell_id"])
    try:
        resolved = mapping[cell_id]
    except KeyError as exc:
        raise KeyError(f"accepted-result mapping is missing {cell_id}") from exc
    source = resolved.get("source_artifact")
    if not isinstance(source, Path):
        source = Path(str(source))
    return source


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
    resume: bool = False,
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
    started_at = _utc_now()
    _write_mapping_manifest(output, mapping)
    _write_progress(output, mapping_coverage=len(mapping), started_at=started_at, status="running")
    _append_log(output, f"reconstruction started; mapping coverage={len(mapping)}/180; resume={resume}")
    observed: dict[str, Mapping[str, Any]] = {}
    # Kept separate by role: this is the historical BTD input required by
    # load_frozen_reference, while final parity maps to the direct-transfer
    # artifact from accepted_results.py.
    replay_reference_grids: dict[str, Any] = {}
    replay_industrial_grids: dict[str, Any] = {}
    loader = LVGridLoader(data_root, mapping_json)

    for fraction in (0.25, 0.50):
        for seed in (42, 123, 2026):
            _write_progress(
                output, mapping_coverage=len(mapping), started_at=started_at,
                current={"seed": seed, "history_fraction": fraction, "stage": "load_reference_grids"},
            )
            references = {name: loader.load(name) for name in CLIENT_NAMES}
            replay_reference_grids = references
            local_cell = next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "local")
            baseline_cells = _block_cells(
                domain="reference_grid", seed=seed, fraction=fraction,
                methods=("local", "fedavg", "fedprox", "fedper"),
            )
            baseline = _run_protected_block(
                output, baseline_cells,
                lambda staging: run_formal_baselines(
                    references, device=device, rounds=10, local_epochs=5, max_epochs=50,
                    batch_size=32, seed=seed, history_fraction=fraction,
                    checkpoint_root=staging, frozen_commits=FROZEN_COMMITS,
                    source_artifact=str(_accepted_source_path(mapping, local_cell)),
                ), f"reference baseline seed={seed} fraction={fraction}",
            )
            if baseline is not None:
                _record_report_metrics(observed, baseline, domain="reference_grid", fraction=fraction, seed=seed, family="baseline")
            _write_progress(
                output, mapping_coverage=len(mapping), started_at=started_at,
                current={"seed": seed, "history_fraction": fraction, "stage": "reference_baselines"},
            )
            # ``reference`` is the historical formal BTD input consumed by
            # load_frozen_reference; ``source_artifact`` remains the exact
            # direct-transfer artifact used for final per-cell parity.
            reference_cell = next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "btd_fl_direct_transfer")
            reference_source = _accepted_source_path(mapping, reference_cell)
            btd_reference_input = direct_transfer_reference_input(history_fraction=fraction, seed=seed)
            if btd_reference_input is not None:
                # Fail before training if the historical input is missing or
                # no longer satisfies the frozen load_frozen_reference schema.
                from code.federated.formal_btd_ablations import load_frozen_reference
                load_frozen_reference(btd_reference_input)
            reference_fomo_cell = next(item for item in matrix_cells() if item["domain"] == "reference_grid" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "fedfomo_style")
            reference_fomo_source = _accepted_source_path(mapping, reference_fomo_cell)
            btd_cells = _block_cells(
                domain="reference_grid", seed=seed, fraction=fraction,
                methods=("btd_fl_direct_transfer",),
            )
            direct = _run_protected_block(
                output, btd_cells,
                lambda staging: run_direct_transfer(
                    references, btd_reference_input, device=device, max_epochs=50, batch_size=32,
                    seed=seed, history_fraction=fraction, checkpoint_root=staging,
                    source_artifact=str(reference_source), frozen_commits=FROZEN_COMMITS,
                ), f"reference BTD seed={seed} fraction={fraction}",
            )
            if direct is not None:
                _record_report_metrics(observed, direct, domain="reference_grid", fraction=fraction, seed=seed, family="btd")
            fomo_cells = _block_cells(
                domain="reference_grid", seed=seed, fraction=fraction,
                methods=("fedfomo_style",),
            )
            fomo = _run_protected_block(
                output, fomo_cells,
                lambda staging: {"scenarios": {
                    target: run_fedfomo_scenario(
                        references, target, 10, 5, 32, device, 1e-12, seed, fraction,
                        CLIENT_NAMES, checkpoint_root=staging, checkpoint_domain="reference_grid",
                        source_artifact=str(reference_fomo_source), frozen_commits=FROZEN_COMMITS,
                    ) for target in CLIENT_NAMES
                }}, f"reference FedFomo seed={seed} fraction={fraction}",
            )
            if fomo is not None:
                _record_report_metrics(observed, fomo, domain="reference_grid", fraction=fraction, seed=seed, family="fedfomo")

            grids = load_external_grids(data_root, mapping_json)
            replay_industrial_grids = grids
            industrial_cell = next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "local")
            industrial_source = _accepted_source_path(mapping, industrial_cell)
            industrial_baseline_cells = _block_cells(
                domain="industrial_external", seed=seed, fraction=fraction,
                methods=("local", "fedavg", "fedprox", "fedper"),
            )
            ext_baseline = _run_protected_block(
                output, industrial_baseline_cells,
                lambda staging: run_external_baselines(
                    grids, history_fraction=fraction, rounds=10, local_epochs=5, max_epochs=50,
                    batch_size=32, device=device, seed=seed, checkpoint_root=staging,
                    source_artifact=str(industrial_source), frozen_commits=FROZEN_COMMITS,
                ), f"industrial baseline seed={seed} fraction={fraction}",
            )
            if ext_baseline is not None:
                _record_report_metrics(observed, ext_baseline, domain="industrial_external", fraction=fraction, seed=seed, family="baseline")
            industrial_btd_cell = _block_cells(
                domain="industrial_external", seed=seed, fraction=fraction,
                methods=("btd_fl_direct_transfer",),
            )
            ext_btd = _run_protected_block(
                output, industrial_btd_cell,
                lambda staging: run_external_btd(
                    grids, history_fraction=fraction, max_epochs=50, batch_size=32,
                    device=device, seed=seed, checkpoint_root=staging,
                    source_artifact=str(_accepted_source_path(mapping, next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "btd_fl_direct_transfer"))),
                    frozen_commits=FROZEN_COMMITS,
                ), f"industrial BTD seed={seed} fraction={fraction}",
            )
            if ext_btd is not None:
                _record_report_metrics(observed, ext_btd, domain="industrial_external", fraction=fraction, seed=seed, family="btd")
            industrial_fomo_source = _accepted_source_path(mapping, next(item for item in matrix_cells() if item["domain"] == "industrial_external" and item["seed"] == seed and item["history_fraction"] == fraction and item["method"] == "fedfomo_style"))
            industrial_fomo_cells = _block_cells(
                domain="industrial_external", seed=seed, fraction=fraction,
                methods=("fedfomo_style",),
            )
            ext_fomo = _run_protected_block(
                output, industrial_fomo_cells,
                lambda staging: run_external_fedfomo(
                    grids, history_fraction=fraction, rounds=10, local_epochs=5, batch_size=32,
                    device=device, seed=seed, max_epochs_metadata=50, checkpoint_root=staging,
                    checkpoint_domain="industrial_external", source_artifact=str(industrial_fomo_source),
                    frozen_commits=FROZEN_COMMITS,
                ), f"industrial FedFomo seed={seed} fraction={fraction}",
            )
            if ext_fomo is not None:
                _record_report_metrics(observed, ext_fomo, domain="industrial_external", fraction=fraction, seed=seed, family="fedfomo")
            _write_progress(
                output, mapping_coverage=len(mapping), started_at=started_at,
                current={"seed": seed, "history_fraction": fraction, "stage": "target_blocks_complete"},
            )

    _append_log(output, "all training blocks complete; starting serialized checkpoint replay")
    _write_progress(
        output, mapping_coverage=len(mapping), started_at=started_at,
        current={"stage": "serialized_checkpoint_replay"},
    )
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
    _write_progress(
        output, mapping_coverage=len(mapping), started_at=started_at,
        current={"stage": "complete"}, status="complete" if parity["parity_passed"] else "parity_failed",
    )
    _append_log(output, f"reconstruction finished; parity_passed={parity['parity_passed']}")
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
    parser.add_argument("--resume", action="store_true", help="resume only from complete, source-verified checkpoint blocks")
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
            resume=args.resume,
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
