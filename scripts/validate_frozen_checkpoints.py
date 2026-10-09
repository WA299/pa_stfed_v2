"""Fail-closed validation of the frozen checkpoint set and parity report."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from code.federated.checkpointing import checkpoint_path, load_checkpoint, matrix_cells, validate_matrix_identity
from code.federated.accepted_results import resolve_accepted_result
DEFAULT_ROOT = ROOT / "artifacts" / "frozen_pretest_checkpoints"


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


EXPECTED_METRICS = tuple(
    (split, scope, metric)
    for split in ("audit", "validation")
    for scope in ("node_macro", "grid_aggregate")
    for metric in ("mae", "rmse", "wape_pct", "smape_pct")
)


def expected_cell_map() -> dict[str, dict[str, Any]]:
    cells = matrix_cells()
    return {str(cell["cell_id"]): cell for cell in cells}


def _validate_parity_report(parity_path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate every parity identity and every numerical comparison.

    A top-level success flag is never trusted.  The report must be explicit,
    complete, finite, and one-to-one with the frozen 180-cell matrix.
    """
    failures: list[str] = []
    accepted_documents: dict[str, Any] = {}
    if parity_path is None:
        return None, ["missing parity report (argument omitted)"]
    if not parity_path.is_file():
        return None, ["missing parity report"]
    try:
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"malformed parity report: {exc}"]
    if not isinstance(parity, Mapping):
        return None, ["parity report must be a JSON object"]
    if parity.get("test_accessed") is not False or parity.get("test_evaluated") is not False:
        failures.append("parity report TEST guardrails are not explicitly false")
    if parity.get("parity_passed") is not True:
        failures.append("parity report top-level parity_passed is not true")
    if parity.get("reconstruction_executed") is not True or parity.get("historical_artifacts_modified") is not False:
        failures.append("parity report does not identify an executed isolated reconstruction")
    cells = parity.get("cells")
    if not isinstance(cells, list):
        return parity, failures + ["parity report cells must be a list"]
    expected = expected_cell_map()
    ids = [item.get("cell_id") if isinstance(item, Mapping) else None for item in cells]
    if len(cells) != len(expected):
        failures.append(f"parity report has {len(cells)} cells, expected {len(expected)}")
    if len(set(ids)) != len(ids):
        failures.append("parity report contains duplicate cell IDs")
    actual_ids = {item for item in ids if item is not None}
    if actual_ids - set(expected):
        failures.append(f"parity report contains extra cells: {sorted(actual_ids - set(expected))}")
    if set(expected) - actual_ids:
        failures.append(f"parity report is missing cells: {sorted(set(expected) - actual_ids)}")
    for index, item in enumerate(cells):
        prefix = f"cells[{index}]"
        if not isinstance(item, Mapping):
            failures.append(f"{prefix} is not an object")
            continue
        cell_id = item.get("cell_id")
        if cell_id not in expected:
            continue
        cell = expected[cell_id]
        try:
            accepted = resolve_accepted_result(cell)
            accepted_path = str(accepted["source_artifact"])
            accepted_keys = accepted["source_json_key"]
        except Exception as exc:
            failures.append(f"{prefix} accepted-result mapping failed: {exc}")
            accepted_path, accepted_keys = None, {}
        for key in ("domain", "target", "seed", "history_fraction", "method", "canonical_method"):
            if item.get(key) != cell[key]:
                failures.append(f"{prefix}.{key} identity mismatch")
        if item.get("parity_passed") is not True or item.get("test_accessed") is not False or item.get("test_evaluated") is not False:
            failures.append(f"{prefix} is not an explicit passed pre-TEST cell")
        if not isinstance(item.get("accepted_artifact"), str) or item.get("accepted_artifact") != accepted_path or not isinstance(item.get("accepted_json_keys"), Mapping) or dict(item.get("accepted_json_keys", {})) != dict(accepted_keys):
            failures.append(f"{prefix} lacks accepted artifact/key provenance")
        comparisons = item.get("metric_comparisons")
        if not isinstance(comparisons, list):
            comparisons = item.get("comparisons")
        if not isinstance(comparisons, list):
            failures.append(f"{prefix} has no numerical metric comparisons")
            continue
        seen: set[tuple[str, str, str]] = set()
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                failures.append(f"{prefix} contains a non-object comparison")
                continue
            key = (str(comparison.get("split")), str(comparison.get("scope")), str(comparison.get("metric")))
            duplicate_key = key in seen
            seen.add(key)
            required_numeric = ("expected", "observed", "absolute_discrepancy", "relative_discrepancy")
            if comparison.get("json_key_path") in (None, "") or not isinstance(comparison.get("accepted_artifact"), str) or comparison.get("parity_passed") is not True:
                failures.append(f"{prefix} has incomplete comparison evidence for {key}")
            if any(name not in comparison or not isinstance(comparison[name], (int, float)) or isinstance(comparison[name], bool) or not math.isfinite(float(comparison[name])) for name in required_numeric):
                failures.append(f"{prefix} has non-finite/missing comparison evidence for {key}")
            if key not in EXPECTED_METRICS:
                failures.append(f"{prefix} has unexpected metric comparison {key}")
            if duplicate_key:
                failures.append(f"{prefix} contains duplicate metric comparison {key}")
            if all(name in comparison and isinstance(comparison[name], (int, float)) and not isinstance(comparison[name], bool) and math.isfinite(float(comparison[name])) for name in ("expected", "observed", "absolute_discrepancy", "relative_discrepancy")):
                expected_value = float(comparison["expected"])
                observed_value = float(comparison["observed"])
                accepted_metric_key = f"{accepted_keys.get(key[0], '')}.{key[1]}.{key[2]}"
                if comparison.get("accepted_artifact") != accepted_path or comparison.get("json_key_path") != accepted_metric_key:
                    failures.append(f"{prefix} source key/path mismatch for {key}")
                abs_error = abs(observed_value - expected_value)
                rel_error = abs_error / max(abs(expected_value), 1e-12)
                try:
                    if accepted_path not in accepted_documents:
                        accepted_documents[accepted_path] = json.loads(Path(accepted_path).read_text(encoding="utf-8"))
                    accepted_value: Any = accepted_documents[accepted_path]
                    for part in accepted_metric_key.split("."):
                        accepted_value = accepted_value[part]
                    accepted_value = float(accepted_value)
                    if not math.isfinite(accepted_value) or accepted_value != expected_value:
                        failures.append(f"{prefix} expected metric is not the accepted source value for {key}")
                except Exception as exc:
                    failures.append(f"{prefix} cannot verify accepted metric source for {key}: {exc}")
                if abs_error > 1e-10 + 1e-6 * abs(expected_value) or abs(float(comparison["absolute_discrepancy"]) - abs_error) > 1e-15 or abs(float(comparison["relative_discrepancy"]) - rel_error) > 1e-12:
                    failures.append(f"{prefix} numerical parity mismatch for {key}")
        if len(comparisons) != len(EXPECTED_METRICS) or seen != set(EXPECTED_METRICS):
            failures.append(f"{prefix} does not contain the complete audit/validation metric contract")
    if not _finite(parity):
        failures.append("parity report contains a non-finite value")
    return dict(parity), failures


def validate_frozen_checkpoints(root: Path = DEFAULT_ROOT, parity_path: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    if parity_path is not None:
        parity_path = Path(parity_path)
    cells = matrix_cells()
    missing, invalid = [], []
    for cell in cells:
        path = checkpoint_path(root, cell)
        if not path.is_file() or not path.with_suffix(path.suffix + ".json").is_file():
            missing.append(cell["cell_id"])
            continue
        try:
            state, metadata = load_checkpoint(path, expected={
                "test_accessed": False, "test_evaluated": False,
                "industrial_test_evaluated": False,
                "reference_grid_tests_evaluated": False,
            }, production=True)
            validate_matrix_identity(metadata, cell)
            if not _finite(metadata):
                raise ValueError("checkpoint metadata is non-finite")
        except Exception as exc:  # fail closed, report every cell
            invalid.append({"cell_id": cell["cell_id"], "error": str(exc)})
    parity, parity_failures = _validate_parity_report(parity_path)
    ready = not missing and not invalid and not parity_failures
    return {
        "schema_version": 1,
        "total_cells": len(cells),
        "missing_cells": missing,
        "invalid_cells": invalid,
        "parity_failures": parity_failures,
        "parity_report_present": parity is not None,
        "test_accessed": False,
        "test_unlock_ready": ready,
        "reconstruction_required": bool(missing or invalid or parity_failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parity-report", type=Path)
    args = parser.parse_args()
    report = validate_frozen_checkpoints(args.checkpoint_root.resolve(), args.parity_report.resolve() if args.parity_report else None)
    print(json.dumps(report, indent=2))
    if not report["test_unlock_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
