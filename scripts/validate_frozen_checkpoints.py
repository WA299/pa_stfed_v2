"""Fail-closed validation of the frozen checkpoint set and parity report."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from code.federated.checkpointing import checkpoint_path, load_checkpoint, matrix_cells, validate_matrix_identity
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


def validate_frozen_checkpoints(root: Path = DEFAULT_ROOT, parity_path: Path | None = None) -> dict[str, Any]:
    cells = matrix_cells()
    missing, invalid = [], []
    for cell in cells:
        path = checkpoint_path(root, cell)
        if not path.is_file() or not path.with_suffix(path.suffix + ".json").is_file():
            missing.append(cell["cell_id"])
            continue
        try:
            _state, metadata = load_checkpoint(path, expected={
                "test_accessed": False, "test_evaluated": False,
                "industrial_test_evaluated": False,
                "reference_grid_tests_evaluated": False,
            })
            validate_matrix_identity(metadata, cell)
            if not _finite(metadata):
                raise ValueError("checkpoint metadata is non-finite")
        except Exception as exc:  # fail closed, report every cell
            invalid.append({"cell_id": cell["cell_id"], "error": str(exc)})
    parity = None
    parity_failures = []
    if parity_path is not None and parity_path.is_file():
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if parity.get("test_accessed") is not False or parity.get("parity_passed") is not True:
            parity_failures.append("parity report is not a passed pre-TEST report")
        for item in parity.get("cells", []):
            if item.get("parity_passed") is not True:
                parity_failures.append(item.get("cell_id", "unknown"))
    elif parity_path is not None:
        parity_failures.append("missing parity report")
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
        "reconstruction_required": bool(missing),
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
