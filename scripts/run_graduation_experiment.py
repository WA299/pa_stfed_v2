"""Single PyCharm-friendly controller for the frozen graduation experiment.

The controller has explicit phases and never upgrades authorization implicitly:

* ``status`` / ``preflight`` are read-only and never load TEST data.
* ``reconstruct`` requires ``--authorize-reconstruction``.
* ``validate`` is fail-closed.
* ``test`` requires both ``--authorize-test`` and a resolved protocol-lock.

Real reconstruction and TEST are intentionally not invoked by this module
during development; the user must select the corresponding phase explicitly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.federated.accepted_results import resolve_all_accepted_results
from code.federated.checkpointing import matrix_cells
from scripts.reconstruct_frozen_checkpoints import (
    DEFAULT_OUTPUT as DEFAULT_CHECKPOINT_ROOT,
    build_reconstruction_plan,
    run_frozen_reconstruction,
)
from scripts.run_final_test_once import (
    DEFAULT_OUTPUT as DEFAULT_TEST_OUTPUT,
    DEFAULT_PARITY,
    DEFAULT_PROTOCOL_LOCK,
    run_once,
)
from scripts.validate_frozen_checkpoints import validate_frozen_checkpoints
from scripts.manage_test_protocol_lock import create_protocol_lock, check_protocol_lock

DEFAULT_DATA_ROOT = ROOT.parent / "pa_stfed_data_v2" / "raw"
DEFAULT_MAPPING = ROOT / "results" / "audits" / "v2_schema_mapping.json"


def _gpu_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        text = subprocess.check_output(
            [executable, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
        return {"available": bool(text), "raw": text}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def status(checkpoint_root: Path, test_output: Path) -> dict[str, Any]:
    progress_path = checkpoint_root / "reconstruction_progress.json"
    test_progress_path = test_output / "test_progress.json"
    reconstruction = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
    testing = json.loads(test_progress_path.read_text(encoding="utf-8")) if test_progress_path.is_file() else None
    current = (reconstruction or {}).get("current") or (testing or {}).get("current")
    completed = (reconstruction or {}).get("completed_cells", 0)
    if testing is not None:
        completed = testing.get("completed_cells", completed)
    return {
        "phase": "test_complete" if (testing or {}).get("completed_cells") == 180 else ("reconstruction" if reconstruction else "not_started"),
        "current_seed": (current or {}).get("seed"),
        "current_history_fraction": (current or {}).get("history_fraction"),
        "current_method": (current or {}).get("method"),
        "current_target": (current or {}).get("target"),
        "completed_cells": completed,
        "total_cells": 180,
        "progress_fraction": completed / 180.0,
        "gpu": _gpu_status(),
        "test_accessed": bool((testing or {}).get("test_accessed", False)),
        "test_evaluated": bool((testing or {}).get("test_evaluated", False)),
        "checkpoint_root": str(checkpoint_root),
        "test_output": str(test_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("status", "preflight", "reconstruct", "validate", "protocol-lock-create", "protocol-lock-check", "test"), default="status")
    parser.add_argument("--authorize-reconstruction", action="store_true")
    parser.add_argument("--authorize-test", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip only complete, source-verified reconstruction/test cells")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint_root = args.checkpoint_root.resolve()
    test_output = args.test_output.resolve()
    if args.phase == "status":
        print(json.dumps(status(checkpoint_root, test_output), ensure_ascii=False, indent=2))
        return
    if args.phase == "preflight":
        mapping = resolve_all_accepted_results()
        cells = matrix_cells()
        print(json.dumps({
            "phase": "preflight", "accepted_mapping_coverage": len(mapping),
            "expected_cells": len(cells), "test_accessed": False,
            "direct_transfer_seed42_25pct_input": "results/federated/formal_25pct/btd_fl_25pct_seed42.json",
            "direct_transfer_seed42_25pct_parity": "results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json",
        }, ensure_ascii=False, indent=2))
        return
    if args.phase == "validate":
        report = validate_frozen_checkpoints(checkpoint_root, args.parity_report.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["test_unlock_ready"]:
            raise SystemExit(1)
        return
    if args.phase == "protocol-lock-create":
        report = create_protocol_lock(
            args.protocol_lock.resolve(), checkpoint_root=checkpoint_root,
            parity_report=args.parity_report.resolve(),
        )
        print(json.dumps({"phase": "protocol-lock-create", **report}, ensure_ascii=False, indent=2))
        return
    if args.phase == "protocol-lock-check":
        report = check_protocol_lock(
            args.protocol_lock.resolve(), checkpoint_root=checkpoint_root,
            parity_report=args.parity_report.resolve(),
        )
        print(json.dumps({"phase": "protocol-lock-check", **report}, ensure_ascii=False, indent=2))
        return
    if args.phase == "reconstruct":
        report = run_frozen_reconstruction(
            checkpoint_root,
            data_root=args.data_root.resolve(), mapping_json=args.mapping_json.resolve(),
            device=args.device, authorize_reconstruction=args.authorize_reconstruction,
            resume=args.resume,
        )
        print(json.dumps({"phase": "reconstruct", "cells": len(report["cells"]), "test_accessed": False}, indent=2))
        return
    if args.phase == "test":
        report = run_once(
            checkpoint_root=checkpoint_root, parity_report=args.parity_report.resolve(),
            protocol_lock=args.protocol_lock.resolve(), output=test_output,
            data_root=args.data_root.resolve(), mapping_json=args.mapping_json.resolve(),
            device=args.device, batch_size=args.batch_size, authorize_test=args.authorize_test,
            resume=args.resume,
        )
        print(json.dumps({"phase": "test", **report}, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
