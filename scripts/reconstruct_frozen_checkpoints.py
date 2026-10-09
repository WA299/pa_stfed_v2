"""Plan or (only when explicitly requested) export frozen pre-TEST checkpoints.

The default command is plan-only.  It never loads TEST targets and never
starts a real reconstruction.  Real execution is intentionally guarded by a
separate explicit flag for the future protocol lock.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from code.federated.checkpointing import matrix_cells
DEFAULT_OUTPUT = ROOT / "artifacts" / "frozen_pretest_checkpoints"


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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true", default=True)
    parser.add_argument("--execute-real", action="store_true", help="Reserved future flag; blocked without protocol lock")
    args = parser.parse_args()
    if args.execute_real:
        raise SystemExit("real reconstruction is intentionally disabled until a protocol-lock commit and explicit TEST preflight authorization")
    plan = build_reconstruction_plan(args.output_root.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "reconstruction_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(path), "cells": len(plan["cells"]), "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
