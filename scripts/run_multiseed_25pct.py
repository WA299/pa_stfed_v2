"""Run the non-historical seeds for the frozen formal 25%-history benchmark."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (123, 2026)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "formal_25pct" / "multiseed")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-btd", action="store_true")
    parser.add_argument("--skip-fedfomo", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--history-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.history_fraction == 0.25 and 42 in args.seeds:
        raise ValueError("seed 42 is historical and cannot be retrained by the 25% multiseed launcher")
    percent = int(round(args.history_fraction * 100))
    if percent not in (25, 50):
        raise ValueError("history_fraction must be 0.25 or 0.50")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for seed in args.seeds:
            common = ["--seed", str(seed), "--device", args.device, "--output-dir", str(args.output_dir), "--data-root", str(args.data_root), "--mapping-json", str(args.mapping_json), "--history-fraction", str(args.history_fraction)]
            if args.synthetic_smoke:
                common.append("--synthetic-smoke")
            if args.force:
                common.append("--force")
            if not args.skip_baselines:
                _run([sys.executable, "scripts/run_formal_btd_fl.py", *common, "--rounds", str(args.rounds), "--local-epochs", str(args.local_epochs), "--max-epochs", str(args.max_epochs), "--baseline-only"])
            if not args.skip_btd:
                _run([sys.executable, "scripts/run_btd_fl_direct_transfer.py", *common, "--max-epochs", str(args.max_epochs), "--batch-size", str(args.batch_size)])
            if not args.skip_fedfomo:
                baseline_name = f"formal_baselines_{percent}pct_seed{seed}" + ("_synthetic_smoke" if args.synthetic_smoke else "") + ".json"
                direct_name = f"btd_fl_direct_transfer_{percent}pct_seed{seed}" + ("_synthetic_smoke" if args.synthetic_smoke else "") + ".json"
                reference_dir = args.output_dir / "smoke" if args.synthetic_smoke else args.output_dir
                _run([sys.executable, "scripts/run_fedfomo_style.py", *common, "--direct-transfer-json", str(reference_dir / direct_name), "--baseline-json", str(reference_dir / baseline_name), "--rounds", str(args.rounds), "--local-epochs", str(args.local_epochs), "--batch-size", str(args.batch_size)])
    if (args.summarize or args.summarize_only) and not args.synthetic_smoke:
        _run([sys.executable, "scripts/summarize_multiseed_25pct.py", "--input-dir", str(args.output_dir), "--history-fraction", str(args.history_fraction)])


if __name__ == "__main__":
    main()
