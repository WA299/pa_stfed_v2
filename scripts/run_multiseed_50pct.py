"""Run the frozen 50%-history leave-one-scarce benchmark."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 123, 2026)


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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "federated" / "formal_50pct" / "multiseed")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-btd", action="store_true")
    parser.add_argument("--skip-fedfomo", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any(seed not in DEFAULT_SEEDS for seed in args.seeds):
        raise ValueError("50% robustness benchmark supports only seeds 42, 123, and 2026")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fraction = "0.5"
    if not args.summarize_only:
        for seed in args.seeds:
            common = ["--seed", str(seed), "--device", args.device, "--output-dir", str(args.output_dir), "--data-root", str(args.data_root), "--mapping-json", str(args.mapping_json), "--history-fraction", fraction]
            if args.synthetic_smoke:
                common.append("--synthetic-smoke")
            if args.force:
                common.append("--force")
            if not args.skip_baselines:
                subprocess.run([sys.executable, "scripts/run_formal_btd_fl.py", *common, "--rounds", str(args.rounds), "--local-epochs", str(args.local_epochs), "--max-epochs", str(args.max_epochs), "--baseline-only"], cwd=ROOT, check=True)
            if not args.skip_btd:
                subprocess.run([sys.executable, "scripts/run_btd_fl_direct_transfer.py", *common, "--max-epochs", str(args.max_epochs), "--batch-size", str(args.batch_size)], cwd=ROOT, check=True)
            if not args.skip_fedfomo:
                suffix = "_synthetic_smoke" if args.synthetic_smoke else ""
                reference_dir = args.output_dir / "smoke" if args.synthetic_smoke else args.output_dir
                baseline = reference_dir / f"formal_baselines_50pct_seed{seed}{suffix}.json"
                direct = reference_dir / f"btd_fl_direct_transfer_50pct_seed{seed}{suffix}.json"
                subprocess.run([sys.executable, "scripts/run_fedfomo_style.py", *common, "--direct-transfer-json", str(direct), "--baseline-json", str(baseline), "--rounds", str(args.rounds), "--local-epochs", str(args.local_epochs), "--batch-size", str(args.batch_size)], cwd=ROOT, check=True)
    if (args.summarize or args.summarize_only) and not args.synthetic_smoke:
        subprocess.run([sys.executable, "scripts/summarize_multiseed_25pct.py", "--input-dir", str(args.output_dir), "--history-fraction", fraction], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
