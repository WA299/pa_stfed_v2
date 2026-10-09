"""Read-only checkpoint inventory and 180-cell TEST preflight.

This module never loads a checkpoint. It only enumerates weight files, hashes
their bytes, and extracts identity fields when the filename contains an exact
verifiable token. It does not read TEST data or result metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "preflight"
DEFAULT_SEARCH_DIRECTORIES = (
    "results",
    "outputs",
    "checkpoints",
    "runs",
    "artifacts",
    "models",
)
WEIGHT_EXTENSIONS = (".pt", ".pth", ".ckpt", ".safetensors")
SEEDS = (42, 123, 2026)
HISTORY_FRACTIONS = (0.25, 0.50)
METHODS = (
    "local",
    "fedavg",
    "fedprox",
    "fedper",
    "fedfomo_style",
    "btd_fl_direct_transfer",
)
REFERENCE_TARGETS = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
INDUSTRIAL_TARGET = "norway_industrial_mvlv"

_SEED_RE = re.compile(r"(?:^|[_-])seed(?:[_-]?)(42|123|2026)(?:$|[_-])", re.IGNORECASE)
_FRACTION_RE = re.compile(r"(?:^|[_-])(25|50)pct(?:$|[_-])", re.IGNORECASE)
_METHOD_TOKENS = {
    "btd_fl_direct_transfer": re.compile(r"(?:^|[_-])btd[_-]fl[_-]direct[_-]transfer(?:$|[_-])", re.IGNORECASE),
    "fedfomo_style": re.compile(r"(?:^|[_-])fedfomo[_-]style(?:$|[_-])", re.IGNORECASE),
    "fedprox": re.compile(r"(?:^|[_-])fedprox(?:$|[_-])", re.IGNORECASE),
    "fedper": re.compile(r"(?:^|[_-])fedper(?:$|[_-])", re.IGNORECASE),
    "fedavg": re.compile(r"(?:^|[_-])fedavg(?:$|[_-])", re.IGNORECASE),
    "local": re.compile(r"(?:^|[_-])(?:scarce[_-])?local(?:$|[_-])", re.IGNORECASE),
}


def _cell_id(domain: str, target: str, seed: int, fraction: float, method: str) -> str:
    return f"{domain}|{target}|seed{seed}|{int(fraction * 100)}pct|{method}"


def build_evaluation_matrix() -> list[dict[str, Any]]:
    """Return the complete matrix without TEST observations or metrics."""
    rows: list[dict[str, Any]] = []
    for fraction in HISTORY_FRACTIONS:
        for seed in SEEDS:
            for target in REFERENCE_TARGETS:
                for method in METHODS:
                    rows.append(
                        {
                            "cell_id": _cell_id("reference", target, seed, fraction, method),
                            "domain": "reference_grid",
                            "target": target,
                            "seed": seed,
                            "history_fraction": fraction,
                            "method": method,
                            "split": "canonical_TEST_locked",
                            "test_data_accessed": False,
                            "evaluation_status": "not_evaluated",
                            "test_metric_value": "",
                        }
                    )
            for method in METHODS:
                rows.append(
                    {
                        "cell_id": _cell_id("industrial", INDUSTRIAL_TARGET, seed, fraction, method),
                        "domain": "industrial_external",
                        "target": INDUSTRIAL_TARGET,
                        "seed": seed,
                        "history_fraction": fraction,
                        "method": method,
                        "split": "canonical_TEST_locked",
                        "test_data_accessed": False,
                        "evaluation_status": "not_evaluated",
                        "test_metric_value": "",
                    }
                )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filename_identity(path: Path) -> dict[str, Any]:
    """Extract only exact identity tokens present in the filename/path."""
    text = "_".join(path.parts)
    stem = path.stem
    seed_matches = _SEED_RE.findall(text)
    fraction_matches = _FRACTION_RE.findall(text)
    metadata: dict[str, Any] = {
        "seed": int(seed_matches[0]) if len(seed_matches) == 1 else None,
        "history_fraction": (int(fraction_matches[0]) / 100.0) if len(fraction_matches) == 1 else None,
        "method": None,
        "target": None,
        "verification": [],
    }
    if len(seed_matches) == 1:
        metadata["verification"].append("seed token is unique in path")
    if len(fraction_matches) == 1:
        metadata["verification"].append("history-fraction token is unique in path")
    method_matches = [name for name, pattern in _METHOD_TOKENS.items() if pattern.search(stem)]
    if len(method_matches) == 1:
        metadata["method"] = method_matches[0]
        metadata["verification"].append("method token is unique in filename")
    target_matches = [target for target in (*REFERENCE_TARGETS, INDUSTRIAL_TARGET) if target.lower() in text.lower()]
    if len(target_matches) == 1:
        metadata["target"] = target_matches[0]
        metadata["verification"].append("target token is unique in path")
    return metadata


def _iter_weight_files(root: Path, search_directories: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for directory in search_directories:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in WEIGHT_EXTENSIONS:
                paths.add(path)
    return sorted(paths)


def inventory_checkpoints(root: Path = ROOT, search_directories: Iterable[str] | None = None) -> dict[str, Any]:
    """Inventory checkpoint bytes without deserializing them."""
    if search_directories is None:
        directories = list(DEFAULT_SEARCH_DIRECTORIES)
        directories.extend(
            path.name for path in root.iterdir()
            if path.is_dir() and path.name.startswith("tmp_")
        )
        search_directories = tuple(sorted(set(directories)))
    else:
        search_directories = tuple(search_directories)
    files = []
    for path in _iter_weight_files(root, search_directories):
        identity = _filename_identity(path.relative_to(root))
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "candidate_identity": identity,
                "deserialized": False,
            }
        )
    matrix = build_evaluation_matrix()
    candidate_cells: set[str] = set()
    for item in files:
        identity = item["candidate_identity"]
        if all(identity.get(field) is not None for field in ("seed", "history_fraction", "method", "target")):
            domain = "industrial_external" if identity["target"] == INDUSTRIAL_TARGET else "reference_grid"
            candidate_cells.add(_cell_id(domain, identity["target"], identity["seed"], identity["history_fraction"], identity["method"]))
    all_cells = {row["cell_id"] for row in matrix}
    candidate_matches = sorted(candidate_cells & all_cells)
    unknown = sorted(candidate_cells - all_cells)
    # This inventory does not accept filename tokens as proof of frozen-model
    # provenance. Until an accepted checkpoint manifest verifies each cell,
    # every cell remains missing for inference readiness.
    missing = sorted(all_cells)
    unverified_files = [
        item["path"] for item in files
        if not all(item["candidate_identity"].get(field) is not None for field in ("seed", "history_fraction", "method", "target"))
    ]
    return {
        "schema_version": 1,
        "project_root": str(root),
        "search_directories": list(search_directories),
        "weight_extensions": list(WEIGHT_EXTENSIONS),
        "checkpoint_count": len(files),
        "checkpoint_files": files,
        "unverified_checkpoint_files": unverified_files,
        "coverage": {
            "reference_cells_total": 144,
            "industrial_cells_total": 36,
            "total_cells": 180,
            "cells_with_candidate_identity_tokens": len(candidate_matches),
            "cells_with_verifiable_checkpoint": 0,
            "cells_missing_verifiable_checkpoint": len(missing),
            "unmatched_verifiable_checkpoint_identities": unknown,
            "missing_cell_ids": missing,
            "accepted_180_cell_inference_possible": False,
            "pre_test_reconstruction_required": True,
        },
        "guardrails": {
            "test_accessed": False,
            "test_observations_read": False,
            "test_metrics_read": False,
            "checkpoint_deserialization_performed": False,
            "reconstruction_executed": False,
        },
        "interpretation": (
            "No checkpoint weights were found in the configured output directories; "
            "accepted 180-cell TEST inference is not currently possible from persisted checkpoints. "
            "Pre-TEST reconstruction would be required, but was not executed."
            if not files
            else "Checkpoint files were found, but no accepted checkpoint manifest was available to verify checkpoint acceptance/frozen provenance; filename identity tokens are candidates only and do not count as persisted inference coverage."
        ),
    }


def write_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["cell_id", "domain", "target", "seed", "history_fraction", "method", "split", "test_data_accessed", "evaluation_status", "test_metric_value"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_inventory(output_dir: Path, inventory: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    coverage = inventory["coverage"]
    lines = [
        "# Checkpoint Readiness Inventory",
        "",
        "This is a read-only preflight inventory. Checkpoint files were hashed as bytes and never deserialized. TEST targets and metrics were not accessed.",
        "",
        f"- Checkpoint weight files found: **{inventory['checkpoint_count']}**",
        f"- Verifiable 180-cell identities: **{coverage['cells_with_verifiable_checkpoint']} / {coverage['total_cells']}**",
        f"- Missing/unverified cells: **{coverage['cells_missing_verifiable_checkpoint']}**",
        f"- Accepted 180-cell TEST inference possible from persisted checkpoints: **{str(coverage['accepted_180_cell_inference_possible']).lower()}**",
        f"- Pre-TEST reconstruction required: **{str(coverage['pre_test_reconstruction_required']).lower()}**",
        "",
        "## Guardrails",
        "",
        "- `test_accessed = false`",
        "- `test_observations_read = false`",
        "- `test_metrics_read = false`",
        "- `checkpoint_deserialization_performed = false`",
        "- `reconstruction_executed = false`",
        "",
        "## Identity policy",
        "",
        "Seed, history fraction, method, and target are recorded only when unique exact tokens are present in the checkpoint filename/path. JSON result metrics are never used to infer checkpoint identity. Unknown or partial identities do not count toward coverage.",
        "",
        "## Interpretation",
        "",
        inventory["interpretation"],
    ]
    (output_dir / "checkpoint_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--search-dir", action="append", dest="search_directories", default=None)
    args = parser.parse_args()
    search_directories = tuple(args.search_directories) if args.search_directories else None
    inventory = inventory_checkpoints(args.root.resolve(), search_directories)
    write_inventory(args.output_dir.resolve(), inventory)
    write_matrix(args.output_dir.resolve() / "evaluation_matrix_180.csv", build_evaluation_matrix())
    print(json.dumps({"checkpoint_count": inventory["checkpoint_count"], **inventory["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
