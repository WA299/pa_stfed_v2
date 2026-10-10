"""Create/check the committed, predeclared final TEST protocol lock.

This helper never authorizes TEST.  Creating the JSON is a preparation step;
the user must commit that file as the protocol-lock commit and later provide
the separate ``--authorize-test`` command-line authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.federated.checkpointing import FROZEN_COMMITS, REFERENCE_TARGETS, checkpoint_path, matrix_cells, sha256_file
from scripts.run_final_test_once import (
    DEFAULT_CHECKPOINT_ROOT, DEFAULT_PARITY, DEFAULT_PROTOCOL_LOCK,
    EXPECTED_REPORTING_COMMIT, FRACTIONS, METHODS, SEEDS,
)
from scripts.validate_frozen_checkpoints import validate_frozen_checkpoints

SCHEMA_VERSION = 1
LOCK_RELATIVE_PATH = "paper/preflight/test_protocol_lock.json"
TEST_SPLITS = {
    "reference_grid": "canonical_test_split_from_frozen_grid_loader",
    "industrial_external": "canonical_industrial_test_split_from_frozen_loader",
}


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _assert_executable_sources_committed(root: Path) -> None:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if not status:
        return
    changed_tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    if changed_tracked:
        raise RuntimeError("working tree has tracked changes; protocol lock does not identify the executed code")
    source_roots = ("code/", "scripts/", "configs/")
    untracked_sources = []
    for line in status.splitlines():
        if line.startswith("?? "):
            path = line[3:].replace("\\", "/").lstrip("./")
            if path.startswith(source_roots):
                untracked_sources.append(path)
    if untracked_sources:
        raise RuntimeError(
            "untracked executable source files make protocol identity ambiguous: "
            + ", ".join(sorted(untracked_sources))
        )


def git_protocol_identity(path: Path, *, repo_root: Path = ROOT) -> dict[str, str]:
    """Require a tracked, clean lock whose containing commit is current HEAD."""
    root = Path(repo_root).resolve()
    lock_path = Path(path).resolve()
    try:
        relative = lock_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("protocol-lock file must be inside the project Git repository") from exc
    if relative != LOCK_RELATIVE_PATH:
        raise ValueError(f"protocol-lock must be at {LOCK_RELATIVE_PATH}")
    if not lock_path.is_file():
        raise FileNotFoundError(f"protocol-lock file is missing: {lock_path}")
    head = _git(root, "rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError("Git HEAD is not a full commit SHA")
    _assert_executable_sources_committed(root)
    _git(root, "ls-files", "--error-unmatch", relative)
    lock_commit = _git(root, "log", "-1", "--format=%H", "--", relative)
    if len(lock_commit) != 40:
        raise RuntimeError("protocol-lock has no verifiable committed Git identity")
    _git(root, "cat-file", "-e", f"{lock_commit}^{{commit}}")
    if lock_commit != head:
        raise RuntimeError(
            "protocol-lock must be committed in the current HEAD commit; "
            f"path commit={lock_commit}, HEAD={head}"
        )
    for args, label in ((["diff", "--quiet", "HEAD", "--", relative], "working tree"),
                        (["diff", "--cached", "--quiet", "HEAD", "--", relative], "index")):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"protocol-lock has uncommitted {label} changes")
    blob = _git(root, "rev-parse", f"{head}:{relative}")
    # Respect Git's path-specific clean filters (notably core.autocrlf on
    # Windows) while still binding the checked file to the committed blob.
    working_blob = _git(root, "hash-object", f"--path={relative}", str(lock_path))
    if working_blob != blob:
        raise RuntimeError("working protocol-lock content does not match the committed HEAD blob")
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"{head}:{relative}"],
        capture_output=True, check=False,
    )
    if committed.returncode != 0:
        raise RuntimeError("committed protocol-lock blob cannot be read")
    return {
        "commit_sha": head,
        "blob_sha": blob,
        "path": relative,
        "content_sha256": hashlib.sha256(committed.stdout).hexdigest(),
    }


def _checkpoint_set_identity(checkpoint_root: Path) -> tuple[str, str]:
    records: list[str] = []
    reconstruction_commits: set[str] = set()
    for cell in matrix_cells():
        path = checkpoint_path(checkpoint_root, cell)
        sidecar_path = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not sidecar_path.is_file():
            raise RuntimeError(f"checkpoint set is incomplete at {cell['cell_id']}")
        digest = sha256_file(path)
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("sha256") != digest or sidecar.get("test_accessed") is not False:
            raise RuntimeError(f"checkpoint integrity/TEST guardrail failed at {cell['cell_id']}")
        commits = sidecar.get("frozen_commits", {})
        for key, expected in FROZEN_COMMITS.items():
            if commits.get(key) != expected:
                raise RuntimeError(f"checkpoint frozen commit mismatch at {cell['cell_id']}: {key}")
        reconstruction_commit = commits.get("reconstruction_code")
        if not isinstance(reconstruction_commit, str) or len(reconstruction_commit) != 40:
            raise RuntimeError(f"checkpoint reconstruction commit is missing at {cell['cell_id']}")
        reconstruction_commits.add(reconstruction_commit)
        records.append(f"{cell['cell_id']}\t{digest}")
    if len(reconstruction_commits) != 1:
        raise RuntimeError("checkpoint cells do not share one reconstruction code commit")
    reconstruction_commit = next(iter(reconstruction_commits))
    _git(ROOT, "cat-file", "-e", f"{reconstruction_commit}^{{commit}}")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest(), reconstruction_commit


def build_lock_document(checkpoint_root: Path, parity_report: Path) -> dict[str, Any]:
    validation = validate_frozen_checkpoints(checkpoint_root, parity_report)
    if validation.get("test_unlock_ready") is not True:
        raise RuntimeError("cannot create protocol lock: checkpoint/parity validation is not ready")
    checkpoint_digest, reconstruction_commit = _checkpoint_set_identity(checkpoint_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "lock_identity": "Git commit containing this tracked file, derived at runtime",
        "reporting_commit": EXPECTED_REPORTING_COMMIT,
        "scientific_method_commit": FROZEN_COMMITS["scientific_method"],
        "accepted_results_commit": FROZEN_COMMITS["accepted_results"],
        "paper_ready_reporting_commit": FROZEN_COMMITS["paper_ready_reporting"],
        "reconstruction_code_commit": reconstruction_commit,
        "matrix_cells": 180,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "history_fractions": list(FRACTIONS),
        "reference_targets": list(REFERENCE_TARGETS),
        "industrial_target": "norway_industrial_mvlv",
        "test_split_contract": dict(TEST_SPLITS),
        "checkpoint_set_sha256": checkpoint_digest,
        "parity_report_path": str(Path(parity_report).resolve()),
        "parity_report_sha256": sha256_file(parity_report),
        "test_authorization": "separate explicit --authorize-test flag required; not granted by this lock",
        "test_opening_authorized": False,
        "test_evaluated": False,
    }


def validate_lock_contract(lock: Mapping[str, Any], *, checkpoint_digest: str,
                           reconstruction_commit: str, parity_path: Path) -> None:
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported protocol-lock schema_version")
    if lock.get("reporting_commit") != EXPECTED_REPORTING_COMMIT:
        raise RuntimeError("protocol-lock reporting commit mismatch")
    frozen = {
        "scientific_method_commit": FROZEN_COMMITS["scientific_method"],
        "accepted_results_commit": FROZEN_COMMITS["accepted_results"],
        "paper_ready_reporting_commit": FROZEN_COMMITS["paper_ready_reporting"],
    }
    for key, value in frozen.items():
        if lock.get(key) != value:
            raise RuntimeError(f"protocol-lock frozen configuration mismatch: {key}")
    exact = {
        "matrix_cells": 180,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "history_fractions": list(FRACTIONS),
        "industrial_target": "norway_industrial_mvlv",
        "test_split_contract": TEST_SPLITS,
        "checkpoint_set_sha256": checkpoint_digest,
        "reconstruction_code_commit": reconstruction_commit,
        "parity_report_path": str(Path(parity_path).resolve()),
        "parity_report_sha256": sha256_file(parity_path),
        "test_opening_authorized": False,
        "test_evaluated": False,
    }
    from code.federated.checkpointing import REFERENCE_TARGETS
    exact["reference_targets"] = list(REFERENCE_TARGETS)
    for key, value in exact.items():
        if lock.get(key) != value:
            raise RuntimeError(f"protocol-lock configuration/identity mismatch: {key}")
    if lock.get("test_authorization") != "separate explicit --authorize-test flag required; not granted by this lock":
        raise RuntimeError("protocol-lock cannot grant or imply TEST authorization")
    if lock.get("lock_identity") != "Git commit containing this tracked file, derived at runtime":
        raise RuntimeError("protocol-lock identity declaration is missing or unresolved")
    for key, value in lock.items():
        if isinstance(value, str) and any(marker in value.upper() for marker in ("PLACEHOLDER", "TODO", "REPLACE_ME", "<COMMIT")):
            raise RuntimeError(f"protocol-lock contains unresolved placeholder in {key}")


def check_protocol_lock(path: Path = DEFAULT_PROTOCOL_LOCK, *,
                        checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
                        parity_report: Path = DEFAULT_PARITY) -> dict[str, Any]:
    identity = git_protocol_identity(path)
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"protocol-lock JSON is malformed: {exc}") from exc
    if not isinstance(lock, Mapping):
        raise RuntimeError("protocol-lock must be a JSON object")
    validation = validate_frozen_checkpoints(checkpoint_root, parity_report)
    if validation.get("test_unlock_ready") is not True:
        raise RuntimeError("checkpoint/parity validation is not ready for this protocol lock")
    checkpoint_digest, reconstruction_commit = _checkpoint_set_identity(checkpoint_root)
    validate_lock_contract(lock, checkpoint_digest=checkpoint_digest,
                           reconstruction_commit=reconstruction_commit, parity_path=parity_report)
    return {"lock": dict(lock), "git_identity": identity, "test_authorized": False}


def create_protocol_lock(path: Path = DEFAULT_PROTOCOL_LOCK, *,
                         checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
                         parity_report: Path = DEFAULT_PARITY) -> dict[str, Any]:
    if path.resolve() != DEFAULT_PROTOCOL_LOCK.resolve():
        raise ValueError(f"protocol-lock output path is fixed: {DEFAULT_PROTOCOL_LOCK}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing protocol lock: {path}")
    _assert_executable_sources_committed(ROOT)
    document = build_lock_document(checkpoint_root, parity_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = __import__("tempfile").mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "created": str(path),
        "test_authorized": False,
        "next_steps": [
            f"git add {LOCK_RELATIVE_PATH}",
            'git commit -m "protocol: lock final TEST evaluation"',
            f"python scripts/manage_test_protocol_lock.py --check --checkpoint-root {checkpoint_root} --parity-report {parity_report}",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="write a non-authorizing lock JSON; a separate Git commit is required")
    mode.add_argument("--check", action="store_true", help="verify lock config and its committed Git identity")
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY)
    args = parser.parse_args()
    if args.create:
        report = create_protocol_lock(args.protocol_lock, checkpoint_root=args.checkpoint_root.resolve(), parity_report=args.parity_report.resolve())
    else:
        report = check_protocol_lock(args.protocol_lock, checkpoint_root=args.checkpoint_root.resolve(), parity_report=args.parity_report.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
