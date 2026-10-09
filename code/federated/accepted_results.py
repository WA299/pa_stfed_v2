"""Explicit, fail-closed mapping from the frozen 180-cell matrix to results.

This module intentionally maps to per-seed artifacts, never aggregate means,
smoke outputs, another seed, or another target.  It is used for pre-TEST
parity only; it does not read TEST data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from code.federated.checkpointing import (
    INDUSTRIAL_TARGET,
    REFERENCE_TARGETS,
    matrix_cells,
)

ROOT = Path(__file__).resolve().parents[2]


def direct_transfer_reference_input(*, history_fraction: float, seed: int) -> Path | None:
    """Return the historical BTD *input* reference for direct-transfer runs.

    The seed-42/25% direct-transfer reconstruction consumes the accepted
    ``btd_fl`` report because ``load_frozen_reference`` intentionally validates
    that schema.  The direct-transfer report remains the separate per-cell
    parity source resolved by :func:`resolve_accepted_result`.
    """
    if abs(float(history_fraction) - 0.25) < 1e-12 and int(seed) == 42:
        return ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_25pct_seed42.json"
    return None


def _source_for(cell: Mapping[str, Any]) -> Path:
    pct = int(round(float(cell["history_fraction"]) * 100))
    seed = int(cell["seed"])
    method = str(cell["method"])
    domain = str(cell["domain"])
    if domain == "industrial_external":
        names = {
            "local": "industrial_baselines",
            "fedavg": "industrial_baselines",
            "fedprox": "industrial_baselines",
            "fedper": "industrial_baselines",
            "fedfomo_style": "industrial_fedfomo_style",
            "btd_fl_direct_transfer": "industrial_btd_direct_transfer",
        }
        stem = names.get(method)
        if stem is None:
            raise ValueError(f"unsupported industrial method: {method}")
        return ROOT / "results" / "federated" / "industrial_external" / f"{stem}_{pct}pct_seed{seed}.json"
    if domain != "reference_grid" or str(cell["target"]) not in REFERENCE_TARGETS:
        raise ValueError(f"unsupported accepted-result cell: {cell}")
    if method in {"local", "fedavg", "fedprox", "fedper"}:
        if pct == 25 and seed == 42:
            return ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_25pct_seed42.json"
        return ROOT / "results" / "federated" / f"formal_{pct}pct" / "multiseed" / f"formal_baselines_{pct}pct_seed{seed}.json"
    if method == "btd_fl_direct_transfer":
        if pct == 25 and seed == 42:
            return ROOT / "results" / "federated" / "formal_25pct" / "btd_fl_direct_transfer_25pct_seed42.json"
        return ROOT / "results" / "federated" / f"formal_{pct}pct" / "multiseed" / f"btd_fl_direct_transfer_{pct}pct_seed{seed}.json"
    if method == "fedfomo_style":
        if pct == 25 and seed == 42:
            return ROOT / "results" / "federated" / "formal_25pct" / "fedfomo_style_25pct_seed42.json"
        return ROOT / "results" / "federated" / f"formal_{pct}pct" / "multiseed" / f"fedfomo_style_{pct}pct_seed{seed}.json"
    raise ValueError(f"unsupported reference method: {method}")


def _metric_key(cell: Mapping[str, Any], split: str) -> str:
    method = str(cell["method"])
    target = str(cell["target"])
    if cell["domain"] == "industrial_external":
        public = {"local": "industrial_scarce_local"}.get(method, method)
        if method in {"local", "fedavg", "fedprox", "fedper"}:
            return f"scenarios.{target}.methods.{public}.{split}"
        if method == "fedfomo_style":
            return f"scenarios.{target}.scarce_target_metrics.{split}"
        return f"scenarios.{target}.metrics.{split}"
    if method in {"local", "fedavg", "fedprox", "fedper"}:
        public = {"local": "scarce_local"}.get(method, method)
        return f"scenarios.{target}.methods.{public}.{split}"
    if method == "fedfomo_style":
        return f"scenarios.{target}.scarce_target_metrics.{split}"
    return f"scenarios.{target}.metrics.{split}"


def _walk(document: Mapping[str, Any], dotted: str) -> Any:
    current: Any = document
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def resolve_accepted_result(cell: Mapping[str, Any], *, require_exists: bool = True) -> dict[str, Any]:
    """Return the one exact frozen per-seed artifact and metric key contract."""
    matches = [item for item in matrix_cells() if item["cell_id"] == cell.get("cell_id")]
    if len(matches) != 1:
        raise ValueError(f"cell is not a unique frozen matrix identity: {cell}")
    canonical = matches[0]
    path = _source_for(canonical)
    if require_exists and not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    fraction_matches = (
        float(document.get("history_fraction", -1)) == float(canonical["history_fraction"])
        if "history_fraction" in document else
        (canonical["domain"] == "reference_grid" and canonical["seed"] == 42 and canonical["history_fraction"] == 0.25 and path.name == "fedfomo_style_25pct_seed42.json")
    )
    if document.get("seed") != canonical["seed"] or not fraction_matches:
        raise ValueError(f"accepted source identity does not match cell: {path}")
    expected_clients = list(REFERENCE_TARGETS) if canonical["domain"] == "reference_grid" else list(REFERENCE_TARGETS) + [INDUSTRIAL_TARGET]
    client_names = document.get("client_grid_names")
    if canonical["domain"] == "reference_grid" and client_names != expected_clients:
        raise ValueError(f"accepted reference source has wrong client set: {path}")
    if canonical["domain"] == "industrial_external" and document.get("primary_targets") not in (None, [INDUSTRIAL_TARGET]):
        raise ValueError(f"accepted industrial source has wrong primary target set: {path}")
    if document.get("test_evaluated") is not False:
        raise ValueError(f"accepted source lacks explicit test_evaluated=false: {path}")
    for flag in ("test_accessed", "industrial_test_evaluated", "reference_grid_tests_evaluated"):
        if flag in document and document[flag] is not False:
            raise ValueError(f"accepted source violates {flag}: {path}")
    if document.get("run_mode") == "synthetic_smoke" or "smoke" in path.parts:
        raise ValueError(f"synthetic smoke artifact cannot be an accepted real mapping: {path}")
    if canonical["domain"] == "industrial_external":
        scenario = document.get("scenarios", {}).get(INDUSTRIAL_TARGET, {})
        metadata = scenario.get("metadata", {})
        for flag in ("test_evaluated", "industrial_test_evaluated", "reference_grid_tests_evaluated"):
            if metadata.get(flag) is not False:
                raise ValueError(f"industrial accepted source lacks safe {flag}: {path}")
    # Verify that both exact per-target metric subtrees exist now, rather than
    # deferring an ambiguous or missing mapping until parity execution.
    keys = {"audit": _metric_key(canonical, "audit"), "validation": _metric_key(canonical, "validation")}
    for split, key in keys.items():
        value = _walk(document, key)
        if not isinstance(value, Mapping) or not all(scope in value for scope in ("node_macro", "grid_aggregate")):
            raise ValueError(f"accepted source metric contract is incomplete for {split}: {path}")
        for scope in ("node_macro", "grid_aggregate"):
            metrics = value.get(scope)
            if not isinstance(metrics, Mapping) or any(metric not in metrics for metric in ("mae", "rmse", "wape_pct", "smape_pct")):
                raise ValueError(f"accepted source metric contract is incomplete for {split}/{scope}: {path}")
    return {
        "source_artifact": path,
        "source_json_key": keys,
        "cell": canonical,
    }


def resolve_all_accepted_results() -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for cell in matrix_cells():
        if cell["cell_id"] in resolved:
            raise ValueError(f"duplicate cell mapping: {cell['cell_id']}")
        resolved[cell["cell_id"]] = resolve_accepted_result(cell)
    if len(resolved) != 180:
        raise ValueError(f"accepted result mapping coverage is {len(resolved)}, expected 180")
    return resolved


__all__ = ["direct_transfer_reference_input", "resolve_accepted_result", "resolve_all_accepted_results"]
