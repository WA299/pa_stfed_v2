"""Safe, explicit checkpoint export and validation for the frozen TEST protocol.

Checkpoint tensors are stored in ``.pt`` files as a plain state dictionary and
identity/provenance is stored in an adjacent JSON sidecar.  The sidecar is
deliberately separate so loading never executes arbitrary pickled objects.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SEEDS = (42, 123, 2026)
HISTORY_FRACTIONS = (0.25, 0.50)
PUBLIC_METHODS = ("local", "fedavg", "fedprox", "fedper", "fedfomo_style", "btd_fl_direct_transfer")
REFERENCE_TARGETS = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
INDUSTRIAL_TARGET = "norway_industrial_mvlv"
FROZEN_MODEL_CLASS = "PUCRSTAttnV2ConditionalUtility"
FROZEN_FEATURE_INDICES = (0, 2, 3, 4, 5, 6)
TOPOLOGY_BUFFERS = (
    "load_bus_mask", "utility_edge_index", "physical_relation_features",
    "utility_prior", "selected_utility",
)
FROZEN_COMMITS = {
    "scientific_method": "1681741779df301dcb64587090ce262d2ad1d443",
    "accepted_results": "e8dd4c3fbc4936175d3e9f0fee5c7c06d6303d3b",
    "paper_ready_reporting": "58dadfa088b78406d8de208faa2bff43a509f49c",
}


def canonical_method(domain: str, method: str) -> str:
    if method != "local":
        return method
    if domain == "reference_grid":
        return "scarce_local"
    if domain == "industrial_external":
        return "industrial_scarce_local"
    raise ValueError(f"unknown checkpoint domain: {domain}")


def matrix_cells() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction in HISTORY_FRACTIONS:
        for seed in SEEDS:
            for target in REFERENCE_TARGETS:
                for method in PUBLIC_METHODS:
                    rows.append(_cell("reference_grid", target, seed, fraction, method))
            rows.extend(_cell("industrial_external", INDUSTRIAL_TARGET, seed, fraction, method) for method in PUBLIC_METHODS)
    return rows


def _cell(domain: str, target: str, seed: int, fraction: float, method: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "target": target,
        "seed": int(seed),
        "history_fraction": float(fraction),
        "method": method,
        "canonical_method": canonical_method(domain, method),
        "cell_id": f"{domain}|{target}|seed{seed}|{int(fraction * 100)}pct|{method}",
    }


def checkpoint_path(root: Path, cell: Mapping[str, Any]) -> Path:
    fraction = int(round(float(cell["history_fraction"]) * 100))
    return root / str(cell["domain"]) / str(cell["target"]) / f"seed{int(cell['seed'])}" / f"{fraction}pct" / f"{cell['method']}.pt"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_state_dict(state_dict: Mapping[str, Any]) -> None:
    import torch

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint must contain tensor values only")
        if not torch.isfinite(value.detach().float()).all().item():
            raise ValueError(f"checkpoint contains non-finite tensor: {name}")


def export_checkpoint(path: Path, state_dict: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a tensor-only checkpoint plus validated sidecar."""
    import torch

    _validate_state_dict(state_dict)
    required = ("method", "domain", "target", "seed", "history_fraction", "model_class", "test_accessed")
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"checkpoint metadata missing required fields: {missing}")
    if bool(metadata["test_accessed"]):
        raise ValueError("checkpoint export cannot access TEST")
    cpu_state = {name: value.detach().cpu().clone() for name, value in state_dict.items()}
    buffer = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
    torch.save(cpu_state, buffer)
    buffer.seek(0)
    payload = buffer.read()
    _atomic_bytes(path, payload)
    digest = sha256_file(path)
    sidecar = {
        "schema_version": 1,
        **_jsonable(dict(metadata)),
        "checkpoint_path": path.as_posix(),
        "sha256": digest,
        "file_size_bytes": int(path.stat().st_size),
        "state_dict_tensor_names": sorted(cpu_state),
        "state_dict_only": True,
        "deserialized_untrusted_objects": False,
    }
    sidecar_path = path.with_suffix(path.suffix + ".json")
    _atomic_bytes(sidecar_path, (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return sidecar


def load_checkpoint(path: Path, expected: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Safely load a tensor-only checkpoint and verify its sidecar/hash."""
    import torch

    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"checkpoint and sidecar are both required: {path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("sha256") != sha256_file(path) or metadata.get("file_size_bytes") != path.stat().st_size:
        raise ValueError(f"checkpoint hash/size mismatch: {path}")
    if metadata.get("state_dict_only") is not True or metadata.get("test_accessed") is not False:
        raise ValueError("checkpoint sidecar violates safe TEST guardrails")
    if expected:
        for key, value in expected.items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(f"checkpoint identity mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    _validate_state_dict(state)
    if sorted(state) != sorted(metadata.get("state_dict_tensor_names", ())):
        raise ValueError("checkpoint tensor-name manifest mismatch")
    return dict(state), metadata


def build_metadata(*, cell: Mapping[str, Any], model: Any, grid: Any, scaler: Any,
                   graph_metadata: Mapping[str, Any], source_artifact: str,
                   frozen_commits: Mapping[str, str], selection: Mapping[str, Any] | None = None,
                   selected_epoch: int | None = None, final_round: int | None = None,
                   environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Construct the complete identity contract without reading TEST data."""
    import torch
    from code.federated.parameter_groups import parameter_groups

    buffers = dict(model.named_buffers())
    buffer_shapes = {name: list(buffers[name].shape) for name in TOPOLOGY_BUFFERS if name in buffers}
    return {
        **dict(cell),
        "method": str(cell["method"]),
        "canonical_method": str(cell["canonical_method"]),
        "model_class": FROZEN_MODEL_CLASS,
        "constructor": {"num_nodes": int(grid.num_nodes), "feature_indices": list(FROZEN_FEATURE_INDICES), "history": 168, "horizon": 1},
        "state_dict_complete": True,
        "local_graph_topology_identity": {"graph_metadata": _jsonable(dict(graph_metadata)), "buffer_shapes": buffer_shapes},
        "target_scaler": {
            "fit_start_index": int(scaler.fit_start_index), "fit_end_index": int(scaler.fit_end_index),
            "fit_timestamp_end": str(scaler.fit_timestamp_end), "fit_split": str(getattr(scaler, "fit_split", "unknown")),
            "fit_load_bus_count": int(getattr(scaler, "fit_load_bus_count", int(np.asarray(grid.load_bus_mask, dtype=bool).sum()))),
            "fit_value_count": int(getattr(scaler, "fit_value_count", 0)),
            "p_mean": _jsonable(getattr(scaler, "p_mean_", None)), "p_scale": _jsonable(getattr(scaler, "p_scale_", None)),
            "q_mean": _jsonable(getattr(scaler, "q_mean_", None)), "q_scale": _jsonable(getattr(scaler, "q_scale_", None)),
        },
        "feature_schema": {"name": "p_calendar", "indices": list(FROZEN_FEATURE_INDICES), "q_placeholder_used": False},
        "history_length": 168,
        "forecast_horizon": 1,
        "target_node_count": int(grid.num_nodes),
        "target_node_order": [str(item) for item in getattr(grid, "node_ids", range(grid.num_nodes))],
        "target_load_bus_mask": np.asarray(grid.load_bus_mask, dtype=bool).tolist(),
        "parameter_groups": {name: list(values) for name, values in parameter_groups(model).items()},
        "source_artifact": str(source_artifact),
        "frozen_commits": dict(frozen_commits or FROZEN_COMMITS),
        "selected_epoch": selected_epoch,
        "final_communication_round": final_round,
        "environment": dict(environment or {"python": os.sys.version, "pytorch": torch.__version__, "cuda": torch.version.cuda}),
        "test_accessed": False,
        "test_evaluated": False,
        "industrial_test_evaluated": False,
        "reference_grid_tests_evaluated": False,
        **dict(selection or {}),
    }


def validate_matrix_identity(metadata: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
    for key in ("method", "domain", "target", "seed", "history_fraction", "canonical_method"):
        if metadata.get(key) != cell.get(key):
            raise ValueError(f"checkpoint {key} does not match matrix cell")


def compare_metric_trees(observed: Mapping[str, Any], expected: Mapping[str, Any],
                         *, absolute_tolerance: float = 1e-10,
                         relative_tolerance: float = 1e-6) -> list[dict[str, Any]]:
    """Compare finite scalar metric trees and return every unexplained mismatch."""
    discrepancies: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    discrepancies.append({"json_key_path": f"{path}.{key}", "reason": "missing_or_extra"})
                else:
                    walk(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
            if not (math.isfinite(float(left)) and math.isfinite(float(right))):
                discrepancies.append({"json_key_path": path, "reason": "non_finite", "observed": left, "expected": right})
                return
            absolute = abs(float(left) - float(right))
            relative = absolute / max(abs(float(right)), 1e-12)
            if absolute > absolute_tolerance + relative_tolerance * abs(float(right)):
                discrepancies.append({"json_key_path": path, "reason": "value_mismatch", "observed": float(left), "expected": float(right), "absolute_discrepancy": absolute, "relative_discrepancy": relative})
            return
        if left != right:
            discrepancies.append({"json_key_path": path, "reason": "value_mismatch", "observed": left, "expected": right})

    walk(observed, expected, "$")
    return discrepancies


def export_model_checkpoint(*, root: Path, cell: Mapping[str, Any], model: Any, grid: Any,
                            scaler: Any, graph_metadata: Mapping[str, Any], source_artifact: str,
                            frozen_commits: Mapping[str, str], selection: Mapping[str, Any] | None = None,
                            selected_epoch: int | None = None, final_round: int | None = None,
                            environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Export an already-selected model without changing training dynamics."""
    path = checkpoint_path(root, cell)
    metadata = build_metadata(
        cell=cell, model=model, grid=grid, scaler=scaler, graph_metadata=graph_metadata,
        source_artifact=source_artifact, frozen_commits=frozen_commits, selection=selection,
        selected_epoch=selected_epoch, final_round=final_round, environment=environment,
    )
    return export_checkpoint(path, model.state_dict(), metadata)


__all__ = [
    "SEEDS", "HISTORY_FRACTIONS", "PUBLIC_METHODS", "REFERENCE_TARGETS", "INDUSTRIAL_TARGET",
    "canonical_method", "matrix_cells", "checkpoint_path", "sha256_file", "export_checkpoint",
    "load_checkpoint", "build_metadata", "validate_matrix_identity", "compare_metric_trees", "export_model_checkpoint", "FROZEN_FEATURE_INDICES", "FROZEN_COMMITS",
]
