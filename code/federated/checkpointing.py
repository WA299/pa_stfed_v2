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
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from code.federated.parameter_groups import (
    FROZEN_TEMPORAL_PARAMETER_NAMES,
    parameter_groups,
)

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
FROZEN_PARAMETER_SHAPES = {
    "gru.weight_ih_l0": [96, 6], "gru.weight_hh_l0": [96, 32],
    "gru.bias_ih_l0": [96], "gru.bias_hh_l0": [96],
    "gru_head.weight": [1, 32], "gru_head.bias": [1],
    "temporal_query.weight": [32, 32], "temporal_query.bias": [32],
    "temporal_key.weight": [32, 32], "temporal_key.bias": [32],
    "temporal_score.weight": [1, 32],
    "temporal_correction.0.weight": [32, 96], "temporal_correction.0.bias": [32],
    "temporal_correction.2.weight": [1, 32], "temporal_correction.2.bias": [1],
    "temporal_gate.0.weight": [32, 96], "temporal_gate.0.bias": [32],
    "temporal_gate.2.weight": [1, 32], "temporal_gate.2.bias": [1],
    "q_projection.weight": [32, 32], "q_projection.bias": [32],
    "k_projection.weight": [32, 32], "k_projection.bias": [32],
    "v_projection.weight": [32, 32], "v_projection.bias": [32],
    "dynamic_scale": [],
    "physical_encoder.0.weight": [16, 3], "physical_encoder.0.bias": [16],
    "physical_encoder.2.weight": [1, 16], "physical_encoder.2.bias": [1],
    "spatial_correction.0.weight": [32, 64], "spatial_correction.0.bias": [32],
    "spatial_correction.2.weight": [1, 32], "spatial_correction.2.bias": [1],
    "spatial_gate.0.weight": [32, 99], "spatial_gate.0.bias": [32],
    "spatial_gate.2.weight": [1, 32], "spatial_gate.2.bias": [1],
}
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


def _current_git_commit() -> str | None:
    """Best-effort identity of the code that emitted a future checkpoint."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip() or None
    except Exception:
        return None


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


def _tensor_specs(state_dict: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Deterministic shape/dtype inventory for the tensor-only state dict."""
    return {
        str(name): {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(state_dict.items())
    }


def _tensor_hash(value: Any) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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
        "schema_version": 2,
        **_jsonable(dict(metadata)),
        "checkpoint_path": path.as_posix(),
        "sha256": digest,
        "file_size_bytes": int(path.stat().st_size),
        "state_dict_tensor_names": sorted(cpu_state),
        "state_dict_tensor_specs": _tensor_specs(cpu_state),
        "state_dict_only": True,
        "deserialized_untrusted_objects": False,
    }
    sidecar_path = path.with_suffix(path.suffix + ".json")
    _atomic_bytes(sidecar_path, (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return sidecar


def load_checkpoint(path: Path, expected: Mapping[str, Any] | None = None, *, production: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Safely load a tensor-only checkpoint and verify its sidecar/hash."""
    import torch

    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"checkpoint and sidecar are both required: {path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("sha256") != sha256_file(path) or metadata.get("file_size_bytes") != path.stat().st_size:
        raise ValueError(f"checkpoint hash/size mismatch: {path}")
    if metadata.get("checkpoint_path") is not None and Path(str(metadata["checkpoint_path"])).resolve() != path.resolve():
        raise ValueError("checkpoint sidecar path mismatch")
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
    if "state_dict_tensor_specs" in metadata and metadata["state_dict_tensor_specs"] != _tensor_specs(state):
        raise ValueError("checkpoint tensor shape/dtype manifest mismatch")
    if production:
        validate_production_checkpoint_metadata(metadata, state)
    return dict(state), metadata


def build_metadata(*, cell: Mapping[str, Any], model: Any, grid: Any, scaler: Any,
                   graph_metadata: Mapping[str, Any], source_artifact: str,
                   frozen_commits: Mapping[str, str], selection: Mapping[str, Any] | None = None,
                   selected_epoch: int | None = None, final_round: int | None = None,
                   epochs_run: int | None = None,
                   environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Construct the complete identity contract without reading TEST data."""
    import torch
    if model.__class__.__name__ != FROZEN_MODEL_CLASS:
        raise ValueError(
            "production checkpoint export requires "
            f"{FROZEN_MODEL_CLASS}, got {model.__class__.__name__}"
        )
    buffers = dict(model.named_buffers())
    buffer_shapes = {name: list(buffers[name].shape) for name in TOPOLOGY_BUFFERS if name in buffers}
    buffer_specs = {
        name: {"shape": list(buffers[name].shape), "dtype": str(buffers[name].dtype)}
        for name in sorted(TOPOLOGY_BUFFERS) if name in buffers
    }
    buffer_hashes = {
        name: _tensor_hash(buffers[name])
        for name in sorted(TOPOLOGY_BUFFERS) if name in buffers
    }
    selection_data = dict(selection or {})
    public_method = str(cell["method"])
    selection_point = (
        "best_calibration_full_model" if public_method == "local" else
        "final_best_full_model" if public_method == "btd_fl_direct_transfer" else
        "fixed_final_round_10"
    )
    source_path = Path(str(source_artifact))
    source_identity = {
        "path": str(source_artifact),
        "exists": bool(source_path.is_file()),
        "sha256": sha256_file(source_path) if source_path.is_file() else None,
    }
    constructor_config = {
        "num_nodes": int(grid.num_nodes), "feature_indices": list(FROZEN_FEATURE_INDICES),
        "history": 168, "horizon": 1, "hidden_size": 32, "input_size": 6,
        "utility_edge_shape": list(buffers["utility_edge_index"].shape) if "utility_edge_index" in buffers else None,
        "relation_feature_shape": list(buffers["physical_relation_features"].shape) if "physical_relation_features" in buffers else None,
    }
    commits = dict(FROZEN_COMMITS)
    commits.update(dict(frozen_commits or {}))
    commits.setdefault("reconstruction_code", _current_git_commit())
    if not commits.get("reconstruction_code"):
        raise ValueError("reconstruction code commit cannot be identified")
    accepted_identity: dict[str, Any] = {}
    try:
        from code.federated.accepted_results import resolve_accepted_result
        accepted = resolve_accepted_result(cell)
        accepted_identity = {
            "artifact": str(accepted["source_artifact"].resolve()),
            "json_keys": dict(accepted["source_json_key"]),
        }
        # The accepted per-cell source is the authoritative parity reference.
        source_identity = {
            "path": accepted_identity["artifact"],
            "exists": True,
            "sha256": sha256_file(Path(accepted_identity["artifact"])),
        }
        source_artifact = accepted_identity["artifact"]
    except Exception as exc:
        raise ValueError(f"accepted per-cell result mapping is required for production export: {exc}") from exc
    source_path = Path(str(source_artifact))
    return {
        **dict(cell),
        "method": str(cell["method"]),
        "canonical_method": str(cell["canonical_method"]),
        "model_class": FROZEN_MODEL_CLASS,
        "constructor": constructor_config,
        "constructor_identity": {"class": FROZEN_MODEL_CLASS, "config": constructor_config},
        "state_dict_complete": True,
        "local_graph_topology_identity": {"graph_metadata": _jsonable(dict(graph_metadata)), "buffer_shapes": buffer_shapes, "buffer_specs": buffer_specs, "buffer_hashes": buffer_hashes, "buffer_names": sorted(buffer_specs)},
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
        "source_artifact": str(source_path.resolve()),
        "source_artifact_identity": {**source_identity, "path": str(source_path.resolve())},
        "accepted_result_artifact": accepted_identity.get("artifact"),
        "accepted_result_json_keys": accepted_identity.get("json_keys"),
        "frozen_commits": commits,
        "selected_epoch": selected_epoch,
        "epochs_run": epochs_run if epochs_run is not None else selection_data.pop("epochs_run", None),
        "selected_training_state_identity": {
            "selection_point": selection_data.pop("selection_point", selection_point),
            "selected_epoch": selected_epoch,
            "final_communication_round": final_round,
        },
        "final_communication_round": final_round,
        "environment": dict(environment or {"python": os.sys.version, "pytorch": torch.__version__, "cuda": torch.version.cuda}),
        "test_accessed": False,
        "test_evaluated": False,
        "industrial_test_evaluated": False,
        "reference_grid_tests_evaluated": False,
        **selection_data,
    }


def validate_production_checkpoint_metadata(
    metadata: Mapping[str, Any], state_dict: Mapping[str, Any] | None = None
) -> None:
    """Validate the non-negotiable production checkpoint contract.

    This deliberately does not instantiate or execute a model.  It validates
    the identity and provenance recorded beside the tensor-only state dict;
    model-constructor/state-shape parity is checked by the reconstruction
    runner when a production model is available.
    """
    required = (
        "schema_version", "model_class", "constructor_identity", "constructor",
        "state_dict_complete", "local_graph_topology_identity", "target_scaler",
        "feature_schema", "history_length", "forecast_horizon", "target_node_count",
        "target_node_order", "target_load_bus_mask", "parameter_groups",
        "source_artifact", "source_artifact_identity", "frozen_commits",
        "selected_training_state_identity", "selected_epoch", "epochs_run",
        "test_accessed", "test_evaluated", "industrial_test_evaluated",
        "reference_grid_tests_evaluated",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"production checkpoint sidecar missing fields: {missing}")
    if metadata.get("model_class") != FROZEN_MODEL_CLASS:
        raise ValueError("generic/non-production model checkpoint rejected")
    if metadata.get("schema_version") != 2:
        raise ValueError("unsupported production checkpoint sidecar schema")
    identity = metadata.get("constructor_identity")
    expected_config = metadata.get("constructor")
    if not isinstance(identity, Mapping) or identity.get("class") != FROZEN_MODEL_CLASS or identity.get("config") != expected_config:
        raise ValueError("constructor identity does not identify the frozen production model")
    if not isinstance(expected_config, Mapping) or expected_config.get("feature_indices") != list(FROZEN_FEATURE_INDICES) or expected_config.get("history") != 168 or expected_config.get("horizon") != 1 or expected_config.get("hidden_size") != 32 or expected_config.get("input_size") != 6:
        raise ValueError("frozen constructor configuration is incomplete")
    if metadata.get("state_dict_complete") is not True:
        raise ValueError("checkpoint state_dict is not marked complete")
    if metadata.get("feature_schema", {}).get("indices") != list(FROZEN_FEATURE_INDICES):
        raise ValueError("checkpoint feature schema is not the frozen P+calendar contract")
    if metadata.get("history_length") != 168 or metadata.get("forecast_horizon") != 1:
        raise ValueError("checkpoint history/horizon does not match frozen contract")
    if not isinstance(metadata.get("target_node_order"), list) or not isinstance(metadata.get("target_load_bus_mask"), list) or len(metadata["target_node_order"]) != int(metadata.get("target_node_count", -1)) or len(metadata["target_load_bus_mask"]) != int(metadata.get("target_node_count", -1)):
        raise ValueError("target node order/mask identity is incomplete")
    if metadata.get("test_accessed") is not False or metadata.get("test_evaluated") is not False:
        raise ValueError("checkpoint TEST guardrails are violated")
    if metadata.get("industrial_test_evaluated") is not False or metadata.get("reference_grid_tests_evaluated") is not False:
        raise ValueError("checkpoint domain TEST guardrails are violated")
    topology = metadata.get("local_graph_topology_identity")
    if not isinstance(topology, Mapping) or set(topology.get("buffer_names", ())) != set(TOPOLOGY_BUFFERS):
        raise ValueError("target-local graph/topology identity is incomplete")
    if not isinstance(metadata.get("target_scaler"), Mapping):
        raise ValueError("target FIT scaler provenance is missing")
    scaler = metadata["target_scaler"]
    for key in ("fit_start_index", "fit_end_index", "fit_split", "fit_load_bus_count", "fit_value_count", "p_mean", "p_scale", "q_mean", "q_scale"):
        if key not in scaler:
            raise ValueError(f"target scaler provenance missing {key}")
    if scaler.get("p_mean") is None or scaler.get("p_scale") is None:
        raise ValueError("target scaler statistics are incomplete")
    if scaler.get("fit_split") not in ("train", "train_fit_only", "scarce_fit"):
        raise ValueError("target scaler is not recorded as a FIT/train-only scaler")
    source_identity = metadata.get("source_artifact_identity")
    if not isinstance(source_identity, Mapping) or source_identity.get("path") != metadata.get("source_artifact") or source_identity.get("exists") is not True or not source_identity.get("sha256"):
        raise ValueError("source artifact identity is incomplete")
    source_path = Path(str(source_identity["path"]))
    if not source_path.is_file() or sha256_file(source_path) != source_identity.get("sha256"):
        raise ValueError("source artifact identity hash cannot be verified")
    commits = metadata.get("frozen_commits")
    if not isinstance(commits, Mapping) or any(commits.get(key) != expected for key, expected in FROZEN_COMMITS.items()) or not isinstance(commits.get("reconstruction_code"), str) or not commits.get("reconstruction_code"):
        raise ValueError("frozen commit identities are incomplete")
    selected = metadata.get("selected_training_state_identity")
    if not isinstance(selected, Mapping) or not selected.get("selection_point"):
        raise ValueError("selected training-state identity is incomplete")
    method = metadata.get("method")
    selection_point = selected.get("selection_point")
    if method == "local" and selection_point != "best_calibration_full_model":
        raise ValueError("local checkpoint is not marked as calibration-selected")
    if method == "btd_fl_direct_transfer" and selection_point != "final_best_full_model":
        raise ValueError("BTD checkpoint is not marked as final selected full state")
    if method in {"fedavg", "fedprox", "fedper", "fedfomo_style"}:
        if selection_point != "fixed_final_round_10" or metadata.get("final_communication_round") != 10:
            raise ValueError("federated checkpoint is not the frozen final round-10 state")
    for epoch_key in ("selected_epoch", "epochs_run"):
        value = metadata.get(epoch_key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f"invalid {epoch_key} metadata")
    if metadata.get("selected_epoch") is not None and metadata.get("epochs_run") is not None and metadata["selected_epoch"] > metadata["epochs_run"]:
        raise ValueError("selected_epoch cannot exceed epochs_run")
    if metadata.get("method") == "btd_fl_direct_transfer":
        names = metadata.get("transferred_parameter_names", [])
        if metadata.get("zero_transfer_fallback"):
            if names:
                raise ValueError("BTD fallback must have an empty transferred parameter list")
        elif set(names) != set(FROZEN_TEMPORAL_PARAMETER_NAMES):
            raise ValueError("BTD transfer is not the complete frozen temporal parameter group")
        if metadata.get("adapted_probe_states_used_for_final_initialization") is not False:
            raise ValueError("adapted probe state cannot initialize final BTD checkpoint")
    if state_dict is not None:
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("production checkpoint has no tensor state")
        names = metadata.get("state_dict_tensor_names")
        if names != sorted(state_dict):
            raise ValueError("production state tensor names are incomplete")
        specs = metadata.get("state_dict_tensor_specs")
        if not isinstance(specs, Mapping) or specs != _tensor_specs(state_dict):
            raise ValueError("production tensor shape/dtype inventory is incomplete")
        for name, shape in FROZEN_PARAMETER_SHAPES.items():
            if name not in state_dict or list(state_dict[name].shape) != shape:
                raise ValueError(f"production model tensor shape mismatch for {name}")
        hashes = topology.get("buffer_hashes")
        if not isinstance(hashes, Mapping) or set(hashes) != set(TOPOLOGY_BUFFERS):
            raise ValueError("target graph buffer hashes are incomplete")
        for name in TOPOLOGY_BUFFERS:
            if name not in state_dict or hashes[name] != _tensor_hash(state_dict[name]):
                raise ValueError(f"target graph buffer identity mismatch for {name}")
        groups = metadata.get("parameter_groups")
        temporal = set(groups.get("temporal", ())) if isinstance(groups, Mapping) else set()
        if not set(FROZEN_TEMPORAL_PARAMETER_NAMES).issubset(temporal):
            raise ValueError("production state does not expose the complete frozen temporal group")


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
                            epochs_run: int | None = None,
                            environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Export an already-selected model without changing training dynamics."""
    path = checkpoint_path(root, cell)
    metadata = build_metadata(
        cell=cell, model=model, grid=grid, scaler=scaler, graph_metadata=graph_metadata,
        source_artifact=source_artifact, frozen_commits=frozen_commits, selection=selection,
        selected_epoch=selected_epoch, final_round=final_round, epochs_run=epochs_run, environment=environment,
    )
    state = model.state_dict()
    metadata["schema_version"] = 2
    metadata["state_dict_tensor_names"] = sorted(state)
    metadata["state_dict_tensor_specs"] = _tensor_specs(state)
    validate_production_checkpoint_metadata(metadata, state)
    return export_checkpoint(path, state, metadata)


__all__ = [
    "SEEDS", "HISTORY_FRACTIONS", "PUBLIC_METHODS", "REFERENCE_TARGETS", "INDUSTRIAL_TARGET",
    "canonical_method", "matrix_cells", "checkpoint_path", "sha256_file", "export_checkpoint",
    "load_checkpoint", "build_metadata", "validate_production_checkpoint_metadata", "validate_matrix_identity", "compare_metric_trees", "export_model_checkpoint", "FROZEN_FEATURE_INDICES", "FROZEN_COMMITS",
]
