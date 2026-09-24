"""Named-parameter-only FedAvg operations and sharing-mode definitions."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .parameter_groups import PARAMETER_GROUPS

SHARING_MODES = (
    "local_only",
    "fedavg_all",
    "temporal_shared_spatial_local",
    "temporal_local_spatial_shared",
)
ALGORITHMS = ("standard", "fedprox", "fedper")


def sharing_groups(mode: str) -> tuple[str, ...]:
    if mode not in SHARING_MODES:
        raise ValueError(f"unknown sharing mode {mode!r}; choose from {SHARING_MODES}")
    return {
        "local_only": (),
        "fedavg_all": PARAMETER_GROUPS,
        "temporal_shared_spatial_local": ("temporal",),
        "temporal_local_spatial_shared": ("spatial",),
    }[mode]


def normalize_sample_weights(sample_counts: Mapping[str, int]) -> dict[str, float]:
    if not sample_counts or any(int(count) <= 0 for count in sample_counts.values()):
        raise ValueError("every client must have a positive local sample count")
    total = float(sum(int(count) for count in sample_counts.values()))
    return {name: int(count) / total for name, count in sample_counts.items()}


def aggregate_named_parameters(
    models: Mapping[str, Any],
    parameter_names: Iterable[str],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """FedAvg selected named parameters and broadcast them to every model.

    No state_dict is read or written; buffers and all unselected parameters
    remain client-local by construction.
    """
    if len(models) < 2 or set(models) != set(weights):
        raise ValueError("models and weights must name the same two or more clients")
    weight_values = np.asarray([float(weights[name]) for name in models], dtype=float)
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0) or not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("aggregation weights must be finite, non-negative, and sum to one")
    parameter_maps = {name: dict(model.named_parameters()) for name, model in models.items()}
    aggregated: dict[str, Any] = {}
    import torch

    with torch.no_grad():
        for parameter_name in parameter_names:
            if any(parameter_name not in values for values in parameter_maps.values()):
                raise ValueError(f"selected parameter {parameter_name!r} is absent from a client")
            parameters = [parameter_maps[name][parameter_name] for name in models]
            shapes = {tuple(parameter.shape) for parameter in parameters}
            if len(shapes) != 1:
                raise ValueError(f"parameter shape mismatch for {parameter_name}: {shapes}")
            reference = parameters[0]
            average = torch.zeros_like(reference)
            for client_name, parameter in zip(models, parameters):
                average.add_(parameter, alpha=float(weights[client_name]))
            for parameter in parameters:
                parameter.copy_(average.to(device=parameter.device, dtype=parameter.dtype))
            aggregated[parameter_name] = average.detach().cpu().clone()
    return aggregated
