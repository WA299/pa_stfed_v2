"""Explicit trainable parameter groups for PUC-RSTAttn V2-CU federation."""
from __future__ import annotations

from typing import Any

TEMPORAL_PREFIXES = (
    "gru.",
    "gru_head.",
    "temporal_query.",
    "temporal_key.",
    "temporal_score.",
    "temporal_correction.",
    "temporal_gate.",
)
SPATIAL_PREFIXES = (
    "q_projection.",
    "k_projection.",
    "v_projection.",
    "physical_encoder.",
    "spatial_correction.",
    "spatial_gate.",
)
SPATIAL_SCALARS = ("dynamic_scale",)
PARAMETER_GROUPS = ("temporal", "spatial")
TOPOLOGY_BUFFER_NAMES = frozenset(
    {
        "load_bus_mask",
        "utility_edge_index",
        "physical_relation_features",
        "utility_prior",
        "selected_utility",
    }
)


def parameter_groups(model: Any) -> dict[str, tuple[str, ...]]:
    """Partition every trainable named parameter into exactly one group."""
    named_parameters = dict(model.named_parameters())
    temporal = tuple(sorted(name for name in named_parameters if name.startswith(TEMPORAL_PREFIXES)))
    spatial = tuple(
        sorted(
            name
            for name in named_parameters
            if name.startswith(SPATIAL_PREFIXES) or name in SPATIAL_SCALARS
        )
    )
    temporal_set, spatial_set = set(temporal), set(spatial)
    all_trainable = {name for name, parameter in named_parameters.items() if parameter.requires_grad}
    if temporal_set & spatial_set:
        raise ValueError(f"parameter groups overlap: {sorted(temporal_set & spatial_set)}")
    if temporal_set | spatial_set != all_trainable:
        missing = sorted(all_trainable - (temporal_set | spatial_set))
        unexpected = sorted((temporal_set | spatial_set) - all_trainable)
        raise ValueError(f"parameter groups do not cover trainables; missing={missing}, unexpected={unexpected}")
    buffers = dict(model.named_buffers())
    missing_buffers = sorted(TOPOLOGY_BUFFER_NAMES - buffers.keys())
    if missing_buffers:
        raise ValueError(f"required local topology buffers are missing: {missing_buffers}")
    if (temporal_set | spatial_set) & buffers.keys():
        raise ValueError("buffers must never be included in trainable parameter groups")
    return {"temporal": temporal, "spatial": spatial}
