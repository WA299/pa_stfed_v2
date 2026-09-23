"""Conditional-utility graph variant of the frozen PUC-RSTAttn V2 model.

Only graph construction differs from :mod:`puc_rstattn_v2`.  The neural
module is the frozen V2 implementation, including its target-wise utility
normalization and aggregate-preserving residual.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from code.audits.strong_temporal_conditional_utility_audit import (
    PAIR_FEATURE_DIM,
    RIDGE_ALPHA,
    SELF_FEATURE_DIM,
    SOURCE_LAGS,
    SELF_HISTORY_LAGS,
    chronological_split,
    compute_conditional_utilities,
    load_bus_indices,
    select_conditional_neighbors,
)
from code.models.puc_rstattn import (
    ANCHOR_LOSS_WEIGHT,
    UtilityGraph,
    build_relation_features,
    formal_train_target_indices,
    normalized_distance_matrices,
)
from code.models.puc_rstattn_v2 import PUCRSTAttnV2

CONDITIONAL_UTILITY_SEMANTICS = "strong_temporal_conditional_predictive_utility"
UTILITY_FIT_SAMPLES = 3578
UTILITY_SELECTION_SAMPLES = 1193
UTILITY_AUDIT_SAMPLES = 1193
UTILITY_TOP_K = 3


def conditional_graph_split(grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed chronological graph fit/selection/audit split."""
    return chronological_split(formal_train_target_indices(grid))


def build_conditional_utility_graph(grid: Any) -> UtilityGraph:
    """Build positive directed conditional-utility edges from formal train only.

    The audit segment is deliberately returned by ``conditional_graph_split``
    but is never passed to utility fitting or graph selection.
    """
    fit_indices, selection_indices, audit_indices = conditional_graph_split(grid)
    loads = load_bus_indices(grid)
    utilities, _records = compute_conditional_utilities(
        grid, fit_indices, selection_indices, loads
    )
    neighbors = select_conditional_neighbors(utilities, loads, top_k=UTILITY_TOP_K)

    sources: list[int] = []
    targets: list[int] = []
    for target in loads:
        for source, _utility in neighbors[int(target)]:
            sources.append(int(source))
            targets.append(int(target))
    edge_index = (
        np.asarray([sources, targets], dtype=np.int64)
        if sources
        else np.empty((2, 0), dtype=np.int64)
    )
    relation_features = build_relation_features(grid, edge_index, utilities, loads)
    positive = [float(value) for value in utilities.values() if float(value) > 0.0]
    counts = {
        str(k): int(sum(len(neighbors[int(target)]) == k for target in loads))
        for k in range(UTILITY_TOP_K + 1)
    }
    diagnostics = {
        "positive_edge_count": int(len(sources)),
        "targets_with_selected_neighbors": counts,
        "mean_positive_utility": float(np.mean(positive)) if positive else 0.0,
        "utility_semantics": CONDITIONAL_UTILITY_SEMANTICS,
        "utility_direction": "source_to_target",
        "utility_top_k": UTILITY_TOP_K,
        "utility_positive_only": True,
        "utility_graph_fit_samples": int(len(fit_indices)),
        "utility_graph_selection_samples": int(len(selection_indices)),
        "utility_graph_audit_samples": int(len(audit_indices)),
        "utility_graph_uses_audit": False,
        "utility_graph_uses_validation": False,
        "utility_graph_uses_test": False,
        "self_feature_dim": SELF_FEATURE_DIM,
        "pair_feature_dim": PAIR_FEATURE_DIM,
        "self_history_lags": list(SELF_HISTORY_LAGS),
        "source_lags": list(SOURCE_LAGS),
        "ridge_alpha": RIDGE_ALPHA,
        "causal_claim": False,
        "graph_audit_indices": audit_indices.copy(),
    }
    return UtilityGraph(edge_index, relation_features, utilities, neighbors, loads, diagnostics)


class PUCRSTAttnV2ConditionalUtility(PUCRSTAttnV2):
    """Frozen V2 neural architecture with a conditional-utility graph."""


PUCRSTAttnV2ConditionalUtilityModel = PUCRSTAttnV2ConditionalUtility
# Short aliases make the diagnostic variant convenient to import without
# changing the frozen V2 module or its public graph builder.
PUCRSTAttnV2CU = PUCRSTAttnV2ConditionalUtility
build_utility_graph = build_conditional_utility_graph

__all__ = [
    "ANCHOR_LOSS_WEIGHT",
    "CONDITIONAL_UTILITY_SEMANTICS",
    "PAIR_FEATURE_DIM",
    "RIDGE_ALPHA",
    "SELF_FEATURE_DIM",
    "SELF_HISTORY_LAGS",
    "SOURCE_LAGS",
    "UTILITY_AUDIT_SAMPLES",
    "UTILITY_FIT_SAMPLES",
    "UTILITY_SELECTION_SAMPLES",
    "UTILITY_TOP_K",
    "PUCRSTAttnV2ConditionalUtility",
    "PUCRSTAttnV2ConditionalUtilityModel",
    "PUCRSTAttnV2CU",
    "build_conditional_utility_graph",
    "build_utility_graph",
    "conditional_graph_split",
]
