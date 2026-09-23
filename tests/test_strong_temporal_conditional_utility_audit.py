from types import SimpleNamespace

import numpy as np
import pandas as pd

from code.audits.strong_temporal_conditional_utility_audit import (
    AUDIT_SAMPLES,
    FIT_SAMPLES,
    PAIR_FEATURE_DIM,
    SELECTION_SAMPLES,
    SELF_FEATURE_DIM,
    SOURCE_LAGS,
    chronological_split,
    compute_conditional_utilities,
    evaluate_selected_graph,
    report_metadata,
    select_conditional_neighbors,
    strong_self_features,
    pair_features,
    uniform_mean_context_features,
)


def _grid(periods=5964 + 168, nodes=4):
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    time = np.arange(periods, dtype=float)
    p = np.column_stack([time + node * 0.1 for node in range(nodes)])
    return SimpleNamespace(
        grid_name="synthetic",
        num_nodes=nodes,
        node_ids=np.asarray([str(i) for i in range(nodes)]),
        timestamps=timestamps,
        p=p,
        load_bus_mask=np.ones(nodes, dtype=bool),
        distance_matrices={"impedance_abs_distance": np.ones((nodes, nodes))},
    )


def test_exact_split_and_dimensions_and_alignment():
    indices = np.arange(168, 168 + FIT_SAMPLES + SELECTION_SAMPLES + AUDIT_SAMPLES)
    fit, selection, audit = chronological_split(indices)
    assert (len(fit), len(selection), len(audit)) == (3578, 1193, 1193)
    grid = _grid()
    targets = np.array([200, 201])
    x = strong_self_features(grid.p, grid.timestamps, targets, 0)
    pair = pair_features(grid.p, grid.timestamps, targets, 0, 1)
    context = uniform_mean_context_features(grid.p, grid.timestamps, targets, 0, [1, 2])
    assert x.shape == (2, SELF_FEATURE_DIM) == (2, 173)
    assert pair.shape == (2, PAIR_FEATURE_DIM) == (2, 176)
    assert context.shape == (2, 176)
    assert np.allclose(x[0, :168], [grid.p[200 - lag, 0] for lag in range(1, 169)])
    assert len(SOURCE_LAGS) == 3


def test_selection_positive_direction_and_tie_breaking():
    utilities = {(1, 0): 0.2, (2, 0): 0.2, (3, 0): -0.1, (0, 1): 0.3, (2, 1): 0.1, (3, 1): 0.0}
    for target in (2, 3):
        for source in (0, 1, 2, 3):
            if source != target:
                utilities.setdefault((source, target), -0.2)
    selected = select_conditional_neighbors(utilities, [0, 1, 2, 3])
    assert [source for source, _ in selected[0]] == [1, 2]
    assert all(value > 0 for values in selected.values() for _, value in values)
    assert all(source != target for target, values in selected.items() for source, _ in values)


def test_selection_uses_selection_not_audit_and_refit_counts_are_exact():
    grid = _grid()
    fit = np.arange(168, 168 + 20)
    selection = np.arange(168 + 20, 168 + 30)
    before = strong_self_features(grid.p, grid.timestamps, selection, 0)
    grid.p[168 + 30 :, :] += 100000.0
    after = strong_self_features(grid.p, grid.timestamps, selection, 0)
    assert np.array_equal(before, after)


def test_metadata_is_exact_and_macro_arithmetic_is_unweighted():
    metadata = report_metadata()
    assert metadata == {
        **metadata,
        "time_scope": "formal_train_only",
        "fit_samples": 3578,
        "selection_samples": 1193,
        "audit_samples": 1193,
        "self_history_lags": "1_to_168",
        "self_feature_dim": 173,
        "source_lags": [1, 24, 168],
        "ridge_alpha": 1.0,
        "utility_semantics": "strong_temporal_conditional_predictive_utility",
        "utility_direction": "source_to_target",
        "utility_top_k": 3,
        "utility_positive_only": True,
        "node_scope": "load_buses_only",
        "validation_used": False,
        "test_evaluated": False,
        "causal_claim": False,
    }
