import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from code.audits.predictive_utility_graph_audit import (
    AUDIT_SAMPLES,
    FIT_SAMPLES,
    LAGS,
    SELECTION_SAMPLES,
    compute_pairwise_utilities,
    feature_matrix,
    fit_feature_scaler,
    formal_train_target_indices,
    graph_diagnostics,
    report_metadata,
    select_electrical_neighbors,
    select_utility_neighbors,
    signed_utility,
    split_train_targets,
)


def _grid(t=5964 + 168, n=5):
    timestamps = pd.date_range("2024-01-01", periods=t, freq="h")
    time = np.arange(t, dtype=float)
    p = np.column_stack(
        [
            np.sin(time / 24) + 0.01 * time,
            np.cos(time / 24) + 0.02 * time,
            np.sin(time / 12) + 0.03 * time,
            np.cos(time / 12) + 0.04 * time,
            np.full(t, 0.0),
        ]
    )
    distance = np.full((n, n), 99.0)
    np.fill_diagonal(distance, 0.0)
    for j, value in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        distance[0, j] = distance[j, 0] = value
    return SimpleNamespace(
        grid_name="synthetic",
        num_nodes=n,
        node_ids=np.asarray([str(i) for i in range(n)]),
        timestamps=timestamps,
        p=p,
        load_bus_mask=np.array([True, True, True, True, False]),
        distance_matrices={"impedance_abs_distance": distance},
        splits={"train": SimpleNamespace(start_index=0, end_index=t, sample_count=t)},
    )


def test_exact_chronological_split_and_lags():
    grid = _grid()
    assert len(formal_train_target_indices(grid)) == 5964
    indices = np.arange(168, 168 + FIT_SAMPLES + SELECTION_SAMPLES + AUDIT_SAMPLES)
    fit, selection, audit = split_train_targets(indices)
    assert (len(fit), len(selection), len(audit)) == (3578, 1193, 1193)
    assert fit[-1] < selection[0] < selection[-1] < audit[0]
    assert np.array_equal(np.concatenate([fit, selection, audit]), indices)
    for lag in LAGS:
        assert np.all(indices - lag >= 0)


def test_feature_dimensions_and_fit_only_scaler():
    grid = _grid()
    target = np.arange(168, 300)
    assert feature_matrix(grid.p, grid.timestamps, target, 0).shape[1] == 8
    assert feature_matrix(grid.p, grid.timestamps, target, 0, [1]).shape[1] == 11
    mean, scale, zero = fit_feature_scaler(np.column_stack([target, np.ones(len(target))]))
    assert np.allclose(mean, [target.mean(), 1.0])
    assert scale[1] == 1.0 and zero[1]


def test_electrical_neighbors_load_only_and_tie_order():
    grid = _grid()
    result = select_electrical_neighbors(grid.distance_matrices["impedance_abs_distance"], [0, 1, 2, 3])
    assert [node for node, _ in result[0]] == [1, 2, 3]
    assert all(node != target and node < 4 for target, values in result.items() for node, _ in values)


def test_utility_neighbors_positive_only_and_stable_tie_break():
    utilities = {(1, 0): 0.2, (2, 0): 0.2, (3, 0): -0.4, (0, 1): 0.1, (2, 1): 0.05, (3, 1): 0.0}
    for target in (2, 3):
        for source in (0, 1, 3, 2):
            if source != target:
                utilities.setdefault((source, target), -0.1)
    selected = select_utility_neighbors(utilities, [0, 1, 2, 3])
    assert [node for node, _ in selected[0]] == [1, 2]
    assert all(value > 0 for values in selected.values() for _, value in values)
    assert np.isclose(signed_utility(2.0, 3.0), -0.5)


def test_pairwise_selection_does_not_read_audit_values():
    grid = _grid()
    fit = np.arange(168, 168 + FIT_SAMPLES)
    selection = np.arange(168 + FIT_SAMPLES, 168 + FIT_SAMPLES + SELECTION_SAMPLES)
    before, _ = compute_pairwise_utilities(grid, fit, selection, np.array([0, 1, 2, 3]))
    grid.p[selection[-1] + 1 :, :] += 100000.0
    after, _ = compute_pairwise_utilities(grid, fit, selection, np.array([0, 1, 2, 3]))
    assert before == after


def test_metadata_if_generated():
    report = report_metadata()
    assert report["time_scope"] == "formal_train_only"
    assert report["fit_samples"] == FIT_SAMPLES
    assert report["selection_samples"] == SELECTION_SAMPLES
    assert report["audit_samples"] == AUDIT_SAMPLES
    assert report["utility_terminology"] == "out_of_sample_incremental_predictive_utility"
    assert report["validation_used"] is False
    assert report["test_used"] is False
    assert report["causal_claim"] is False
