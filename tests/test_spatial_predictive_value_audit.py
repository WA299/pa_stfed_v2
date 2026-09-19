import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from code.audits.audit_spatial_predictive_value import (
    LAGS,
    MODEL_NAMES,
    _feature_matrix,
    _metrics,
    choose_electrical_neighbors,
    choose_statistical_neighbors,
    chronological_audit_split,
    fit_feature_scaler,
    valid_target_indices,
)


def _grid():
    t, n = 420, 5
    timestamps = pd.date_range("2024-01-01", periods=t, freq="h")
    x = np.arange(t, dtype=float)
    p = np.column_stack([x, 2 * x, x[::-1], np.sin(x), np.full(t, 7.0)])
    d = np.full((n, n), 99.0)
    np.fill_diagonal(d, 0.0)
    for j, value in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        d[0, j] = d[j, 0] = value
    return SimpleNamespace(
        p=p,
        timestamps=timestamps,
        distance_matrices={"impedance_abs_distance": d},
        load_bus_mask=np.array([True, True, True, True, False]),
    )


def test_chronological_split_and_valid_lags():
    indices = valid_target_indices(0, 300)
    fit, holdout = chronological_audit_split(indices)
    assert len(fit) == int(len(indices) * 0.8)
    assert fit[-1] < holdout[0]
    assert np.array_equal(indices, np.concatenate([fit, holdout]))
    assert indices[0] >= 168 and indices[-1] < 300
    for lag in LAGS:
        assert np.all(indices - lag >= 0)


def test_fit_only_scaler_and_zero_variance():
    xfit = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 5.0], [5.0, 6.0, 5.0]])
    mean, scale, zero = fit_feature_scaler(xfit)
    assert np.allclose(mean, [3.0, 4.0, 5.0])
    assert np.allclose(scale[:2], np.std(xfit[:, :2], axis=0))
    assert zero.tolist() == [False, False, True] and scale[2] == 1.0
    mean2, scale2, zero2 = fit_feature_scaler(xfit)
    assert np.allclose(mean, mean2) and np.allclose(scale, scale2) and np.array_equal(zero, zero2)


def test_neighbor_selection_and_dimensions():
    grid = _grid(); load = np.flatnonzero(grid.load_bus_mask)
    electrical = choose_electrical_neighbors(grid, load)
    assert [j for j, _ in electrical[0]] == [1, 2, 3]
    assert all(len(v) <= 3 and all(i != j and j in load for j, _ in v) for i, v in electrical.items())
    indices = valid_target_indices(0, 350); fit, _ = chronological_audit_split(indices)
    statistical = choose_statistical_neighbors(grid, load, fit)
    assert all(len(v) <= 3 and all(i != j and j in load for j, _ in v) for i, v in statistical.items())
    dims = {}
    for kind in MODEL_NAMES:
        neighbors = electrical if kind == "electrical_neighbors" else statistical if kind == "statistical_neighbors" else {}
        dims[kind] = _feature_matrix(grid.p, grid.timestamps, np.arange(200, 220), 0, kind, load, neighbors).shape[1]
    assert dims == {"self_only": 8, "grid_context": 11, "electrical_neighbors": 17, "statistical_neighbors": 17}


def test_holdout_does_not_change_selection_or_scaler():
    grid = _grid(); load = np.flatnonzero(grid.load_bus_mask)
    indices = valid_target_indices(0, 350); fit, holdout = chronological_audit_split(indices)
    electrical_before = choose_electrical_neighbors(grid, load)
    statistical_before = choose_statistical_neighbors(grid, load, fit)
    scaler_before = fit_feature_scaler(grid.p[fit[:20]])
    grid.p[holdout, :] *= 1000.0
    assert electrical_before == choose_electrical_neighbors(grid, load)
    assert statistical_before == choose_statistical_neighbors(grid, load, fit)
    scaler_after = fit_feature_scaler(grid.p[fit[:20]])
    assert all(np.allclose(a, b) for a, b in zip(scaler_before, scaler_after))


def test_metrics_and_metadata():
    metrics = _metrics(np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 2.0]))
    assert all(np.isfinite(list(metrics.values())))
    assert np.isclose(metrics["mae"], 1.0)
    assert np.isclose(metrics["rmse"], np.sqrt(5.0 / 3.0))
    assert np.isclose(metrics["wape"], 3.0 / 7.0 * 100.0)
    report = json.loads(Path("results/audits/spatial_predictive_value.json").read_text(encoding="utf-8"))
    assert report["time_scope"] == "formal_train_only"
    assert report["validation_evaluated"] is False and report["test_evaluated"] is False
    assert report["ridge_alpha"] == 1.0 and report["lags"] == [1, 24, 168]
