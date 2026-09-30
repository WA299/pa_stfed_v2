from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from code.audits.btd_full_backbone_bridge import build_scarce_target_graph
from code.audits.federated_transfer_benefit import IndexDataset, scarce_split
from code.data.industrial_mvlv_loader import (
    IndustrialMVLVLoader,
    load_industrial_mvlv,
    parse_industrial_load_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("INDUSTRIAL_MVLV_DATA_ROOT", str(REPO_ROOT.parent / "pa_stfed_data_v2" / "raw" / "norway_industrial_mvlv")))
requires_archive = pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="industrial MV/LV archive is not available")


@requires_archive
def test_industrial_schema_and_common_interval():
    grid = load_industrial_mvlv(DATA_ROOT)
    assert (grid.num_nodes, grid.num_edges, grid.num_timesteps) == (76, 75, 13344)
    assert int(grid.load_bus_mask.sum()) == 45
    assert grid.p.shape == (13344, 76)
    assert np.all(grid.q == 0)
    assert grid.metadata["q_available"] is False
    assert grid.metadata["q_placeholder_zero"] is True
    assert grid.metadata["test_locked"] is True
    assert grid.metadata["test_evaluated"] is False
    assert str(grid.timestamps[0]) == "2020-09-07 01:00:00"
    assert str(grid.timestamps[-1]) == "2022-03-17 00:00:00"
    assert grid.metadata["common_interval"]["missing_hour_count"] == 0


@requires_archive
def test_topology_and_raw_zero_electrical_values_are_preserved():
    grid = load_industrial_mvlv(DATA_ROOT)
    assert grid.metadata["connected"] and grid.metadata["tree"]
    assert len(grid.metadata["source_slack_bus_ids"]) == 2
    assert grid.metadata["r_zero_count"] == 63
    assert grid.metadata["x_zero_count"] == 63
    assert np.count_nonzero(grid.edge_features[:, 0] == 0) == 63
    assert np.count_nonzero(grid.edge_features[:, 1] == 0) == 63
    assert grid.metadata["electrical_diagnostics"]["load_load_impedance_path_normalization_median_positive"]
    assert grid.metadata["electrical_diagnostics"]["load_load_impedance_path_normalization_median"] == pytest.approx(
        np.median(grid.distance_matrices["impedance_abs_distance"][np.ix_(np.flatnonzero(grid.load_bus_mask), np.flatnonzero(grid.load_bus_mask))][~np.eye(45, dtype=bool)])
    )
    assert np.isfinite(grid.distance_matrices["impedance_abs_distance"]).all()


@requires_archive
def test_archive_late_starting_files_are_exact():
    grid = load_industrial_mvlv(DATA_ROOT)
    earliest = min(item["start_time"] for item in grid.metadata["load_file_metadata"].values())
    late = sorted(
        bus_id for bus_id, item in grid.metadata["load_file_metadata"].items()
        if item["start_time"] > earliest
    )
    assert late == ["r1v0.415b41", "r1v0.415b9", "r2v0.415b3", "r2v0.415b4"]


@requires_archive
def test_duplicate_policy_and_dynamic_feature_selection():
    grid = load_industrial_mvlv(DATA_ROOT)
    assert all(item["duplicate_row_count"] == 1 for item in grid.metadata["load_file_metadata"].values())
    assert all(item["duplicate_values_identical"] for item in grid.metadata["load_file_metadata"].values())
    ds = IndexDataset(grid, np.asarray([168], dtype=np.int64))
    assert ds.feature_indices == (0, 2, 3, 4, 5, 6)
    assert ds[0]["x"].shape == (168, 76, 6)


@requires_archive
def test_canonical_and_scarce_splits_are_exact():
    grid = load_industrial_mvlv(DATA_ROOT)
    assert {name: item.sample_count for name, item in grid.splits.items()} == {
        "train": 9340, "validation": 2001, "test": 2003
    }
    split25 = scarce_split(grid, 0.25)
    split50 = scarce_split(grid, 0.50)
    assert (split25.available_raw_hours, len(split25.eligible_indices), len(split25.fit_indices), len(split25.calibration_indices), len(split25.audit_indices)) == (2335, 2167, 1300, 433, 434)
    assert (split50.available_raw_hours, len(split50.eligible_indices), len(split50.fit_indices), len(split50.calibration_indices), len(split50.audit_indices)) == (4670, 4502, 2701, 900, 901)
    assert split25.eligible_indices[0] - 168 >= split25.available_start
    assert split50.eligible_indices[0] - 168 >= split50.available_start


@requires_archive
def test_fit_only_graph_compatibility_for_both_history_fractions():
    grid = load_industrial_mvlv(DATA_ROOT)
    for fraction in (0.25, 0.50):
        graph = build_scarce_target_graph(grid, scarce_split(grid, fraction))
        assert len(graph.load_indices) == 45
        assert np.isfinite(graph.relation_features).all()
        assert graph.diagnostics["graph_uses_target_fit_only"] is True
        assert graph.diagnostics["graph_uses_calibration"] is False
        assert graph.diagnostics["graph_uses_audit"] is False
        assert graph.diagnostics["graph_uses_validation"] is False
        assert graph.diagnostics["graph_uses_test"] is False


def test_conflicting_duplicate_values_raise():
    path = Path("r1v0.415b1.txt")
    pd.DataFrame(
        {
            "Bus_ID": ["r1v0.415b1", "r1v0.415b1"],
            "Timestamp": ["01/06/2021 00:00:00", "01/06/2021 00:00:00"],
            "Load_kWh": [1.0, 2.0],
        }
    ).to_csv(path, sep=";", index=False)
    try:
        with pytest.raises(ValueError, match="conflicting duplicate"):
            parse_industrial_load_file(path, {"r1v0.415b1"})
    finally:
        path.unlink(missing_ok=True)


def test_identical_duplicate_values_deduplicate():
    path = Path("r1v0.415b1.txt")
    pd.DataFrame(
        {
            "Bus_ID": ["r1v0.415b1", "r1v0.415b1", "r1v0.415b1"],
            "Timestamp": ["01/06/2021 00:00:00", "01/06/2021 00:00:00", "01/06/2021 01:00:00"],
            "Load_kWh": [1.0, 1.0, 2.0],
        }
    ).to_csv(path, sep=";", index=False)
    try:
        bus_id, timestamps, values, duplicates = parse_industrial_load_file(path, {"r1v0.415b1"})
        assert bus_id == "r1v0.415b1"
        assert len(timestamps) == 2 and len(values) == 2
        assert list(duplicates.values()) == [1]
    finally:
        path.unlink(missing_ok=True)
