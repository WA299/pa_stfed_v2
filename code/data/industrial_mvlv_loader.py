"""Loader and schema audit primitives for the Norwegian industrial MV/LV archive.

This module deliberately stops at a validated :class:`LVGridData` object.  It
does not train a model or evaluate a forecasting split.  Industrial load files
contain hourly ``Load_kWh`` energy measurements; their values are preserved in
those original units.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from code.data.lv_grid_loader import (
    DYNAMIC_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    LVGridData,
    STATIC_FEATURE_NAMES,
    TimeSplit,
)


DEFAULT_DATA_ROOT = Path(r"E:\pa_stfed_data_v2\raw\norway_industrial_mvlv")
GRID_NAME = "norway_industrial_mvlv"
LOAD_FILE_GLOB = "*.txt"
EXPECTED_LOAD_COUNT = 45
EXPECTED_NODE_COUNT = 76
EXPECTED_EDGE_COUNT = 75
EXPECTED_COMMON_ROWS = 13344
EXPECTED_COMMON_START = "2020-09-07 01:00:00"
EXPECTED_COMMON_END = "2022-03-17 00:00:00"


def _column(frame: pd.DataFrame, name: str) -> str:
    wanted = name.strip().lower()
    for value in frame.columns:
        if str(value).strip().lower() == wanted:
            return str(value)
    raise ValueError(f"required column {name!r} is missing")


def _id(value: Any) -> str:
    if value is None or pd.isna(value):
        raise ValueError("bus ID cannot be empty")
    return str(value).strip()


def _time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hours = timestamps.hour.to_numpy(dtype=float) + timestamps.minute.to_numpy(dtype=float) / 60.0
    weekday = timestamps.dayofweek.to_numpy(dtype=float)
    hour_angle = 2.0 * np.pi * hours / 24.0
    day_angle = 2.0 * np.pi * weekday / 7.0
    return np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(day_angle),
            np.cos(day_angle),
            (weekday >= 5).astype(float),
        ]
    ).astype(np.float32)


def _split(timestamps: pd.DatetimeIndex) -> dict[str, TimeSplit]:
    count = len(timestamps)
    train_end = int(count * 0.70)
    validation_end = train_end + int(count * 0.15)
    ranges = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, count),
    }
    result: dict[str, TimeSplit] = {}
    for name, (start, end) in ranges.items():
        result[name] = TimeSplit(
            name=name,
            start_index=start,
            end_index=end,
            sample_count=end - start,
            start_time=timestamps[start].isoformat(sep=" "),
            end_time=timestamps[end - 1].isoformat(sep=" "),
        )
    return result


def _json_number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    value = value.item() if isinstance(value, (np.integer, np.floating)) else value
    return value if not isinstance(value, float) or np.isfinite(value) else None


class IndustrialMVLVLoader:
    """Parse and validate one Norwegian industrial MV/LV archive."""

    def __init__(self, data_root: str | Path = DEFAULT_DATA_ROOT):
        self.data_root = Path(data_root)

    def load(self) -> LVGridData:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"industrial MV/LV directory not found: {self.data_root}")
        bus = pd.read_csv(self.data_root / "bus.csv", sep=";", dtype=str, encoding="utf-8-sig")
        branch = pd.read_csv(self.data_root / "branch.csv", sep=";", dtype=str, encoding="utf-8-sig")
        topology = self._parse_topology(bus, branch)
        series, series_meta = self._parse_loads(set(topology["node_ids"]))
        timestamps, values_by_bus, common_meta = self._common_matrix(series)

        node_ids = np.asarray(topology["node_ids"], dtype=object)
        node_to_index = {value: index for index, value in enumerate(node_ids.tolist())}
        p = np.zeros((len(timestamps), len(node_ids)), dtype=np.float32)
        load_bus_mask = np.zeros(len(node_ids), dtype=bool)
        for bus_id, values in values_by_bus.items():
            index = node_to_index[bus_id]
            p[:, index] = np.asarray(values, dtype=np.float32)
            load_bus_mask[index] = True
        static_features = np.asarray(topology["static_features"], dtype=np.float32).copy()
        static_features[:, 0] = load_bus_mask.astype(np.float32)
        static_features[:, 3] = load_bus_mask.astype(np.float32)
        q = np.zeros_like(p, dtype=np.float32)
        dynamic_features = np.concatenate(
            [p[:, :, None], q[:, :, None], np.repeat(_time_features(timestamps)[:, None, :], len(node_ids), axis=1)],
            axis=2,
        ).astype(np.float32)
        splits = _split(timestamps)

        metadata = dict(topology["metadata"])
        metadata.update(
            {
                "dataset_role": "external_industrial_target_domain",
                "load_value_unit": "Load_kWh per hourly interval",
                "load_values_converted_to_mw": False,
                "timestamp_count": int(len(timestamps)),
                "timestamp_start": timestamps[0].isoformat(sep=" "),
                "timestamp_end": timestamps[-1].isoformat(sep=" "),
                "dynamic_feature_names": list(DYNAMIC_FEATURE_NAMES),
                "static_feature_names": list(STATIC_FEATURE_NAMES),
                "edge_feature_names": list(EDGE_FEATURE_NAMES),
                "q_available": False,
                "q_placeholder_zero": True,
                "q_used_by_model": False,
                "physical_line_length_available": False,
                "physical_line_length_used_by_model": False,
                "test_locked": True,
                "test_evaluated": False,
                "load_file_count": len(series_meta),
                "load_bus_ids": sorted(values_by_bus),
                "load_file_metadata": series_meta,
                "load_bus_count": int(load_bus_mask.sum()),
                "load_count_by_region": dict(Counter(bus_id.split("v", 1)[0] for bus_id in values_by_bus)),
                "load_count_by_voltage": dict(
                    Counter(topology["bus_voltage_by_id"][bus_id] for bus_id in values_by_bus)
                ),
                "zero_load_count_by_bus": {
                    bus_id: int(item["zero_load_count"]) for bus_id, item in series_meta.items()
                },
                "zero_load_fraction_by_bus": {
                    bus_id: float(item["zero_load_fraction"]) for bus_id, item in series_meta.items()
                },
                "electrical_diagnostics": self._electrical_diagnostics(
                    topology["distances"], np.flatnonzero(load_bus_mask)
                ),
                "common_interval": common_meta,
                "split": {
                    name: {
                        "start_index": value.start_index,
                        "end_index": value.end_index,
                        "sample_count": value.sample_count,
                        "start_time": value.start_time,
                        "end_time": value.end_time,
                    }
                    for name, value in splits.items()
                },
            }
        )
        return LVGridData(
            grid_name=GRID_NAME,
            node_ids=node_ids,
            timestamps=timestamps,
            p=p,
            q=q,
            dynamic_features=dynamic_features,
            static_features=static_features,
            edge_index=topology["edge_index"],
            edge_features=topology["edge_features"],
            distance_matrices=topology["distances"],
            load_bus_mask=load_bus_mask,
            consumer_metadata_by_bus={bus_id: [{"source": "industrial_load_file"}] for bus_id in values_by_bus},
            branch_type_mapping={},
            branch_type_values=[None] * EXPECTED_EDGE_COUNT,
            splits=splits,
            metadata=metadata,
        )

    @staticmethod
    def _electrical_diagnostics(distances: dict[str, np.ndarray], loads: np.ndarray) -> dict[str, Any]:
        loads = np.asarray(loads, dtype=int)
        off_diagonal = ~np.eye(len(loads), dtype=bool)
        impedance = np.asarray(distances["impedance_abs_distance"], dtype=float)[np.ix_(loads, loads)]
        hop = np.asarray(distances["hop_distance"], dtype=float)[np.ix_(loads, loads)]
        impedance_values = impedance[off_diagonal]
        hop_values = hop[off_diagonal]
        zero_count = int(np.count_nonzero(impedance_values == 0.0))
        positive = impedance_values[impedance_values > 0.0]
        normalization_median = float(np.median(impedance_values))
        return {
            "load_load_pair_count": int(len(impedance_values)),
            "load_load_zero_impedance_path_count": zero_count,
            "load_load_zero_impedance_path_fraction": float(zero_count / len(impedance_values)),
            "load_load_impedance_path_median": float(np.median(impedance_values)),
            "load_load_impedance_path_positive_median": float(np.median(positive)) if len(positive) else None,
            "load_load_impedance_path_normalization_median": normalization_median,
            "load_load_impedance_path_normalization_median_positive": bool(np.isfinite(normalization_median) and normalization_median > 0),
            "hop_distance_min": float(np.min(hop_values)),
            "hop_distance_median": float(np.median(hop_values)),
            "hop_distance_max": float(np.max(hop_values)),
        }

    def _parse_topology(self, bus: pd.DataFrame, branch: pd.DataFrame) -> dict[str, Any]:
        bus_col = _column(bus, "BUS_I")
        bus_ids = [_id(value) for value in bus[bus_col]]
        if len(set(bus_ids)) != len(bus_ids):
            raise ValueError("duplicate BUS_I values in bus.csv")
        f_col, t_col = _column(branch, "F_BUS"), _column(branch, "T_BUS")
        r_col, x_col = _column(branch, "BR_R"), _column(branch, "BR_X")
        node_set = set(bus_ids)
        if len(bus_ids) != EXPECTED_NODE_COUNT:
            raise ValueError(f"industrial archive identity requires {EXPECTED_NODE_COUNT} buses, found {len(bus_ids)}")
        if len(branch) != EXPECTED_EDGE_COUNT:
            raise ValueError(f"industrial archive identity requires {EXPECTED_EDGE_COUNT} branch rows, found {len(branch)}")
        edges: list[tuple[str, str]] = []
        physical_pairs: set[frozenset[str]] = set()
        r_values: list[float] = []
        x_values: list[float] = []
        graph = nx.Graph()
        graph.add_nodes_from(bus_ids)
        for _, row in branch.iterrows():
            left, right = _id(row[f_col]), _id(row[t_col])
            if left not in node_set or right not in node_set:
                raise ValueError(f"branch endpoint missing from bus.csv: {left}, {right}")
            pair = frozenset((left, right))
            if pair in physical_pairs:
                raise ValueError(f"duplicate physical branch pair: {left}, {right}")
            physical_pairs.add(pair)
            r_value = float(pd.to_numeric(row[r_col], errors="coerce"))
            x_value = float(pd.to_numeric(row[x_col], errors="coerce"))
            if not np.isfinite(r_value) or not np.isfinite(x_value):
                raise ValueError("BR_R and BR_X must be finite")
            edges.append((left, right)); r_values.append(r_value); x_values.append(x_value)
            graph.add_edge(left, right)
        if not nx.is_connected(graph):
            raise ValueError("industrial topology is disconnected")
        if not nx.is_tree(graph):
            raise ValueError("industrial topology must be a tree")

        type_col, kv_col = _column(bus, "BUS_TYPE"), _column(bus, "BASE_KV")
        source_ids = [_id(value) for value in bus.loc[pd.to_numeric(bus[type_col], errors="coerce") == 3, bus_col]]
        if len(source_ids) != 2:
            raise ValueError(f"industrial archive identity requires 2 source/slack buses, found {len(source_ids)}")
        root_depth: dict[str, int] = {}
        for node in bus_ids:
            root_depth[node] = min(nx.shortest_path_length(graph, node, source) for source in source_ids)
        node_to_index = {value: index for index, value in enumerate(bus_ids)}
        edge_index = np.asarray(
            [[node_to_index[left] for left, _ in edges], [node_to_index[right] for _, right in edges]], dtype=np.int64
        )
        edge_features = np.column_stack(
            [np.asarray(r_values, dtype=np.float32), np.asarray(x_values, dtype=np.float32),
             np.full(len(edges), np.nan, dtype=np.float32), np.full(len(edges), np.nan, dtype=np.float32)]
        )
        path_attrs = {
            frozenset((left, right)): (r, x, float(np.hypot(r, x)))
            for (left, right), r, x in zip(edges, r_values, x_values)
        }
        distances = {
            "hop_distance": np.zeros((len(bus_ids), len(bus_ids)), dtype=np.float32),
            "cumulative_r_distance": np.zeros((len(bus_ids), len(bus_ids)), dtype=np.float32),
            "cumulative_x_distance": np.zeros((len(bus_ids), len(bus_ids)), dtype=np.float32),
            "impedance_abs_distance": np.zeros((len(bus_ids), len(bus_ids)), dtype=np.float32),
            "physical_line_length_distance": np.full((len(bus_ids), len(bus_ids)), np.nan, dtype=np.float32),
        }
        np.fill_diagonal(distances["physical_line_length_distance"], 0.0)
        for left in bus_ids:
            for right in bus_ids:
                path = nx.shortest_path(graph, left, right)
                attrs = [path_attrs[frozenset((a, b))] for a, b in zip(path, path[1:])]
                i, j = node_to_index[left], node_to_index[right]
                distances["hop_distance"][i, j] = len(path) - 1
                distances["cumulative_r_distance"][i, j] = sum(item[0] for item in attrs)
                distances["cumulative_x_distance"][i, j] = sum(item[1] for item in attrs)
                distances["impedance_abs_distance"][i, j] = sum(item[2] for item in attrs)
        kv_values = [str(value).strip() for value in bus[kv_col]]
        static_rows = []
        for node, kv in zip(bus_ids, kv_values):
            static_rows.append([0.0, float(graph.degree[node]), float(root_depth[node]), 0.0])
        metadata = {
            "grid_name": GRID_NAME,
            "node_order_source": "bus.csv row order",
            "edge_order_source": "branch.csv row order",
            "bus_count": len(bus_ids),
            "edge_count": len(edges),
            "connected": True,
            "tree": True,
            "source_slack_bus_ids": source_ids,
            "source_slack_count": len(source_ids),
            "root_depth_definition": "minimum hop distance to BUS_TYPE=3 source/slack buses",
            "base_kv_counts": dict(Counter(kv_values)),
            "r_nonzero_count": int(np.count_nonzero(np.asarray(r_values) != 0)),
            "r_zero_count": int(np.count_nonzero(np.asarray(r_values) == 0)),
            "x_nonzero_count": int(np.count_nonzero(np.asarray(x_values) != 0)),
            "x_zero_count": int(np.count_nonzero(np.asarray(x_values) == 0)),
            "edge_r_x_values_preserved": True,
            "physical_line_length_available": False,
        }
        return {"node_ids": bus_ids, "edge_index": edge_index, "edge_features": edge_features,
                "static_features": np.asarray(static_rows, dtype=np.float32), "distances": distances,
                "bus_voltage_by_id": dict(zip(bus_ids, kv_values)),
                "metadata": metadata}

    def _parse_loads(self, bus_ids: set[str]) -> tuple[dict[str, pd.Series], dict[str, dict[str, Any]]]:
        paths = sorted(self.data_root.glob(LOAD_FILE_GLOB), key=lambda path: path.name)
        if len(paths) != EXPECTED_LOAD_COUNT:
            raise ValueError(f"expected {EXPECTED_LOAD_COUNT} load files, found {len(paths)}")
        series: dict[str, pd.Series] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for path in paths:
            raw_row_count = len(pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig"))
            bus_id, timestamps, values, duplicate_groups = parse_industrial_load_file(path, bus_ids)
            series[bus_id] = pd.Series(values, index=timestamps)
            metadata[bus_id] = {
                "file": path.name,
                "bus_id": bus_id,
                "row_count_raw": int(raw_row_count),
                "row_count_after_deduplication": int(len(values)),
                "start_time": timestamps[0].isoformat(sep=" "),
                "end_time": timestamps[-1].isoformat(sep=" "),
                "duplicate_timestamps": duplicate_groups,
                "duplicate_row_count": int(sum(duplicate_groups.values())),
                "duplicate_values_identical": True,
                "nan_count": 0,
                "negative_count": 0,
                "zero_load_count": int(np.count_nonzero(values == 0)),
                "zero_load_fraction": float(np.mean(values == 0)),
            }
        if len(series) != EXPECTED_LOAD_COUNT:
            raise ValueError("industrial archive identity requires 45 unique measured load buses")
        return series, metadata

    @staticmethod
    def _common_matrix(series: dict[str, pd.Series]) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], dict[str, Any]]:
        common = set.intersection(*(set(item.index) for item in series.values()))
        if not common:
            raise ValueError("load files have no common timestamp interval")
        timestamps = pd.DatetimeIndex(sorted(common))
        expected = pd.date_range(timestamps[0], timestamps[-1], freq="h")
        if not timestamps.equals(expected):
            raise ValueError("common industrial interval is not a complete hourly cadence")
        if len(timestamps) != EXPECTED_COMMON_ROWS or str(timestamps[0]) != EXPECTED_COMMON_START or str(timestamps[-1]) != EXPECTED_COMMON_END:
            raise ValueError(f"unexpected common interval: {timestamps[0]} to {timestamps[-1]} ({len(timestamps)} rows)")
        values = {bus_id: item.reindex(timestamps).to_numpy(dtype=float) for bus_id, item in series.items()}
        if any(np.isnan(value).any() for value in values.values()):
            raise ValueError("common interval contains an unavailable load value; no imputation is permitted")
        return timestamps, values, {
            "start_time": timestamps[0].isoformat(sep=" "),
            "end_time": timestamps[-1].isoformat(sep=" "),
            "row_count": int(len(timestamps)),
            "hourly_cadence": True,
            "missing_hour_count": 0,
            "all_load_buses_present": True,
        }


def load_industrial_mvlv(data_root: str | Path = DEFAULT_DATA_ROOT) -> LVGridData:
    """Convenience loader for the industrial MV/LV archive."""
    return IndustrialMVLVLoader(data_root).load()


# Descriptive aliases for callers that use the generic external-grid wording.
IndustrialMVLVData = LVGridData
load_industrial_grid = load_industrial_mvlv


def parse_industrial_load_file(
    path: str | Path, bus_ids: set[str] | Iterable[str]
) -> tuple[str, pd.DatetimeIndex, np.ndarray, dict[str, int]]:
    """Parse one load file, enforcing the archive's duplicate policy."""
    path = Path(path)
    frame = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    id_col, ts_col, value_col = _column(frame, "Bus_ID"), _column(frame, "Timestamp"), _column(frame, "Load_kWh")
    ids = frame[id_col].map(_id)
    unique_ids = ids.unique().tolist()
    if len(unique_ids) != 1:
        raise ValueError(f"Bus_ID is not constant in {path.name}")
    bus_id = unique_ids[0]
    if path.stem != bus_id:
        raise ValueError(f"load filename {path.stem!r} disagrees with Bus_ID {bus_id!r}")
    if bus_id not in set(bus_ids):
        raise ValueError(f"load Bus_ID {bus_id!r} does not exist in bus.csv")
    timestamps = pd.to_datetime(frame[ts_col], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"invalid timestamp in {path.name}")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if values.isna().any():
        raise ValueError(f"NaN/non-numeric load in {path.name}")
    if (values < 0).any():
        raise ValueError(f"negative load in {path.name}")
    duplicate_groups: dict[str, int] = {}
    keep = np.ones(len(frame), dtype=bool)
    duplicate_mask = timestamps.duplicated(keep=False)
    for timestamp in timestamps[duplicate_mask].drop_duplicates():
        positions = np.flatnonzero(timestamps.to_numpy() == timestamp.to_datetime64())
        duplicate_groups[timestamp.isoformat(sep=" ")] = int(len(positions) - 1)
        duplicate_values = values.iloc[positions].to_numpy(dtype=float)
        if not np.all(duplicate_values == duplicate_values[0]):
            raise ValueError(f"conflicting duplicate timestamp values in {path.name} at {timestamp}")
        keep[positions[1:]] = False
    timestamps = timestamps.iloc[keep]
    values = values.iloc[keep]
    order = np.argsort(timestamps.to_numpy())
    timestamps = pd.DatetimeIndex(timestamps.iloc[order])
    values = values.iloc[order].to_numpy(dtype=float)
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"timestamps are not chronological after cleanup in {path.name}")
    return bus_id, timestamps, values, duplicate_groups


__all__ = [
    "DEFAULT_DATA_ROOT", "GRID_NAME", "IndustrialMVLVLoader", "IndustrialMVLVData",
    "load_industrial_mvlv", "load_industrial_grid", "parse_industrial_load_file",
]
