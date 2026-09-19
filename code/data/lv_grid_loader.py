"""Canonical loader for the four Norway LV grids.

The loader preserves the physical bus topology, aggregates validated consumer
profiles by bus, and exposes deterministic time/static/edge features. It does
not train models or calculate test metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd


LV_DATASET = "norway_4_lv_grids"
DYNAMIC_FEATURE_NAMES = (
    "p",
    "q",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "weekend",
)
STATIC_FEATURE_NAMES = (
    "is_load_bus",
    "degree",
    "root_depth",
    "consumer_profile_count",
)
EDGE_FEATURE_NAMES = ("r", "x", "length_km", "branch_type_code")
DISTANCE_NAMES = (
    "hop_distance",
    "cumulative_r_distance",
    "cumulative_x_distance",
    "impedance_abs_distance",
    "physical_line_length_distance",
)


def _normalise_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _find_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    wanted = {name.strip().lower() for name in names}
    for column in frame.columns:
        if str(column).strip().lower() in wanted:
            return str(column)
    return None


def _is_index_column(column: Any) -> bool:
    text = str(column).strip().lower()
    return text == "" or text.startswith("unnamed:")


@dataclass(frozen=True)
class TimeSplit:
    name: str
    start_index: int
    end_index: int
    sample_count: int
    start_time: str
    end_time: str

    @property
    def slice(self) -> slice:
        return slice(self.start_index, self.end_index)


@dataclass
class TrainOnlyScaler:
    """Small scaler interface whose fit operation is restricted to train rows."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    train_sample_count_: int | None = None

    def fit(self, values: np.ndarray, train_end: int) -> "TrainOnlyScaler":
        if train_end <= 0 or train_end > len(values):
            raise ValueError("train_end must select a non-empty prefix of values")
        train_values = np.asarray(values[:train_end], dtype=float)
        self.mean_ = np.nanmean(train_values, axis=0)
        scale = np.nanstd(train_values, axis=0)
        self.scale_ = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        self.train_sample_count_ = int(train_end)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before inverse_transform")
        return np.asarray(values, dtype=float) * self.scale_ + self.mean_


@dataclass
class LVGridData:
    grid_name: str
    node_ids: np.ndarray
    timestamps: pd.DatetimeIndex
    p: np.ndarray
    q: np.ndarray
    dynamic_features: np.ndarray
    static_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    distance_matrices: dict[str, np.ndarray]
    load_bus_mask: np.ndarray
    consumer_metadata_by_bus: dict[str, list[dict[str, Any]]]
    branch_type_mapping: dict[str, int]
    branch_type_values: list[str | None]
    splits: dict[str, TimeSplit]
    metadata: dict[str, Any]

    @property
    def num_timesteps(self) -> int:
        return int(len(self.timestamps))

    @property
    def num_nodes(self) -> int:
        return int(len(self.node_ids))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def fit_scaler(self, values: np.ndarray | None = None) -> TrainOnlyScaler:
        """Fit a scaler on the chronological train prefix only."""
        selected = self.dynamic_features if values is None else values
        return TrainOnlyScaler().fit(selected, self.splits["train"].end_index)


class LVGridLoader:
    """Load verified Norway LV grids into one canonical representation."""

    def __init__(self, data_root: str | Path, mapping_path: str | Path):
        self.data_root = Path(data_root)
        self.mapping_path = Path(mapping_path)
        report = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        self._mapping_by_name = {grid["name"]: grid for grid in report.get("lv_grids", [])}

    def available_grids(self) -> list[str]:
        root = self.data_root / LV_DATASET
        return sorted((path.name for path in root.iterdir() if path.is_dir()), key=_sort_key)

    def load_all(self) -> dict[str, LVGridData]:
        return {name: self.load(name) for name in self.available_grids()}

    def load(self, grid_name: str) -> LVGridData:
        grid_root = self.data_root / LV_DATASET / grid_name
        if not grid_root.is_dir():
            raise FileNotFoundError(f"LV grid directory not found: {grid_name}")
        mapping = self._mapping_by_name.get(grid_name)
        if mapping is None or not mapping.get("profile_to_bus_mapping_verified"):
            raise ValueError(f"Verified profile-to-bus mapping is missing for {grid_name}")
        return self._load_grid(grid_root, mapping)

    @staticmethod
    def _read_profiles(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
        frame = pd.read_csv(path, header=None, skiprows=1, dtype=str, encoding="utf-8-sig")
        timestamps = pd.to_datetime(
            frame.iloc[:, 0], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        if timestamps.isna().any():
            raise ValueError(f"Invalid timestamp in {path.name}")
        values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return pd.DatetimeIndex(timestamps), values

    @staticmethod
    def _aggregate_profiles(values: np.ndarray, mappings: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for item in mappings:
            bus = _normalise_id(item.get("mapped_bus_i"))
            position = int(item["profile_position"]) - 1
            if bus is None or position < 0 or position >= values.shape[1]:
                raise ValueError(f"Invalid profile mapping at position {item.get('profile_position')}")
            grouped[bus].append(values[:, position])
        aggregated: dict[str, np.ndarray] = {}
        for bus, profiles in grouped.items():
            with np.errstate(invalid="ignore", over="ignore"):
                aggregated[bus] = np.sum(np.vstack(profiles), axis=0)
        return aggregated

    @staticmethod
    def _time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
        hours = timestamps.hour.to_numpy(dtype=float) + timestamps.minute.to_numpy(dtype=float) / 60.0
        day_of_week = timestamps.dayofweek.to_numpy(dtype=float)
        hour_angle = 2.0 * np.pi * hours / 24.0
        day_angle = 2.0 * np.pi * day_of_week / 7.0
        return np.column_stack(
            [
                np.sin(hour_angle),
                np.cos(hour_angle),
                np.sin(day_angle),
                np.cos(day_angle),
                (day_of_week >= 5).astype(float),
            ]
        )

    @staticmethod
    def _read_topology(
        root: Path,
        mapping: dict[str, Any],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, int],
        list[str | None],
        dict[str, list[dict[str, Any]]],
        dict[str, Any],
    ]:
        bus = pd.read_csv(root / "mpc_bus.csv", dtype=str, encoding="utf-8-sig")
        branch = pd.read_csv(root / "mpc_branch.csv", dtype=str, encoding="utf-8-sig")
        bus_id_col = _find_column(bus, ["bus_i"])
        fbus_col = _find_column(branch, ["fbus"])
        tbus_col = _find_column(branch, ["tbus"])
        if bus_id_col is None or fbus_col is None or tbus_col is None:
            raise ValueError(f"Topology identifiers missing for {root.name}")

        node_ids = [_normalise_id(value) for value in bus[bus_id_col]]
        node_ids = [value for value in node_ids if value is not None]
        node_to_index = {bus_id: index for index, bus_id in enumerate(node_ids)}
        if len(node_to_index) != len(node_ids):
            raise ValueError(f"Duplicate bus IDs in {root.name}")

        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        edge_rows: list[tuple[str, str, dict[str, Any]]] = []
        for _, row in branch.iterrows():
            fbus = _normalise_id(row[fbus_col])
            tbus = _normalise_id(row[tbus_col])
            if fbus not in node_to_index or tbus not in node_to_index:
                raise ValueError(f"Branch endpoint missing from bus.csv in {root.name}")
            edge_rows.append((fbus, tbus, row.to_dict()))
            graph.add_edge(fbus, tbus)
        if not nx.is_tree(graph):
            raise ValueError(f"Expected a connected tree topology for {root.name}")

        bus_type_col = _find_column(bus, ["type"])
        explicit_slack = []
        if bus_type_col is not None:
            for _, row in bus.iterrows():
                if _json_scalar(pd.to_numeric(row[bus_type_col], errors="coerce")) == 3:
                    bus_id = _normalise_id(row[bus_id_col])
                    if bus_id is not None:
                        explicit_slack.append(bus_id)
        directed = nx.DiGraph()
        directed.add_nodes_from(node_ids)
        directed.add_edges_from((fbus, tbus) for fbus, tbus, _ in edge_rows)
        directed_sources = [node for node in directed if directed.in_degree(node) == 0]
        if len(explicit_slack) == 1:
            roots = explicit_slack
            root_resolution = "explicit_bus_type_3"
        elif len(explicit_slack) > 1:
            roots = explicit_slack
            root_resolution = "multiple_explicit_bus_type_3"
        elif len(directed_sources) == 1:
            roots = directed_sources
            root_resolution = "unique_directed_branch_source_without_type_3"
        else:
            raise ValueError(f"Could not resolve a root bus for {root.name}")
        root_bus = roots[0]
        root_depth = nx.single_source_shortest_path_length(graph, root_bus)
        degree = dict(graph.degree())

        extra_files = sorted(set(root.glob("branch*_extra.csv")) | set(root.glob("branch_extra.csv")))
        extra_path = extra_files[0] if extra_files else None
        extra = pd.read_csv(extra_path, dtype=str, encoding="utf-8-sig") if extra_path else pd.DataFrame()
        extra_f = _find_column(extra, ["fbus", "from bus", "from_bus"]) if not extra.empty else None
        extra_t = _find_column(extra, ["tbus", "to bus", "to_bus"]) if not extra.empty else None
        extra_pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        if extra_f and extra_t:
            for _, row in extra.iterrows():
                fbus = _normalise_id(row[extra_f])
                tbus = _normalise_id(row[extra_t])
                if fbus is not None and tbus is not None:
                    extra_pools[tuple(sorted((fbus, tbus)))].append(row.to_dict())

        mpc_feature_cols = [
            column for column in branch.columns if column not in {fbus_col, tbus_col} and not _is_index_column(column)
        ]
        extra_feature_cols = [
            column
            for column in extra.columns
            if column not in {extra_f, extra_t} and not _is_index_column(column)
        ]
        length_field = next((column for column in extra_feature_cols if "length" in str(column).lower()), None)
        branch_type_field = next((column for column in extra_feature_cols if "type" in str(column).lower()), None)
        r_field = next((column for column in mpc_feature_cols if str(column).lower() == "r"), None)
        x_field = next((column for column in mpc_feature_cols if str(column).lower() == "x"), None)

        edge_features_rows: list[list[float]] = []
        branch_type_values: list[str | None] = []
        path_attrs: dict[frozenset[str], dict[str, float]] = {}
        for fbus, tbus, row in edge_rows:
            extra_row = None
            pool = extra_pools.get(tuple(sorted((fbus, tbus))), [])
            if pool:
                extra_row = pool.pop(0)
            raw_type = _normalise_id(extra_row.get(branch_type_field)) if extra_row and branch_type_field else None
            branch_type_values.append(raw_type)
            r_value = float(pd.to_numeric(row.get(r_field), errors="coerce")) if r_field else np.nan
            x_value = float(pd.to_numeric(row.get(x_field), errors="coerce")) if x_field else np.nan
            length_value = (
                float(pd.to_numeric(extra_row.get(length_field), errors="coerce"))
                if extra_row is not None and length_field
                else np.nan
            )
            edge_features_rows.append([r_value, x_value, length_value, np.nan])
            path_attrs[frozenset((fbus, tbus))] = {
                "r": r_value,
                "x": x_value,
                "length": length_value,
                "impedance_abs": float(np.hypot(r_value, x_value)),
            }

        unique_types = sorted({value for value in branch_type_values if value is not None})
        branch_type_mapping = {value: index for index, value in enumerate(unique_types)}
        for index, value in enumerate(branch_type_values):
            edge_features_rows[index][3] = branch_type_mapping.get(value, np.nan)
        edge_features = np.asarray(edge_features_rows, dtype=np.float32)
        edge_index = np.asarray(
            [[node_to_index[fbus] for fbus, _, _ in edge_rows], [node_to_index[tbus] for _, tbus, _ in edge_rows]],
            dtype=np.int64,
        )

        distances = {
            name: np.zeros((len(node_ids), len(node_ids)), dtype=np.float32) for name in DISTANCE_NAMES
        }
        for source in node_ids:
            for target in node_ids:
                path = nx.shortest_path(graph, source, target)
                hop = len(path) - 1
                totals = {key: 0.0 for key in ("r", "x", "impedance_abs", "length")}
                for left, right in zip(path, path[1:]):
                    attrs = path_attrs[frozenset((left, right))]
                    for key in totals:
                        totals[key] += attrs[key]
                i = node_to_index[source]
                j = node_to_index[target]
                distances["hop_distance"][i, j] = hop
                distances["cumulative_r_distance"][i, j] = totals["r"]
                distances["cumulative_x_distance"][i, j] = totals["x"]
                distances["impedance_abs_distance"][i, j] = totals["impedance_abs"]
                distances["physical_line_length_distance"][i, j] = totals["length"]

        consumer_metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in mapping["profile_mapping"]:
            bus_id = _normalise_id(item["mapped_bus_i"])
            consumer_metadata[bus_id].append(
                {
                    "profile_position": int(item["profile_position"]),
                    "consumer": item.get("consumer"),
                    "consumer_id": item.get("consumer_id"),
                    "p_raw_header": item.get("p_raw_header"),
                    "q_raw_header": item.get("q_raw_header"),
                }
            )

        static_rows = []
        load_bus_mask = np.zeros(len(node_ids), dtype=bool)
        for node in node_ids:
            is_load = node in consumer_metadata
            load_bus_mask[node_to_index[node]] = is_load
            static_rows.append(
                [
                    float(is_load),
                    float(degree[node]),
                    float(root_depth[node]),
                    float(len(consumer_metadata.get(node, []))),
                ]
            )
        static_features = np.asarray(static_rows, dtype=np.float32)
        metadata = {
            "grid_name": root.name,
            "node_ids": node_ids,
            "node_order_source": "mpc_bus.csv row order",
            "edge_order_source": "mpc_branch.csv row order",
            "root_bus": root_bus,
            "root_resolution": root_resolution,
            "explicit_slack_bus_ids": explicit_slack,
            "directed_branch_source_candidates": directed_sources,
            "bus_count": len(node_ids),
            "edge_count": len(edge_rows),
            "dynamic_feature_names": list(DYNAMIC_FEATURE_NAMES),
            "static_feature_names": list(STATIC_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "branch_type_mapping": branch_type_mapping,
            "consumer_profile_count": len(mapping["profile_mapping"]),
            "physical_load_bus_count": int(load_bus_mask.sum()),
            "length_source": extra_path.name if extra_path else None,
            "branch_type_source": extra_path.name if extra_path else None,
        }
        return (
            np.asarray(node_ids, dtype=object),
            edge_index,
            edge_features,
            np.asarray(static_features, dtype=np.float32),
            branch_type_mapping,
            branch_type_values,
            dict(consumer_metadata),
            {"metadata": metadata, "degree": degree, "root_depth": root_depth, "distances": distances},
        )

    def _load_grid(self, root: Path, mapping: dict[str, Any]) -> LVGridData:
        p_timestamps, p_profiles = self._read_profiles(root / "p_load.csv")
        q_timestamps, q_profiles = self._read_profiles(root / "q_load.csv")
        if not p_timestamps.equals(q_timestamps):
            raise ValueError(f"P/Q timestamps are misaligned for {root.name}")
        if p_profiles.shape != q_profiles.shape:
            raise ValueError(f"P/Q profile shapes are misaligned for {root.name}")
        p_by_bus = self._aggregate_profiles(p_profiles, mapping["profile_mapping"])
        q_by_bus = self._aggregate_profiles(q_profiles, mapping["profile_mapping"])
        (
            node_ids,
            edge_index,
            edge_features,
            static_features,
            branch_type_mapping,
            branch_type_values,
            consumer_metadata,
            topology,
        ) = self._read_topology(root, mapping)
        node_to_index = {str(node): index for index, node in enumerate(node_ids)}
        p = np.zeros((len(p_timestamps), len(node_ids)), dtype=np.float32)
        q = np.zeros_like(p)
        for bus, index in node_to_index.items():
            if bus in p_by_bus:
                p[:, index] = p_by_bus[bus]
                q[:, index] = q_by_bus[bus]
        time_features = self._time_features(p_timestamps)
        repeated_time = np.repeat(time_features[:, None, :], len(node_ids), axis=1).astype(np.float32)
        dynamic_features = np.concatenate([p[:, :, None], q[:, :, None], repeated_time], axis=2)
        splits = self._make_splits(p_timestamps)
        metadata = dict(topology["metadata"])
        metadata.update(
            {
                "timestamp_count": len(p_timestamps),
                "timestamp_start": p_timestamps[0].isoformat(sep=" "),
                "timestamp_end": p_timestamps[-1].isoformat(sep=" "),
                "p_q_timestamp_aligned": True,
                "p_q_profile_shape_aligned": True,
                "time_order_preserved": bool(p_timestamps.is_monotonic_increasing),
                "split": {
                    name: {
                        "start_index": split.start_index,
                        "end_index": split.end_index,
                        "sample_count": split.sample_count,
                        "start_time": split.start_time,
                        "end_time": split.end_time,
                    }
                    for name, split in splits.items()
                },
            }
        )
        return LVGridData(
            grid_name=root.name,
            node_ids=node_ids,
            timestamps=p_timestamps,
            p=p,
            q=q,
            dynamic_features=dynamic_features,
            static_features=static_features,
            edge_index=edge_index,
            edge_features=edge_features,
            distance_matrices=topology["distances"],
            load_bus_mask=static_features[:, 0].astype(bool),
            consumer_metadata_by_bus=consumer_metadata,
            branch_type_mapping=branch_type_mapping,
            branch_type_values=branch_type_values,
            splits=splits,
            metadata=metadata,
        )

    @staticmethod
    def _make_splits(timestamps: pd.DatetimeIndex) -> dict[str, TimeSplit]:
        sample_count = len(timestamps)
        train_end = int(sample_count * 0.70)
        validation_end = train_end + int(sample_count * 0.15)
        ranges = {
            "train": (0, train_end),
            "validation": (train_end, validation_end),
            "test": (validation_end, sample_count),
        }
        return {
            name: TimeSplit(
                name=name,
                start_index=start,
                end_index=end,
                sample_count=end - start,
                start_time=timestamps[start].isoformat(sep=" "),
                end_time=timestamps[end - 1].isoformat(sep=" "),
            )
            for name, (start, end) in ranges.items()
        }


def load_lv_grid(
    grid_name: str,
    data_root: str | Path,
    mapping_path: str | Path,
) -> LVGridData:
    """Convenience wrapper for loading one canonical LV grid."""
    return LVGridLoader(data_root, mapping_path).load(grid_name)


def load_all_lv_grids(
    data_root: str | Path,
    mapping_path: str | Path,
) -> dict[str, LVGridData]:
    """Convenience wrapper for loading all available canonical LV grids."""
    return LVGridLoader(data_root, mapping_path).load_all()


def build_validation_report(loader: LVGridLoader) -> dict[str, Any]:
    grids = []
    for grid_name in loader.available_grids():
        data = loader.load(grid_name)
        checks = {
            "node_count": int(data.num_nodes),
            "edge_count": int(data.num_edges),
            "timestamps": int(data.num_timesteps),
            "timestamps_8760": data.num_timesteps == 8760,
            "p_shape": list(data.p.shape),
            "q_shape": list(data.q.shape),
            "p_q_shape_aligned": data.p.shape == data.q.shape,
            "p_q_timestamp_aligned": bool(data.metadata["p_q_timestamp_aligned"]),
            "load_bus_mask_count": int(data.load_bus_mask.sum()),
            "load_bus_mask_expected": int(data.metadata["physical_load_bus_count"]),
            "load_bus_mask_count_correct": int(data.load_bus_mask.sum()) == int(data.metadata["physical_load_bus_count"]),
            "distance_matrices": {
                name: {
                    "shape": list(matrix.shape),
                    "symmetric": bool(np.allclose(matrix, matrix.T, equal_nan=True)),
                    "diagonal_zero": bool(np.allclose(np.diag(matrix), 0.0, equal_nan=True)),
                }
                for name, matrix in data.distance_matrices.items()
            },
            "dynamic_feature_shape": list(data.dynamic_features.shape),
            "static_feature_shape": list(data.static_features.shape),
            "edge_feature_shape": list(data.edge_features.shape),
            "split_sample_counts": {name: split.sample_count for name, split in data.splits.items()},
        }
        grids.append(
            {
                "grid_name": grid_name,
                "metadata": data.metadata,
                "checks": checks,
                "all_checks_pass": all(
                    [
                        checks["node_count"] in (39, 50, 56, 80),
                        checks["edge_count"] == checks["node_count"] - 1,
                        checks["timestamps_8760"],
                        checks["p_q_shape_aligned"],
                        checks["p_q_timestamp_aligned"],
                        checks["load_bus_mask_count_correct"],
                    ]
                    + [
                        item["symmetric"] and item["diagonal_zero"]
                        for item in checks["distance_matrices"].values()
                    ]
                ),
            }
        )
    return {
        "audit": "v2_loader_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "Loader validation only; no model training and no test metrics.",
            "All four physical LV bus topologies are retained.",
            "Time order is preserved and split chronologically 70/15/15 without shuffle.",
        ],
        "grids": grids,
        "all_checks_pass": all(grid["all_checks_pass"] for grid in grids),
    }


def render_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Canonical LV Loader Validation",
        "",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        "",
        "No model training or test metrics are performed.",
        "",
        "| Grid | Nodes | Edges | Timestamps | P/Q aligned | Load buses | Distances valid | All checks |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for grid in report["grids"]:
        checks = grid["checks"]
        distance_valid = all(
            item["symmetric"] and item["diagonal_zero"] for item in checks["distance_matrices"].values()
        )
        lines.append(
            f"| {grid['grid_name']} | {checks['node_count']} | {checks['edge_count']} | {checks['timestamps']} | "
            f"{checks['p_q_shape_aligned'] and checks['p_q_timestamp_aligned']} | {checks['load_bus_mask_count']} | "
            f"{distance_valid} | {grid['all_checks_pass']} |"
        )
    lines.extend(["", "## Shapes and splits", ""])
    for grid in report["grids"]:
        checks = grid["checks"]
        lines.extend(
            [
                f"### {grid['grid_name']}",
                "",
                f"Dynamic shape `(time, node, feature)`: `{checks['dynamic_feature_shape']}`; "
                f"static shape `(node, feature)`: `{checks['static_feature_shape']}`; "
                f"edge feature shape `(edge, feature)`: `{checks['edge_feature_shape']}`.",
                f"Split sample counts: `{checks['split_sample_counts']}`.",
                "",
                "| Distance matrix | Shape | Symmetric | Diagonal zero |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name, item in checks["distance_matrices"].items():
            lines.append(f"| `{name}` | `{item['shape']}` | {item['symmetric']} | {item['diagonal_zero']} |")
        lines.append("")
    lines.append(f"All checks pass: **{report['all_checks_pass']}**.")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument(
        "--mapping-json", type=Path, default=repo_root / "results" / "audits" / "v2_schema_mapping.json"
    )
    parser.add_argument(
        "--output-json", type=Path, default=repo_root / "results" / "audits" / "v2_loader_validation.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=repo_root / "results" / "audits" / "v2_loader_validation.md"
    )
    args = parser.parse_args()
    report = build_validation_report(LVGridLoader(args.data_root, args.mapping_json))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_validation_markdown(report), encoding="utf-8")
    print(f"grids={len(report['grids'])} all_checks_pass={report['all_checks_pass']}")


if __name__ == "__main__":
    main()
