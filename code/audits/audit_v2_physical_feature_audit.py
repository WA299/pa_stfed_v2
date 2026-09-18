"""Audit physical topology and P/Q information independence for LV grids."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


LV_DATASET = "norway_4_lv_grids"
HIGH_CORRELATION_THRESHOLD = 0.99
RATIO_NEAR_CONSTANT_CV = 1e-3


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=repo_root.parent / "pa_stfed_data_v2" / "raw"
    )
    parser.add_argument(
        "--mapping-json", type=Path, default=repo_root / "results" / "audits" / "v2_schema_mapping.json"
    )
    parser.add_argument(
        "--output-json", type=Path, default=repo_root / "results" / "audits" / "v2_physical_feature_audit.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=repo_root / "results" / "audits" / "v2_physical_feature_audit.md"
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def normalise_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_profiles(path: Path) -> tuple[pd.Series, np.ndarray]:
    frame = pd.read_csv(path, header=None, skiprows=1, dtype=str, encoding="utf-8-sig")
    dates = pd.to_datetime(frame.iloc[:, 0], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return dates, values


def aggregate_profiles(values: np.ndarray, mappings: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in mappings:
        if item.get("mapped_bus_i") is None:
            raise ValueError("Verified mapping contains a missing mapped_bus_i")
        position = int(item["profile_position"]) - 1
        if position < 0 or position >= values.shape[1]:
            raise ValueError(f"Profile position {position + 1} is outside the profile matrix")
        grouped[str(item["mapped_bus_i"])].append(values[:, position])
    result: dict[str, np.ndarray] = {}
    for bus, series in grouped.items():
        with np.errstate(invalid="ignore", over="ignore"):
            result[bus] = np.sum(np.vstack(series), axis=0)
    return result


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def pq_bus_relation(p: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    pair_mask = np.isfinite(p) & np.isfinite(q)
    pair_count = int(pair_mask.sum())
    correlation = None
    if pair_count >= 2 and np.std(p[pair_mask]) > 0 and np.std(q[pair_mask]) > 0:
        correlation = safe_float(np.corrcoef(p[pair_mask], q[pair_mask])[0, 1])

    ratio_mask = pair_mask & (p != 0)
    ratios = q[ratio_mask] / p[ratio_mask]
    ratio_summary = finite_summary(ratios)
    ratio_cv = None
    if ratio_summary["count"]:
        if ratio_summary["mean"] == 0 and ratio_summary["std"] == 0:
            ratio_cv = 0.0
        elif ratio_summary["mean"] != 0:
            ratio_cv = ratio_summary["std"] / abs(ratio_summary["mean"])
    ratio_near_constant = bool(
        ratio_summary["count"] >= 2 and ratio_cv is not None and ratio_cv <= RATIO_NEAR_CONSTANT_CV
    )
    high_correlation = bool(correlation is not None and abs(correlation) >= HIGH_CORRELATION_THRESHOLD)
    scaling_candidate = bool(high_correlation and ratio_near_constant)
    return {
        "sample_count": int(len(p)),
        "finite_pq_pair_count": pair_count,
        "p_zero_excluded_count": int(np.sum(np.isfinite(p) & (p == 0))),
        "pearson_correlation": correlation,
        "high_correlation": high_correlation,
        "q_over_p": {
            **ratio_summary,
            "cv": safe_float(ratio_cv),
            "near_constant": ratio_near_constant,
        },
        "fixed_power_factor_scaling_candidate": scaling_candidate,
        "q_independent_dynamic_signal_by_rule": not scaling_candidate,
    }


def find_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    wanted = {name.strip().lower() for name in names}
    for column in frame.columns:
        if str(column).strip().lower() in wanted:
            return str(column)
    return None


def is_index_column(column: Any) -> bool:
    text = str(column).strip().lower()
    return text == "" or text.startswith("unnamed:")


def edge_key(fbus: str, tbus: str, directed: bool = True) -> tuple[str, str]:
    return (fbus, tbus) if directed else tuple(sorted((fbus, tbus)))


def match_edge_extra(
    branch: pd.DataFrame, extra: pd.DataFrame, fbus_col: str, tbus_col: str
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    extra_f = find_column(extra, ["fbus", "from bus", "from_bus"])
    extra_t = find_column(extra, ["tbus", "to bus", "to_bus"])
    if extra_f is None or extra_t is None:
        return [None] * len(branch), {"status": "endpoint_columns_missing", "unmatched_count": len(branch)}

    directed_pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    undirected_pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for _, row in extra.iterrows():
        fbus = normalise_id(row[extra_f])
        tbus = normalise_id(row[extra_t])
        if fbus is not None and tbus is not None:
            record = row.to_dict()
            directed_pools[edge_key(fbus, tbus)].append(record)
            undirected_pools[edge_key(fbus, tbus, directed=False)].append(record)

    matches: list[dict[str, Any] | None] = []
    directed_count = 0
    undirected_count = 0
    unmatched = 0
    for _, row in branch.iterrows():
        fbus = normalise_id(row[fbus_col])
        tbus = normalise_id(row[tbus_col])
        match = None
        if fbus is not None and tbus is not None:
            direct_pool = directed_pools.get(edge_key(fbus, tbus), [])
            if direct_pool:
                match = direct_pool.pop(0)
                directed_count += 1
            else:
                reverse_pool = undirected_pools.get(edge_key(fbus, tbus, directed=False), [])
                if reverse_pool:
                    match = reverse_pool.pop(0)
                    undirected_count += 1
        if match is None:
            unmatched += 1
        matches.append(match)
    return matches, {
        "status": "matched" if unmatched == 0 else "partial",
        "directed_match_count": directed_count,
        "undirected_match_count": undirected_count,
        "unmatched_count": unmatched,
        "extra_unconsumed_count": max(len(extra) - directed_count - undirected_count, 0),
        "extra_endpoint_columns": [extra_f, extra_t],
    }


def feature_summary(values: list[Any], source: str) -> dict[str, Any]:
    series = pd.Series(values, dtype=object)
    missing_mask = series.isna() | series.astype(str).str.strip().eq("")
    nonmissing = series[~missing_mask]
    numeric = pd.to_numeric(nonmissing, errors="coerce")
    numeric_field = len(nonmissing) == 0 or numeric.notna().all()
    result: dict[str, Any] = {
        "source": source,
        "count": int(len(series)),
        "missing_count": int(missing_mask.sum()),
        "zero_count": None,
        "negative_count": None,
        "min": None,
        "mean": None,
        "max": None,
        "numeric": bool(numeric_field),
    }
    if numeric_field:
        numbers = numeric.to_numpy(dtype=float)
        result.update(
            {
                "zero_count": int(np.sum(numbers == 0)),
                "negative_count": int(np.sum(numbers < 0)),
                "min": safe_float(np.min(numbers)) if len(numbers) else None,
                "mean": safe_float(np.mean(numbers)) if len(numbers) else None,
                "max": safe_float(np.max(numbers)) if len(numbers) else None,
            }
        )
    else:
        counts = Counter(str(value) for value in nonmissing)
        result["unique_values"] = dict(sorted(counts.items()))
    return result


def topology_sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def topology_audit(root: Path, mapping_grid: dict[str, Any]) -> dict[str, Any]:
    bus = pd.read_csv(root / "mpc_bus.csv", dtype=str, encoding="utf-8-sig")
    branch = pd.read_csv(root / "mpc_branch.csv", dtype=str, encoding="utf-8-sig")
    extra_files = sorted(set(root.glob("branch*_extra.csv")) | set(root.glob("branch_extra.csv")))
    extra_path = extra_files[0] if extra_files else None
    extra = pd.read_csv(extra_path, dtype=str, encoding="utf-8-sig") if extra_path else pd.DataFrame()

    bus_id_col = find_column(bus, ["bus_i"])
    fbus_col = find_column(branch, ["fbus"])
    tbus_col = find_column(branch, ["tbus"])
    if bus_id_col is None or fbus_col is None or tbus_col is None:
        raise ValueError(f"Required topology endpoint fields are missing in {root.name}")
    node_ids = [normalise_id(value) for value in bus[bus_id_col]]
    node_ids = [value for value in node_ids if value is not None]

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    directed = nx.DiGraph()
    directed.add_nodes_from(node_ids)
    edge_pairs: list[tuple[str, str]] = []
    for _, row in branch.iterrows():
        fbus = normalise_id(row[fbus_col])
        tbus = normalise_id(row[tbus_col])
        if fbus is None or tbus is None:
            continue
        edge_pairs.append((fbus, tbus))
        graph.add_edge(fbus, tbus)
        directed.add_edge(fbus, tbus)

    components = [sorted(component, key=topology_sort_key) for component in nx.connected_components(graph)]
    components.sort(key=lambda component: topology_sort_key(component[0]) if component else (2, ""))
    bus_type_col = find_column(bus, ["type"])
    explicit_slack = []
    if bus_type_col is not None:
        for _, row in bus.iterrows():
            if safe_float(row[bus_type_col]) == 3:
                identifier = normalise_id(row[bus_id_col])
                if identifier is not None:
                    explicit_slack.append(identifier)
    directed_sources = sorted([node for node in directed if directed.in_degree(node) == 0], key=topology_sort_key)
    if len(explicit_slack) == 1:
        distance_roots = explicit_slack
        root_resolution = "explicit_bus_type_3"
    elif len(explicit_slack) > 1:
        distance_roots = explicit_slack
        root_resolution = "multiple_explicit_bus_type_3"
    elif len(directed_sources) == 1:
        distance_roots = directed_sources
        root_resolution = "unique_directed_branch_source_without_type_3"
    else:
        distance_roots = []
        root_resolution = "not_confirmed"

    distances: dict[str, int] = {}
    if distance_roots:
        for root_node in distance_roots:
            distances.update(
                {
                    node: min(distances.get(node, distance), distance)
                    for node, distance in nx.single_source_shortest_path_length(graph, root_node).items()
                }
            )
    degree = dict(graph.degree())

    extra_matches, extra_match_summary = match_edge_extra(branch, extra, fbus_col, tbus_col)
    mpc_feature_cols = [column for column in branch.columns if column not in {fbus_col, tbus_col} and not is_index_column(column)]
    extra_f = find_column(extra, ["fbus", "from bus", "from_bus"]) if not extra.empty else None
    extra_t = find_column(extra, ["tbus", "to bus", "to_bus"]) if not extra.empty else None
    extra_feature_cols = [
        column
        for column in extra.columns
        if column not in {extra_f, extra_t} and not is_index_column(column)
    ]

    edge_records: list[dict[str, Any]] = []
    feature_values: dict[str, list[Any]] = {column: [] for column in mpc_feature_cols + extra_feature_cols}
    feature_sources: dict[str, str] = {column: "mpc_branch.csv" for column in mpc_feature_cols}
    feature_sources.update({column: extra_path.name if extra_path else "missing" for column in extra_feature_cols})
    r_field = next((column for column in mpc_feature_cols if str(column).lower() == "r"), None)
    x_field = next((column for column in mpc_feature_cols if str(column).lower() == "x"), None)
    length_field = next((column for column in extra_feature_cols if "length" in str(column).lower()), None)
    branch_type_field = next((column for column in extra_feature_cols if "type" in str(column).lower()), None)
    for row_index, (_, row) in enumerate(branch.iterrows()):
        fbus = normalise_id(row[fbus_col])
        tbus = normalise_id(row[tbus_col])
        extra_row = extra_matches[row_index] if row_index < len(extra_matches) else None
        attrs: dict[str, Any] = {}
        for column in mpc_feature_cols:
            value = json_scalar(row[column])
            attrs[column] = value
            feature_values[column].append(value)
        for column in extra_feature_cols:
            value = json_scalar(extra_row.get(column)) if extra_row is not None else None
            attrs[column] = value
            feature_values[column].append(value)
        edge_records.append(
            {
                "edge_index": row_index,
                "fbus": fbus,
                "tbus": tbus,
                "features": attrs,
            }
        )

    path_edge_attrs: dict[frozenset[str], dict[str, float | None]] = {}
    for edge in edge_records:
        features = edge["features"]
        r_value = safe_float(features.get(r_field)) if r_field else None
        x_value = safe_float(features.get(x_field)) if x_field else None
        length_value = safe_float(features.get(length_field)) if length_field else None
        path_edge_attrs[frozenset((edge["fbus"], edge["tbus"]))] = {
            "r": r_value,
            "x": x_value,
            "impedance_abs": safe_float(np.hypot(r_value, x_value))
            if r_value is not None and x_value is not None
            else None,
            "physical_line_length": length_value,
        }

    node_records: dict[str, Any] = {}
    bus_feature_cols = [column for column in bus.columns if column != bus_id_col and not is_index_column(column)]
    bus_by_id = {normalise_id(row[bus_id_col]): row for _, row in bus.iterrows()}
    load_profile_counts = Counter(str(item["mapped_bus_i"]) for item in mapping_grid["profile_mapping"])
    for node in sorted(node_ids, key=topology_sort_key):
        row = bus_by_id[node]
        root_path_features = {
            "hop_distance": distances.get(node),
            "cumulative_r_distance": None,
            "cumulative_x_distance": None,
            "cumulative_impedance_abs_distance": None,
            "cumulative_physical_line_length": None,
        }
        if distance_roots and node in distances:
            root_node = min(distance_roots, key=lambda candidate: nx.shortest_path_length(graph, candidate, node))
            path = nx.shortest_path(graph, root_node, node)
            totals = {key: 0.0 for key in ("r", "x", "impedance_abs", "physical_line_length")}
            complete = {key: True for key in totals}
            for left, right in zip(path, path[1:]):
                attrs = path_edge_attrs.get(frozenset((left, right)), {})
                for key in totals:
                    value = attrs.get(key)
                    if value is None:
                        complete[key] = False
                    else:
                        totals[key] += value
            root_path_features.update(
                {
                    "cumulative_r_distance": safe_float(totals["r"]) if complete["r"] else None,
                    "cumulative_x_distance": safe_float(totals["x"]) if complete["x"] else None,
                    "cumulative_impedance_abs_distance": safe_float(totals["impedance_abs"])
                    if complete["impedance_abs"]
                    else None,
                    "cumulative_physical_line_length": safe_float(totals["physical_line_length"])
                    if complete["physical_line_length"]
                    else None,
                }
            )
        node_records[node] = {
            "degree": int(degree.get(node, 0)),
            "is_distance_root": node in distance_roots,
            "is_explicit_slack": node in explicit_slack,
            "is_physical_load_bus": node in load_profile_counts,
            "consumer_profile_count": int(load_profile_counts.get(node, 0)),
            "root_depth_hops": root_path_features["hop_distance"],
            "cumulative_distances": root_path_features,
            "bus_static_fields": {column: json_scalar(row[column]) for column in bus_feature_cols},
        }

    path_features = {
        "hop_distance": {
            "constructible_for_all_nodes": bool(distance_roots and len(distances) == len(node_ids)),
            "missing_nodes": [node for node in node_ids if node not in distances],
        },
    }
    for name, key in (
        ("cumulative_r_distance", "cumulative_r_distance"),
        ("cumulative_x_distance", "cumulative_x_distance"),
        ("cumulative_impedance_abs_distance", "cumulative_impedance_abs_distance"),
        ("cumulative_physical_line_length", "cumulative_physical_line_length"),
    ):
        missing_nodes = [node for node, item in node_records.items() if item["cumulative_distances"][key] is None]
        path_features[name] = {
            "constructible_for_all_nodes": len(missing_nodes) == 0 and bool(distance_roots),
            "missing_nodes": missing_nodes,
        }

    feature_audit = {
        "available_mpc_branch_features": mpc_feature_cols,
        "available_extra_edge_features": extra_feature_cols,
        "feature_sources": feature_sources,
        "summaries": {
            column: feature_summary(values, feature_sources[column]) for column, values in feature_values.items()
        },
        "r_field": r_field,
        "x_field": x_field,
        "length_field": length_field,
        "branch_type_field": branch_type_field,
        "extra_file": extra_path.name if extra_path else None,
    }
    depth_values = [value for value in distances.values()]
    degree_values = list(degree.values())
    return {
        "name": root.name,
        "relative_root": root.name,
        "bus_count": int(len(node_ids)),
        "branch_count": int(len(branch)),
        "topology": {
            "connected_components": len(components),
            "component_sizes": [len(component) for component in components],
            "components": components,
            "connected": len(components) == 1,
            "is_radial_tree": bool(len(components) == 1 and len(branch) == len(node_ids) - 1 and nx.is_tree(graph)),
            "self_loop_count": int(nx.number_of_selfloops(graph)),
            "parallel_edge_pair_count": int(
                len(edge_pairs) - len({edge_key(fbus, tbus, directed=False) for fbus, tbus in edge_pairs})
            ),
            "degree_summary": finite_summary(np.asarray(degree_values, dtype=float)),
            "root_slack": {
                "explicit_slack_bus_ids_type_3": sorted(explicit_slack, key=topology_sort_key),
                "directed_branch_source_candidates": directed_sources,
                "distance_root_bus_ids_used": distance_roots,
                "resolution": root_resolution,
                "slack_explicitly_confirmed": len(explicit_slack) == 1,
            },
            "root_depth_summary": finite_summary(np.asarray(depth_values, dtype=float)) if depth_values else finite_summary(np.array([])),
            "node_records": node_records,
        },
        "node_static_features": {
            "available_mpc_bus_fields": bus_feature_cols,
            "mpc_bus_field_summaries": {
                column: feature_summary(bus[column].tolist(), "mpc_bus.csv") for column in bus_feature_cols
            },
            "topology_derived_fields": [
                "degree",
                "is_distance_root",
                "is_explicit_slack",
                "is_physical_load_bus",
                "consumer_profile_count",
                "root_depth_hops",
                "cumulative_r_distance",
                "cumulative_x_distance",
                "cumulative_impedance_abs_distance",
                "cumulative_physical_line_length",
            ],
            "bus_field_source": "mpc_bus.csv",
        },
        "edge_features": feature_audit,
        "edge_extra_match": extra_match_summary,
        "path_feature_constructibility": path_features,
        "pq_information": {},
    }


def pq_audit_for_grid(root: Path, mapping_grid: dict[str, Any]) -> dict[str, Any]:
    if not mapping_grid.get("profile_to_bus_mapping_verified"):
        raise ValueError(f"P/Q mapping is not verified for {mapping_grid['name']}")
    p_dates, p_values = read_profiles(root / "p_load.csv")
    q_dates, q_values = read_profiles(root / "q_load.csv")
    if not p_dates.equals(q_dates) or p_values.shape != q_values.shape:
        raise ValueError(f"P/Q arrays are not aligned for {mapping_grid['name']}")
    p_by_bus = aggregate_profiles(p_values, mapping_grid["profile_mapping"])
    q_by_bus = aggregate_profiles(q_values, mapping_grid["profile_mapping"])
    buses = sorted(p_by_bus, key=topology_sort_key)
    per_bus = {
        bus: pq_bus_relation(p_by_bus[bus], q_by_bus[bus]) for bus in buses
    }
    high_count = sum(item["high_correlation"] for item in per_bus.values())
    ratio_count = sum(item["q_over_p"]["near_constant"] for item in per_bus.values())
    scaling_count = sum(item["fixed_power_factor_scaling_candidate"] for item in per_bus.values())
    independent_count = sum(item["q_independent_dynamic_signal_by_rule"] for item in per_bus.values())
    total = len(per_bus)
    return {
        "aggregation": "sum profiles sharing mapped physical bus, separately for P and Q",
        "thresholds": {
            "high_correlation_abs_pearson": HIGH_CORRELATION_THRESHOLD,
            "q_over_p_near_constant_cv": RATIO_NEAR_CONSTANT_CV,
        },
        "per_bus": per_bus,
        "summary": {
            "physical_load_bus_count": total,
            "highly_correlated_bus_count": high_count,
            "highly_correlated_bus_ratio": high_count / total if total else None,
            "q_over_p_near_constant_bus_count": ratio_count,
            "q_over_p_near_constant_bus_ratio": ratio_count / total if total else None,
            "fixed_power_factor_scaling_candidate_bus_count": scaling_count,
            "fixed_power_factor_scaling_candidate_ratio": scaling_count / total if total else None,
            "q_independent_dynamic_signal_bus_count_by_rule": independent_count,
            "q_independent_dynamic_signal_ratio_by_rule": independent_count / total if total else None,
            "q_independent_dynamic_information_assessment": (
                "evaluate per-bus values and rule counts; no conclusion is imposed"
            ),
        },
    }


def build_report(data_root: Path, mapping_path: Path) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_by_name = {grid["name"]: grid for grid in mapping.get("lv_grids", [])}
    lv_root = data_root / LV_DATASET
    grids = []
    for grid_root in sorted(path for path in lv_root.iterdir() if path.is_dir()):
        mapping_grid = mapping_by_name.get(grid_root.name)
        if mapping_grid is None:
            raise ValueError(f"No verified mapping found for {grid_root.name}")
        topology = topology_audit(grid_root, mapping_grid)
        topology["pq_information"] = pq_audit_for_grid(grid_root, mapping_grid)
        grids.append(topology)

    signatures = [
        {
            "name": grid["name"],
            "bus_count": grid["bus_count"],
            "branch_count": grid["branch_count"],
            "connected_components": grid["topology"]["connected_components"],
            "is_radial_tree": grid["topology"]["is_radial_tree"],
            "maximum_depth": grid["topology"]["root_depth_summary"]["max"],
            "average_depth": grid["topology"]["root_depth_summary"]["mean"],
            "branch_types": grid["edge_features"]["summaries"].get(
                grid["edge_features"].get("branch_type_field"), {}
            ).get("unique_values", {}),
        }
        for grid in grids
    ]
    heterogeneity = {
        "bus_count_values": sorted({item["bus_count"] for item in signatures}),
        "branch_count_values": sorted({item["branch_count"] for item in signatures}),
        "maximum_depth_values": sorted({item["maximum_depth"] for item in signatures}),
        "average_depth_values": sorted({item["average_depth"] for item in signatures}),
        "branch_type_sets_differ": len({tuple(sorted(item["branch_types"])) for item in signatures}) > 1,
        "obvious_structural_heterogeneity": len({item["bus_count"] for item in signatures}) > 1
        or len({item["maximum_depth"] for item in signatures}) > 1
        or len({tuple(sorted(item["branch_types"])) for item in signatures}) > 1,
        "basis": "compare actual node/edge counts, root depths, and branch-type sets",
    }
    node_fields = sorted(
        {
            field
            for grid in grids
            for field in grid["node_static_features"]["available_mpc_bus_fields"]
        }
    )
    edge_fields = sorted(
        {
            field
            for grid in grids
            for field in grid["edge_features"]["available_mpc_branch_features"]
            + grid["edge_features"]["available_extra_edge_features"]
        }
    )
    path_features = [
        "hop_distance",
        "cumulative_r_distance",
        "cumulative_x_distance",
        "cumulative_impedance_abs_distance",
        "cumulative_physical_line_length",
    ]
    q_grid_assessment = {
        grid["name"]: (
            "not_independent_by_fixed_pf_rule"
            if grid["pq_information"]["summary"]["fixed_power_factor_scaling_candidate_ratio"] == 1.0
            else "potentially_independent_by_fixed_pf_rule"
        )
        for grid in grids
    }
    return {
        "audit": "v2_physical_feature_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "Four norway_4_lv_grids only; industrial data is not processed.",
            "P/Q series use the verified profile-to-bus mapping and aggregate consumers per physical bus.",
            "No neural network is trained and no test metric is calculated.",
        ],
        "pq_definitions": {
            "pearson": "Pearson correlation over finite P/Q pairs",
            "q_over_p": "Q/P over finite samples with P != 0",
            "high_correlation_rule": f"abs(Pearson) >= {HIGH_CORRELATION_THRESHOLD}",
            "near_constant_ratio_rule": f"Q/P CV <= {RATIO_NEAR_CONSTANT_CV}",
            "fixed_power_factor_candidate_rule": "high correlation and near-constant Q/P",
        },
        "topology_definitions": {
            "radial_tree": "one connected component, edge_count = node_count - 1, and networkx tree check",
            "cumulative_r": "sum of branch r along the selected root-to-node path",
            "cumulative_x": "sum of branch x along the selected root-to-node path",
            "cumulative_impedance_abs": "sum(sqrt(r^2 + x^2)) along the path",
            "cumulative_physical_line_length": "sum of extra-file Length [km] along the path",
        },
        "lv_grids": grids,
        "cross_grid_topology_comparison": {
            "grid_signatures": signatures,
            "heterogeneity": heterogeneity,
        },
        "conclusions": {
            "p_as_prediction_target": {
                "assessment": "yes_for_physical_load_forecasting_scope",
                "basis": "P profiles are time-aligned, positionally mapped, and aggregated per verified physical bus.",
            },
            "q_as_independent_dynamic_input": {
                "assessment": "grid_dependent; see per-grid fixed_power_factor_scaling_candidate ratios",
                "per_grid_rule_assessment": q_grid_assessment,
                "basis": "independent-dynamic status is reported from actual Pearson and Q/P statistics, not assumed.",
            },
            "available_node_static_features": {
                "directly_constructible_topology_features": [
                    "degree",
                    "is_distance_root",
                    "is_explicit_slack",
                    "is_physical_load_bus",
                    "consumer_profile_count",
                    "root_depth_hops",
                    "cumulative_r_distance",
                    "cumulative_x_distance",
                    "cumulative_impedance_abs_distance",
                    "cumulative_physical_line_length",
                ],
                "mpc_bus_fields_present": node_fields,
                "mpc_bus_static_or_constraint_candidates": [
                    field
                    for field in ("type", "Gs", "Bs", "area", "basekV", "zone", "Vmax", "Vmin", "maxVm", "minVm")
                    if field in node_fields
                ],
                "requires_semantic_or_target_leakage_review": [
                    field for field in ("Pd", "Qd", "Vm", "Va") if field in node_fields
                ],
            },
            "available_edge_features": edge_fields,
            "constructible_electrical_distance_features": {
                feature: all(
                    grid["path_feature_constructibility"][feature]["constructible_for_all_nodes"]
                    for grid in grids
                )
                for feature in path_features
            },
            "topology_structural_heterogeneity": {
                "assessment": heterogeneity["obvious_structural_heterogeneity"],
                "basis": heterogeneity["basis"],
            },
        },
    }


def md_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Physical Features and P/Q Information Audit",
        "",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        "",
        "Only the four LV grids are processed. No neural network or test metric is used.",
        "",
    ]
    for grid in report["lv_grids"]:
        topology = grid["topology"]
        pq = grid["pq_information"]
        lines.extend(
            [
                f"## {grid['name']}",
                "",
                f"Bus count: {grid['bus_count']}; branch count: {grid['branch_count']}; "
                f"components: {topology['connected_components']}; connected: **{md_value(topology['connected'])}**; "
                f"radial/tree: **{md_value(topology['is_radial_tree'])}**.",
                f"Explicit type=3 slack: {md_value(topology['root_slack']['explicit_slack_bus_ids_type_3'])}; "
                f"distance root used: {md_value(topology['root_slack']['distance_root_bus_ids_used'])}; "
                f"resolution: `{topology['root_slack']['resolution']}`.",
                f"Depth max/mean: {md_value(topology['root_depth_summary']['max'])} / {md_value(topology['root_depth_summary']['mean'])}.",
                "",
                "### P/Q relation by physical bus",
                "",
                "| Bus | Samples | Pearson P-Q | High corr | Q/P mean | Q/P std | Q/P min | Q/P max | Q/P CV | Q/P near-constant | Fixed-PF candidate | Q independent by rule |",
                "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for bus, item in pq["per_bus"].items():
            ratio = item["q_over_p"]
            lines.append(
                f"| `{bus}` | {item['sample_count']} | {md_value(item['pearson_correlation'])} | "
                f"{md_value(item['high_correlation'])} | {md_value(ratio['mean'])} | {md_value(ratio['std'])} | "
                f"{md_value(ratio['min'])} | {md_value(ratio['max'])} | {md_value(ratio['cv'])} | "
                f"{md_value(ratio['near_constant'])} | {md_value(item['fixed_power_factor_scaling_candidate'])} | "
                f"{md_value(item['q_independent_dynamic_signal_by_rule'])} |"
            )
        summary = pq["summary"]
        lines.extend(
            [
                "",
                f"Highly correlated buses: {summary['highly_correlated_bus_count']}/{summary['physical_load_bus_count']} "
                f"({summary['highly_correlated_bus_ratio']:.3f}); Q/P near-constant buses: "
                f"{summary['q_over_p_near_constant_bus_count']}/{summary['physical_load_bus_count']} "
                f"({summary['q_over_p_near_constant_bus_ratio']:.3f}); fixed-PF candidates: "
                f"{summary['fixed_power_factor_scaling_candidate_bus_count']}/{summary['physical_load_bus_count']} "
                f"({summary['fixed_power_factor_scaling_candidate_ratio']:.3f}).",
                f"Q independent-dynamic signal by rule: {summary['q_independent_dynamic_signal_bus_count_by_rule']}/"
                f"{summary['physical_load_bus_count']} ({summary['q_independent_dynamic_signal_ratio_by_rule']:.3f}); "
                "this is a reported rule outcome, not a pre-decided conclusion.",
                "",
                "### Topology and root distances",
                "",
                "| Bus | Degree | Root depth | Cum. R | Cum. X | Cum. |Z| | Cum. length (km) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for bus, item in topology["node_records"].items():
            distance = item["cumulative_distances"]
            lines.append(
                f"| `{bus}` | {item['degree']} | {md_value(item['root_depth_hops'])} | "
                f"{md_value(distance['cumulative_r_distance'])} | {md_value(distance['cumulative_x_distance'])} | "
                f"{md_value(distance['cumulative_impedance_abs_distance'])} | "
                f"{md_value(distance['cumulative_physical_line_length'])} |"
            )
        lines.extend(
            [
                "",
                "### Node static features",
                "",
                f"Available `mpc_bus.csv` fields: {md_value(grid['node_static_features']['available_mpc_bus_fields'])}.",
                f"Topology-derived fields: {md_value(grid['node_static_features']['topology_derived_fields'])}.",
                "",
                "### Edge feature audit",
                "",
                f"Extra edge match: `{grid['edge_extra_match']['status']}`; "
                f"directed={grid['edge_extra_match']['directed_match_count']}; "
                f"undirected={grid['edge_extra_match']['undirected_match_count']}; "
                f"unmatched={grid['edge_extra_match']['unmatched_count']}.",
                "",
                "| Feature | Source | Missing | Zero | Negative | Min | Mean | Max | Numeric |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for feature, item in grid["edge_features"]["summaries"].items():
            lines.append(
                f"| `{feature}` | `{item['source']}` | {item['missing_count']} | {md_value(item['zero_count'])} | "
                f"{md_value(item['negative_count'])} | {md_value(item['min'])} | {md_value(item['mean'])} | "
                f"{md_value(item['max'])} | {md_value(item['numeric'])} |"
            )
        lines.extend(
            [
                "",
                "### Path feature constructibility",
                "",
            ]
        )
        for feature, item in grid["path_feature_constructibility"].items():
            lines.append(
                f"- `{feature}`: constructible_for_all_nodes={md_value(item['constructible_for_all_nodes'])}; "
                f"missing_nodes={md_value(item['missing_nodes'])}"
            )
        lines.append("")

    comparison = report["cross_grid_topology_comparison"]
    lines.extend(
        [
            "## Cross-grid topology comparison",
            "",
            "| Grid | Buses | Branches | Components | Tree | Max depth | Mean depth | Branch types |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for item in comparison["grid_signatures"]:
        lines.append(
            f"| {item['name']} | {item['bus_count']} | {item['branch_count']} | {item['connected_components']} | "
            f"{md_value(item['is_radial_tree'])} | {md_value(item['maximum_depth'])} | "
            f"{md_value(item['average_depth'])} | {md_value(list(item['branch_types']))} |"
        )
    lines.extend(
        [
            "",
            f"Obvious structural heterogeneity: **{md_value(comparison['heterogeneity']['obvious_structural_heterogeneity'])}**.",
            f"Basis: {comparison['heterogeneity']['basis']}.",
            "",
        ]
    )
    conclusions = report["conclusions"]
    q_grid_text = "; ".join(
        f"{name}: {status}"
        for name, status in conclusions["q_as_independent_dynamic_input"]["per_grid_rule_assessment"].items()
    )
    lines.extend(
        [
            "## Conclusions",
            "",
            f"- P as prediction target: **{conclusions['p_as_prediction_target']['assessment']}**. "
            f"{conclusions['p_as_prediction_target']['basis']}",
            f"- Q as independent dynamic input: **{conclusions['q_as_independent_dynamic_input']['assessment']}**. "
            f"Per-grid rule outcomes: {q_grid_text}.",
            f"- Directly constructible node static features: {md_value(conclusions['available_node_static_features']['directly_constructible_topology_features'])}.",
            f"- Present mpc_bus fields: {md_value(conclusions['available_node_static_features']['mpc_bus_fields_present'])}; "
            f"static/constraint candidates: {md_value(conclusions['available_node_static_features']['mpc_bus_static_or_constraint_candidates'])}; "
            f"review before use: {md_value(conclusions['available_node_static_features']['requires_semantic_or_target_leakage_review'])}.",
            f"- Available edge features: {md_value(conclusions['available_edge_features'])}.",
            f"- Constructible electrical-distance features: {conclusions['constructible_electrical_distance_features']}.",
            f"- Obvious topology structural heterogeneity: **{md_value(conclusions['topology_structural_heterogeneity']['assessment'])}**. "
            f"{conclusions['topology_structural_heterogeneity']['basis']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root.resolve(), args.mapping_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"lv_grids={len(report['lv_grids'])}")


if __name__ == "__main__":
    main()
