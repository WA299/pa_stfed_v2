"""Audit V2 schema, timestamps, and load-profile-to-bus mappings.

The audit is intentionally limited to schema and identifier relationships. It
does not clean data, calculate forecasting metrics, or train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LV_DATASET = "norway_4_lv_grids"
INDUSTRIAL_DATASET = "norway_industrial_mvlv"
LV_FILES = ("p_load.csv", "q_load.csv")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root.parent / "pa_stfed_data_v2" / "raw"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "results" / "audits" / "v2_schema_mapping.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=repo_root / "results" / "audits" / "v2_schema_mapping.md",
    )
    return parser.parse_args()


def normalise_identifier(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def scalar_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).strip()


def read_csv_header(path: Path, delimiter: str = ",") -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle, delimiter=delimiter))


def json_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return value.isoformat(sep=" ")


def interval_values(timestamps: pd.Series) -> list[float]:
    if timestamps.empty or timestamps.isna().any():
        return []
    differences = timestamps.diff().dropna().dt.total_seconds().div(60)
    return sorted({float(value) for value in differences.tolist()})


def inspect_time_file(
    path: Path,
    *,
    timestamp_column: int | str = 0,
    timestamp_format: str | None = None,
) -> dict[str, Any]:
    try:
        frame = pd.read_csv(
            path,
            usecols=[timestamp_column],
            dtype=str,
            encoding="utf-8-sig",
            sep=";" if path.suffix.lower() == ".txt" else ",",
        )
    except Exception as exc:  # Keep the audit useful when one source is malformed.
        return {
            "status": "read_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }

    raw = frame.iloc[:, 0]
    parsed = pd.to_datetime(raw, format=timestamp_format, errors="coerce")
    intervals = interval_values(parsed)
    valid = parsed.dropna()
    row_count = int(len(parsed))
    invalid_count = int(parsed.isna().sum())
    unique_count = int(parsed.nunique(dropna=True))
    return {
        "status": "success",
        "row_count": row_count,
        "invalid_timestamp_count": invalid_count,
        "date_parseable": invalid_count == 0,
        "first_timestamp": json_timestamp(parsed.iloc[0] if row_count else None),
        "last_timestamp": json_timestamp(parsed.iloc[-1] if row_count else None),
        "min_timestamp": json_timestamp(valid.min() if not valid.empty else None),
        "max_timestamp": json_timestamp(valid.max() if not valid.empty else None),
        "unique_timestamp_count": unique_count,
        "observed_interval_minutes": intervals,
        "interval_consistent": len(intervals) == 1,
        "hourly_interval": intervals == [60.0],
        "strict_8760_hourly": (
            row_count == 8760
            and invalid_count == 0
            and unique_count == 8760
            and intervals == [60.0]
        ),
    }


def read_bus_ids(path: Path, delimiter: str, column: str) -> set[str]:
    frame = pd.read_csv(path, sep=delimiter, dtype=str, encoding="utf-8-sig")
    return {
        identifier
        for identifier in (normalise_identifier(value) for value in frame[column])
        if identifier is not None
    }


def read_load_bus_rows(path: Path) -> list[dict[str, str | None]]:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if "bus_i" not in frame.columns:
        raise ValueError("load_bus_extra.csv has no bus_i column")
    rows: list[dict[str, str | None]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "bus_i": normalise_identifier(row.get("bus_i")),
                "consumer": scalar_text(row.get("Consumer type")),
                "consumer_id": scalar_text(row.get("ID")),
            }
        )
    return rows


def profile_mapping(
    p_headers: list[str], q_headers: list[str], load_rows: list[dict[str, str | None]], bus_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p_profiles = p_headers[1:] if p_headers and p_headers[0].strip().lower() == "date" else p_headers
    q_profiles = q_headers[1:] if q_headers and q_headers[0].strip().lower() == "date" else q_headers
    mapping: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    max_count = max(len(p_profiles), len(q_profiles), len(load_rows))
    for position in range(max_count):
        load_row = load_rows[position] if position < len(load_rows) else None
        bus_i = load_row["bus_i"] if load_row else None
        has_p = position < len(p_profiles)
        has_q = position < len(q_profiles)
        bus_exists = bus_i in bus_ids if bus_i is not None else False
        status = "mapped" if has_p and has_q and load_row and bus_exists else "unmapped"
        item = {
            "profile_position": position + 1,
            "p_raw_header": p_profiles[position] if has_p else None,
            "q_raw_header": q_profiles[position] if has_q else None,
            "mapped_bus_i": bus_i,
            "consumer": load_row["consumer"] if load_row else None,
            "consumer_id": load_row["consumer_id"] if load_row else None,
            "bus_exists_in_mpc_bus": bus_exists,
            "p_q_columns_present": has_p and has_q,
            "status": status,
            "mapping_basis": "profile column position -> load_bus_extra row order",
        }
        mapping.append(item)
        if status != "mapped":
            unmapped.append(item)
    return mapping, unmapped


def duplicate_bus_summary(load_rows: list[dict[str, str | None]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in load_rows:
        bus_i = row["bus_i"]
        if bus_i is not None:
            counts[bus_i] = counts.get(bus_i, 0) + 1
    return [
        {"bus_i": bus_i, "consumer_count": count}
        for bus_i, count in sorted(counts.items(), key=lambda item: item[0])
        if count > 1
    ]


def audit_lv_grid(root: Path, name: str) -> dict[str, Any]:
    p_path = root / "p_load.csv"
    q_path = root / "q_load.csv"
    p_headers = read_csv_header(p_path)
    q_headers = read_csv_header(q_path)
    load_rows = read_load_bus_rows(root / "load_bus_extra.csv")
    bus_ids = read_bus_ids(root / "mpc_bus.csv", ",", "bus_i")
    mapping, unmapped = profile_mapping(p_headers, q_headers, load_rows, bus_ids)
    duplicates = duplicate_bus_summary(load_rows)

    p_time = inspect_time_file(p_path, timestamp_column=0, timestamp_format="%Y-%m-%d %H:%M:%S")
    q_time = inspect_time_file(q_path, timestamp_column=0, timestamp_format="%Y-%m-%d %H:%M:%S")
    p_profile_count = max(len(p_headers) - 1, 0)
    q_profile_count = max(len(q_headers) - 1, 0)
    time_alignment = {
        "row_count_equal": p_time.get("row_count") == q_time.get("row_count"),
        "first_timestamp_equal": p_time.get("first_timestamp") == q_time.get("first_timestamp"),
        "last_timestamp_equal": p_time.get("last_timestamp") == q_time.get("last_timestamp"),
        "timestamp_schema_aligned": (
            p_time.get("row_count") == q_time.get("row_count")
            and p_time.get("first_timestamp") == q_time.get("first_timestamp")
            and p_time.get("last_timestamp") == q_time.get("last_timestamp")
        ),
    }
    p_q_one_to_one = (
        p_profile_count == q_profile_count == len(load_rows)
        and not unmapped
        and time_alignment["timestamp_schema_aligned"]
    )
    raw_header_match = p_headers[1:] == q_headers[1:]
    load_bus_count = len({row["bus_i"] for row in load_rows if row["bus_i"] is not None})
    return {
        "name": name,
        "relative_root": name,
        "bus_count": int(len(pd.read_csv(root / "mpc_bus.csv", dtype=str))),
        "branch_count": int(len(pd.read_csv(root / "mpc_branch.csv", dtype=str))),
        "consumer_count": len(load_rows),
        "p_load_profile_count": p_profile_count,
        "q_load_profile_count": q_profile_count,
        "unique_load_bus_count": load_bus_count,
        "multiple_consumers_per_bus_observed": bool(duplicates),
        "multiple_consumers_per_bus_allowed_by_mapping": True,
        "buses_with_multiple_consumers": duplicates,
        "p_load": {
            "relative_path": "p_load.csv",
            "raw_header": p_headers,
            "time": p_time,
        },
        "q_load": {
            "relative_path": "q_load.csv",
            "raw_header": q_headers,
            "time": q_time,
        },
        "p_q": {
            "profile_count_equal": p_profile_count == q_profile_count,
            "load_bus_extra_count_matches": len(load_rows) == p_profile_count == q_profile_count,
            "raw_header_tokens_equal_by_position": raw_header_match,
            "mapping_uses_raw_column_position": True,
            "timestamp_alignment": time_alignment,
            "profile_one_to_one": p_q_one_to_one,
            "note": (
                "Raw duplicate headers and suffix-like headers are not treated as bus IDs; "
                "mapping is positional against load_bus_extra.csv."
            ),
        },
        "profile_mapping": mapping,
        "unmapped_profiles": unmapped,
        "mapping_anomalies": [
            "p_q_profile_count_mismatch" if p_profile_count != q_profile_count else None,
            "load_bus_extra_count_mismatch" if len(load_rows) != p_profile_count else None,
            "unmapped_profile" if unmapped else None,
        ],
    }


def audit_industrial(root: Path, source_resolution: str = "named_directory") -> dict[str, Any]:
    bus_frame = pd.read_csv(root / "bus.csv", sep=";", dtype=str, encoding="utf-8-sig")
    bus_ids = {
        identifier
        for identifier in (normalise_identifier(value) for value in bus_frame["BUS_I"])
        if identifier is not None
    }
    load_files = sorted(root.glob("*.txt"), key=lambda path: path.name)
    file_records: list[dict[str, Any]] = []
    mapping_anomalies: list[dict[str, Any]] = []
    for path in load_files:
        frame = pd.read_csv(
            path,
            sep=";",
            usecols=["Bus_ID", "Timestamp"],
            dtype=str,
            encoding="utf-8-sig",
        )
        file_bus_ids = sorted(
            {
                identifier
                for identifier in (normalise_identifier(value) for value in frame["Bus_ID"])
                if identifier is not None
            }
        )
        missing = sorted(set(file_bus_ids) - bus_ids)
        time = inspect_time_file(
            path,
            timestamp_column="Timestamp",
            timestamp_format="%d/%m/%Y %H:%M:%S",
        )
        record = {
            "relative_path": path.name,
            "unique_bus_ids": file_bus_ids,
            "unique_bus_id_count": len(file_bus_ids),
            "bus_ids_exist_in_bus_csv": not missing,
            "missing_bus_ids": missing,
            "time": time,
        }
        file_records.append(record)
        if missing:
            mapping_anomalies.append({"file": path.name, "missing_bus_ids": missing})
        if len(file_bus_ids) != 1:
            mapping_anomalies.append(
                {"file": path.name, "issue": "expected_one_unique_Bus_ID", "unique_bus_ids": file_bus_ids}
            )

    temperature_files = sorted(
        path.name for path in root.rglob("*") if path.is_file() and "temp" in path.name.lower()
    )
    weather_files = sorted(
        path.name for path in root.rglob("*") if path.is_file() and "weather" in path.name.lower()
    )
    return {
        "name": INDUSTRIAL_DATASET,
        "relative_root": INDUSTRIAL_DATASET,
        "source_resolution": source_resolution,
        "bus_count": int(len(bus_frame)),
        "bus_id_column": "BUS_I",
        "load_txt_file_count": len(load_files),
        "load_txt_files": file_records,
        "temperature_weather_check": {
            "temperature_files": temperature_files,
            "weather_files": weather_files,
            "checked_by_filename_only": True,
            "numeric_values_analyzed": False,
        },
        "mapping_anomalies": mapping_anomalies,
    }


def clean_anomaly_lists(report: dict[str, Any]) -> None:
    for grid in report["lv_grids"]:
        grid["mapping_anomalies"] = [item for item in grid["mapping_anomalies"] if item is not None]


def resolve_industrial_root(data_root: Path) -> tuple[Path | None, str]:
    named_root = data_root / INDUSTRIAL_DATASET
    if named_root.is_dir():
        return named_root, "named_directory"
    candidates = []
    for candidate in data_root.iterdir() if data_root.is_dir() else []:
        if not candidate.is_dir():
            continue
        if (
            (candidate / "bus.csv").is_file()
            and (candidate / "branch.csv").is_file()
            and (candidate / "gen.csv").is_file()
            and any(candidate.glob("*.txt"))
        ):
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0], "unique_signature_match"
    return None, "not_found_or_ambiguous"


def build_report(data_root: Path) -> dict[str, Any]:
    lv_root = data_root / LV_DATASET
    industrial_root, industrial_resolution = resolve_industrial_root(data_root)
    lv_grids = []
    if lv_root.is_dir():
        for grid_root in sorted(path for path in lv_root.iterdir() if path.is_dir()):
            lv_grids.append(audit_lv_grid(grid_root, grid_root.name))
    industrial = (
        audit_industrial(industrial_root, industrial_resolution)
        if industrial_root is not None and industrial_root.is_dir()
        else None
    )
    report = {
        "audit": "v2_schema_mapping",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "Schema and identifier mapping only.",
            "No model training, forecasting metrics, data cleaning, or complete statistical audit.",
            "All paths are relative labels; source drive paths are not written to this report.",
        ],
        "lv_grids": lv_grids,
        "industrial_mvlv": industrial,
    }
    clean_anomaly_lists(report)
    return report


def md_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Schema and ID Mapping Audit",
        "",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        "",
        "Scope is limited to schema, timestamps, and identifier mappings; no model or forecasting analysis is performed.",
        "",
    ]
    for grid in report["lv_grids"]:
        lines.extend(
            [
                f"## LV grid: {grid['name']}",
                "",
                (
                    f"bus_count={grid['bus_count']}; branch_count={grid['branch_count']}; "
                    f"consumer_count={grid['consumer_count']}; "
                    f"unique_load_bus_count={grid['unique_load_bus_count']}"
                ),
                f"Multiple consumers on one bus observed: **{md_value(grid['multiple_consumers_per_bus_observed'])}**.",
                f"Multiple consumers per bus allowed by row mapping: **{md_value(grid['multiple_consumers_per_bus_allowed_by_mapping'])}**.",
                f"P/Q profile one-to-one: **{md_value(grid['p_q']['profile_one_to_one'])}**.",
                f"Unmapped profile count: **{len(grid['unmapped_profiles'])}**.",
                "",
                "| File | Date parseable | First | Last | Rows | Interval (min) | Strict 8760 hourly |",
                "| --- | --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for key in ("p_load", "q_load"):
            time = grid[key]["time"]
            lines.append(
                f"| `{grid[key]['relative_path']}` | {md_value(time.get('date_parseable'))} | "
                f"{md_value(time.get('first_timestamp'))} | {md_value(time.get('last_timestamp'))} | "
                f"{md_value(time.get('row_count'))} | {md_value(time.get('observed_interval_minutes'))} | "
                f"{md_value(time.get('strict_8760_hourly'))} |"
            )
        lines.extend(
            [
                "",
                "Profile mapping uses raw CSV column position -> `load_bus_extra.csv` row order:",
                "",
                "| Position | P raw header | Q raw header | bus_i | Consumer | ID | Bus exists | Status |",
                "| ---: | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in grid["profile_mapping"]:
            lines.append(
                f"| {item['profile_position']} | `{md_value(item['p_raw_header'])}` | "
                f"`{md_value(item['q_raw_header'])}` | `{md_value(item['mapped_bus_i'])}` | "
                f"{md_value(item['consumer'])} | `{md_value(item['consumer_id'])}` | "
                f"{md_value(item['bus_exists_in_mpc_bus'])} | {item['status']} |"
            )
        lines.extend(["", "Mapping anomalies: " + (", ".join(grid["mapping_anomalies"]) or "none"), ""])

    industrial = report["industrial_mvlv"]
    if industrial is not None:
        lines.extend(
            [
                "## Industrial MV/LV",
                "",
                f"bus_count={industrial['bus_count']}; load TXT file count={industrial['load_txt_file_count']}.",
                f"Temperature files: {md_value(industrial['temperature_weather_check']['temperature_files'])}; "
                f"weather files: {md_value(industrial['temperature_weather_check']['weather_files'])}.",
                "",
                "| Load TXT | Unique Bus_ID | Bus IDs exist in bus.csv | First | Last | Interval (min) |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for record in industrial["load_txt_files"]:
            time = record["time"]
            lines.append(
                f"| `{record['relative_path']}` | {md_value(record['unique_bus_ids'])} | "
                f"{md_value(record['bus_ids_exist_in_bus_csv'])} | {md_value(time.get('first_timestamp'))} | "
                f"{md_value(time.get('last_timestamp'))} | {md_value(time.get('observed_interval_minutes'))} |"
            )
        lines.extend(
            [
                "",
                "Industrial mapping anomalies: "
                + (md_value(industrial["mapping_anomalies"]) if industrial["mapping_anomalies"] else "none"),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")

    lv_anomalies = sum(len(grid["mapping_anomalies"]) for grid in report["lv_grids"])
    industrial_anomalies = len(report["industrial_mvlv"]["mapping_anomalies"]) if report["industrial_mvlv"] else 0
    print(
        f"lv_grids={len(report['lv_grids'])} "
        f"industrial_txt={report['industrial_mvlv']['load_txt_file_count'] if report['industrial_mvlv'] else 0} "
        f"mapping_anomalies={lv_anomalies + industrial_anomalies}"
    )


if __name__ == "__main__":
    main()
