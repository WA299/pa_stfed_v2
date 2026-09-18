"""List the on-disk structure of the V2 Norway datasets.

This audit intentionally performs structure inspection only. It does not alter
the source data, calculate forecasting metrics, clean data, or train models.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TABLE_EXTENSIONS = {".csv", ".txt", ".tsv", ".tab"}
DELIMITER_CANDIDATES = (";", ",", "\t", "|")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root.parent / "pa_stfed_data_v2" / "raw"
    default_output_json = repo_root / "results" / "audits" / "v2_dataset_structure.json"
    default_output_md = repo_root / "results" / "audits" / "v2_dataset_structure.md"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-json", type=Path, default=default_output_json)
    parser.add_argument("--output-md", type=Path, default=default_output_md)
    return parser.parse_args()


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_paths(root: Path) -> tuple[list[Path], list[Path]]:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: relative_posix(path, root),
    )
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: relative_posix(path, root),
    )
    return directories, files


def sample_text(path: Path, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes)
    return raw.decode("utf-8-sig", errors="replace")


def detect_delimiter(path: Path) -> str | None:
    sample = sample_text(path)
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return None

    first_line = lines[0]
    counts = {delimiter: first_line.count(delimiter) for delimiter in DELIMITER_CANDIDATES}
    best_delimiter, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count:
        return best_delimiter

    try:
        return csv.Sniffer().sniff("\n".join(lines[:10]), delimiters=";,\t|").delimiter
    except csv.Error:
        return None


def clean_column_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def inspect_table(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"attempted": True}
    delimiter = detect_delimiter(path)
    if delimiter is None:
        # A one-column CSV/TXT has no delimiter, but pandas can still safely
        # expose its header and shape with comma as a separator.
        delimiter = ","

    try:
        frame = pd.read_csv(
            path,
            sep=delimiter,
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
            on_bad_lines="error",
        )
        result.update(
            {
                "status": "success",
                "fields": [clean_column_name(column) for column in frame.columns],
                "shape": [int(frame.shape[0]), int(frame.shape[1])],
                "delimiter": "\\t" if delimiter == "\t" else delimiter,
            }
        )
    except Exception as exc:  # A structure listing should continue after one bad table.
        result.update(
            {
                "status": "error",
                "fields": [],
                "shape": None,
                "delimiter": "\\t" if delimiter == "\t" else delimiter,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
        )
    return result


def inspect_dataset(label: str, root: Path) -> dict[str, Any]:
    directories, files = iter_paths(root)
    file_records: list[dict[str, Any]] = []
    for path in files:
        extension = path.suffix.lower()
        record: dict[str, Any] = {
            "relative_path": relative_posix(path, root),
            "extension": extension,
            "size_bytes": int(path.stat().st_size),
        }
        if extension in TABLE_EXTENSIONS:
            record["table"] = inspect_table(path)
        file_records.append(record)

    table_records = [record for record in file_records if "table" in record]
    successful_tables = sum(record["table"]["status"] == "success" for record in table_records)
    return {
        "name": label,
        "directories": [relative_posix(path, root) for path in directories],
        "files": file_records,
        "summary": {
            "directory_count": len(directories),
            "file_count": len(files),
            "table_file_count": len(table_records),
            "table_read_success_count": successful_tables,
        },
    }


def build_report(data_root: Path) -> dict[str, Any]:
    dataset_specs = (
        ("norway_4_lv_grids", data_root / "norway_4_lv_grids"),
        ("norway_industrial_mvlv", data_root / "norway_industrial_mvlv"),
    )
    datasets: list[dict[str, Any]] = []
    missing: list[str] = []
    for label, root in dataset_specs:
        if root.is_dir():
            datasets.append(inspect_dataset(label, root))
        else:
            missing.append(label)

    return {
        "audit": "v2_dataset_structure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_roots": [label for label, _ in dataset_specs],
        "datasets": datasets,
        "missing_datasets": missing,
        "constraints": [
            "Recursive file and directory structure listing only.",
            "No source data is modified.",
            "No forecasting metrics, model training, data cleaning, or full audit is performed.",
            "Paths in this report are relative to each dataset root.",
        ],
    }


def format_shape(shape: Iterable[int] | None) -> str:
    if shape is None:
        return "-"
    values = list(shape)
    return "(" + ", ".join(str(value) for value in values) + ")"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Dataset Structure",
        "",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        "",
        "This report lists relative paths, extensions, file sizes, and safely readable table headers/shapes.",
        "",
    ]
    for dataset in report["datasets"]:
        summary = dataset["summary"]
        lines.extend(
            [
                f"## {dataset['name']}",
                "",
                (
                    f"Directories: {summary['directory_count']}  "
                    f"Files: {summary['file_count']}  "
                    f"Tables read successfully: {summary['table_read_success_count']}/"
                    f"{summary['table_file_count']}"
                ),
                "",
                "| Relative path | Extension | Size (bytes) | Table status | Fields | Shape |",
                "| --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for record in dataset["files"]:
            table = record.get("table", {})
            fields = ", ".join(table.get("fields", []))
            if len(fields) > 240:
                fields = fields[:237] + "..."
            lines.append(
                f"| `{record['relative_path']}` | `{record['extension'] or '[none]'}` | "
                f"{record['size_bytes']} | {table.get('status', '-')} | "
                f"{fields or '-'} | {format_shape(table.get('shape'))} |"
            )
        lines.append("")

    if report["missing_datasets"]:
        lines.extend(["## Missing datasets", "", ", ".join(report["missing_datasets"]), ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    report = build_report(data_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")

    total_files = sum(dataset["summary"]["file_count"] for dataset in report["datasets"])
    print(f"datasets={len(report['datasets'])} files={total_files}")
    for dataset in report["datasets"]:
        summary = dataset["summary"]
        print(
            f"{dataset['name']}: files={summary['file_count']} "
            f"tables_read={summary['table_read_success_count']}/{summary['table_file_count']}"
        )


if __name__ == "__main__":
    main()
