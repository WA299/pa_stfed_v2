"""Build the paper-ready package from accepted, frozen result JSON files only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "paper_ready"
SEEDS = (42, 123, 2026)
REFERENCE_CLIENTS = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
REFERENCE_METHODS = (
    "scarce_local",
    "fedavg",
    "fedprox",
    "fedper",
    "fedfomo_style",
    "btd_fl_direct_transfer",
)
INDUSTRIAL_METHODS = (
    "industrial_scarce_local",
    "fedavg",
    "fedprox",
    "fedper",
    "fedfomo_style",
    "btd_fl_direct_transfer",
)
LABELS = {
    "scarce_local": "Scarce Local",
    "industrial_scarce_local": "Industrial Local",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedper": "FedPer",
    "fedfomo_style": "FedFomo-style",
    "btd_fl_direct_transfer": "BTD-FL",
}
CENTRAL_ABLATION = Path("results/ablations/puc_rstattn_v2_ablation_summary.json")
CENTRAL_SUMMARY = Path("results/centralized/centralized_validation_summary.json")
REF25_PATH = Path("results/federated/formal_25pct/multiseed/multiseed_25pct_summary.json")
REF50_PATH = Path("results/federated/formal_50pct/multiseed/multiseed_50pct_summary.json")
IND25_PATH = Path("results/federated/industrial_external/industrial_external_25pct_summary.json")
IND50_PATH = Path("results/federated/industrial_external/industrial_external_50pct_summary.json")
SCIENTIFIC_METHOD_COMMIT = "1681741779df301dcb64587090ce262d2ad1d443"
ACCEPTED_RESULTS_COMMIT = "e8dd4c3fbc4936175d3e9f0fee5c7c06d6303d3b"
REPORTING_COMMIT_PLACEHOLDER = "REPORTING_COMMIT_TO_BE_RECORDED_AFTER_THIS_PACKAGE_IS_COMMITTED"


def resolve_json_path(data: Any, key_path: str) -> Any:
    """Resolve a dot-separated object path; missing keys always raise."""
    current = data
    for part in key_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(key_path)
        current = current[part]
    return current


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {label}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    _reject_nonfinite(data, str(path))
    return data


def require_false(data: Mapping[str, Any], key_path: str, path: Path) -> None:
    try:
        value = resolve_json_path(data, key_path)
    except KeyError as exc:
        raise ValueError(f"missing TEST guardrail {key_path}: {path}") from exc
    if value is not False:
        raise ValueError(f"TEST guardrail {key_path} is not false: {path}")


def validate_source_test_guardrail(path: Path, data: Mapping[str, Any]) -> tuple[str, bool]:
    """Validate known frozen schemas. Unknown or incomplete schemas fail closed."""
    name = path.name
    if name in {CENTRAL_ABLATION.name, CENTRAL_SUMMARY.name}:
        guardrail = "metadata.test_evaluated"
    elif name in {REF25_PATH.name, REF50_PATH.name}:
        guardrail = "guardrails.test_evaluated"
    elif name.startswith(("btd_", "fedfomo_style_", "formal_baselines_")):
        guardrail = "test_evaluated"
    else:
        raise ValueError(f"unknown TEST provenance schema: {path}")
    require_false(data, guardrail, path)
    return guardrail, False


def _same_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)
    )


def _assert_exact_set(actual: Iterable[Any], expected: Sequence[Any], label: str) -> None:
    actual_tuple = tuple(actual)
    if len(actual_tuple) != len(expected) or set(actual_tuple) != set(expected):
        raise ValueError(f"{label} mismatch: {actual_tuple!r}")


def reference_underlying_paths(root: Path, fraction: float) -> list[Path]:
    if fraction == 0.25:
        base = root / "results/federated/formal_25pct"
        multi = base / "multiseed"
        return [
            base / "btd_fl_25pct_seed42.json",
            base / "btd_fl_direct_transfer_25pct_seed42.json",
            base / "fedfomo_style_25pct_seed42.json",
            *[
                multi / f"{prefix}_25pct_seed{seed}.json"
                for seed in (123, 2026)
                for prefix in ("formal_baselines", "btd_fl_direct_transfer", "fedfomo_style")
            ],
        ]
    base = root / "results/federated/formal_50pct/multiseed"
    return [
        base / f"{prefix}_50pct_seed{seed}.json"
        for seed in SEEDS
        for prefix in ("formal_baselines", "btd_fl_direct_transfer", "fedfomo_style")
    ]


def validate_reference(root: Path, relative_path: Path, fraction: float) -> dict[str, Any]:
    path = root / relative_path
    data = load_json(path)
    validate_source_test_guardrail(path, data)
    expected_experiment = f"multiseed_formal_{int(fraction * 100)}pct_summary"
    if data.get("experiment") != expected_experiment:
        raise ValueError(f"wrong reference experiment identity: {path}")
    if data.get("seeds") != list(SEEDS):
        raise ValueError(f"reference seed set is not exactly {SEEDS}: {path}")
    _assert_exact_set(data.get("methods", ()), REFERENCE_METHODS, "reference methods")
    if tuple(data.get("client_grid_names", ())) != REFERENCE_CLIENTS:
        raise ValueError(f"reference client set mismatch: {path}")
    if fraction == 0.25:
        if relative_path != REF25_PATH or "history_fraction" in data:
            raise ValueError("historical 25% summary identity must be path/name/experiment based")
    elif not _same_number(data.get("history_fraction"), 0.5):
        raise ValueError("50% summary history_fraction must be exactly 0.5")
    for artifact_path in reference_underlying_paths(root, fraction):
        artifact = load_json(artifact_path)
        validate_source_test_guardrail(artifact_path, artifact)
        if artifact.get("seed") not in SEEDS:
            raise ValueError(f"invalid underlying reference seed: {artifact_path}")
        recorded_fraction = artifact.get("history_fraction")
        if recorded_fraction is not None and not _same_number(recorded_fraction, fraction):
            raise ValueError(f"underlying reference fraction mismatch: {artifact_path}")
    _validate_reference_arithmetic(data)
    return data


def _validate_reference_arithmetic(data: Mapping[str, Any]) -> None:
    for method in REFERENCE_METHODS:
        for split in ("audit", "validation"):
            for scope in ("node_macro", "grid_aggregate"):
                for metric_name in ("mae", "rmse", "wape_pct", "smape_pct"):
                    item = data["metrics"][method][split][scope][metric_name]
                    values = [float(item["per_seed"][str(seed)]) for seed in SEEDS]
                    if not _same_number(item["mean"], statistics.fmean(values)):
                        raise ValueError("reference mean arithmetic mismatch")
                    if not _same_number(item["std"], statistics.pstdev(values)):
                        raise ValueError("reference std arithmetic mismatch")
    for comparator in REFERENCE_METHODS[:-1]:
        item = data["btd_vs_comparators"][comparator]
        values = [float(item["per_seed"][str(seed)]["relative_improvement"]["validation"]) for seed in SEEDS]
        if not _same_number(item["mean_relative_improvement"], statistics.fmean(values)):
            raise ValueError("reference comparison mean mismatch")
        if not _same_number(item["std_relative_improvement"], statistics.pstdev(values)):
            raise ValueError("reference comparison std mismatch")
        if item["seeds_btd_wins"] != sum(value > 0 for value in values):
            raise ValueError("reference comparison win-count mismatch")


def industrial_artifact_path(root: Path, fraction: float, kind: str, seed: int) -> Path:
    pct = int(fraction * 100)
    return root / "results/federated/industrial_external" / f"industrial_{kind}_{pct}pct_seed{seed}.json"


def validate_industrial(root: Path, relative_path: Path, fraction: float) -> dict[str, Any]:
    path = root / relative_path
    data = load_json(path)
    if data.get("seeds") != list(SEEDS):
        raise ValueError(f"industrial seed set is not exactly {SEEDS}: {path}")
    _assert_exact_set(data.get("methods", {}).keys(), INDUSTRIAL_METHODS, "industrial methods")
    if not _same_number(data.get("history_fraction"), fraction):
        raise ValueError(f"industrial history fraction mismatch: {path}")
    from code.federated.industrial_result_validation import load_valid_external_result

    contracts = {
        "baselines": ("industrial_baselines", 10, 5),
        "btd_direct_transfer": ("industrial_btd_direct_transfer", 0, 0),
        "fedfomo_style": ("industrial_fedfomo_style", 10, 5),
    }
    for seed in SEEDS:
        for kind, (artifact_type, rounds, local_epochs) in contracts.items():
            load_valid_external_result(
                industrial_artifact_path(root, fraction, kind, seed),
                artifact_type=artifact_type,
                seed=seed,
                fraction=fraction,
                run_mode="real",
                rounds=rounds,
                local_epochs=local_epochs,
                max_epochs=50,
                batch_size=32,
            )
    _validate_industrial_arithmetic(data)
    return data


def _validate_industrial_arithmetic(data: Mapping[str, Any]) -> None:
    for method in INDUSTRIAL_METHODS:
        for split in ("audit", "validation"):
            for scope in ("node_macro", "grid_aggregate"):
                for metric_name in ("mae", "rmse", "wape_pct", "smape_pct"):
                    item = data["methods"][method][split][scope][metric_name]
                    values = [float(value) for value in item["values"]]
                    if len(values) != len(SEEDS):
                        raise ValueError("industrial summary must contain exactly three values")
                    if not _same_number(item["mean"], statistics.fmean(values)):
                        raise ValueError("industrial mean arithmetic mismatch")
                    if not _same_number(item["std"], statistics.pstdev(values)):
                        raise ValueError("industrial std arithmetic mismatch")
    for comparator in INDUSTRIAL_METHODS[:-1]:
        item = data["comparisons_btd_vs_comparators"][comparator]
        values = [float(item["per_seed"][str(seed)]["relative_improvement"]) for seed in SEEDS]
        if not _same_number(item["mean_relative_improvement"], statistics.fmean(values)):
            raise ValueError("industrial comparison mean mismatch")
        if not _same_number(item["std_relative_improvement"], statistics.pstdev(values)):
            raise ValueError("industrial comparison std mismatch")
        if item["seed_win_count"] != sum(value > 0 for value in values):
            raise ValueError("industrial comparison win-count mismatch")
    behavior = data["btd_donor_behavior"]
    benefits = [
        float(value)
        for seed in map(str, SEEDS)
        for value in behavior["by_seed"][seed]["benefits"].values()
    ]
    if behavior["positive_benefit_count"] != sum(value > 0 for value in benefits):
        raise ValueError("industrial positive-benefit count mismatch")
    if behavior["non_positive_benefit_count"] != sum(value <= 0 for value in benefits):
        raise ValueError("industrial non-positive-benefit count mismatch")
    if behavior["fallback_count"] != sum(
        bool(behavior["by_seed"][str(seed)]["zero_transfer_fallback"]) for seed in SEEDS
    ):
        raise ValueError("industrial fallback count mismatch")


@dataclass(frozen=True)
class SourceValue:
    path: Path
    key: str
    value: float | int
    guardrail_path: str


@dataclass
class BuildContext:
    root: Path
    out: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    def source(self, relative_path: Path, data: Mapping[str, Any], key: str) -> SourceValue:
        value = resolve_json_path(data, key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"paper number is not finite numeric: {relative_path}:{key}")
        if relative_path in (IND25_PATH, IND50_PATH):
            guardrail_path = "underlying_real_artifacts[*].test_evaluated"
        else:
            guardrail_path, _ = validate_source_test_guardrail(self.root / relative_path, data)
        return SourceValue(relative_path, key, value, guardrail_path)

    def record(
        self,
        *,
        output_artifact: str,
        output_section: str,
        row: str,
        column: str | None,
        plotted_quantity: str | None,
        method: str,
        domain: str,
        history_fraction: float | None,
        seed_scope: str,
        split: str,
        metric: str,
        scope: str,
        source: SourceValue,
    ) -> float | int:
        self.entries.append(
            {
                "output_artifact": output_artifact,
                "output_section": output_section,
                "row": row,
                "column": column,
                "plotted_quantity": plotted_quantity,
                "method": method,
                "dataset_domain": domain,
                "history_fraction": history_fraction,
                "seed_scope": seed_scope,
                "split": split,
                "metric": metric,
                "scope": scope,
                "value": source.value,
                "source_artifact_path": source.path.as_posix(),
                "json_key_path": source.key,
                "source_frozen_status": "accepted frozen",
                "source_test_guardrail_path": source.guardrail_path,
                "source_test_guardrail_value": False,
            }
        )
        return source.value

    def record_derived(
        self,
        *,
        output_artifact: str,
        output_section: str,
        row: str,
        column: str,
        method: str,
        domain: str,
        history_fraction: float | None,
        seed_scope: str,
        split: str,
        metric: str,
        scope: str,
        value: float,
        operation: str,
        sources: Sequence[SourceValue],
    ) -> float:
        if not math.isfinite(value):
            raise ValueError("derived paper number is not finite")
        primary = sources[0]
        self.entries.append(
            {
                "output_artifact": output_artifact,
                "output_section": output_section,
                "row": row,
                "column": column,
                "plotted_quantity": None,
                "method": method,
                "dataset_domain": domain,
                "history_fraction": history_fraction,
                "seed_scope": seed_scope,
                "split": split,
                "metric": metric,
                "scope": scope,
                "value": value,
                "source_artifact_path": primary.path.as_posix(),
                "json_key_path": primary.key,
                "source_frozen_status": "derived from accepted frozen sources",
                "source_test_guardrail_path": primary.guardrail_path,
                "source_test_guardrail_value": False,
                "derivation": {
                    "operation": operation,
                    "sources": [
                        {"source_artifact_path": item.path.as_posix(), "json_key_path": item.key, "value": item.value}
                        for item in sources
                    ],
                },
            }
        )
        return value


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.17g}"


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _table_number(
    ctx: BuildContext,
    data: Mapping[str, Any],
    path: Path,
    key: str,
    *,
    artifact: str,
    section: str,
    row: str,
    column: str,
    method: str,
    domain: str,
    fraction: float | None,
    split: str,
    metric_name: str,
    scope: str,
) -> float | int:
    return ctx.record(
        output_artifact=artifact,
        output_section=section,
        row=row,
        column=column,
        plotted_quantity=None,
        method=method,
        domain=domain,
        history_fraction=fraction,
        seed_scope="mean/std across seeds 42/123/2026" if "mean" in column.lower() or "std" in column.lower() else "seeds 42/123/2026",
        split=split,
        metric=metric_name,
        scope=scope,
        source=ctx.source(path, data, key),
    )


def build_centralized(ctx: BuildContext, ablation: Mapping[str, Any], centralized: Mapping[str, Any]) -> None:
    artifact = "tables/table_centralized_main.csv"
    specs = (
        ("GRU anchor", "gru_anchor_only", CENTRAL_ABLATION, ablation),
        ("temporal_residual_only", "temporal_residual_only", CENTRAL_ABLATION, ablation),
        ("Graph WaveNet", "graph_wavenet", CENTRAL_SUMMARY, centralized),
        ("PUC-RSTAttn V2", "full_puc_rstattn_v2", CENTRAL_ABLATION, ablation),
    )
    headers = ("Method", "Split", "Node MAE", "Node RMSE", "Node WAPE %", "Node sMAPE %", "Source", "JSON key path")
    rows: list[list[Any]] = []
    for label, method, path, data in specs:
        base = f"macro_across_grids.node_macro.{method}.metrics"
        values = []
        for metric_name, column in (("mae", "Node MAE"), ("rmse", "Node RMSE"), ("wape_pct", "Node WAPE %"), ("smape_pct", "Node sMAPE %")):
            value = _table_number(
                ctx, data, path, f"{base}.{metric_name}", artifact=artifact,
                section="Centralized architecture evidence", row=label, column=column,
                method=method, domain="four Norway reference grids", fraction=None,
                split="development validation", metric_name=metric_name, scope="node_macro",
            )
            values.append(fmt(value))
        rows.append([label, "development validation", *values, path.as_posix(), base])
    write_csv(ctx.out / artifact, headers, rows)
    (ctx.out / "tables/table_centralized_main.md").write_text(
        "# Centralized Architecture Evidence\n\n" + markdown_table(headers, rows) + "\n",
        encoding="utf-8",
    )


def build_reference_tables(ctx: BuildContext, ref25: Mapping[str, Any], ref50: Mapping[str, Any]) -> None:
    main_artifact = "tables/table_reference_federated_main.csv"
    main_headers = (
        "Method", "25% validation MAE mean", "25% validation MAE std", "25% audit MAE mean", "25% audit MAE std",
        "50% validation MAE mean", "50% validation MAE std", "50% audit MAE mean", "50% audit MAE std",
    )
    main_rows: list[list[Any]] = []
    for method in REFERENCE_METHODS:
        row: list[Any] = [LABELS[method]]
        for fraction, data, path in ((0.25, ref25, REF25_PATH), (0.5, ref50, REF50_PATH)):
            for split in ("validation", "audit"):
                for stat in ("mean", "std"):
                    column = f"{int(fraction * 100)}% {split} MAE {stat}"
                    key = f"metrics.{method}.{split}.node_macro.mae.{stat}"
                    value = _table_number(
                        ctx, data, path, key, artifact=main_artifact,
                        section="Reference-grid main metrics", row=LABELS[method], column=column,
                        method=method, domain="four Norway reference grids", fraction=fraction,
                        split=split, metric_name="mae", scope="node_macro",
                    )
                    row.append(fmt(value))
        main_rows.append(row)
    write_csv(ctx.out / main_artifact, main_headers, main_rows)

    comparison_artifact = "tables/table_reference_federated_comparisons.csv"
    comparison_headers = (
        "Comparator", "25% mean relative validation-MAE improvement", "25% std across seeds", "25% seed wins / 3",
        "50% mean relative validation-MAE improvement", "50% std across seeds", "50% seed wins / 3",
    )
    comparison_rows: list[list[Any]] = []
    for comparator in REFERENCE_METHODS[:-1]:
        row = [LABELS[comparator]]
        for fraction, data, path in ((0.25, ref25, REF25_PATH), (0.5, ref50, REF50_PATH)):
            for field_name, column, metric_name in (
                ("mean_relative_improvement", f"{int(fraction*100)}% mean relative validation-MAE improvement", "relative_improvement"),
                ("std_relative_improvement", f"{int(fraction*100)}% std across seeds", "relative_improvement_std"),
                ("seeds_btd_wins", f"{int(fraction*100)}% seed wins / 3", "seed_win_count"),
            ):
                key = f"btd_vs_comparators.{comparator}.{field_name}"
                value = _table_number(
                    ctx, data, path, key, artifact=comparison_artifact,
                    section="BTD-FL reference-grid comparisons", row=LABELS[comparator], column=column,
                    method="btd_fl_direct_transfer", domain="four Norway reference grids", fraction=fraction,
                    split="development validation", metric_name=metric_name, scope="node_macro",
                )
                row.append(fmt(value))
        comparison_rows.append(row)
    write_csv(ctx.out / comparison_artifact, comparison_headers, comparison_rows)
    md = (
        "# Reference-grid Scarce-history Benchmark\n\n"
        "## Main Metrics\n\n" + markdown_table(main_headers, main_rows) +
        "\n\n## BTD-FL Relative Comparisons\n\n" + markdown_table(comparison_headers, comparison_rows) + "\n"
    )
    (ctx.out / "tables/table_reference_federated_main.md").write_text(md, encoding="utf-8")


def build_industrial_tables(ctx: BuildContext, ind25: Mapping[str, Any], ind50: Mapping[str, Any]) -> None:
    main_artifact = "tables/table_industrial_external_main.csv"
    metric_specs = (
        ("node_macro", "mae", "node-macro MAE"),
        ("node_macro", "rmse", "node-macro RMSE"),
        ("node_macro", "wape_pct", "node WAPE %"),
        ("node_macro", "smape_pct", "node sMAPE %"),
        ("grid_aggregate", "mae", "grid-aggregate MAE"),
    )
    headers = ["Method"]
    for fraction in (0.25, 0.5):
        for _, _, label in metric_specs:
            headers.extend((f"{int(fraction*100)}% validation {label} mean", f"{int(fraction*100)}% validation {label} std"))
    rows: list[list[Any]] = []
    for method in INDUSTRIAL_METHODS:
        row: list[Any] = [LABELS[method]]
        for fraction, data, path in ((0.25, ind25, IND25_PATH), (0.5, ind50, IND50_PATH)):
            for scope, metric_name, label in metric_specs:
                for stat in ("mean", "std"):
                    column = f"{int(fraction*100)}% validation {label} {stat}"
                    key = f"methods.{method}.validation.{scope}.{metric_name}.{stat}"
                    value = _table_number(
                        ctx, data, path, key, artifact=main_artifact,
                        section="External industrial main metrics", row=LABELS[method], column=column,
                        method=method, domain="norway_industrial_mvlv", fraction=fraction,
                        split="external industrial validation", metric_name=metric_name, scope=scope,
                    )
                    row.append(fmt(value))
        rows.append(row)
    write_csv(ctx.out / main_artifact, headers, rows)

    comparison_artifact = "tables/table_industrial_external_comparisons.csv"
    comparison_headers = (
        "Comparator", "25% BTD mean relative validation-MAE improvement", "25% std", "25% seed wins / 3",
        "50% BTD mean relative validation-MAE improvement", "50% std", "50% seed wins / 3",
    )
    comparison_rows: list[list[Any]] = []
    for comparator in INDUSTRIAL_METHODS[:-1]:
        row = [LABELS[comparator]]
        for fraction, data, path in ((0.25, ind25, IND25_PATH), (0.5, ind50, IND50_PATH)):
            for field_name, column, metric_name in (
                ("mean_relative_improvement", f"{int(fraction*100)}% BTD mean relative validation-MAE improvement", "relative_improvement"),
                ("std_relative_improvement", f"{int(fraction*100)}% std", "relative_improvement_std"),
                ("seed_win_count", f"{int(fraction*100)}% seed wins / 3", "seed_win_count"),
            ):
                key = f"comparisons_btd_vs_comparators.{comparator}.{field_name}"
                value = _table_number(
                    ctx, data, path, key, artifact=comparison_artifact,
                    section="BTD-FL external industrial comparisons", row=LABELS[comparator], column=column,
                    method="btd_fl_direct_transfer", domain="norway_industrial_mvlv", fraction=fraction,
                    split="external industrial validation", metric_name=metric_name, scope="node_macro",
                )
                row.append(fmt(value))
        comparison_rows.append(row)
    write_csv(ctx.out / comparison_artifact, comparison_headers, comparison_rows)
    md = (
        "# Industrial External Validation Benchmark\n\n"
        "The primary endpoint is node-macro MAE. Node WAPE and sMAPE are reported descriptively and are not uniformly improved.\n\n"
        "## Main Metrics\n\n" + markdown_table(headers, rows) +
        "\n\n## BTD-FL Relative Comparisons\n\n" + markdown_table(comparison_headers, comparison_rows) + "\n"
    )
    (ctx.out / "tables/table_industrial_external_main.md").write_text(md, encoding="utf-8")


def build_ablations(ctx: BuildContext) -> None:
    direct_path = Path("results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json")
    adapted_path = Path("results/federated/formal_25pct/btd_fl_25pct_seed42.json")
    no_benefit_path = Path("results/federated/formal_25pct/ablations/btd_no_benefit_selection_25pct_seed42.json")
    full_model_path = Path("results/federated/formal_25pct/ablations/btd_full_model_transfer_25pct_seed42.json")
    sources = {path: load_json(ctx.root / path) for path in (direct_path, adapted_path, no_benefit_path, full_model_path)}
    for path, data in sources.items():
        validate_source_test_guardrail(ctx.root / path, data)
    specs = (
        ("Final BTD direct raw temporal transfer", direct_path, "benefit-selected current-run positive argmax", "raw donor temporal state", "no", "four_target_unweighted_macro"),
        ("No benefit selection", no_benefit_path, "fixed lexicographic donor", "target-adapted donor temporal state", "yes", "four_target_unweighted_macro"),
        ("Full-model transfer", full_model_path, "benefit-selected donor frozen from accepted BTD", "raw full trainable donor state", "no", "four_target_unweighted_macro"),
        ("Historical adapted-state BTD", adapted_path, "benefit-selected current-run positive argmax", "target-adapted donor temporal state", "yes", "four_scenario_macro"),
    )
    artifact = "tables/table_btd_ablations.csv"
    headers = (
        "Method", "WHO selection", "WHAT state transferred", "Probe adapted state reused in final initialization?",
        "Audit node-MAE", "Audit relative difference vs final direct BTD",
        "Validation node-MAE", "Validation relative difference vs final direct BTD", "Source artifact",
    )
    final_data = sources[direct_path]
    final_refs = {
        split: ctx.source(direct_path, final_data, f"four_target_unweighted_macro.{split}.node_macro.mae")
        for split in ("audit", "validation")
    }
    rows: list[list[Any]] = []
    for label, path, who, what, reused, macro_key in specs:
        data = sources[path]
        row = [label, who, what, reused]
        for split in ("audit", "validation"):
            metric_key = f"{macro_key}.{split}.node_macro.mae" if macro_key == "four_target_unweighted_macro" else f"{macro_key}.{split}.btd_fl.node_macro.mae"
            source = ctx.source(path, data, metric_key)
            value = ctx.record(
                output_artifact=artifact, output_section="BTD-FL mechanism ablations", row=label,
                column=f"{split.title()} node-MAE", plotted_quantity=None, method=label,
                domain="four Norway reference grids", history_fraction=0.25, seed_scope="seed 42",
                split=split, metric="mae", scope="node_macro", source=source,
            )
            relative = (float(value) - float(final_refs[split].value)) / float(final_refs[split].value)
            ctx.record_derived(
                output_artifact=artifact, output_section="BTD-FL mechanism ablations", row=label,
                column=f"{split.title()} relative difference vs final direct BTD", method=label,
                domain="four Norway reference grids", history_fraction=0.25, seed_scope="seed 42",
                split=split, metric="relative_difference", scope="node_macro",
                value=relative, operation="(row_mae - final_direct_mae) / final_direct_mae",
                sources=(source, final_refs[split]),
            )
            row.extend((fmt(value), fmt(relative)))
        row.append(path.as_posix())
        rows.append(row)
    write_csv(ctx.out / artifact, headers, rows)
    (ctx.out / "tables/table_btd_ablations.md").write_text(
        "# BTD-FL Mechanism Ablations\n\n" + markdown_table(headers, rows) +
        "\n\nThe historical adapted-state formulation reused the target-adapted probe state; the final formulation discards it and transfers the selected donor's raw temporal state. Values are descriptive seed-42 comparisons.\n",
        encoding="utf-8",
    )


def build_donor_table(
    ctx: BuildContext,
    ref25: Mapping[str, Any],
    ref50: Mapping[str, Any],
    ind25: Mapping[str, Any],
    ind50: Mapping[str, Any],
) -> None:
    artifact = "tables/table_btd_donor_behavior.csv"
    headers = ("Domain", "History", "Seed", "Target", "Selected donor", "Selected benefit", "All donor benefits", "Fallback")
    rows: list[list[Any]] = []
    for fraction, data, path in ((0.25, ref25, REF25_PATH), (0.5, ref50, REF50_PATH)):
        by_target = data["donor_selection_robustness"]["by_target"]
        for target in REFERENCE_CLIENTS:
            for seed in map(str, SEEDS):
                item = by_target[target][seed]
                benefit_key = f"donor_selection_robustness.by_target.{target}.{seed}.selected_calibration_benefit"
                selected_benefit = ctx.record(
                    output_artifact=artifact, output_section="Reference-grid donor behavior", row=f"{target}/{seed}",
                    column="Selected benefit", plotted_quantity=None, method="btd_fl_direct_transfer",
                    domain="four Norway reference grids", history_fraction=fraction, seed_scope=f"seed {seed}",
                    split="calibration", metric="target_conditioned_directed_benefit", scope="target",
                    source=ctx.source(path, data, benefit_key),
                )
                for donor, benefit in item["calibration_benefit_by_donor"].items():
                    ctx.record(
                        output_artifact=artifact, output_section="Reference-grid donor behavior", row=f"{target}/{seed}",
                        column="All donor benefits", plotted_quantity=f"benefit from {donor}", method="btd_fl_direct_transfer",
                        domain="four Norway reference grids", history_fraction=fraction, seed_scope=f"seed {seed}",
                        split="calibration", metric="target_conditioned_directed_benefit", scope="donor_target_edge",
                        source=ctx.source(path, data, f"donor_selection_robustness.by_target.{target}.{seed}.calibration_benefit_by_donor.{donor}"),
                    )
                rows.append(["reference-grid", f"{int(fraction*100)}%", seed, target, item["selected_donor"], fmt(selected_benefit), json.dumps(item["calibration_benefit_by_donor"], sort_keys=True), item["zero_transfer_fallback"]])
        behavior = data["donor_selection_robustness"]
        for field_name, label in (("positive_directed_benefit_count", "positive"), ("non_positive_directed_benefit_count", "non-positive"), ("fallback_occurrence_count", "fallback")):
            ctx.record(
                output_artifact=artifact, output_section="Reference-grid donor summary", row=f"{int(fraction*100)}% summary",
                column=f"{label} count", plotted_quantity=None, method="btd_fl_direct_transfer",
                domain="four Norway reference grids", history_fraction=fraction, seed_scope="seeds 42/123/2026",
                split="calibration", metric=f"{label}_count", scope="directed_edges" if label != "fallback" else "target_runs",
                source=ctx.source(path, data, f"donor_selection_robustness.{field_name}"),
            )
        rows.append(["reference-grid summary", f"{int(fraction*100)}%", "all", "all", "", "", json.dumps({"positive": behavior["positive_directed_benefit_count"], "non_positive": behavior["non_positive_directed_benefit_count"]}), behavior["fallback_occurrence_count"]])
    for fraction, data, path in ((0.25, ind25, IND25_PATH), (0.5, ind50, IND50_PATH)):
        behavior = data["btd_donor_behavior"]
        for seed in map(str, SEEDS):
            item = behavior["by_seed"][seed]
            selected_benefit = ctx.record(
                output_artifact=artifact, output_section="Industrial donor behavior", row=f"industrial/{seed}",
                column="Selected benefit", plotted_quantity=None, method="btd_fl_direct_transfer",
                domain="norway_industrial_mvlv", history_fraction=fraction, seed_scope=f"seed {seed}",
                split="calibration", metric="target_conditioned_directed_benefit", scope="target",
                source=ctx.source(path, data, f"btd_donor_behavior.by_seed.{seed}.selected_calibration_benefit"),
            )
            for donor in REFERENCE_CLIENTS:
                ctx.record(
                    output_artifact=artifact, output_section="Industrial donor behavior", row=f"industrial/{seed}",
                    column="All donor benefits", plotted_quantity=f"benefit from {donor}", method="btd_fl_direct_transfer",
                    domain="norway_industrial_mvlv", history_fraction=fraction, seed_scope=f"seed {seed}",
                    split="calibration", metric="target_conditioned_directed_benefit", scope="donor_target_edge",
                    source=ctx.source(path, data, f"btd_donor_behavior.by_seed.{seed}.benefits.{donor}"),
                )
            rows.append(["industrial", f"{int(fraction*100)}%", seed, "norway_industrial_mvlv", item["selected_donor"], fmt(selected_benefit), json.dumps(item["benefits"], sort_keys=True), item["zero_transfer_fallback"]])
        for field_name, label in (("positive_benefit_count", "positive"), ("non_positive_benefit_count", "non-positive"), ("fallback_count", "fallback")):
            ctx.record(
                output_artifact=artifact, output_section="Industrial donor summary", row=f"{int(fraction*100)}% summary",
                column=f"{label} count", plotted_quantity=None, method="btd_fl_direct_transfer",
                domain="norway_industrial_mvlv", history_fraction=fraction, seed_scope="seeds 42/123/2026",
                split="calibration", metric=f"{label}_count", scope="directed_edges" if label != "fallback" else "target_runs",
                source=ctx.source(path, data, f"btd_donor_behavior.{field_name}"),
            )
        rows.append(["industrial summary", f"{int(fraction*100)}%", "all", "norway_industrial_mvlv", "", "", json.dumps({"positive": behavior["positive_benefit_count"], "non_positive": behavior["non_positive_benefit_count"]}), behavior["fallback_count"]])
    write_csv(ctx.out / artifact, headers, rows)
    (ctx.out / "tables/table_btd_donor_behavior.md").write_text(
        "# BTD Target-conditioned Directed Transfer Benefits\n\n" + markdown_table(headers, rows) +
        "\n\nBenefits are target-conditioned directed calibration quantities, not causal effects. The zero-transfer fallback was never triggered in these accepted runs; its performance is therefore not empirically established here.\n",
        encoding="utf-8",
    )


def build_figures(ctx: BuildContext, ref25: Mapping[str, Any], ref50: Mapping[str, Any], ind25: Mapping[str, Any], ind50: Mapping[str, Any]) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = plt.get_cmap("tab10").colors
    x = np.arange(2)
    figure_name = "figures/fig_history_fraction_mae"
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
    for axis_index, (ax, domain, summaries, paths, industrial) in enumerate((
        (axes[0], "four Norway reference grids", (ref25, ref50), (REF25_PATH, REF50_PATH), False),
        (axes[1], "norway_industrial_mvlv", (ind25, ind50), (IND25_PATH, IND50_PATH), True),
    )):
        for method_index, reference_method in enumerate(REFERENCE_METHODS):
            method = "industrial_scarce_local" if industrial and reference_method == "scarce_local" else reference_method
            values, errors = [], []
            for fraction, data, path in zip((0.25, 0.5), summaries, paths):
                base = f"{'methods' if industrial else 'metrics'}.{method}.validation.node_macro.mae"
                for stat, target in (("mean", values), ("std", errors)):
                    target.append(float(ctx.record(
                        output_artifact=f"{figure_name}.pdf", output_section="Figure 1 panel data",
                        row=f"panel {axis_index + 1}: {LABELS[method]}", column=None,
                        plotted_quantity=f"{int(fraction*100)}% validation node-macro MAE {stat}", method=method,
                        domain=domain, history_fraction=fraction, seed_scope="mean/std across seeds 42/123/2026",
                        split="development validation" if not industrial else "external industrial validation",
                        metric="mae", scope="node_macro", source=ctx.source(path, data, f"{base}.{stat}"),
                    )))
            ax.errorbar(x, values, yerr=errors, marker="o", capsize=2.5, linewidth=1.2, color=colors[method_index], label=LABELS[method])
        ax.set_xticks(x, ("25%", "50%"))
        ax.set_xlabel("Target available-history fraction")
        ax.set_ylabel("Development-validation node-macro MAE" if not industrial else "External-validation node-macro MAE")
        ax.grid(axis="y", color="0.88", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=7.4, ncol=2)
    fig.savefig(ctx.out / f"{figure_name}.png", dpi=300)
    fig.savefig(ctx.out / f"{figure_name}.pdf")
    plt.close(fig)

    figure_name = "figures/fig_btd_relative_improvement"
    comparisons = ("scarce_local", "fedavg", "fedfomo_style")
    categories = (
        ("Reference 25%", ref25, REF25_PATH, "btd_vs_comparators", False, 0.25),
        ("Reference 50%", ref50, REF50_PATH, "btd_vs_comparators", False, 0.5),
        ("Industrial 25%", ind25, IND25_PATH, "comparisons_btd_vs_comparators", True, 0.25),
        ("Industrial 50%", ind50, IND50_PATH, "comparisons_btd_vs_comparators", True, 0.5),
    )
    fig, ax = plt.subplots(figsize=(7.3, 3.5), constrained_layout=True)
    positions = np.arange(len(categories)); width = 0.23
    for index, comparator in enumerate(comparisons):
        values = []
        for label, data, path, section, industrial, fraction in categories:
            key_comp = "industrial_scarce_local" if industrial and comparator == "scarce_local" else comparator
            source = ctx.source(path, data, f"{section}.{key_comp}.mean_relative_improvement")
            percent = float(source.value) * 100.0
            ctx.record_derived(
                output_artifact=f"{figure_name}.pdf", output_section="Figure 2 data", row=label,
                column="BTD relative validation-MAE improvement (%)", method="btd_fl_direct_transfer",
                domain="norway_industrial_mvlv" if industrial else "four Norway reference grids",
                history_fraction=fraction, seed_scope="mean across seeds 42/123/2026",
                split="external industrial validation" if industrial else "development validation",
                metric="relative_improvement_pct", scope="node_macro", value=percent,
                operation="fraction * 100", sources=(source,),
            )
            values.append(percent)
        ax.bar(positions + (index - 1) * width, values, width, color=colors[index], label=LABELS[comparator])
    ax.set_xticks(positions, [item[0] for item in categories])
    ax.set_ylabel("BTD relative validation-MAE improvement (%)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(ctx.out / f"{figure_name}.png", dpi=300)
    fig.savefig(ctx.out / f"{figure_name}.pdf")
    plt.close(fig)

    figure_name = "figures/fig_industrial_benefit_matrix"
    donors = REFERENCE_CLIENTS
    row_items: list[tuple[str, str, float, Mapping[str, Any], Path]] = []
    for fraction, data, path in ((0.25, ind25, IND25_PATH), (0.5, ind50, IND50_PATH)):
        for seed in map(str, SEEDS):
            row_items.append((f"{seed} / {int(fraction*100)}%", seed, fraction, data, path))
    matrix = np.zeros((len(row_items), len(donors)), dtype=float)
    selected_columns: list[int] = []
    marker_data: list[dict[str, Any]] = []
    for row_index, (label, seed, fraction, data, path) in enumerate(row_items):
        item = data["btd_donor_behavior"]["by_seed"][seed]
        selected_columns.append(donors.index(item["selected_donor"]))
        marker_data.append({"row": label, "selected_donor": item["selected_donor"], "selected_column": selected_columns[-1]})
        for column_index, donor in enumerate(donors):
            value = float(ctx.record(
                output_artifact=f"{figure_name}.pdf", output_section="Figure 3 matrix", row=label,
                column=None, plotted_quantity=f"benefit from {donor}", method="btd_fl_direct_transfer",
                domain="norway_industrial_mvlv", history_fraction=fraction, seed_scope=f"seed {seed}",
                split="calibration", metric="target_conditioned_directed_benefit", scope="donor_target_edge",
                source=ctx.source(path, data, f"btd_donor_behavior.by_seed.{seed}.benefits.{donor}"),
            ))
            matrix[row_index, column_index] = value
    limit = float(np.max(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(6.3, 3.7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(4), ("39", "50", "56", "80"))
    ax.set_yticks(range(len(row_items)), [item[0] for item in row_items])
    ax.set_xlabel("Reference-grid donor")
    ax.set_ylabel("Seed / target history")
    for row_index, selected_column in enumerate(selected_columns):
        ax.scatter(selected_column, row_index, marker="s", s=360, facecolors="none", edgecolors="black", linewidths=1.5)
    fig.colorbar(image, ax=ax, label="Target-conditioned directed calibration benefit")
    fig.savefig(ctx.out / f"{figure_name}.png", dpi=300)
    fig.savefig(ctx.out / f"{figure_name}.pdf")
    plt.close(fig)
    (ctx.out / "figures/fig_industrial_benefit_matrix_marker_data.json").write_text(
        json.dumps(marker_data, indent=2) + "\n", encoding="utf-8"
    )


def _digest_number(
    ctx: BuildContext, data: Mapping[str, Any], path: Path, key: str, *, section: str, row: str,
    method: str, domain: str, fraction: float | None, split: str, metric_name: str, scope: str,
) -> float | int:
    return ctx.record(
        output_artifact="paper_results_summary.md", output_section=section, row=row, column="reported value",
        plotted_quantity=None, method=method, domain=domain, history_fraction=fraction,
        seed_scope="mean/std across seeds 42/123/2026" if fraction is not None else "unweighted macro across four grids",
        split=split, metric=metric_name, scope=scope, source=ctx.source(path, data, key),
    )


def write_results_digest(
    ctx: BuildContext, ablation: Mapping[str, Any], centralized: Mapping[str, Any],
    ref25: Mapping[str, Any], ref50: Mapping[str, Any], ind25: Mapping[str, Any], ind50: Mapping[str, Any],
) -> None:
    gru_ref = ctx.source(CENTRAL_ABLATION, ablation, "macro_across_grids.node_macro.gru_anchor_only.metrics.mae")
    temporal_ref = ctx.source(CENTRAL_ABLATION, ablation, "macro_across_grids.node_macro.temporal_residual_only.metrics.mae")
    gru = _digest_number(ctx, ablation, CENTRAL_ABLATION, gru_ref.key, section="Temporal dominance", row="GRU anchor", method="gru_anchor_only", domain="four Norway reference grids", fraction=None, split="development validation", metric_name="mae", scope="node_macro")
    temporal = _digest_number(ctx, ablation, CENTRAL_ABLATION, temporal_ref.key, section="Temporal dominance", row="temporal_residual_only", method="temporal_residual_only", domain="four Norway reference grids", fraction=None, split="development validation", metric_name="mae", scope="node_macro")
    temporal_gain = (float(gru) - float(temporal)) / float(gru)
    ctx.record_derived(output_artifact="paper_results_summary.md", output_section="Temporal dominance", row="temporal_residual_only vs GRU", column="relative improvement", method="temporal_residual_only", domain="four Norway reference grids", history_fraction=None, seed_scope="unweighted macro across four grids", split="development validation", metric="relative_improvement", scope="node_macro", value=temporal_gain, operation="(gru_mae - temporal_mae) / gru_mae", sources=(gru_ref, temporal_ref))
    temporal_gain_source = ctx.record_derived(output_artifact="paper_results_summary.md", output_section="Temporal dominance", row="temporal_residual_only vs GRU", column="relative improvement (%)", method="temporal_residual_only", domain="four Norway reference grids", history_fraction=None, seed_scope="unweighted macro across four grids", split="development validation", metric="relative_improvement_pct", scope="node_macro", value=temporal_gain * 100.0, operation="(gru_mae - temporal_mae) / gru_mae * 100", sources=(gru_ref, temporal_ref))
    gwn = _digest_number(ctx, centralized, CENTRAL_SUMMARY, "macro_across_grids.node_macro.graph_wavenet.metrics.mae", section="Temporal dominance", row="Graph WaveNet", method="graph_wavenet", domain="four Norway reference grids", fraction=None, split="development validation", metric_name="mae", scope="node_macro")
    spatial_values = {}
    for method in ("utility_uniform_spatial", "utility_prior_no_physics", "full_puc_rstattn_v2"):
        spatial_values[method] = _digest_number(ctx, ablation, CENTRAL_ABLATION, f"macro_across_grids.node_macro.{method}.metrics.mae", section="Physical/spatial contribution", row=method, method=method, domain="four Norway reference grids", fraction=None, split="development validation", metric_name="mae", scope="node_macro")

    def summary_values(data: Mapping[str, Any], path: Path, fraction: float, industrial: bool) -> dict[str, Any]:
        method_section = "methods" if industrial else "metrics"
        local_method = "industrial_scarce_local" if industrial else "scarce_local"
        domain = "norway_industrial_mvlv" if industrial else "four Norway reference grids"
        split = "external industrial validation" if industrial else "development validation"
        result: dict[str, Any] = {}
        for method in (local_method, "fedavg", "fedprox", "fedfomo_style", "btd_fl_direct_transfer"):
            for stat in ("mean", "std"):
                result[f"{method}_{stat}"] = _digest_number(ctx, data, path, f"{method_section}.{method}.validation.node_macro.mae.{stat}", section=f"{'Industrial external' if industrial else 'Reference-grid'} {int(fraction*100)}%", row=LABELS[method], method=method, domain=domain, fraction=fraction, split=split, metric_name="mae", scope="node_macro")
        comparison_section = "comparisons_btd_vs_comparators" if industrial else "btd_vs_comparators"
        for comparator in (local_method, "fedavg", "fedfomo_style"):
            key_comparator = comparator
            for field_name in ("mean_relative_improvement", "seed_win_count" if industrial else "seeds_btd_wins"):
                key = f"{comparison_section}.{key_comparator}.{field_name}"
                result[f"{comparator}_{field_name}"] = _digest_number(ctx, data, path, key, section=f"{'Industrial external' if industrial else 'Reference-grid'} {int(fraction*100)}%", row=f"BTD vs {LABELS[comparator]}", method="btd_fl_direct_transfer", domain=domain, fraction=fraction, split=split, metric_name="relative_improvement" if field_name == "mean_relative_improvement" else "seed_win_count", scope="node_macro")
                if field_name == "mean_relative_improvement":
                    source = ctx.source(path, data, key)
                    result[f"{comparator}_{field_name}_pct"] = ctx.record_derived(
                        output_artifact="paper_results_summary.md", output_section=f"{'Industrial external' if industrial else 'Reference-grid'} {int(fraction*100)}%",
                        row=f"BTD vs {LABELS[comparator]}", column="relative improvement (%)", method="btd_fl_direct_transfer",
                        domain=domain, history_fraction=fraction, seed_scope="mean across seeds 42/123/2026", split=split,
                        metric="relative_improvement_pct", scope="node_macro", value=float(source.value) * 100.0,
                        operation="fraction * 100", sources=(source,),
                    )
        return result

    r25 = summary_values(ref25, REF25_PATH, 0.25, False)
    r50 = summary_values(ref50, REF50_PATH, 0.5, False)
    i25 = summary_values(ind25, IND25_PATH, 0.25, True)
    i50 = summary_values(ind50, IND50_PATH, 0.5, True)

    adapted = load_json(ctx.root / "results/federated/formal_25pct/btd_fl_25pct_seed42.json")
    direct = load_json(ctx.root / "results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json")
    no_benefit = load_json(ctx.root / "results/federated/formal_25pct/ablations/btd_no_benefit_selection_25pct_seed42.json")
    full_model = load_json(ctx.root / "results/federated/formal_25pct/ablations/btd_full_model_transfer_25pct_seed42.json")
    mechanism = {}
    for label, data, path, key in (
        ("direct", direct, Path("results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json"), "four_target_unweighted_macro.validation.node_macro.mae"),
        ("adapted", adapted, Path("results/federated/formal_25pct/btd_fl_25pct_seed42.json"), "four_scenario_macro.validation.btd_fl.node_macro.mae"),
        ("no_benefit", no_benefit, Path("results/federated/formal_25pct/ablations/btd_no_benefit_selection_25pct_seed42.json"), "four_target_unweighted_macro.validation.node_macro.mae"),
        ("full_model", full_model, Path("results/federated/formal_25pct/ablations/btd_full_model_transfer_25pct_seed42.json"), "four_target_unweighted_macro.validation.node_macro.mae"),
    ):
        mechanism[label] = _digest_number(ctx, data, path, key, section="BTD mechanism ablations", row=label, method=label, domain="four Norway reference grids", fraction=0.25, split="development validation", metric_name="mae", scope="node_macro")

    counts: dict[str, tuple[int, int, int]] = {}
    for label, data, path, prefix, fields, fraction, domain in (
        ("reference 25%", ref25, REF25_PATH, "donor_selection_robustness", ("positive_directed_benefit_count", "non_positive_directed_benefit_count", "fallback_occurrence_count"), 0.25, "four Norway reference grids"),
        ("reference 50%", ref50, REF50_PATH, "donor_selection_robustness", ("positive_directed_benefit_count", "non_positive_directed_benefit_count", "fallback_occurrence_count"), 0.5, "four Norway reference grids"),
        ("industrial 25%", ind25, IND25_PATH, "btd_donor_behavior", ("positive_benefit_count", "non_positive_benefit_count", "fallback_count"), 0.25, "norway_industrial_mvlv"),
        ("industrial 50%", ind50, IND50_PATH, "btd_donor_behavior", ("positive_benefit_count", "non_positive_benefit_count", "fallback_count"), 0.5, "norway_industrial_mvlv"),
    ):
        values = []
        for field_name in fields:
            values.append(int(_digest_number(ctx, data, path, f"{prefix}.{field_name}", section="Benefit/fallback behavior", row=label, method="btd_fl_direct_transfer", domain=domain, fraction=fraction, split="calibration", metric_name=field_name, scope="directed_edges" if "fallback" not in field_name else "target_runs")))
        counts[label] = tuple(values)  # type: ignore[assignment]

    text = f"""# Frozen Results Digest

All values below come from accepted frozen JSON artifacts. Three-seed summaries describe consistency and robustness; no inferential test is reported. Reference-grid canonical validation is development validation, and industrial canonical validation is external industrial validation.

## 1. Temporal dominance

The GRU anchor achieved development-validation node-macro MAE {fmt(gru)}, while `temporal_residual_only` achieved {fmt(temporal)}, a relative reduction of {fmt(temporal_gain_source)}%. Graph WaveNet achieved {fmt(gwn)} on the same four-grid macro endpoint.

## 2. Physical/spatial contribution

Against temporal-only MAE {fmt(temporal)}, uniform spatial utility, utility-prior without physics, and full PUC-RSTAttn V2 yielded {fmt(spatial_values['utility_uniform_spatial'])}, {fmt(spatial_values['utility_prior_no_physics'])}, and {fmt(spatial_values['full_puc_rstattn_v2'])}, respectively. These exact ablations support a weak/inconsistent spatial contribution on this endpoint.

## 3. Reference-grid 25%

BTD-FL achieved {fmt(r25['btd_fl_direct_transfer_mean'])} +/- {fmt(r25['btd_fl_direct_transfer_std'])} development-validation node-macro MAE, versus Local {fmt(r25['scarce_local_mean'])} +/- {fmt(r25['scarce_local_std'])}, FedAvg {fmt(r25['fedavg_mean'])} +/- {fmt(r25['fedavg_std'])}, and FedFomo-style {fmt(r25['fedfomo_style_mean'])} +/- {fmt(r25['fedfomo_style_std'])}. Relative improvements were {fmt(r25['scarce_local_mean_relative_improvement_pct'])}%, {fmt(r25['fedavg_mean_relative_improvement_pct'])}%, and {fmt(r25['fedfomo_style_mean_relative_improvement_pct'])}%, with {r25['scarce_local_seeds_btd_wins']}/3, {r25['fedavg_seeds_btd_wins']}/3, and {r25['fedfomo_style_seeds_btd_wins']}/3 seed wins.

## 4. Reference-grid 50%

BTD-FL achieved {fmt(r50['btd_fl_direct_transfer_mean'])} +/- {fmt(r50['btd_fl_direct_transfer_std'])}, versus Local {fmt(r50['scarce_local_mean'])} +/- {fmt(r50['scarce_local_std'])}, FedAvg {fmt(r50['fedavg_mean'])} +/- {fmt(r50['fedavg_std'])}, and FedFomo-style {fmt(r50['fedfomo_style_mean'])} +/- {fmt(r50['fedfomo_style_std'])}. Relative improvements were {fmt(r50['scarce_local_mean_relative_improvement_pct'])}%, {fmt(r50['fedavg_mean_relative_improvement_pct'])}%, and {fmt(r50['fedfomo_style_mean_relative_improvement_pct'])}%, with {r50['scarce_local_seeds_btd_wins']}/3, {r50['fedavg_seeds_btd_wins']}/3, and {r50['fedfomo_style_seeds_btd_wins']}/3 seed wins.

## 5. FedFomo-style mechanism comparison

BTD-FL reduced validation node-macro MAE relative to FedFomo-style by {fmt(r25['fedfomo_style_mean_relative_improvement_pct'])}% at 25% and {fmt(r50['fedfomo_style_mean_relative_improvement_pct'])}% at 50%, winning in {r25['fedfomo_style_seeds_btd_wins']}/3 and {r50['fedfomo_style_seeds_btd_wins']}/3 seeds.

## 6. BTD mechanism ablations

At seed 42 and 25% history, final benefit-selected raw temporal transfer achieved validation MAE {fmt(mechanism['direct'])}; removing benefit selection gave {fmt(mechanism['no_benefit'])}; transferring the full trainable donor state gave {fmt(mechanism['full_model'])}; and the historical benefit-selected formulation that reused the target-adapted probe state gave {fmt(mechanism['adapted'])}. This separates WHO is selected, WHAT is transferred, and whether adapted probe state is reused.

## 7. Industrial external 25%

On hourly industrial load/energy measurements, BTD-FL achieved {fmt(i25['btd_fl_direct_transfer_mean'])} +/- {fmt(i25['btd_fl_direct_transfer_std'])} external-validation node-macro MAE, versus Industrial Local {fmt(i25['industrial_scarce_local_mean'])} +/- {fmt(i25['industrial_scarce_local_std'])}. The relative improvement was {fmt(i25['industrial_scarce_local_mean_relative_improvement_pct'])}% with {i25['industrial_scarce_local_seed_win_count']}/3 wins. Versus FedFomo-style, it was {fmt(i25['fedfomo_style_mean_relative_improvement_pct'])}% with {i25['fedfomo_style_seed_win_count']}/3 wins.

## 8. Industrial external 50%

BTD-FL achieved {fmt(i50['btd_fl_direct_transfer_mean'])} +/- {fmt(i50['btd_fl_direct_transfer_std'])}, versus Industrial Local {fmt(i50['industrial_scarce_local_mean'])} +/- {fmt(i50['industrial_scarce_local_std'])}. The relative improvement was {fmt(i50['industrial_scarce_local_mean_relative_improvement_pct'])}% with {i50['industrial_scarce_local_seed_win_count']}/3 wins. Versus FedFomo-style, it was {fmt(i50['fedfomo_style_mean_relative_improvement_pct'])}% with {i50['fedfomo_style_seed_win_count']}/3 wins. Node WAPE and sMAPE were not uniformly improved.

## 9. Negative-transfer evidence

At 25%, industrial FedAvg and FedProx validation MAE were {fmt(i25['fedavg_mean'])} and {fmt(i25['fedprox_mean'])}, both above Industrial Local {fmt(i25['industrial_scarce_local_mean'])}. At 50%, they were {fmt(i50['fedavg_mean'])} and {fmt(i50['fedprox_mean'])}, versus Local {fmt(i50['industrial_scarce_local_mean'])}. In addition, the industrial 25% benefit graph contained {counts['industrial 25%'][1]} non-positive directed edge; the reference 50% graph contained {counts['reference 50%'][1]}.

## 10. Benefit/fallback behavior

Positive/non-positive/fallback counts were {counts['reference 25%'][0]}/{counts['reference 25%'][1]}/{counts['reference 25%'][2]} for reference 25%, {counts['reference 50%'][0]}/{counts['reference 50%'][1]}/{counts['reference 50%'][2]} for reference 50%, {counts['industrial 25%'][0]}/{counts['industrial 25%'][1]}/{counts['industrial 25%'][2]} for industrial 25%, and {counts['industrial 50%'][0]}/{counts['industrial 50%'][1]}/{counts['industrial 50%'][2]} for industrial 50%. The fallback was never triggered in these accepted runs, so fallback performance is not empirically established.

Canonical TEST remains locked; no TEST result is reported.
"""
    (ctx.out / "paper_results_summary.md").write_text(text, encoding="utf-8")


def write_captions(out: Path) -> None:
    (out / "paper_captions.md").write_text(
        """# Paper-ready Captions

**Table A.** Centralized architecture evidence on development validation across four reference grids. Metrics are unweighted grid-level node macros from explicit accepted method entries.

**Table B.** Reference-grid three-seed consistency and robustness at 25% and 50% target history. Canonical validation is development validation; the separate comparison table reports BTD-FL relative node-MAE improvement and seed wins.

**Table C.** External industrial validation on hourly industrial load/energy measurements. Absolute MAE is compared only within the industrial target domain. Node WAPE and sMAPE are not uniformly improved.

**Table D.** Seed-42 BTD-FL mechanism ablations separating WHO is selected, WHAT state is transferred, and whether target-adapted probe state is reused.

**Table E.** Target-conditioned directed transfer benefit and selected donor behavior across history fractions and seeds. These calibration quantities are not causal effects.

**Figure 1.** Mean +/- population standard deviation of validation node-macro MAE across seeds 42, 123, and 2026. Reference-grid development validation and external industrial validation use separate axes because their units and scales are not comparable.

**Figure 2.** BTD-FL relative validation node-MAE improvement (%) versus Local, FedAvg, and FedFomo-style. Values summarize three-seed consistency, not an inferential test.

**Figure 3.** Industrial target-conditioned directed calibration benefit by seed and available-history fraction. Columns follow donors 39, 50, 56, and 80; black outlines mark the frozen selected donor.
""",
        encoding="utf-8",
    )


def write_test_protocol(out: Path) -> None:
    (out / "test_opening_protocol.md").write_text(
        f"""# Canonical TEST Opening Protocol

Opening TEST is an evaluation event, not a development iteration.

## Frozen identities

- Scientific method/code frozen commit: `{SCIENTIFIC_METHOD_COMMIT}`
- Accepted development and external-result commit: `{ACCEPTED_RESULTS_COMMIT}`
- Paper-ready reporting/manifest commit: `{REPORTING_COMMIT_PLACEHOLDER}`

The reporting placeholder must be replaced with the SHA of the committed paper-ready package before execution, without changing methods, data selection, or reported development/external results.

## Preregistered evaluation matrix

- Seeds: 42, 123, 2026. No seed may be dropped.
- Target available-history fractions: 0.25 and 0.50.
- Methods: Local, FedAvg, FedProx, FedPer, FedFomo-style, and BTD-FL.
- Reference targets: each of `39_bus_semi_urban_reference_grid`, `50_bus_rural_reference_grid`, `56_bus_semi_urban_reference_grid`, and `80_bus_rural_reference_grid` is evaluated on its frozen canonical TEST split.
- Industrial target: `norway_industrial_mvlv` is evaluated on its Stage-1 frozen canonical TEST indices `[11341, 13344)` only.

This is 3 seeds x 2 history fractions x 6 methods for each domain. Reference reporting includes all four target grids; industrial reporting includes the single industrial target.

## Selection prohibition

No reference or industrial TEST observation, target, loss, or metric may be used for donor selection, FedFomo-style peer weighting, communication-round selection, epoch selection, early stopping, graph construction, scaling, hyperparameter selection, or model revision. Frozen protocol-selected states are evaluated once. There is no retraining, tuning, donor reselection, seed replacement, or reporting-rule change in response to TEST.

## One-shot execution order

1. Verify the three frozen commit identities and a clean accepted-artifact checksum manifest.
2. Verify all development/external artifacts still state `test_evaluated=false`; industrial artifacts must also state `industrial_test_evaluated=false` and `reference_grid_tests_evaluated=false`.
3. Materialize the complete evaluation matrix before reading any TEST target.
4. Evaluate reference-grid TEST for all matrix cells in deterministic seed, fraction, method, target order.
5. Evaluate industrial TEST for all matrix cells in deterministic seed, fraction, method order.
6. Write raw per-cell artifacts before aggregation; never overwrite development/external artifacts.
7. Validate completeness, finite metrics, commit identity, split provenance, and exact matrix membership.
8. Produce the frozen aggregate report once, including every seed and every method.

## Output contract

Write only under `results/final_test_opening/`. Per-cell filenames are:

- `reference_test_<fraction>pct_seed<seed>_<method>_<target>.json`
- `industrial_test_<fraction>pct_seed<seed>_<method>.json`

Aggregate filenames are `reference_test_summary.json`, `industrial_test_summary.json`, and `final_test_opening_manifest.json`. Every artifact must include: scientific commit, accepted-results commit, reporting-manifest commit, seed, history fraction, method, domain, target, exact TEST index bounds, checkpoint/protocol identity, metric definitions and units, `test_evaluated=true`, TEST access timestamp, and flags proving TEST was not used for selection. Industrial artifacts additionally record `industrial_test_evaluated=true` and `reference_grid_tests_evaluated` with the actual domain-specific state.

## Failure and reporting rules

Negative TEST results remain in the complete report and cause no method changes. Partial failures are recorded with error provenance; no partial aggregate is presented as complete, and successful cells are not rerun selectively. Missing cells trigger a halted release, not seed dropping. All six methods, both fractions, all seeds, and all target grids are reported without selective omission. No further development begins from TEST outcomes within this preregistered evaluation event.

This document defines future execution only. It contains no TEST result value and authorizes no TEST access in the reporting commit.
""",
        encoding="utf-8",
    )


def validate_manifest_entries(root: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "output_artifact", "output_section", "row", "method", "dataset_domain", "history_fraction",
        "seed_scope", "split", "metric", "scope", "value", "source_artifact_path", "json_key_path",
        "source_frozen_status", "source_test_guardrail_path", "source_test_guardrail_value",
    }
    for entry in entries:
        missing = required - set(entry)
        if missing:
            raise ValueError(f"manifest entry missing fields: {sorted(missing)}")
        path = root / str(entry["source_artifact_path"])
        data = load_json(path)
        if entry["source_test_guardrail_value"] is not False:
            raise ValueError("manifest TEST provenance is not false")
        if "derivation" not in entry:
            resolved = resolve_json_path(data, str(entry["json_key_path"]))
            if not _same_number(resolved, entry["value"]):
                raise ValueError(f"manifest value does not resolve: {entry}")
        else:
            derivation = entry["derivation"]
            sources = derivation["sources"]
            resolved_values = []
            for source in sources:
                source_data = load_json(root / source["source_artifact_path"])
                resolved = resolve_json_path(source_data, source["json_key_path"])
                if not _same_number(resolved, source["value"]):
                    raise ValueError("derived manifest source does not resolve")
                resolved_values.append(float(resolved))
            operation = derivation["operation"]
            if operation == "fraction * 100":
                expected = resolved_values[0] * 100.0
            elif operation == "(row_mae - final_direct_mae) / final_direct_mae":
                expected = (resolved_values[0] - resolved_values[1]) / resolved_values[1]
            elif operation == "(gru_mae - temporal_mae) / gru_mae":
                expected = (resolved_values[0] - resolved_values[1]) / resolved_values[0]
            elif operation == "(gru_mae - temporal_mae) / gru_mae * 100":
                expected = (resolved_values[0] - resolved_values[1]) / resolved_values[0] * 100.0
            else:
                raise ValueError(f"unknown manifest derivation: {operation}")
            if not _same_number(expected, entry["value"]):
                raise ValueError("derived manifest arithmetic mismatch")


def write_manifest(ctx: BuildContext) -> None:
    validate_manifest_entries(ctx.root, ctx.entries)
    payload = {
        "description": "Per-number provenance for paper-ready tables, figures, and quantitative result digest.",
        "canonical_test_metrics_included": False,
        "entry_count": len(ctx.entries),
        "entries": ctx.entries,
    }
    manifest_dir = ctx.out / "manifests"
    (manifest_dir / "paper_results_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Paper Results Manifest", "",
        f"Entries: {len(ctx.entries)}. Every entry is numeric and traces to accepted frozen JSON. Canonical TEST metrics are excluded.", "",
        "| Output | Section | Row | Column/quantity | Value | Source | JSON key | TEST guardrail |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for entry in ctx.entries:
        location = entry["column"] or entry["plotted_quantity"]
        lines.append(
            f"| `{entry['output_artifact']}` | {entry['output_section']} | {entry['row']} | {location} | {entry['value']} | "
            f"`{entry['source_artifact_path']}` | `{entry['json_key_path']}` | `{entry['source_test_guardrail_path']}=false` |"
        )
    (manifest_dir / "paper_results_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: Path = ROOT, out: Path = DEFAULT_OUT, *, figures: bool = True) -> BuildContext:
    for directory in (out / "tables", out / "figures", out / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    ctx = BuildContext(root=root, out=out)
    ablation = load_json(root / CENTRAL_ABLATION)
    centralized = load_json(root / CENTRAL_SUMMARY)
    validate_source_test_guardrail(root / CENTRAL_ABLATION, ablation)
    validate_source_test_guardrail(root / CENTRAL_SUMMARY, centralized)
    ref25 = validate_reference(root, REF25_PATH, 0.25)
    ref50 = validate_reference(root, REF50_PATH, 0.5)
    ind25 = validate_industrial(root, IND25_PATH, 0.25)
    ind50 = validate_industrial(root, IND50_PATH, 0.5)
    build_centralized(ctx, ablation, centralized)
    build_reference_tables(ctx, ref25, ref50)
    build_industrial_tables(ctx, ind25, ind50)
    build_ablations(ctx)
    build_donor_table(ctx, ref25, ref50, ind25, ind50)
    if figures:
        build_figures(ctx, ref25, ref50, ind25, ind50)
    write_results_digest(ctx, ablation, centralized, ref25, ref50, ind25, ind50)
    write_captions(out)
    write_test_protocol(out)
    write_manifest(ctx)
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true", help="regenerate reporting files without rendering figures")
    args = parser.parse_args()
    ctx = build(figures=not args.no_figures)
    print(f"paper-ready package: {ctx.out}")
    print(f"manifest entries: {len(ctx.entries)}")


if __name__ == "__main__":
    main()
