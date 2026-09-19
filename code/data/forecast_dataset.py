"""Supervised next-hour forecasting windows built on the canonical LV loader."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:  # Supports both package imports and direct CLI execution.
    from .lv_grid_loader import LVGridData, LVGridLoader
except ImportError:  # pragma: no cover - exercised by the file-path CLI.
    from lv_grid_loader import LVGridData, LVGridLoader


FEATURE_MODES = {
    "p_calendar": (0, 2, 3, 4, 5, 6),
    "pq_calendar": (0, 1, 2, 3, 4, 5, 6),
}
FEATURE_NAMES = {
    "p_calendar": ("p", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "weekend"),
    "pq_calendar": (
        "p",
        "q",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "weekend",
    ),
}
HISTORY_LENGTH = 168
FORECAST_HORIZON = 1


def _iso(value: Any) -> str:
    return value.isoformat(sep=" ")


@dataclass
class ForecastFeatureScaler:
    """Feature-wise scaler fitted only on canonical train timestamps."""

    feature_names: tuple[str, ...]
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    fit_split: str | None = None
    fit_start_index: int | None = None
    fit_end_index: int | None = None
    fit_timestamp_end: str | None = None
    fit_value_count: int | None = None

    def fit(self, dataset: "ForecastWindowDataset", split: str = "train") -> "ForecastFeatureScaler":
        if split != "train":
            raise ValueError("ForecastFeatureScaler may only be fitted on the train split")
        train_split = dataset.grid.splits["train"]
        raw = dataset.grid.dynamic_features[train_split.start_index : train_split.end_index]
        values = raw[:, :, dataset.feature_indices].reshape(-1, len(dataset.feature_indices)).astype(float)
        self.mean_ = np.nanmean(values, axis=0)
        standard_deviation = np.nanstd(values, axis=0)
        self.scale_ = np.where(np.isfinite(standard_deviation) & (standard_deviation > 0), standard_deviation, 1.0)
        self.feature_names = tuple(dataset.feature_names)
        self.fit_split = "train"
        self.fit_start_index = train_split.start_index
        self.fit_end_index = train_split.end_index
        self.fit_timestamp_end = _iso(dataset.grid.timestamps[train_split.end_index - 1])
        self.fit_value_count = int(values.shape[0])
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
class ForecastWindowDataset:
    """Lazy supervised windows for one feature mode and one LV grid.

    ``x`` windows are sliced from the canonical loader on demand. Target
    indices are chronological and are assigned to splits by target timestamp.
    """

    grid: LVGridData
    feature_mode: str
    target_indices: np.ndarray
    split_labels: np.ndarray
    history_length: int = HISTORY_LENGTH
    forecast_horizon: int = FORECAST_HORIZON

    def __post_init__(self) -> None:
        if self.feature_mode not in FEATURE_MODES:
            raise ValueError(f"Unsupported feature mode: {self.feature_mode}")
        if self.history_length != HISTORY_LENGTH or self.forecast_horizon != FORECAST_HORIZON:
            raise ValueError("V2 main task requires history_length=168 and forecast_horizon=1")
        self.feature_indices = FEATURE_MODES[self.feature_mode]
        self.feature_names = FEATURE_NAMES[self.feature_mode]
        self.load_bus_mask = self.grid.load_bus_mask.copy()

    @classmethod
    def from_grid(
        cls,
        grid: LVGridData,
        feature_mode: str,
        history_length: int = HISTORY_LENGTH,
        forecast_horizon: int = FORECAST_HORIZON,
    ) -> "ForecastWindowDataset":
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"Unsupported feature mode: {feature_mode}")
        if history_length != HISTORY_LENGTH or forecast_horizon != FORECAST_HORIZON:
            raise ValueError("V2 main task requires history_length=168 and forecast_horizon=1")
        first_target = history_length
        last_target_exclusive = len(grid.timestamps) - forecast_horizon + 1
        target_indices = np.arange(first_target, last_target_exclusive, dtype=np.int64)
        split_labels = np.asarray(
            [cls._split_for_target(grid, int(index)) for index in target_indices], dtype=object
        )
        return cls(
            grid=grid,
            feature_mode=feature_mode,
            target_indices=target_indices,
            split_labels=split_labels,
            history_length=history_length,
            forecast_horizon=forecast_horizon,
        )

    @staticmethod
    def _split_for_target(grid: LVGridData, target_index: int) -> str:
        for name, split in grid.splits.items():
            if split.start_index <= target_index < split.end_index:
                return name
        raise ValueError(f"Target index {target_index} is outside all canonical splits")

    def __len__(self) -> int:
        return int(len(self.target_indices))

    @property
    def num_nodes(self) -> int:
        return self.grid.num_nodes

    @property
    def num_features(self) -> int:
        return len(self.feature_indices)

    @property
    def history_shape(self) -> tuple[int, int, int]:
        return (self.history_length, self.num_nodes, self.num_features)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (self.num_nodes,)

    @property
    def x_shape(self) -> tuple[int, int, int, int]:
        return (len(self), self.history_length, self.num_nodes, self.num_features)

    @property
    def y_shape(self) -> tuple[int, int]:
        return (len(self), self.num_nodes)

    @property
    def target_timestamps(self) -> np.ndarray:
        return np.asarray([self.grid.timestamps[index] for index in self.target_indices], dtype=object)

    @property
    def first_target_timestamp(self) -> str:
        return _iso(self.grid.timestamps[int(self.target_indices[0])])

    @property
    def last_target_timestamp(self) -> str:
        return _iso(self.grid.timestamps[int(self.target_indices[-1])])

    def split(self, split_name: str) -> "ForecastWindowDataset":
        if split_name not in self.grid.splits:
            raise KeyError(f"Unknown split: {split_name}")
        mask = self.split_labels == split_name
        return ForecastWindowDataset(
            grid=self.grid,
            feature_mode=self.feature_mode,
            target_indices=self.target_indices[mask],
            split_labels=self.split_labels[mask],
            history_length=self.history_length,
            forecast_horizon=self.forecast_horizon,
        )

    def get_window(self, index: int) -> dict[str, Any]:
        target_index = int(self.target_indices[index])
        history_start = target_index - self.history_length
        history_end = target_index
        history = self.grid.dynamic_features[history_start:history_end, :, self.feature_indices]
        target = self.grid.p[target_index, :]
        return {
            "x": history,
            "y": target,
            "load_bus_mask": self.load_bus_mask.copy(),
            "target_timestamp": self.grid.timestamps[target_index],
            "target_index": target_index,
            "history_start_index": history_start,
            "history_end_index": history_end,
            "feature_names": self.feature_names,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_window(index)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self.get_window(index)

    def fit_scaler(self) -> ForecastFeatureScaler:
        return ForecastFeatureScaler(tuple(self.feature_names)).fit(self, split="train")

    def metadata(self) -> dict[str, Any]:
        return {
            "grid_name": self.grid.grid_name,
            "feature_mode": self.feature_mode,
            "feature_names": list(self.feature_names),
            "history_length": self.history_length,
            "forecast_horizon": self.forecast_horizon,
            "target": "next-hour active power P",
            "num_samples": len(self),
            "num_nodes": self.num_nodes,
            "num_features": self.num_features,
            "history_shape": list(self.history_shape),
            "target_shape": list(self.target_shape),
            "x_shape": list(self.x_shape),
            "y_shape": list(self.y_shape),
            "first_target_timestamp": self.first_target_timestamp,
            "last_target_timestamp": self.last_target_timestamp,
            "load_bus_count": int(self.load_bus_mask.sum()),
            "branch_type_code_used": False,
            "target_indices": {
                name: [
                    int(indices[0]),
                    int(indices[-1]),
                    int(len(indices)),
                ]
                for name in ("train", "validation", "test")
                for indices in [self.target_indices[self.split_labels == name]]
                if len(indices)
            },
        }


def build_forecast_datasets(grid: LVGridData) -> dict[str, ForecastWindowDataset]:
    return {mode: ForecastWindowDataset.from_grid(grid, mode) for mode in FEATURE_MODES}


def _validate_mode_dataset(dataset: ForecastWindowDataset) -> dict[str, Any]:
    split_checks: dict[str, Any] = {}
    no_future_leakage = True
    target_split_bounds = True
    for split_name in ("train", "validation", "test"):
        subset = dataset.split(split_name)
        split = dataset.grid.splits[split_name]
        target_indices = subset.target_indices
        in_bounds = bool(np.all((target_indices >= split.start_index) & (target_indices < split.end_index)))
        target_split_bounds = target_split_bounds and in_bounds
        if len(subset):
            first = subset[0]
            last = subset[len(subset) - 1]
            no_future_leakage = no_future_leakage and first["history_end_index"] <= first["target_index"]
            no_future_leakage = no_future_leakage and last["history_end_index"] <= last["target_index"]
            split_checks[split_name] = {
                "sample_count": len(subset),
                "first_target_timestamp": subset.first_target_timestamp,
                "last_target_timestamp": subset.last_target_timestamp,
                "first_target_index": int(first["target_index"]),
                "last_target_index": int(last["target_index"]),
                "target_in_split_bounds": in_bounds,
                "history_shape": list(first["x"].shape),
                "target_shape": list(first["y"].shape),
                "first_history_end_index": int(first["history_end_index"]),
                "first_target_index_for_leakage_check": int(first["target_index"]),
            }
        else:
            split_checks[split_name] = {
                "sample_count": 0,
                "target_in_split_bounds": in_bounds,
            }

    scaler = dataset.fit_scaler()
    train_end = dataset.grid.splits["train"].end_index
    scaler_train_only = (
        scaler.fit_split == "train"
        and scaler.fit_start_index == dataset.grid.splits["train"].start_index
        and scaler.fit_end_index == train_end
        and scaler.fit_timestamp_end == _iso(dataset.grid.timestamps[train_end - 1])
    )
    return {
        "metadata": dataset.metadata(),
        "split_checks": split_checks,
        "target_split_bounds_valid": target_split_bounds,
        "no_future_leakage": no_future_leakage,
        "scaler": {
            "feature_names": list(scaler.feature_names),
            "fit_split": scaler.fit_split,
            "fit_start_index": scaler.fit_start_index,
            "fit_end_index": scaler.fit_end_index,
            "fit_timestamp_end": scaler.fit_timestamp_end,
            "fit_value_count": scaler.fit_value_count,
            "train_only": scaler_train_only,
        },
        "branch_type_code_used": False,
        "test_labels_evaluated": False,
        "all_checks_pass": target_split_bounds and no_future_leakage and scaler_train_only,
    }


def build_validation_report(loader: LVGridLoader) -> dict[str, Any]:
    grids: list[dict[str, Any]] = []
    for grid_name in loader.available_grids():
        grid = loader.load(grid_name)
        modes = {mode: _validate_mode_dataset(dataset) for mode, dataset in build_forecast_datasets(grid).items()}
        grids.append({"grid_name": grid_name, "modes": modes, "all_checks_pass": all(item["all_checks_pass"] for item in modes.values())})
    return {
        "audit": "v2_forecast_dataset_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "Supervised next-hour windows only; no model training and no test metric.",
            "All physical buses remain in each graph; loss/metric masks are load_bus_mask only.",
            "Target split is determined by target timestamp; validation history may use prior train observations.",
        ],
        "task": {
            "history_length": HISTORY_LENGTH,
            "forecast_horizon": FORECAST_HORIZON,
            "target": "next-hour active power P",
            "feature_modes": {mode: list(names) for mode, names in FEATURE_NAMES.items()},
            "split": "chronological 70/15/15; no shuffle",
        },
        "grids": grids,
        "all_checks_pass": all(grid["all_checks_pass"] for grid in grids),
    }


def render_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Forecasting Window Dataset Validation",
        "",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        "",
        "Task: history 168 hours, horizon 1 hour, target next-hour active P. No model training or test metric.",
        "",
        "| Grid | Mode | Samples | History shape | Target shape | Train/Val/Test | Split bounds | No future leakage | Train-only scaler | All checks |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for grid in report["grids"]:
        for mode, item in grid["modes"].items():
            metadata = item["metadata"]
            split_counts = "/".join(str(item["split_checks"][name]["sample_count"]) for name in ("train", "validation", "test"))
            lines.append(
                f"| {grid['grid_name']} | {mode} | {metadata['num_samples']} | `{metadata['history_shape']}` | "
                f"`{metadata['target_shape']}` | {split_counts} | {item['target_split_bounds_valid']} | "
                f"{item['no_future_leakage']} | {item['scaler']['train_only']} | {item['all_checks_pass']} |"
            )
    lines.extend(["", "## Target timestamp ranges", ""])
    for grid in report["grids"]:
        lines.append(f"### {grid['grid_name']}")
        lines.append("")
        for mode, item in grid["modes"].items():
            lines.append(f"#### {mode}")
            lines.append("")
            lines.append("| Split | Samples | First target | Last target | Target bounds |")
            lines.append("| --- | ---: | --- | --- | --- |")
            for split_name in ("train", "validation", "test"):
                split = item["split_checks"][split_name]
                lines.append(
                    f"| {split_name} | {split['sample_count']} | {split.get('first_target_timestamp', '-')} | "
                    f"{split.get('last_target_timestamp', '-')} | {split['target_in_split_bounds']} |"
                )
            lines.append("")
            lines.append(
                f"Scaler: split={item['scaler']['fit_split']}, source indices "
                f"[{item['scaler']['fit_start_index']}, {item['scaler']['fit_end_index']}) ending "
                f"{item['scaler']['fit_timestamp_end']}; test labels evaluated: {item['test_labels_evaluated']}."
            )
            lines.append("")
    lines.append(f"All checks pass: **{report['all_checks_pass']}**.")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=repo_root / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-json", type=Path, default=repo_root / "results" / "audits" / "v2_forecast_dataset_validation.json")
    parser.add_argument("--output-md", type=Path, default=repo_root / "results" / "audits" / "v2_forecast_dataset_validation.md")
    args = parser.parse_args()
    report = build_validation_report(LVGridLoader(args.data_root, args.mapping_json))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_validation_markdown(report), encoding="utf-8")
    print(f"grids={len(report['grids'])} all_checks_pass={report['all_checks_pass']}")


if __name__ == "__main__":
    main()
