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
    """Train-only P/Q scaler that preserves calendar values and load masks.

    P and Q statistics are computed from train timestamps and physical load
    buses only. Calendar features are deliberately passed through unchanged.
    """

    feature_names: tuple[str, ...]
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    p_mean_: float | None = None
    p_scale_: float | None = None
    q_mean_: float | None = None
    q_scale_: float | None = None
    load_bus_mask_: np.ndarray | None = None
    fit_split: str | None = None
    fit_start_index: int | None = None
    fit_end_index: int | None = None
    fit_timestamp_end: str | None = None
    fit_value_count: int | None = None
    fit_value_count_by_feature: dict[str, int] | None = None
    fit_load_bus_count: int | None = None

    @staticmethod
    def _channel_stats(values: np.ndarray, channel_name: str) -> tuple[float, float, int]:
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            raise ValueError(f"No finite train load-bus values available for {channel_name}")
        mean = float(np.mean(finite_values))
        scale = float(np.std(finite_values))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return mean, scale, int(finite_values.size)

    def fit(self, dataset: "ForecastWindowDataset", split: str = "train") -> "ForecastFeatureScaler":
        if split != "train":
            raise ValueError("ForecastFeatureScaler may only be fitted on the train split")
        train_split = dataset.grid.splits["train"]
        raw = dataset.grid.dynamic_features[train_split.start_index : train_split.end_index]
        load_bus_mask = np.asarray(dataset.grid.load_bus_mask, dtype=bool)
        if load_bus_mask.ndim != 1 or not np.any(load_bus_mask):
            raise ValueError("At least one physical load bus is required for scaling")

        self.p_mean_ = self.p_scale_ = self.q_mean_ = self.q_scale_ = None
        stats_by_feature: dict[str, tuple[float, float, int]] = {}
        if "p" in dataset.feature_names:
            self.p_mean_, self.p_scale_, p_count = self._channel_stats(raw[:, load_bus_mask, 0], "p")
            stats_by_feature["p"] = (self.p_mean_, self.p_scale_, p_count)
        if "q" in dataset.feature_names:
            self.q_mean_, self.q_scale_, q_count = self._channel_stats(raw[:, load_bus_mask, 1], "q")
            stats_by_feature["q"] = (self.q_mean_, self.q_scale_, q_count)

        # Keep the legacy vector attributes, with identity entries for calendar
        # features so callers can inspect the complete feature layout.
        self.mean_ = np.asarray(
            [stats_by_feature.get(name, (0.0, 1.0, 0))[0] for name in dataset.feature_names]
        )
        self.scale_ = np.asarray(
            [stats_by_feature.get(name, (0.0, 1.0, 0))[1] for name in dataset.feature_names]
        )
        self.feature_names = tuple(dataset.feature_names)
        self.load_bus_mask_ = load_bus_mask.copy()
        self.fit_split = "train"
        self.fit_start_index = train_split.start_index
        self.fit_end_index = train_split.end_index
        self.fit_timestamp_end = _iso(dataset.grid.timestamps[train_split.end_index - 1])
        self.fit_value_count_by_feature = {
            name: int(stats_by_feature[name][2]) for name in stats_by_feature
        }
        self.fit_value_count = int(train_split.sample_count * int(load_bus_mask.sum()))
        self.fit_load_bus_count = int(load_bus_mask.sum())
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.load_bus_mask_ is None:
            raise RuntimeError("fit must be called before transform")
        transformed = np.asarray(values, dtype=float).copy()
        if transformed.ndim < 2 or transformed.shape[-1] != len(self.feature_names):
            raise ValueError("Feature values must have shape (..., nodes, num_features)")
        for position, name in enumerate(self.feature_names):
            if name == "p":
                channel = (transformed[..., :, position] - self.p_mean_) / self.p_scale_
                channel[..., ~self.load_bus_mask_] = 0.0
                transformed[..., :, position] = channel
            elif name == "q":
                channel = (transformed[..., :, position] - self.q_mean_) / self.q_scale_
                channel[..., ~self.load_bus_mask_] = 0.0
                transformed[..., :, position] = channel
        return transformed

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.load_bus_mask_ is None:
            raise RuntimeError("fit must be called before inverse_transform")
        restored = np.asarray(values, dtype=float).copy()
        if restored.ndim < 2 or restored.shape[-1] != len(self.feature_names):
            raise ValueError("Feature values must have shape (..., nodes, num_features)")
        for position, name in enumerate(self.feature_names):
            if name == "p":
                channel = restored[..., :, position] * self.p_scale_ + self.p_mean_
                channel[..., ~self.load_bus_mask_] = 0.0
                restored[..., :, position] = channel
            elif name == "q":
                channel = restored[..., :, position] * self.q_scale_ + self.q_mean_
                channel[..., ~self.load_bus_mask_] = 0.0
                restored[..., :, position] = channel
        return restored

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        """Transform active-power targets using the train-only P scaler."""
        if self.p_mean_ is None or self.p_scale_ is None or self.load_bus_mask_ is None:
            raise RuntimeError("P scaler is unavailable; fit a dataset containing P first")
        transformed = (np.asarray(values, dtype=float) - self.p_mean_) / self.p_scale_
        transformed = transformed.copy()
        if transformed.ndim < 1 or transformed.shape[-1] != len(self.load_bus_mask_):
            raise ValueError("Target values must have shape (..., nodes)")
        transformed[..., ~self.load_bus_mask_] = 0.0
        return transformed

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Restore active-power targets and keep non-load buses at exactly zero."""
        if self.p_mean_ is None or self.p_scale_ is None or self.load_bus_mask_ is None:
            raise RuntimeError("P scaler is unavailable; fit a dataset containing P first")
        restored = np.asarray(values, dtype=float) * self.p_scale_ + self.p_mean_
        restored = restored.copy()
        if restored.ndim < 1 or restored.shape[-1] != len(self.load_bus_mask_):
            raise ValueError("Target values must have shape (..., nodes)")
        restored[..., ~self.load_bus_mask_] = 0.0
        return restored


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
    train_load_value_count = dataset.grid.splits["train"].sample_count * int(dataset.load_bus_mask.sum())
    scaler_load_only = (
        scaler.fit_load_bus_count == int(dataset.load_bus_mask.sum())
        and scaler.fit_value_count == train_load_value_count
        and all(
            count == train_load_value_count
            for count in (scaler.fit_value_count_by_feature or {}).values()
        )
    )
    sample = dataset[0]["x"]
    transformed_sample = scaler.transform(sample)
    calendar_positions = [
        position
        for position, name in enumerate(dataset.feature_names)
        if name in {"hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "weekend"}
    ]
    calendar_passthrough = all(
        np.array_equal(transformed_sample[..., position], sample[..., position])
        for position in calendar_positions
    )
    non_load_zero_after_transform = True
    for position, name in enumerate(dataset.feature_names):
        if name in {"p", "q"}:
            non_load_zero_after_transform = non_load_zero_after_transform and bool(
                np.all(transformed_sample[..., ~dataset.load_bus_mask, position] == 0.0)
            )
    target = dataset[0]["y"]
    target_roundtrip = np.allclose(
        scaler.inverse_transform_target(scaler.transform_target(target)),
        target,
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
            "fit_value_count_by_feature": scaler.fit_value_count_by_feature,
            "fit_load_bus_count": scaler.fit_load_bus_count,
            "load_bus_only": scaler_load_only,
            "calendar_passthrough": calendar_passthrough,
            "non_load_zero_after_transform": non_load_zero_after_transform,
            "target_inverse_roundtrip": target_roundtrip,
            "p_mean": scaler.p_mean_,
            "p_scale": scaler.p_scale_,
            "q_mean": scaler.q_mean_,
            "q_scale": scaler.q_scale_,
            "train_only": scaler_train_only,
        },
        "branch_type_code_used": False,
        "test_labels_evaluated": False,
        "all_checks_pass": (
            target_split_bounds
            and no_future_leakage
            and scaler_train_only
            and scaler_load_only
            and calendar_passthrough
            and non_load_zero_after_transform
            and target_roundtrip
        ),
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
        "| Grid | Mode | Samples | History shape | Target shape | Train/Val/Test | Split bounds | No future leakage | Train load-only scaler | Calendar unchanged | Target roundtrip | All checks |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for grid in report["grids"]:
        for mode, item in grid["modes"].items():
            metadata = item["metadata"]
            split_counts = "/".join(str(item["split_checks"][name]["sample_count"]) for name in ("train", "validation", "test"))
            lines.append(
                f"| {grid['grid_name']} | {mode} | {metadata['num_samples']} | `{metadata['history_shape']}` | "
                f"`{metadata['target_shape']}` | {split_counts} | {item['target_split_bounds_valid']} | "
                f"{item['no_future_leakage']} | {item['scaler']['train_only'] and item['scaler']['load_bus_only']} | "
                f"{item['scaler']['calendar_passthrough']} | {item['scaler']['target_inverse_roundtrip']} | "
                f"{item['all_checks_pass']} |"
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
                f"{item['scaler']['fit_timestamp_end']}; load-bus-only={item['scaler']['load_bus_only']}, "
                f"calendar-unchanged={item['scaler']['calendar_passthrough']}, "
                f"target-roundtrip={item['scaler']['target_inverse_roundtrip']}; "
                f"test labels evaluated: {item['test_labels_evaluated']}."
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
