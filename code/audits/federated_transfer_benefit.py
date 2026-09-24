"""Leakage-safe directed cross-grid temporal transfer-benefit utilities."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from code.data.forecast_dataset import ForecastFeatureScaler
from code.models.centralized_gru import evaluate_validation, masked_scaled_mae
from code.models.puc_rstattn_v2_ablation import PUCRSTAttnV2Ablation
from code.federated.parameter_groups import parameter_groups
from scripts.run_puc_rstattn_v2 import _batch_indices, _to_batch

HISTORY = 168
HISTORY_FRACTION = 0.25
ANCHOR_WEIGHT = 0.2
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class ScarceSplit:
    train_start: int
    train_end: int
    available_start: int
    available_end: int
    eligible_indices: np.ndarray
    fit_indices: np.ndarray
    calibration_indices: np.ndarray
    audit_indices: np.ndarray

    @property
    def raw_train_hours(self) -> int: return self.train_end - self.train_start
    @property
    def available_raw_hours(self) -> int: return self.available_end - self.available_start


class IndexDataset:
    def __init__(self, grid: Any, indices: np.ndarray, history_start_index: int | None = None):
        self.grid = grid
        self.target_indices = np.asarray(indices, dtype=np.int64)
        self.feature_indices = (0, 2, 3, 4, 5, 6)
        self.load_bus_mask = np.asarray(grid.load_bus_mask, dtype=bool)
        self.history_start_index = history_start_index

    def __len__(self) -> int: return int(len(self.target_indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = int(self.target_indices[index])
        start = target - HISTORY
        if self.history_start_index is not None:
            start = max(start, self.history_start_index)
        return {"x": self.grid.dynamic_features[start:target, :, self.feature_indices],
                "y": self.grid.p[target, :], "target_index": target,
                "history_start_index": target-HISTORY, "history_end_index": target}


def scarce_split(grid: Any, history_fraction: float = HISTORY_FRACTION) -> ScarceSplit:
    if not 0 < history_fraction <= 1: raise ValueError("history_fraction must be in (0, 1]")
    train = grid.splits["train"]
    available_hours = int(math.ceil(history_fraction * (train.end_index - train.start_index)))
    available_start = train.end_index - available_hours
    eligible = np.arange(max(train.start_index, available_start + HISTORY), train.end_index, dtype=np.int64)
    fit_end, cal_end = int(0.60 * len(eligible)), int(0.80 * len(eligible))
    if fit_end <= 0 or cal_end <= fit_end or cal_end >= len(eligible): raise ValueError("target train interval is too short")
    return ScarceSplit(train.start_index, train.end_index, available_start, train.end_index,
                       eligible, eligible[:fit_end], eligible[fit_end:cal_end], eligible[cal_end:])


def donor_split(grid: Any) -> tuple[np.ndarray, np.ndarray]:
    train = grid.splits["train"]; eligible = np.arange(train.start_index + HISTORY, train.end_index, dtype=np.int64); cut = int(0.8 * len(eligible)); return eligible[:cut], eligible[cut:]


def fit_fit_only_scaler(grid: Any, fit_indices: np.ndarray, scaler_start_index: int | None = None) -> ForecastFeatureScaler:
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    if fit_indices.size == 0: raise ValueError("fit_indices must be non-empty")
    train = grid.splits["train"]; end = int(fit_indices.max()) + 1; start = max(int(train.start_index), int(fit_indices.min()) - HISTORY)
    if scaler_start_index is not None: start = int(scaler_start_index)
    mask = np.asarray(grid.load_bus_mask, dtype=bool); values = np.asarray(grid.p[start:end, :], dtype=float)[:, mask]
    scaler = ForecastFeatureScaler(("p", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "weekend"))
    scaler.p_mean_ = float(np.nanmean(values)); scaler.p_scale_ = float(np.nanstd(values)); scaler.p_scale_ = scaler.p_scale_ if np.isfinite(scaler.p_scale_) and scaler.p_scale_ > 0 else 1.0
    scaler.mean_ = np.asarray([scaler.p_mean_, 0, 0, 0, 0, 0], dtype=float); scaler.scale_ = np.asarray([scaler.p_scale_, 1, 1, 1, 1, 1], dtype=float); scaler.load_bus_mask_ = mask.copy()
    scaler.fit_split = "train_fit_only"; scaler.fit_start_index = start; scaler.fit_end_index = end; scaler.fit_timestamp_end = str(grid.timestamps[end - 1]); scaler.fit_load_bus_count = int(mask.sum()); scaler.fit_value_count = int(values.size); scaler.fit_value_count_by_feature = {"p": int(values.size)}
    return scaler


def benefit(local_mae: float, transferred_mae: float) -> float: return float((local_mae - transferred_mae) / (local_mae + 1e-12))


def select_donor(calibration_benefits: Mapping[str, float]) -> tuple[str | None, float]:
    if not calibration_benefits: return None, 0.0
    donor, value = max(calibration_benefits.items(), key=lambda item: (item[1], item[0])); return (donor, float(value)) if value > 0 else (None, 0.0)


def transfer_temporal_parameters(target: Any, donor: Any) -> tuple[str, ...]:
    target_params, donor_params = dict(target.named_parameters()), dict(donor.named_parameters()); names = parameter_groups(target)["temporal"]
    if names != parameter_groups(donor)["temporal"]: raise ValueError("temporal groups differ")
    import torch
    with torch.no_grad():
        for name in names: target_params[name].copy_(donor_params[name])
    return names


def _evaluate(model: Any, grid: Any, indices: np.ndarray, scaler: Any, device: Any, history_start_index: int | None = None) -> dict[str, Any]:
    import torch
    dataset = IndexDataset(grid, indices, history_start_index); predictions, actual = [], []; model.eval()
    with torch.no_grad():
        for batch in _batch_indices(len(dataset), BATCH_SIZE):
            x, y = _to_batch(dataset, scaler, batch); predictions.append(scaler.inverse_transform_target(model(torch.from_numpy(x).to(device)).cpu().numpy())); actual.append(scaler.inverse_transform_target(y))
    return evaluate_validation(np.concatenate(actual), np.concatenate(predictions), grid.load_bus_mask)


def make_proxy(grid: Any, seed: int = SEED) -> Any:
    import torch
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    return PUCRSTAttnV2Ablation("temporal_residual_only", grid.num_nodes, np.empty((2, 0), dtype=np.int64), np.empty((0, 3), dtype=np.float32), grid.load_bus_mask, np.empty(0, dtype=np.float32))


def train_proxy(grid: Any, fit_indices: np.ndarray, calibration_indices: np.ndarray, scaler: Any, initial_state: Mapping[str, Any] | None = None, device: str = "cpu", max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, history_start_index: int | None = None) -> tuple[Any, dict[str, Any]]:
    import torch
    model = make_proxy(grid).to(device)
    if initial_state is not None: model.load_state_dict(copy.deepcopy(initial_state))
    fit_ds = IndexDataset(grid, fit_indices, history_start_index); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE); load_mask = torch.as_tensor(grid.load_bus_mask, dtype=torch.bool, device=device); best_state, best_mae, stale = None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in _batch_indices(len(fit_ds), BATCH_SIZE):
            x, y = _to_batch(fit_ds, scaler, batch); optimizer.zero_grad(set_to_none=True); final, _, details = model(torch.from_numpy(x).to(device), return_details=True); target = torch.from_numpy(y).to(device); loss = masked_scaled_mae(final, target, load_mask) + ANCHOR_WEIGHT * masked_scaled_mae(details["y_gru"], target, load_mask); loss.backward(); optimizer.step()
        current = _evaluate(model, grid, calibration_indices, scaler, device, history_start_index)["node_macro"]["mae"]
        if current < best_mae: best_mae = current; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, {"best_calibration_node_mae": float(best_mae), "epochs_run": epoch}


__all__ = ["HISTORY", "HISTORY_FRACTION", "MAX_EPOCHS", "PATIENCE", "BATCH_SIZE", "LEARNING_RATE", "SEED", "ScarceSplit", "IndexDataset", "scarce_split", "donor_split", "fit_fit_only_scaler", "benefit", "select_donor", "transfer_temporal_parameters", "make_proxy", "train_proxy", "_evaluate"]
