"""Centralized shared-node GRU smoke baseline utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class SharedNodeGRU(nn.Module):
        """One GRU shared by all nodes, with independent node sequences."""

        def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.gru = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=False,
            )
            self.output = nn.Linear(self.hidden_size, 1)

        def forward(self, x: Tensor) -> Tensor:
            if x.ndim != 4:
                raise ValueError("x must have shape (batch, history, nodes, features)")
            batch_size, history_length, node_count, feature_count = x.shape
            if feature_count != self.input_size:
                raise ValueError(f"expected {self.input_size} features, got {feature_count}")
            node_sequences = x.permute(1, 0, 2, 3).reshape(
                history_length, batch_size * node_count, feature_count
            )
            _, hidden = self.gru(node_sequences)
            last_hidden = hidden[-1].reshape(batch_size, node_count, self.hidden_size)
            return self.output(last_hidden).squeeze(-1)

else:

    class SharedNodeGRU:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("SharedNodeGRU requires PyTorch; install torch to run training")


def masked_scaled_mae(prediction: Tensor, target: Tensor, load_bus_mask: Tensor) -> Tensor:
    """Mean absolute error over load buses only, in scaled target space."""
    if not TORCH_AVAILABLE:
        raise ImportError("masked_scaled_mae requires PyTorch")
    mask = load_bus_mask.to(device=prediction.device, dtype=torch.bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must both have shape (batch, nodes)")
    if mask.ndim != 1 or mask.shape[0] != prediction.shape[1] or not bool(mask.any()):
        raise ValueError("load_bus_mask must select at least one node")
    return torch.abs(prediction[:, mask] - target[:, mask]).mean()


def _masked_numpy(values: np.ndarray, load_bus_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(load_bus_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != mask.shape[0]:
        raise ValueError("values must have shape (samples, nodes)")
    return values[:, mask]


def _safe_wape(actual: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-8) -> float:
    denominator = float(np.sum(np.abs(actual)))
    return float(100.0 * np.sum(np.abs(prediction - actual)) / max(denominator, epsilon))


def _smape(actual: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-8) -> float:
    denominator = np.abs(actual) + np.abs(prediction)
    return float(100.0 * np.mean(2.0 * np.abs(prediction - actual) / np.maximum(denominator, epsilon)))


def regression_metrics(actual: np.ndarray, prediction: np.ndarray, load_bus_mask: np.ndarray) -> dict[str, float]:
    """Compute masked metrics in original P units; WAPE/sMAPE are percentages."""
    actual_load = _masked_numpy(actual, load_bus_mask)
    prediction_load = _masked_numpy(prediction, load_bus_mask)
    errors = prediction_load - actual_load
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "wape_pct": _safe_wape(actual_load, prediction_load),
        "smape_pct": _smape(actual_load, prediction_load),
    }


def evaluate_validation(actual: np.ndarray, prediction: np.ndarray, load_bus_mask: np.ndarray) -> dict[str, Any]:
    """Return node-macro and grid-aggregate validation metrics."""
    mask = np.asarray(load_bus_mask, dtype=bool)
    actual_load = _masked_numpy(actual, mask)
    prediction_load = _masked_numpy(prediction, mask)
    node_metrics = [
        regression_metrics(actual_load[:, [node]], prediction_load[:, [node]], np.array([True]))
        for node in range(actual_load.shape[1])
    ]
    node_macro = {
        metric: float(np.mean([item[metric] for item in node_metrics]))
        for metric in ("mae", "rmse", "wape_pct", "smape_pct")
    }
    aggregate_actual = np.sum(actual_load, axis=1, keepdims=True)
    aggregate_prediction = np.sum(prediction_load, axis=1, keepdims=True)
    return {
        "node_macro": node_macro,
        "grid_aggregate": regression_metrics(aggregate_actual, aggregate_prediction, np.array([True])),
        "node_count": int(actual_load.shape[1]),
        "sample_count": int(actual_load.shape[0]),
    }


def persistence_1h_predictions(p_values: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """Predict each target with the immediately preceding raw P observation."""
    p_values = np.asarray(p_values, dtype=float)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if p_values.ndim != 2 or np.any(target_indices <= 0) or np.any(target_indices >= len(p_values)):
        raise ValueError("invalid P array or target indices for persistence baseline")
    return p_values[target_indices - 1].copy()


REPORT_METRICS = ("mae", "rmse", "wape_pct", "smape_pct")


def best_epoch_summary(training_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the epoch with minimum validation node-macro MAE."""
    if not training_history:
        raise ValueError("training_history must contain at least one epoch")
    best = min(training_history, key=lambda item: float(item["validation"]["node_macro"]["mae"]))
    return {
        "best_epoch": int(best["epoch"]),
        "best_validation_node_macro_mae": float(best["validation"]["node_macro"]["mae"]),
        "best_train_scaled_mae": float(best["train_scaled_mae"]),
        "epochs_run": int(len(training_history)),
    }


def compare_gru_persistence(
    gru: dict[str, Any], persistence: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Compare each metric factually, without an overall model judgement."""
    comparison: dict[str, dict[str, str]] = {}
    for scope in ("node_macro", "grid_aggregate"):
        comparison[scope] = {}
        for metric in REPORT_METRICS:
            gru_value = float(gru[scope][metric])
            persistence_value = float(persistence[scope][metric])
            if np.isclose(gru_value, persistence_value, rtol=1e-12, atol=1e-12):
                relation = "equal"
            elif gru_value < persistence_value:
                relation = "gru_lower"
            else:
                relation = "gru_higher"
            comparison[scope][metric] = relation
    return comparison


