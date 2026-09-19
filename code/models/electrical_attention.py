"""Shared-GRU global attention with an impedance-distance prior."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("ElectricalAttentionBaseline requires PyTorch; install torch to run the model")


def normalize_impedance_abs_distance(distance_matrix: Any) -> tuple[np.ndarray, float]:
    """Normalize one loader-provided impedance path-distance matrix.

    The scale is the median of strictly positive off-diagonal entries.  No
    fallback is allowed when the matrix does not contain a valid positive path.
    """

    distance = np.asarray(distance_matrix, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("impedance_abs_distance must be a square matrix")
    if not np.all(np.isfinite(distance)):
        raise ValueError("impedance_abs_distance must contain only finite values")
    if np.any(distance < 0):
        raise ValueError("impedance_abs_distance must be non-negative")
    diagonal = np.diag(distance)
    if not np.allclose(diagonal, 0.0):
        raise ValueError("impedance_abs_distance diagonal must be zero")
    off_diagonal = ~np.eye(distance.shape[0], dtype=bool)
    positive = distance[off_diagonal & (distance > 0)]
    if positive.size == 0:
        raise ValueError("impedance_abs_distance has no positive off-diagonal distance")
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("impedance_abs_distance median positive scale must be positive")
    normalized = distance / scale
    normalized = np.log1p(normalized)
    np.fill_diagonal(normalized, 0.0)
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
        raise ValueError("normalized impedance distance must be finite and non-negative")
    return normalized.astype(np.float32), scale


def _inverse_softplus(value: float) -> float:
    if value <= 0 or not np.isfinite(value):
        raise ValueError("initial lambda must be finite and positive")
    return float(np.log(np.expm1(value)))


if TORCH_AVAILABLE:

    class ElectricalAttentionBaseline(nn.Module):
        """Shared GRU and global attention biased only by impedance path distance."""

        def __init__(
            self,
            distance_matrix: Any,
            input_size: int = 6,
            hidden_size: int = 32,
            num_layers: int = 1,
            lambda_initial: float = 1.0,
        ) -> None:
            super().__init__()
            if int(input_size) != 6:
                raise ValueError("electrical attention requires the six p_calendar features")
            if int(hidden_size) != 32:
                raise ValueError("electrical attention requires hidden_size=32")
            if int(num_layers) != 1:
                raise ValueError("electrical attention requires num_layers=1")
            normalized_distance, scale = normalize_impedance_abs_distance(distance_matrix)
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.num_nodes = int(normalized_distance.shape[0])
            self.distance_scale = float(scale)
            self.gru = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=False,
            )
            self.query_projection = nn.Linear(self.hidden_size, self.hidden_size)
            self.key_projection = nn.Linear(self.hidden_size, self.hidden_size)
            self.value_projection = nn.Linear(self.hidden_size, self.hidden_size)
            self.spatial_projection = nn.Linear(self.hidden_size, self.hidden_size)
            self.output = nn.Linear(self.hidden_size, 1)
            self.raw_lambda = nn.Parameter(torch.tensor(_inverse_softplus(float(lambda_initial))))
            self.register_buffer("normalized_distance", torch.from_numpy(normalized_distance))
            self.last_attention_weights: Tensor | None = None
            self.last_lambda: Tensor | None = None

        @property
        def lambda_value(self) -> Tensor:
            return F.softplus(self.raw_lambda)

        def _temporal_encode(self, x: Tensor) -> Tensor:
            batch_size, history_length, node_count, feature_count = x.shape
            node_sequences = x.permute(1, 0, 2, 3).reshape(
                history_length, batch_size * node_count, feature_count
            )
            _, hidden = self.gru(node_sequences)
            return hidden[-1].reshape(batch_size, node_count, self.hidden_size)

        def _electrical_attention(self, temporal_hidden: Tensor) -> tuple[Tensor, Tensor]:
            queries = self.query_projection(temporal_hidden)
            keys = self.key_projection(temporal_hidden)
            values = self.value_projection(temporal_hidden)
            dynamic_score = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.hidden_size)
            lambda_value = self.lambda_value
            scores = dynamic_score - lambda_value * self.normalized_distance.unsqueeze(0)
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, values)
            spatial_hidden = temporal_hidden + self.spatial_projection(attended)
            return spatial_hidden, weights

        def forward(self, x: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
            if x.ndim != 4:
                raise ValueError("x must have shape (batch, history, nodes, features)")
            _, history_length, node_count, feature_count = x.shape
            if node_count != self.num_nodes:
                raise ValueError(f"expected {self.num_nodes} nodes, got {node_count}")
            if feature_count != self.input_size:
                raise ValueError(f"expected {self.input_size} features, got {feature_count}")
            if history_length <= 0:
                raise ValueError("history length must be positive")
            temporal_hidden = self._temporal_encode(x)
            spatial_hidden, weights = self._electrical_attention(temporal_hidden)
            self.last_attention_weights = weights
            self.last_lambda = self.lambda_value.detach()
            prediction = self.output(spatial_hidden).squeeze(-1)
            return (prediction, weights) if return_attention else prediction

else:

    class ElectricalAttentionBaseline:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
