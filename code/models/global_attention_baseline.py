"""Topology-free shared-GRU plus global scaled dot-product attention baseline."""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("GlobalAttentionBaseline requires PyTorch; install torch to run the model")


if TORCH_AVAILABLE:

    class GlobalAttentionBaseline(nn.Module):
        """Shared-node GRU followed by global single-head self-attention."""

        def __init__(
            self,
            input_size: int = 6,
            hidden_size: int = 32,
            num_layers: int = 1,
        ) -> None:
            super().__init__()
            if int(input_size) != 6:
                raise ValueError("global attention baseline requires the six p_calendar features")
            if int(hidden_size) != 32:
                raise ValueError("global attention baseline requires hidden_size=32")
            if int(num_layers) != 1:
                raise ValueError("global attention baseline requires num_layers=1")
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
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
            self.last_attention_weights: Tensor | None = None

        def _temporal_encode(self, x: Tensor) -> Tensor:
            batch_size, history_length, node_count, feature_count = x.shape
            node_sequences = x.permute(1, 0, 2, 3).reshape(
                history_length, batch_size * node_count, feature_count
            )
            _, hidden = self.gru(node_sequences)
            return hidden[-1].reshape(batch_size, node_count, self.hidden_size)

        def _global_attention(
            self,
            temporal_hidden: Tensor,
            source_mask: Tensor | None = None,
            candidate_mask: Tensor | None = None,
        ) -> tuple[Tensor, Tensor]:
            queries = self.query_projection(temporal_hidden)
            keys = self.key_projection(temporal_hidden)
            values = self.value_projection(temporal_hidden)
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.hidden_size)
            node_count = temporal_hidden.shape[1]
            allowed = torch.ones((node_count, node_count), dtype=torch.bool, device=scores.device)
            if candidate_mask is not None:
                if not isinstance(candidate_mask, torch.Tensor):
                    raise TypeError("candidate_mask must be a torch.Tensor or None")
                if candidate_mask.ndim != 2 or candidate_mask.shape != (node_count, node_count):
                    raise ValueError("candidate_mask must have shape (nodes, nodes)")
                if candidate_mask.dtype != torch.bool:
                    raise ValueError("candidate_mask must have bool dtype")
                candidate_mask = candidate_mask.to(device=scores.device)
                allowed = allowed & candidate_mask
            if source_mask is not None:
                if not isinstance(source_mask, torch.Tensor):
                    raise TypeError("source_mask must be a torch.Tensor or None")
                if source_mask.ndim != 1 or source_mask.shape[0] != temporal_hidden.shape[1]:
                    raise ValueError("source_mask must have shape (nodes,)")
                if source_mask.dtype != torch.bool:
                    raise ValueError("source_mask must have bool dtype")
                if not bool(source_mask.any()):
                    raise ValueError("source_mask must select at least one source node")
                source_mask = source_mask.to(device=scores.device)
                allowed = allowed & source_mask.view(1, -1)
            if not bool(allowed.any(dim=-1).all()):
                raise ValueError("each query row must have at least one allowed source")
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, values)
            spatial_hidden = temporal_hidden + self.spatial_projection(attended)
            return spatial_hidden, weights

        def forward(
            self,
            x: Tensor,
            return_attention: bool = False,
            source_mask: Tensor | None = None,
            candidate_mask: Tensor | None = None,
        ) -> Tensor | tuple[Tensor, Tensor]:
            if x.ndim != 4:
                raise ValueError("x must have shape (batch, history, nodes, features)")
            _, history_length, _, feature_count = x.shape
            if feature_count != self.input_size:
                raise ValueError(f"expected {self.input_size} features, got {feature_count}")
            if history_length <= 0:
                raise ValueError("history length must be positive")
            temporal_hidden = self._temporal_encode(x)
            spatial_hidden, weights = self._global_attention(
                temporal_hidden,
                source_mask=source_mask,
                candidate_mask=candidate_mask,
            )
            self.last_attention_weights = weights
            prediction = self.output(spatial_hidden).squeeze(-1)
            return (prediction, weights) if return_attention else prediction

else:

    class GlobalAttentionBaseline:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
