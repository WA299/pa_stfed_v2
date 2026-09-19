"""Lightweight fixed-topology STGCN baseline.

The model uses only binary physical adjacency from the canonical loader.  Edge
attributes, electrical distances, attention, and federated-learning concerns
are intentionally outside this baseline.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without torch.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("STGCNBaseline requires PyTorch; install torch to run the model")


def build_binary_physical_normalized_adjacency(
    edge_index: Any,
    num_nodes: int,
    *,
    dtype: Any = None,
) -> Tensor:
    """Build symmetric normalized adjacency from a directed edge index.

    The loader's branch orientation is discarded: every edge contributes both
    directions, then a unit self-loop is added before symmetric normalization.
    """

    _require_torch()
    if int(num_nodes) <= 0:
        raise ValueError("num_nodes must be positive")
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes)):
        raise ValueError("edge_index contains a node outside num_nodes")
    adjacency = torch.zeros((int(num_nodes), int(num_nodes)), dtype=dtype or torch.float32)
    if edges.shape[1]:
        source, target = edges[0], edges[1]
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)
    normalized = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    return normalized


if TORCH_AVAILABLE:

    class STGCNBaseline(nn.Module):
        """Small STGCN-style forecaster with a shared graph transform."""

        def __init__(
            self,
            edge_index: Any,
            num_nodes: int,
            input_size: int = 6,
            hidden_dim: int = 32,
            temporal_kernel_size: int = 3,
        ) -> None:
            super().__init__()
            if int(input_size) != 6:
                raise ValueError("STGCN baseline requires the six p_calendar features")
            if int(hidden_dim) <= 0:
                raise ValueError("hidden_dim must be positive")
            if int(temporal_kernel_size) <= 0 or int(temporal_kernel_size) % 2 == 0:
                raise ValueError("temporal_kernel_size must be a positive odd integer")
            self.input_size = int(input_size)
            self.hidden_dim = int(hidden_dim)
            self.num_nodes = int(num_nodes)
            self.temporal_kernel_size = int(temporal_kernel_size)
            self.input_projection = nn.Linear(self.input_size, self.hidden_dim)
            padding = self.temporal_kernel_size // 2
            self.temporal_conv1 = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=(self.temporal_kernel_size, 1),
                padding=(padding, 0),
            )
            self.graph_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.temporal_conv2 = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=(self.temporal_kernel_size, 1),
                padding=(padding, 0),
            )
            self.output_projection = nn.Linear(self.hidden_dim, 1)
            self.register_buffer(
                "adjacency_norm",
                build_binary_physical_normalized_adjacency(edge_index, self.num_nodes),
            )

        def forward(self, x: Tensor) -> Tensor:
            if x.ndim != 4:
                raise ValueError("x must have shape (batch, history, nodes, features)")
            batch_size, history_length, node_count, feature_count = x.shape
            if node_count != self.num_nodes:
                raise ValueError(f"expected {self.num_nodes} nodes, got {node_count}")
            if feature_count != self.input_size:
                raise ValueError(f"expected {self.input_size} features, got {feature_count}")
            if history_length < self.temporal_kernel_size:
                raise ValueError("history length must be at least the temporal kernel size")

            hidden = self.input_projection(x)
            hidden = hidden.permute(0, 3, 1, 2)
            hidden = torch.relu(self.temporal_conv1(hidden))
            hidden = torch.einsum("ij,bhtj->bhti", self.adjacency_norm, hidden)
            hidden = torch.relu(self.graph_projection(hidden.permute(0, 2, 3, 1)))
            hidden = hidden.permute(0, 3, 1, 2)
            hidden = torch.relu(self.temporal_conv2(hidden))
            last_hidden = hidden[:, :, -1, :].permute(0, 2, 1)
            return self.output_projection(last_hidden).squeeze(-1)

else:

    class STGCNBaseline:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
