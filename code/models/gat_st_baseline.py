"""Shared-GRU plus single-head physical-mask GAT baseline."""

from __future__ import annotations

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
        raise ImportError("GATSTBaseline requires PyTorch; install torch to run the model")


def build_bidirectional_self_loop_edge_index(edge_index: Any, num_nodes: int) -> Tensor:
    """Expand canonical directed branches into unique bidirectional edges + self-loops."""

    _require_torch()
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError("edge_index contains a node outside num_nodes")
    reverse = edges[[1, 0]]
    loops = torch.arange(num_nodes, dtype=torch.long, device=edges.device).repeat(2, 1)
    expanded = torch.cat((edges, reverse, loops), dim=1)
    return torch.unique(expanded, dim=1)


def build_physical_attention_mask(edge_index: Any, num_nodes: int) -> Tensor:
    """Return [N, N] bool mask allowing self and physical one-hop neighbors."""

    _require_torch()
    expanded = build_bidirectional_self_loop_edge_index(edge_index, num_nodes)
    mask = torch.zeros((int(num_nodes), int(num_nodes)), dtype=torch.bool, device=expanded.device)
    mask[expanded[0], expanded[1]] = True
    return mask


if TORCH_AVAILABLE:

    class GATSTBaseline(nn.Module):
        """One shared temporal GRU followed by one masked single-head GAT layer."""

        def __init__(
            self,
            edge_index: Any,
            num_nodes: int,
            input_size: int = 6,
            hidden_size: int = 32,
            num_layers: int = 1,
            negative_slope: float = 0.2,
        ) -> None:
            super().__init__()
            if int(input_size) != 6:
                raise ValueError("GAT-ST baseline requires the six p_calendar features")
            if int(hidden_size) != 32:
                raise ValueError("GAT-ST baseline requires hidden_size=32")
            if int(num_layers) != 1:
                raise ValueError("GAT-ST baseline requires num_layers=1")
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.num_nodes = int(num_nodes)
            self.gru = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=False,
            )
            self.gat_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.attention_src = nn.Parameter(torch.empty(self.hidden_size))
            self.attention_dst = nn.Parameter(torch.empty(self.hidden_size))
            nn.init.xavier_uniform_(self.gat_projection.weight)
            nn.init.xavier_uniform_(self.attention_src.unsqueeze(0))
            nn.init.xavier_uniform_(self.attention_dst.unsqueeze(0))
            self.leaky_relu = nn.LeakyReLU(float(negative_slope))
            self.spatial_projection = nn.Linear(self.hidden_size, self.hidden_size)
            self.output = nn.Linear(self.hidden_size, 1)
            self.register_buffer(
                "attention_mask",
                build_physical_attention_mask(edge_index, self.num_nodes),
            )
            expanded_edges = build_bidirectional_self_loop_edge_index(edge_index, self.num_nodes)
            self.register_buffer("edge_index", expanded_edges)
            self.last_attention_weights: Tensor | None = None

        def _temporal_encode(self, x: Tensor) -> Tensor:
            batch_size, history_length, node_count, feature_count = x.shape
            node_sequences = x.permute(1, 0, 2, 3).reshape(
                history_length, batch_size * node_count, feature_count
            )
            _, hidden = self.gru(node_sequences)
            return hidden[-1].reshape(batch_size, node_count, self.hidden_size)

        def _spatial_attention(self, temporal_hidden: Tensor) -> tuple[Tensor, Tensor]:
            projected = self.gat_projection(temporal_hidden)
            source_score = (projected * self.attention_src).sum(dim=-1)
            target_score = (projected * self.attention_dst).sum(dim=-1)
            scores = self.leaky_relu(source_score.unsqueeze(-1) + target_score.unsqueeze(-2))
            scores = scores.masked_fill(~self.attention_mask.unsqueeze(0), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            aggregated = torch.bmm(weights, projected)
            spatial_hidden = temporal_hidden + self.spatial_projection(aggregated)
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
            spatial_hidden, weights = self._spatial_attention(temporal_hidden)
            self.last_attention_weights = weights
            prediction = self.output(spatial_hidden).squeeze(-1)
            return (prediction, weights) if return_attention else prediction

else:

    class GATSTBaseline:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
