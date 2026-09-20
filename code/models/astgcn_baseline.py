"""ASTGCN-r recent-component adaptation for the centralized V2 protocol.

The model accepts repository tensors in ``(B, T, N, F)`` order and uses the
ASTGCN implementation convention ``(B, N, F, T)`` inside each block.
"""
from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Tensor = Any
    nn = None
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("ASTGCNBaseline requires PyTorch")


def build_physical_adjacency(
    edge_index: Any,
    num_nodes: int,
    dtype: Any = None,
) -> Tensor:
    """Build binary undirected physical adjacency without self-loops."""
    _require_torch()
    dtype = dtype or torch.float32
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    adjacency = torch.zeros((int(num_nodes), int(num_nodes)), dtype=dtype)
    if edges.numel():
        if edges.min() < 0 or edges.max() >= num_nodes:
            raise ValueError("edge_index contains a node outside [0, num_nodes)")
        adjacency[edges[0], edges[1]] = 1.0
        adjacency[edges[1], edges[0]] = 1.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


def build_scaled_laplacian(adjacency: Tensor) -> Tensor:
    """Return ``2L/lambda_max - I`` for combinatorial ``L = D - A``."""
    _require_torch()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not torch.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    degree = adjacency.sum(dim=1)
    laplacian = torch.diag(degree) - adjacency
    lambda_max = torch.linalg.eigvalsh(laplacian).max().real
    if not torch.isfinite(lambda_max) or float(lambda_max) <= 0.0:
        raise ValueError("scaled Laplacian requires a graph with a positive eigenvalue")
    identity = torch.eye(
        adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
    )
    return 2.0 * laplacian / lambda_max - identity


def chebyshev_supports(
    scaled_laplacian: Tensor,
    k: int = 3,
) -> list[Tensor]:
    """Build ``[T_0, ..., T_{K-1}]`` using the Chebyshev recurrence."""
    _require_torch()
    if k < 1:
        raise ValueError("k must be positive")
    if (
        scaled_laplacian.ndim != 2
        or scaled_laplacian.shape[0] != scaled_laplacian.shape[1]
    ):
        raise ValueError("scaled_laplacian must be a square matrix")
    identity = torch.eye(
        scaled_laplacian.shape[0],
        device=scaled_laplacian.device,
        dtype=scaled_laplacian.dtype,
    )
    supports = [identity]
    if k > 1:
        supports.append(scaled_laplacian)
    for _ in range(2, k):
        supports.append(
            2.0 * scaled_laplacian @ supports[-1] - supports[-2]
        )
    return supports


# Retain the previous public alias for downstream imports.
build_chebyshev_supports = chebyshev_supports


if TORCH_AVAILABLE:

    class TemporalAttentionLayer(nn.Module):
        """Temporal attention from Guo et al. (AAAI 2019)."""

        def __init__(self, num_nodes: int, in_channels: int, timesteps: int):
            super().__init__()
            self.U1 = nn.Parameter(torch.empty(num_nodes))
            self.U2 = nn.Parameter(torch.empty(in_channels, num_nodes))
            self.U3 = nn.Parameter(torch.empty(in_channels))
            self.be = nn.Parameter(torch.empty(1, timesteps, timesteps))
            self.Ve = nn.Parameter(torch.empty(timesteps, timesteps))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.U1)
            nn.init.xavier_uniform_(self.U2)
            nn.init.uniform_(self.U3)
            nn.init.xavier_uniform_(self.be)
            nn.init.xavier_uniform_(self.Ve)

        def forward(self, x: Tensor) -> Tensor:
            # x: (B, N, F, T)
            lhs = torch.matmul(x.permute(0, 3, 2, 1), self.U1)
            lhs = torch.matmul(lhs, self.U2)
            rhs = torch.einsum("f,bnft->bnt", self.U3, x)
            product = torch.matmul(lhs, rhs)
            attention = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
            return torch.softmax(attention, dim=1)


    class SpatialAttentionLayer(nn.Module):
        """Spatial attention from Guo et al. (AAAI 2019)."""

        def __init__(self, num_nodes: int, in_channels: int, timesteps: int):
            super().__init__()
            self.W1 = nn.Parameter(torch.empty(timesteps))
            self.W2 = nn.Parameter(torch.empty(in_channels, timesteps))
            self.W3 = nn.Parameter(torch.empty(in_channels))
            self.bs = nn.Parameter(torch.empty(1, num_nodes, num_nodes))
            self.Vs = nn.Parameter(torch.empty(num_nodes, num_nodes))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.W1)
            nn.init.xavier_uniform_(self.W2)
            nn.init.uniform_(self.W3)
            nn.init.xavier_uniform_(self.bs)
            nn.init.xavier_uniform_(self.Vs)

        def forward(self, x: Tensor) -> Tensor:
            # x: (B, N, F, T)
            lhs = torch.matmul(x, self.W1)
            lhs = torch.matmul(lhs, self.W2)
            rhs = torch.einsum("f,bnft->bnt", self.W3, x).transpose(1, 2)
            product = torch.matmul(lhs, rhs)
            attention = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
            return torch.softmax(attention, dim=1)


    class ChebConvWithSpatialAttention(nn.Module):
        """Chebyshev graph convolution directly modulated by spatial attention."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            chebyshev_k: int,
        ):
            super().__init__()
            self.Theta = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(in_channels, out_channels))
                    for _ in range(chebyshev_k)
                ]
            )
            for theta in self.Theta:
                nn.init.xavier_uniform_(theta)

        def forward(
            self,
            x: Tensor,
            spatial_attention: Tensor,
            supports: list[Tensor],
        ) -> Tensor:
            # x: (B, N, F_in, T); output: (B, N, F_out, T)
            outputs = []
            for time_index in range(x.shape[-1]):
                graph_signal = x[..., time_index]
                output = graph_signal.new_zeros(
                    graph_signal.shape[0],
                    graph_signal.shape[1],
                    self.Theta[0].shape[1],
                )
                for support, theta in zip(supports, self.Theta):
                    support_with_attention = support.unsqueeze(0) * spatial_attention
                    propagated = torch.matmul(
                        support_with_attention.transpose(1, 2), graph_signal
                    )
                    output = output + torch.matmul(propagated, theta)
                outputs.append(output.unsqueeze(-1))
            return torch.cat(outputs, dim=-1)


    class ASTGCNBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_nodes: int,
            timesteps: int,
            chebyshev_k: int,
        ):
            super().__init__()
            self.temporal_attention = TemporalAttentionLayer(
                num_nodes, in_channels, timesteps
            )
            self.spatial_attention = SpatialAttentionLayer(
                num_nodes, in_channels, timesteps
            )
            self.cheb_conv = ChebConvWithSpatialAttention(
                in_channels, hidden_channels, chebyshev_k
            )
            self.temporal_conv = nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
            )
            self.residual_conv = nn.Conv2d(
                in_channels, hidden_channels, kernel_size=(1, 1)
            )
            self.layer_norm = nn.LayerNorm(hidden_channels)
            self.temporal_attention_weights: Tensor | None = None
            self.spatial_attention_weights: Tensor | None = None

        def forward(self, x: Tensor, supports: list[Tensor]) -> Tensor:
            temporal_attention = self.temporal_attention(x)
            batch, nodes, channels, timesteps = x.shape
            temporal_attended = torch.matmul(
                x.reshape(batch, nodes * channels, timesteps),
                temporal_attention,
            ).reshape(batch, nodes, channels, timesteps)
            spatial_attention = self.spatial_attention(temporal_attended)
            spatial = self.cheb_conv(x, spatial_attention, supports)
            temporal = self.temporal_conv(spatial.permute(0, 2, 1, 3))
            residual = self.residual_conv(x.permute(0, 2, 1, 3))
            activated = torch.relu(temporal + residual)
            output = self.layer_norm(activated.permute(0, 3, 2, 1))
            self.temporal_attention_weights = temporal_attention
            self.spatial_attention_weights = spatial_attention
            return output.permute(0, 2, 3, 1)


    class ASTGCNBaseline(nn.Module):
        def __init__(
            self,
            edge_index: Any,
            num_nodes: int,
            input_size: int = 6,
            hidden_channels: int = 32,
            chebyshev_k: int = 3,
            num_blocks: int = 2,
            history_length: int = 168,
        ):
            super().__init__()
            self.num_nodes = int(num_nodes)
            self.input_size = int(input_size)
            self.hidden_channels = int(hidden_channels)
            self.chebyshev_k = int(chebyshev_k)
            self.num_blocks = int(num_blocks)
            self.history_length = int(history_length)

            adjacency = build_physical_adjacency(edge_index, num_nodes)
            scaled_laplacian = build_scaled_laplacian(adjacency)
            supports = chebyshev_supports(scaled_laplacian, chebyshev_k)
            self.register_buffer("adjacency", adjacency)
            self.register_buffer("scaled_laplacian", scaled_laplacian)
            for index, support in enumerate(supports):
                self.register_buffer(f"cheb_support_{index}", support)

            blocks = []
            in_channels = input_size
            for _ in range(num_blocks):
                blocks.append(
                    ASTGCNBlock(
                        in_channels,
                        hidden_channels,
                        num_nodes,
                        history_length,
                        chebyshev_k,
                    )
                )
                in_channels = hidden_channels
            self.blocks = nn.ModuleList(blocks)
            self.final_conv = nn.Conv2d(
                in_channels=history_length,
                out_channels=1,
                kernel_size=(1, hidden_channels),
            )

        @property
        def cheb_supports(self) -> list[Tensor]:
            return [
                getattr(self, f"cheb_support_{index}")
                for index in range(self.chebyshev_k)
            ]

        def forward(self, x: Tensor) -> Tensor:
            expected = (
                "input must have shape "
                f"(B, {self.history_length}, {self.num_nodes}, {self.input_size})"
            )
            if (
                x.ndim != 4
                or x.shape[1] != self.history_length
                or x.shape[2] != self.num_nodes
                or x.shape[3] != self.input_size
            ):
                raise ValueError(expected)
            hidden = x.permute(0, 2, 3, 1)
            for block in self.blocks:
                hidden = block(hidden, self.cheb_supports)
            output = self.final_conv(hidden.permute(0, 3, 1, 2))
            return output.squeeze(1).squeeze(-1)

else:

    class ASTGCNBaseline:
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()
