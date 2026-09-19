"""Compact ASTGCN core reimplementation for centralized baselines."""
from __future__ import annotations
from typing import Any
try:
    import torch
    from torch import Tensor, nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None; Tensor = Any; nn = None; TORCH_AVAILABLE = False

def _require_torch():
    if not TORCH_AVAILABLE: raise ImportError("ASTGCNBaseline requires PyTorch")

def build_physical_adjacency(edge_index: Any, num_nodes: int, dtype=None) -> Tensor:
    _require_torch(); dtype = dtype or torch.float32
    e = torch.as_tensor(edge_index, dtype=torch.long)
    if e.ndim != 2 or e.shape[0] != 2: raise ValueError("edge_index must have shape (2,E)")
    a = torch.zeros((num_nodes, num_nodes), dtype=dtype)
    if e.numel():
        a[e[0], e[1]] = 1.; a[e[1], e[0]] = 1.
    a.fill_diagonal_(1.); return a

def build_scaled_laplacian(adjacency: Tensor) -> Tensor:
    _require_torch()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]: raise ValueError("adjacency must be square")
    deg = adjacency.sum(1); inv = deg.clamp_min(1e-12).pow(-.5)
    norm = inv[:, None] * adjacency * inv[None, :]
    lap = torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype) - norm
    eig = torch.linalg.eigvalsh(lap).max().real.clamp_min(1e-6)
    eye = torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
    return (2.0 / eig) * lap - eye

def chebyshev_supports(scaled_laplacian: Tensor, k: int = 3) -> list[Tensor]:
    _require_torch()
    if k < 1: raise ValueError("k must be positive")
    n = scaled_laplacian.shape[0]; eye = torch.eye(n, device=scaled_laplacian.device, dtype=scaled_laplacian.dtype)
    out = [eye]
    if k > 1: out.append(scaled_laplacian)
    for _ in range(2, k): out.append(2 * scaled_laplacian @ out[-1] - out[-2])
    return out

build_chebyshev_supports = chebyshev_supports

if TORCH_AVAILABLE:
 class _Block(nn.Module):
    def __init__(self, channels: int, k: int):
        super().__init__(); self.channels = channels; self.k = k
        self.temporal_q = nn.Linear(channels, channels); self.temporal_k = nn.Linear(channels, channels)
        self.spatial_q = nn.Linear(channels, channels); self.spatial_k = nn.Linear(channels, channels)
        self.cheb = nn.ModuleList([nn.Linear(channels, channels) for _ in range(k)])
        self.temporal = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.residual = nn.Linear(channels, channels)
    def forward(self, x: Tensor, supports: list[Tensor]):
        td = x.mean(2); ta = torch.softmax(torch.matmul(self.temporal_q(td), self.temporal_k(td).transpose(1, 2)) / self.channels**.5, dim=-1)
        x = x + torch.einsum('bij,bjnc->binc', ta, x)
        nd = x.mean(1); sa = torch.softmax(torch.matmul(self.spatial_q(nd), self.spatial_k(nd).transpose(1, 2)) / self.channels**.5, dim=-1)
        x = x + torch.einsum('bij,btjc->btic', sa, x)
        z = sum(lin(torch.einsum('ij,btjc->btic', s, x)) for s, lin in zip(supports, self.cheb))
        z = torch.relu(self.temporal(z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        self.temporal_attention, self.spatial_attention = ta, sa
        return torch.relu(z + self.residual(x))

 class ASTGCNBaseline(nn.Module):
    def __init__(self, edge_index: Any, num_nodes: int, input_size: int = 6, hidden_channels: int = 32, chebyshev_k: int = 3, num_blocks: int = 2):
        super().__init__(); self.num_nodes = int(num_nodes); self.input_size = int(input_size); self.hidden_channels = int(hidden_channels); self.chebyshev_k = int(chebyshev_k); self.num_blocks = int(num_blocks)
        adj = build_physical_adjacency(edge_index, self.num_nodes); lap = build_scaled_laplacian(adj); supports = chebyshev_supports(lap, chebyshev_k)
        self.register_buffer('adjacency', adj); self.register_buffer('scaled_laplacian', lap)
        for i, s in enumerate(supports): self.register_buffer(f'cheb_support_{i}', s)
        self.input_projection = nn.Linear(input_size, hidden_channels); self.blocks = nn.ModuleList([_Block(hidden_channels, chebyshev_k) for _ in range(num_blocks)]); self.output = nn.Linear(hidden_channels, 1)
    @property
    def cheb_supports(self): return [getattr(self, f'cheb_support_{i}') for i in range(self.chebyshev_k)]
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[2] != self.num_nodes or x.shape[3] != self.input_size: raise ValueError('input shape mismatch')
        h = self.input_projection(x)
        for b in self.blocks: h = b(h, self.cheb_supports)
        return self.output(h[:, -1]).squeeze(-1)
else:
 class ASTGCNBaseline:
    def __init__(self,*args,**kwargs): _require_torch()
