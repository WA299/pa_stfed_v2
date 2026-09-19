"""Compact Graph WaveNet-style centralized baseline."""
from __future__ import annotations
from typing import Any
try:
    import torch
    from torch import Tensor, nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None; Tensor = Any; nn = None; TORCH_AVAILABLE = False

def normalized_physical_support(edge_index: Any, num_nodes: int) -> Tensor:
    if not TORCH_AVAILABLE: raise ImportError("GraphWaveNetBaseline requires PyTorch")
    e = torch.as_tensor(edge_index, dtype=torch.long)
    a = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if e.numel():
        a[e[0], e[1]] = 1; a[e[1], e[0]] = 1
    a.fill_diagonal_(1)
    d = a.sum(1).clamp_min(1e-12).pow(-0.5)
    return d[:, None] * a * d[None, :]

if TORCH_AVAILABLE:
 class GraphWaveNetBaseline(nn.Module):
    def __init__(self, edge_index: Any, num_nodes: int, input_size: int = 6,
                 residual_channels: int = 32, dilation_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 kernel_size: int = 2, blocks: int = 4, layers_per_block: int = 2,
                 adaptive_embedding_dim: int = 10):
        super().__init__(); self.num_nodes=int(num_nodes); self.input_size=int(input_size)
        self.residual_channels=residual_channels; self.kernel_size=kernel_size
        self.blocks=blocks; self.layers_per_block=layers_per_block
        self.receptive_field = 1 + blocks * (2**layers_per_block - 1) * (kernel_size-1)
        self.start_conv = nn.Conv2d(input_size, residual_channels, (1,1))
        self.filter_convs = nn.ModuleList(); self.gate_convs=nn.ModuleList()
        self.residual_convs=nn.ModuleList(); self.skip_convs=nn.ModuleList(); self.graph_convs=nn.ModuleList()
        for b in range(blocks):
            for l in range(layers_per_block):
                dil=2**l
                self.filter_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dil)))
                self.gate_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dil)))
                self.graph_convs.append(nn.Conv2d(dilation_channels,residual_channels,(1,1)))
                self.residual_convs.append(nn.Conv2d(dilation_channels,residual_channels,(1,1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels,skip_channels,(1,1)))
        self.end = nn.Sequential(nn.ReLU(), nn.Conv2d(skip_channels,end_channels,(1,1)), nn.ReLU(), nn.Conv2d(end_channels,1,(1,1)))
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, adaptive_embedding_dim)*0.1)
        self.nodevec2 = nn.Parameter(torch.randn(adaptive_embedding_dim, num_nodes)*0.1)
        self.register_buffer('predefined_support', normalized_physical_support(edge_index,num_nodes))

    def adaptive_adjacency(self) -> Tensor:
        return torch.softmax(torch.relu(self.nodevec1 @ self.nodevec2), dim=1)

    def graph_propagate(self, x: Tensor, support: Tensor) -> Tensor:
        return torch.einsum('ij,bcjt->bcit', support, x)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[2] != self.num_nodes or x.shape[3] != self.input_size:
            raise ValueError('x must have shape (batch,time,nodes,features)')
        h = self.start_conv(x.permute(0,3,2,1))
        supports = (self.predefined_support, self.adaptive_adjacency())
        skip = None; idx=0
        for _ in range(self.blocks):
            for _ in range(self.layers_per_block):
                residual=h; dil=2**(idx % self.layers_per_block); pad=(self.kernel_size-1)*dil
                hp=torch.nn.functional.pad(h,(pad,0,0,0))
                z=torch.tanh(self.filter_convs[idx](hp))*torch.sigmoid(self.gate_convs[idx](hp))
                s=self.skip_convs[idx](z); skip=s if skip is None else skip[..., -s.shape[-1]:]+s
                g=sum(self.graph_propagate(z,sp) for sp in supports)
                h=self.residual_convs[idx](g)+residual[..., -g.shape[-1]:]
                idx+=1
        out=self.end(skip if skip is not None else h)
        return out[..., -1].squeeze(1).squeeze(-1)
else:
 class GraphWaveNetBaseline:
    def __init__(self,*args,**kwargs): raise ImportError("GraphWaveNetBaseline requires PyTorch")
