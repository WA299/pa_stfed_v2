"""ASTGCN-r recent-component adaptation for the centralized V2 protocol."""
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
    _require_torch(); dtype=dtype or torch.float32; e=torch.as_tensor(edge_index,dtype=torch.long); a=torch.zeros((int(num_nodes),int(num_nodes)),dtype=dtype)
    if e.numel(): a[e[0],e[1]]=1.; a[e[1],e[0]]=1.
    a.fill_diagonal_(1.); return a
def build_scaled_laplacian(adjacency: Tensor) -> Tensor:
    _require_torch(); d=adjacency.sum(1).clamp_min(1e-12); norm=d.pow(-.5)[:,None]*adjacency*d.pow(-.5)[None,:]; lap=torch.eye(adjacency.shape[0],device=adjacency.device,dtype=adjacency.dtype)-norm; eig=torch.linalg.eigvalsh(lap).max().real.clamp_min(1e-6); return 2*lap/eig-torch.eye(adjacency.shape[0],device=adjacency.device,dtype=adjacency.dtype)
def chebyshev_supports(scaled_laplacian: Tensor, k: int=3) -> list[Tensor]:
    _require_torch(); n=scaled_laplacian.shape[0]; eye=torch.eye(n,device=scaled_laplacian.device,dtype=scaled_laplacian.dtype); out=[eye]
    if k>1: out.append(scaled_laplacian)
    for _ in range(2,k): out.append(2*scaled_laplacian@out[-1]-out[-2])
    return out
build_chebyshev_supports=chebyshev_supports
if TORCH_AVAILABLE:
 class _ASTBlock(nn.Module):
    def __init__(self,c,n,t,k):
     super().__init__(); self.c=c; self.U1=nn.Parameter(torch.randn(n)*.1); self.U2=nn.Parameter(torch.randn(c,n)*.1); self.U3=nn.Parameter(torch.randn(c)*.1); self.be=nn.Parameter(torch.zeros(1,t,t)); self.Ve=nn.Parameter(torch.eye(t)); self.W1=nn.Parameter(torch.randn(t)*.1); self.W2=nn.Parameter(torch.randn(c,t)*.1); self.W3=nn.Parameter(torch.randn(c)*.1); self.bs=nn.Parameter(torch.zeros(1,n,n)); self.Vs=nn.Parameter(torch.eye(n)); self.theta=nn.ModuleList([nn.Linear(c,c,bias=False) for _ in range(k)]); self.temporal_conv=nn.Conv2d(c,c,(3,1),padding=(1,0)); self.residual=nn.Linear(c,c); self.norm=nn.LayerNorm(c)
    def temporal_attention(self,x):
     lhs=torch.einsum('btnc,n,c->bt',x,self.U1,self.U3); rhs=torch.einsum('btnc,cn,c->bt',x,self.U2,self.U3); return torch.softmax(torch.einsum('bts,tu->btu',lhs.unsqueeze(-1)+rhs.unsqueeze(1)+self.be,self.Ve),-1)
    def spatial_attention(self,x):
     p=x.mean(1); lhs=torch.einsum('bnc,n,c->bn',p,self.U1,self.W3); rhs=torch.einsum('bnc,ct,t->bn',p,self.W2,self.W1); return torch.softmax(torch.einsum('bij,jk->bik',lhs.unsqueeze(-1)+rhs.unsqueeze(1)+self.bs,self.Vs),-1)
    def forward(self,x,supports):
     ta=self.temporal_attention(x); x_t=torch.einsum('btu,bunc->btnc',ta,x); sa=self.spatial_attention(x_t); graph=0
     for support,theta in zip(supports,self.theta): graph=graph+torch.einsum('bij,btjc->btic',support.unsqueeze(0)*sa,theta(x_t))
     y=self.temporal_conv(graph.permute(0,3,1,2)).permute(0,2,3,1); self.temporal_attention_weights=ta; self.spatial_attention_weights=sa; return self.norm(torch.relu(y+self.residual(x_t)))
 class ASTGCNBaseline(nn.Module):
    def __init__(self,edge_index,num_nodes,input_size=6,hidden_channels=32,chebyshev_k=3,num_blocks=2,history_length=168):
     super().__init__(); self.num_nodes=int(num_nodes); self.input_size=input_size; self.hidden_channels=hidden_channels; self.chebyshev_k=chebyshev_k; self.num_blocks=num_blocks; self.history_length=history_length; adj=build_physical_adjacency(edge_index,num_nodes); lap=build_scaled_laplacian(adj); supports=chebyshev_supports(lap,chebyshev_k); self.register_buffer('adjacency',adj); self.register_buffer('scaled_laplacian',lap)
     for i,s in enumerate(supports): self.register_buffer(f'cheb_support_{i}',s)
     self.input_projection=nn.Linear(input_size,hidden_channels); self.blocks=nn.ModuleList([_ASTBlock(hidden_channels,num_nodes,history_length,chebyshev_k) for _ in range(num_blocks)]); self.output=nn.Linear(hidden_channels,1)
    @property
    def cheb_supports(self): return [getattr(self,f'cheb_support_{i}') for i in range(self.chebyshev_k)]
    def forward(self,x):
     if x.ndim!=4 or x.shape[1]!=self.history_length or x.shape[2]!=self.num_nodes or x.shape[3]!=self.input_size: raise ValueError('input must be (B,168,N,6)')
     h=torch.relu(self.input_projection(x))
     for block in self.blocks: h=block(h,self.cheb_supports)
     return self.output(h[:,-1]).squeeze(-1)
else:
 class ASTGCNBaseline:
  def __init__(self,*args,**kwargs): _require_torch()
