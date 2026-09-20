"""Compact Graph WaveNet core adaptation with diffusion graph convolution."""
from __future__ import annotations
from typing import Any
try:
 import torch
 from torch import Tensor,nn
 TORCH_AVAILABLE=True
except ImportError:
 torch=None; Tensor=Any; nn=None; TORCH_AVAILABLE=False
def normalized_physical_support(edge_index,num_nodes):
 if not TORCH_AVAILABLE: raise ImportError('GraphWaveNetBaseline requires PyTorch')
 e=torch.as_tensor(edge_index,dtype=torch.long); a=torch.zeros((num_nodes,num_nodes));
 if e.numel(): a[e[0],e[1]]=1.; a[e[1],e[0]]=1.
 a.fill_diagonal_(1.); d=a.sum(1).clamp_min(1e-12).pow(-.5); return d[:,None]*a*d[None,:]
if TORCH_AVAILABLE:
 class DiffusionGraphConv(nn.Module):
  def __init__(self,in_channels,out_channels,num_supports,order=2):
   super().__init__(); self.order=order; self.num_supports=num_supports; self.projection=nn.Conv2d(in_channels*(1+num_supports*order),out_channels,(1,1))
  def forward(self,x,supports):
   pieces=[x]
   for support in supports:
    z=x
    for _ in range(self.order): z=torch.einsum('ij,bcjt->bcit',support,z); pieces.append(z)
   return self.projection(torch.cat(pieces,dim=1))
 class GraphWaveNetBaseline(nn.Module):
  def __init__(self,edge_index,num_nodes,input_size=6,residual_channels=32,dilation_channels=32,skip_channels=64,end_channels=128,kernel_size=2,blocks=4,layers_per_block=2,adaptive_embedding_dim=10,dropout=0.3):
   super().__init__(); self.num_nodes=int(num_nodes); self.input_size=input_size; self.residual_channels=residual_channels; self.kernel_size=kernel_size; self.blocks=blocks; self.layers_per_block=layers_per_block; self.dropout_rate=dropout; self.receptive_field=1+blocks*(2**layers_per_block-1)*(kernel_size-1)
   self.start_conv=nn.Conv2d(input_size,residual_channels,(1,1)); self.filter_convs=nn.ModuleList(); self.gate_convs=nn.ModuleList(); self.graph_convs=nn.ModuleList(); self.residual_convs=nn.ModuleList(); self.skip_convs=nn.ModuleList(); self.norms=nn.ModuleList()
   for _ in range(blocks):
    for layer in range(layers_per_block):
     dil=2**layer; self.filter_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dil))); self.gate_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dil))); self.graph_convs.append(DiffusionGraphConv(dilation_channels,residual_channels,2,2)); self.residual_convs.append(nn.Conv2d(residual_channels,residual_channels,(1,1))); self.skip_convs.append(nn.Conv2d(dilation_channels,skip_channels,(1,1))); self.norms.append(nn.BatchNorm2d(residual_channels))
   self.end=nn.Sequential(nn.ReLU(),nn.Conv2d(skip_channels,end_channels,(1,1)),nn.ReLU(),nn.Conv2d(end_channels,1,(1,1))); self.nodevec1=nn.Parameter(torch.randn(num_nodes,adaptive_embedding_dim)*.1); self.nodevec2=nn.Parameter(torch.randn(adaptive_embedding_dim,num_nodes)*.1); self.register_buffer('predefined_support',normalized_physical_support(edge_index,num_nodes))
  def adaptive_adjacency(self): return torch.softmax(torch.relu(self.nodevec1@self.nodevec2),dim=1)
  def temporal_forward(self,x):
   h=self.start_conv(x.permute(0,3,2,1)); skips=[]; idx=0
   for _ in range(self.blocks):
    for _ in range(self.layers_per_block):
     residual=h; dil=2**(idx%self.layers_per_block); padded=torch.nn.functional.pad(h,((self.kernel_size-1)*dil,0,0,0)); z=torch.tanh(self.filter_convs[idx](padded))*torch.sigmoid(self.gate_convs[idx](padded)); skips.append(self.skip_convs[idx](z)); g=self.graph_convs[idx](z,(self.predefined_support,self.adaptive_adjacency())); h=self.norms[idx](self.residual_convs[idx](g)+residual[...,-g.shape[-1]:]); h=torch.nn.functional.dropout(h,p=self.dropout_rate,training=self.training); idx+=1
   return h,skips
  def forward(self,x):
   if x.ndim!=4 or x.shape[2]!=self.num_nodes or x.shape[3]!=self.input_size: raise ValueError('x must be (B,T,N,F)')
   h,skips=self.temporal_forward(x); skip=skips[-1]
   for s in skips[:-1]: skip=skip[...,-s.shape[-1]:]+s
   return self.end(skip)[...,-1].squeeze(1).squeeze(-1)
else:
 class GraphWaveNetBaseline:
  def __init__(self,*args,**kwargs): raise ImportError('GraphWaveNetBaseline requires PyTorch')
