from types import SimpleNamespace
import numpy as np
import torch
import pytest

from code.audits.federated_transfer_benefit import (
    HISTORY, benefit, fit_fit_only_scaler, make_proxy, scarce_split,
    select_donor, transfer_temporal_parameters,
)

def _grid(nodes=3, train_end=1200):
    rng=np.random.default_rng(3); total=train_end+80; dynamic=rng.normal(size=(total,nodes,7)).astype(np.float32)
    return SimpleNamespace(num_nodes=nodes,load_bus_mask=np.ones(nodes,dtype=bool),dynamic_features=dynamic,p=dynamic[:,:,0].copy(),timestamps=np.arange(total),splits={'train':SimpleNamespace(start_index=100,end_index=train_end),'validation':SimpleNamespace(start_index=train_end,end_index=train_end+40),'test':SimpleNamespace(start_index=train_end+40,end_index=total)})

def test_scarcity_recent_raw_train_and_chronological_disjoint():
    grid=_grid(); split=scarce_split(grid)
    expected=train_end=grid.splits['train'].end_index; assert split.available_start == train_end-int(np.ceil(.25*(train_end-grid.splits['train'].start_index))); assert split.eligible_indices[0] >= split.available_start+HISTORY; assert split.eligible_indices[-1] < train_end
    assert len(set(split.fit_indices)&set(split.calibration_indices))==0 and len(set(split.calibration_indices)&set(split.audit_indices))==0

def test_fit_only_scaler_and_benefit_selection():
    grid=_grid(); split=scarce_split(grid); scaler=fit_fit_only_scaler(grid,split.fit_indices); assert scaler.fit_split=='train_fit_only'; assert scaler.fit_end_index == int(split.fit_indices.max())+1; assert benefit(10,8) == pytest.approx(.2); assert select_donor({'a':-.1,'b':0}) == (None,0.0); assert select_donor({'a':-.1,'b':.2}) == ('b',.2)

def test_temporal_transfer_only_preserves_topology_and_excludes_spatial():
    a,b=_grid(3),_grid(3); ma,mb=make_proxy(a),make_proxy(b); before={k:v.clone() for k,v in ma.named_buffers()}; names=transfer_temporal_parameters(ma,mb); assert set(names)==set(__import__('code.federated.parameter_groups',fromlist=['parameter_groups']).parameter_groups(ma)['temporal']); assert all(torch.equal(v,before[k]) for k,v in ma.named_buffers())
