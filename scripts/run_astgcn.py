"""Validation-only ASTGCN centralized baseline runner (manual training entry point)."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from typing import Any
import numpy as np
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from code.data.forecast_dataset import ForecastWindowDataset
from code.data.lv_grid_loader import LVGridLoader
from code.models.centralized_gru import TORCH_AVAILABLE,best_epoch_summary,evaluate_validation,masked_scaled_mae,persistence_1h_predictions
from code.models.astgcn_baseline import ASTGCNBaseline
GRID_NAMES=("39_bus_semi_urban_reference_grid","50_bus_rural_reference_grid","56_bus_semi_urban_reference_grid","80_bus_rural_reference_grid")
SHORT={GRID_NAMES[0]:"39bus",GRID_NAMES[1]:"50bus",GRID_NAMES[2]:"56bus",GRID_NAMES[3]:"80bus"}
FEATURE_MODE="p_calendar"; INPUT_SIZE=6; HIDDEN_CHANNELS=32; CHEBYSHEV_K=3; NUM_BLOCKS=2; DEFAULT_SEED=42; DEFAULT_MAX_EPOCHS=50; DEFAULT_PATIENCE=8; DEFAULT_BATCH_SIZE=32; DEFAULT_LEARNING_RATE=1e-3
def output_paths(grid_name, output_dir):
    if grid_name not in GRID_NAMES: raise ValueError("unsupported grid")
    return output_dir/f"astgcn_{SHORT[grid_name]}_p_calendar.json", output_dir/f"astgcn_{SHORT[grid_name]}_p_calendar.md"
def build_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--grid',choices=GRID_NAMES,default=GRID_NAMES[0]); p.add_argument('--device',default='cpu'); p.add_argument('--seed',type=int,default=DEFAULT_SEED); p.add_argument('--max-epochs',type=int,default=DEFAULT_MAX_EPOCHS); p.add_argument('--patience',type=int,default=DEFAULT_PATIENCE); p.add_argument('--batch-size',type=int,default=DEFAULT_BATCH_SIZE); p.add_argument('--learning-rate',type=float,default=DEFAULT_LEARNING_RATE); p.add_argument('--data-root',type=Path,default=REPO_ROOT.parent/'pa_stfed_data_v2'/'raw'); p.add_argument('--mapping-json',type=Path,default=REPO_ROOT/'results/audits/v2_schema_mapping.json'); p.add_argument('--output-dir',type=Path,default=REPO_ROOT/'results/centralized'); return p
def _batches(n,b): return [np.arange(i,min(i+b,n)) for i in range(0,n,b)]
def _batch(ds,sc,inds):
    w=[ds[int(i)] for i in inds]; return np.stack([sc.transform(z['x']) for z in w]).astype('float32'),np.stack([sc.transform_target(z['y']) for z in w]).astype('float32')
def _seed(s): random.seed(s); np.random.seed(s); import torch; torch.manual_seed(s)
def run_training(args):
    if not TORCH_AVAILABLE: raise RuntimeError('PyTorch required')
    import torch
    _seed(args.seed); grid=LVGridLoader(args.data_root,args.mapping_json).load(args.grid); ds=ForecastWindowDataset.from_grid(grid,FEATURE_MODE); sc=ds.fit_scaler(); tr=ds.split('train'); va=ds.split('validation'); dev=torch.device(args.device)
    model=ASTGCNBaseline(grid.edge_index,grid.num_nodes,INPUT_SIZE,HIDDEN_CHANNELS,CHEBYSHEV_K,NUM_BLOCKS).to(dev); opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate); mask=torch.from_numpy(grid.load_bus_mask); best=float('inf'); state=None; hist=[]; stale=0
    def evaluate(data):
        pred=[]; actual=[]; model.eval()
        with torch.no_grad():
            for inds in _batches(len(data),args.batch_size):
                x,y=_batch(data,sc,inds); pred.append(sc.inverse_transform_target(model(torch.from_numpy(x).to(dev)).cpu().numpy())); actual.append(sc.inverse_transform_target(y))
        return evaluate_validation(np.concatenate(actual),np.concatenate(pred),data.load_bus_mask)
    for epoch in range(1,args.max_epochs+1):
        model.train(); losses=[]
        for inds in _batches(len(tr),args.batch_size):
            x,y=_batch(tr,sc,inds); opt.zero_grad(); loss=masked_scaled_mae(model(torch.from_numpy(x).to(dev)),torch.from_numpy(y).to(dev),mask); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        val=evaluate(va); hist.append({'epoch':epoch,'train_scaled_mae':float(np.mean(losses)),'validation':val})
        if val['node_macro']['mae']<best: best=val['node_macro']['mae']; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=args.patience: break
    if state: model.load_state_dict(state)
    val=evaluate(va); pa=evaluate_validation(grid.p[va.target_indices],persistence_1h_predictions(grid.p,va.target_indices),grid.load_bus_mask)
    return {'config':{'model':'ASTGCN','implementation':'ASTGCN_core_reimplementation','grid_name':args.grid,'feature_mode':'p_calendar','history_length':168,'forecast_horizon':1,'topology_used':True,'graph_type':'physical_binary_undirected','chebyshev_k':3,'hidden_channels':32,'num_blocks':2,'electrical_distance_used':False,'electrical_edge_features_used':False,'adaptive_graph_used':False,'test_evaluated':False},'validation':{'astgcn':val,'persistence_1h':pa},'summary':best_epoch_summary(hist),'training_history':hist,'test_evaluated':False}
def main():
    a=build_parser().parse_args(); j,m=output_paths(a.grid,a.output_dir); r=run_training(a); j.parent.mkdir(parents=True,exist_ok=True); j.write_text(json.dumps(r,indent=2)+'\n'); m.write_text('# ASTGCN centralized baseline\n\nValidation-only report.\n'); print(f'wrote {j} and {m}')
if __name__=='__main__': main()
