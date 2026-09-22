"""Summarize existing centralized validation JSON files only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

GRIDS=("39bus","50bus","56bus","80bus")
STANDARD=("persistence_1h","shared_gru","stgcn","astgcn","graph_wavenet","puc_rstattn_v2")
FILES={
 "shared_gru":("gru_{g}_p_calendar.json","gru"),
 "astgcn":("astgcn_{g}_p_calendar.json","astgcn"),
 "graph_wavenet":("graph_wavenet_{g}_p_calendar.json","graph_wavenet"),
}
PUC_V2_FILES="puc_rstattn_v2_{g}_p_calendar.json"
STGCN_CANDIDATES=(
 "stgcn_{g}_p_calendar_binary_physical.json",
 "stgcn_{g}_p_calendar.json",
)
METRICS=("mae","rmse","wape_pct","smape_pct")
def _validation(path): return json.loads(path.read_text(encoding="utf-8")).get("validation",{})
def _config(path): return json.loads(path.read_text(encoding="utf-8")).get("config",{})
def _is_standard_stgcn(path):
 c=_config(path)
 adjacency_mode=c.get("adjacency_mode")
 adjacency_type=c.get("adjacency_type")
 adjacency_ok=(adjacency_mode == "binary_physical" if adjacency_mode else adjacency_type == "binary_physical_normalized")
 return (
  c.get("feature_mode") == "p_calendar"
  and c.get("topology_used") is True
  and c.get("electrical_edge_features_used") is False
  and adjacency_ok
 )
def _stgcn_candidates(indir,grid):
 return [indir/name.format(g=grid) for name in STGCN_CANDIDATES if (indir/name.format(g=grid)).exists()]
def _same_metrics(left,right):
 for scope in ("node_macro","grid_aggregate"):
  for metric in METRICS:
   if not np.isclose(float(left[scope][metric]),float(right[scope][metric]),rtol=1e-10,atol=1e-12): return False
 return True
def _select_stgcn(indir,grid):
 valid=[]
 for path in _stgcn_candidates(indir,grid):
  if _is_standard_stgcn(path): valid.append(path)
 if not valid:return None
 selected=valid[0]
 selected_result=_validation(selected).get("stgcn")
 for other in valid[1:]:
  other_result=_validation(other).get("stgcn")
  if selected_result is None or other_result is None or not _same_metrics(selected_result,other_result):
   raise ValueError(f"ambiguous standard STGCN results for {grid}: {selected.name} vs {other.name}")
 return selected
def _model_paths(indir,grid,model):
 if model == "stgcn":
  selected=_select_stgcn(indir,grid)
  return [selected] if selected is not None else []
 if model == "puc_rstattn_v2":
  path=indir/PUC_V2_FILES.format(g=grid)
  return [path] if path.exists() else []
 fn,_=FILES[model]
 path=indir/fn.format(g=grid)
 return [path] if path.exists() else []

def _validate_puc_v2(path):
 data=json.loads(path.read_text(encoding="utf-8"))
 config=data.get("config",{})
 if config.get("model") != "PUC-RSTAttn-V2" or config.get("feature_mode") != "p_calendar" or config.get("test_evaluated") is not False:
  raise ValueError(f"invalid frozen PUC-RSTAttn V2 result: {path.name}")
 if "final" not in data.get("validation",{}):
  raise ValueError(f"missing final validation in frozen PUC-RSTAttn V2 result: {path.name}")
 return data
def _persistence_sources(indir,grid):
 for model in (*FILES.keys(),"stgcn"):
  for p in _model_paths(indir,grid,model):
   if "persistence_1h" in _validation(p): yield model,_validation(p)["persistence_1h"]
def _consistent_persistence(indir,grid):
 sources=list(_persistence_sources(indir,grid))
 if not sources:return None
 ref=sources[0][1]
 for name,item in sources[1:]:
  for scope in ("node_macro","grid_aggregate"):
   for metric in METRICS:
    if not np.isclose(float(ref[scope][metric]),float(item[scope][metric]),rtol=1e-10,atol=1e-12): raise ValueError(f"inconsistent persistence_1h for {grid}: {sources[0][0]} vs {name}, {scope}.{metric}")
 return ref
def summarize(indir):
 puc_paths=[indir/PUC_V2_FILES.format(g=g) for g in GRIDS]
 existing_puc=[p for p in puc_paths if p.exists()]
 if existing_puc and len(existing_puc) != len(GRIDS):
  missing=", ".join(p.name for p in puc_paths if not p.exists())
  raise ValueError(f"missing frozen PUC-RSTAttn V2 results: {missing}")
 per={g:{} for g in GRIDS}
 for g in GRIDS:
  per[g]["persistence_1h"]=_consistent_persistence(indir,g)
  for model in (*FILES.keys(),"stgcn","puc_rstattn_v2"):
   paths=_model_paths(indir,g,model)
   if not paths: per[g][model]=None; continue
   v=_validation(paths[0]) if model != "puc_rstattn_v2" else _validate_puc_v2(paths[0]).get("validation",{})
   validation_key="stgcn" if model == "stgcn" else ("final" if model == "puc_rstattn_v2" else FILES[model][1])
   per[g][model]=v.get(validation_key)
 macro={"node_macro":{},"grid_aggregate":{}}
 for model in STANDARD:
  for scope in macro:
   vals=[per[g][model][scope] for g in GRIDS if per[g].get(model) and scope in per[g][model]]
   macro[scope][model]={"metrics":{m:float(np.mean([x[m] for x in vals])) for m in METRICS} if vals else None,"available_grid_count":len(vals),"expected_grid_count":4}
 return {"metadata":{"cross_grid_aggregation":"unweighted_macro_mean","micro_wape_computed":False,"test_evaluated":False,"wape_unit":"percent"},"standard_baselines":per,"macro_across_grids":macro,"diagnostic_variants":{}}
def render_markdown(report):
 lines=["# Centralized Validation Summary","","Validation-only; test entries are ignored."]
 for scope,title in (("node_macro","Per-grid node macro"),("grid_aggregate","Per-grid grid aggregate")):
  lines += ["",f"## {title}","","| Grid | Model | MAE | RMSE | WAPE (%) | sMAPE (%) |","|---|---|---:|---:|---:|---:|"]
  for g in GRIDS:
   for model in STANDARD:
    item=report["standard_baselines"][g].get(model)
    if item is None or scope not in item: lines.append(f"| {g} | {model} | missing | missing | missing | missing |"); continue
    values=item[scope]; lines.append(f"| {g} | {model} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
 for scope in ("node_macro","grid_aggregate"):
  lines += ["",f"## Cross-grid {scope}","","| Model | MAE | RMSE | WAPE (%) | sMAPE (%) | Available | Expected |","|---|---:|---:|---:|---:|---:|---:|"]
  for model in STANDARD:
   item=report["macro_across_grids"][scope][model]
   if item["metrics"] is None: lines.append(f"| {model} | missing | missing | missing | missing | {item['available_grid_count']} | {item['expected_grid_count']} |")
   else: m=item["metrics"]; lines.append(f"| {model} | {m['mae']:.6g} | {m['rmse']:.6g} | {m['wape_pct']:.6g} | {m['smape_pct']:.6g} | {item['available_grid_count']} | {item['expected_grid_count']} |")
 lines += ["","Missing entries are explicit; diagnostic variants are excluded from standard_baselines."]
 return "\n".join(lines)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=Path('results/centralized')); p.add_argument('--output-json',type=Path,default=Path('results/centralized/centralized_validation_summary.json')); p.add_argument('--output-md',type=Path,default=Path('results/centralized/centralized_validation_summary.md')); a=p.parse_args(); r=summarize(a.input_dir); a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8'); a.output_md.write_text(render_markdown(r)+'\n',encoding='utf-8')
if __name__=='__main__': main()
