"""Summarize existing centralized validation JSON files only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

GRIDS=("39bus","50bus","56bus","80bus")
STANDARD=("persistence_1h","shared_gru","stgcn","astgcn","graph_wavenet")
FILES={
 "shared_gru":("gru_{g}_p_calendar.json","gru"),
 "stgcn":("stgcn_{g}_p_calendar.json","stgcn"),
 "astgcn":("astgcn_{g}_p_calendar.json","astgcn"),
 "graph_wavenet":("graph_wavenet_{g}_p_calendar.json","graph_wavenet"),
}
METRICS=("mae","rmse","wape_pct","smape_pct")
def _validation(path): return json.loads(path.read_text(encoding="utf-8")).get("validation",{})
def _persistence_sources(indir,grid):
 for model,(fn,_) in FILES.items():
  p=indir/fn.format(g=grid)
  if p.exists() and "persistence_1h" in _validation(p): yield model,_validation(p)["persistence_1h"]
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
 per={g:{} for g in GRIDS}
 for g in GRIDS:
  per[g]["persistence_1h"]=_consistent_persistence(indir,g)
  for model,(fn,validation_key) in FILES.items():
   p=indir/fn.format(g=g)
   if not p.exists(): per[g][model]=None; continue
   v=_validation(p); per[g][model]=v.get(validation_key)
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
