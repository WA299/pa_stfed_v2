from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
GRIDS=("39bus","50bus","56bus","80bus")
STANDARD=("persistence_1h","shared_gru","stgcn","astgcn","graph_wavenet")
FILES={"stgcn":"stgcn_{g}_p_calendar.json","astgcn":"astgcn_{g}_p_calendar.json","graph_wavenet":"graph_wavenet_{g}_p_calendar.json","shared_gru":"centralized_gru_{g}_p_calendar.json"}
def load(path):
 d=json.loads(path.read_text()); return d.get("validation",{})
def summarize(indir):
 rows={g:{} for g in GRIDS}
 for g in GRIDS:
  for model,fn in FILES.items():
   p=indir/fn.format(g=g)
   if p.exists():
    v=load(p); key=model if model in v else next((k for k in v if k not in ("persistence_1h",)),None)
    rows[g][model]=v.get(key) if key else None
   else: rows[g][model]=None
  for m in ("persistence_1h",): rows[g][m]=None
 macro={}
 for model in STANDARD:
  vals=[rows[g][model]["node_macro"] for g in GRIDS if rows[g].get(model)]
  macro[model]={k:float(np.mean([v[k] for v in vals])) for k in ("mae","rmse","wape_pct","smape_pct")} if vals else None
 return {"metadata":{"cross_grid_aggregation":"unweighted_macro_mean","test_evaluated":False},"standard_baselines":rows,"macro_across_grids_node":macro,"diagnostic_variants":{}}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=Path('results/centralized')); p.add_argument('--output-json',type=Path,default=Path('results/centralized/centralized_validation_summary.json')); p.add_argument('--output-md',type=Path,default=Path('results/centralized/centralized_validation_summary.md')); a=p.parse_args(); r=summarize(a.input_dir); a.output_json.write_text(json.dumps(r,indent=2)+'\n'); a.output_md.write_text('# Centralized validation summary\n\n'+json.dumps(r,indent=2)+'\n')
if __name__=='__main__': main()
