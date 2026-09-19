"""Train-only spatial predictive value audit using fixed Ridge regressions."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from code.data.lv_grid_loader import LVGridLoader, LVGridData

LAGS = (1, 24, 168)
MODEL_NAMES = ("self_only", "grid_context", "electrical_neighbors", "statistical_neighbors")

def _finite(x):
    x = float(x); return x if np.isfinite(x) else None

def calendar_features(timestamps, indices):
    ts = timestamps[indices]
    hour = np.asarray([x.hour for x in ts], float); dow = np.asarray([x.dayofweek for x in ts], float)
    return np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24), np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), (dow >= 5).astype(float)])

def valid_target_indices(train_start: int, train_end: int, max_lag: int = 168):
    return np.arange(train_start + max_lag, train_end, dtype=int) if train_end - train_start > max_lag else np.empty(0, dtype=int)

def chronological_audit_split(indices, fit_fraction=.8):
    indices = np.asarray(indices, dtype=int); cut = int(len(indices) * fit_fraction)
    if cut <= 0 or cut >= len(indices): raise ValueError("audit split requires non-empty fit and holdout")
    return indices[:cut], indices[cut:]

def fit_feature_scaler(x):
    x = np.asarray(x, dtype=float); mean = x.mean(axis=0); scale = x.std(axis=0)
    zero = (~np.isfinite(scale)) | (scale <= 0); return mean, np.where(zero, 1.0, scale), zero

def hour_of_week_residual(p, timestamps, fit_time_indices):
    p = np.asarray(p, dtype=float); idx = np.asarray(fit_time_indices, dtype=int)
    hours = np.asarray([timestamps[i].dayofweek * 24 + timestamps[i].hour for i in idx], int)
    means = np.full((168, p.shape[1]), np.nan)
    for h in range(168):
        m = hours == h
        if np.any(m): means[h] = np.nanmean(p[idx[m]], axis=0)
    return np.asarray([p[i] - means[h] for i, h in zip(idx, hours)]), means

def choose_electrical_neighbors(grid: LVGridData, load_indices, max_neighbors=3):
    d = np.asarray(grid.distance_matrices["impedance_abs_distance"], float); load_indices = np.asarray(load_indices, int); out = {}
    for i in load_indices:
        others = [int(j) for j in load_indices if int(j) != int(i)]; others.sort(key=lambda j: (float(d[i, j]), j))
        out[int(i)] = [(j, float(d[i, j])) for j in others[:max_neighbors]]
    return out

def choose_statistical_neighbors(grid, load_indices, fit_time_indices, max_neighbors=3):
    residual, _ = hour_of_week_residual(grid.p, grid.timestamps, fit_time_indices); out = {}
    for i in load_indices:
        vals = []
        for j in load_indices:
            if int(j) == int(i): continue
            a, b = residual[:, i], residual[:, j]; m = np.isfinite(a) & np.isfinite(b)
            corr = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0 else -np.inf
            vals.append((corr, int(j)))
        vals.sort(key=lambda z: (-z[0], z[1])); out[int(i)] = [(j, None if not np.isfinite(c) else float(c)) for c, j in vals[:max_neighbors]]
    return out

def _metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float); err = pred - y; denom = float(np.sum(np.abs(y)))
    values = (float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), float(np.sum(np.abs(err)) / denom * 100) if denom > 0 else 0.0)
    if not all(np.isfinite(values)): raise ValueError("non-finite metric")
    return dict(zip(("mae", "rmse", "wape"), values))

def _feature_row(p, timestamps, t, i, kind, load_indices, neighbors):
    vals = [p[t-l, i] for l in LAGS] + calendar_features(timestamps, np.asarray([t]))[0].tolist()
    if kind == "grid_context":
        agg = p[:, load_indices].sum(axis=1); vals += [agg[t-l] for l in LAGS]
    elif kind in ("electrical_neighbors", "statistical_neighbors"):
        vals += [p[t-l, j] for j, _ in neighbors.get(int(i), []) for l in LAGS]
    return np.asarray(vals, float)

def _feature_matrix(p, timestamps, target_idx, i, kind, load_indices, neighbors):
    """Vectorized equivalent of _feature_row; keeps all lag access within train."""
    target_idx = np.asarray(target_idx, dtype=int)
    x = np.column_stack([p[target_idx-l, i] for l in LAGS] + [calendar_features(timestamps, target_idx)])
    if kind == "grid_context":
        agg = p[:, load_indices].sum(axis=1)
        x = np.column_stack([x] + [agg[target_idx-l] for l in LAGS])
    elif kind in ("electrical_neighbors", "statistical_neighbors"):
        extra = [p[target_idx-l, j] for j, _ in neighbors.get(int(i), []) for l in LAGS]
        if extra: x = np.column_stack([x] + extra)
    return np.asarray(x, dtype=float)

def audit_grid(grid: LVGridData):
    train = grid.splits["train"]; load = np.flatnonzero(grid.load_bus_mask).astype(int)
    idx = valid_target_indices(train.start_index, train.end_index); fit_idx, hold_idx = chronological_audit_split(idx)
    elec = choose_electrical_neighbors(grid, load); stat = choose_statistical_neighbors(grid, load, fit_idx); neighbors = {"electrical_neighbors": elec, "statistical_neighbors": stat}
    per_model = {}
    for kind in MODEL_NAMES:
        bus_metrics = {}; dims = {}
        for i in load:
            nmap = neighbors.get(kind, {}); Xfit = _feature_matrix(grid.p, grid.timestamps, fit_idx, int(i), kind, load, nmap); Xhold = _feature_matrix(grid.p, grid.timestamps, hold_idx, int(i), kind, load, nmap)
            mean, scale, zero = fit_feature_scaler(Xfit); model = Ridge(alpha=1.0).fit((Xfit-mean)/scale, grid.p[fit_idx, i]); pred = model.predict((Xhold-mean)/scale)
            bus_metrics[str(grid.node_ids[i])] = {**_metrics(grid.p[hold_idx, i], pred), "node_index": int(i), "zero_variance_feature_count": int(zero.sum())}; dims[str(grid.node_ids[i])] = int(Xfit.shape[1])
        summary = {m: float(np.mean([v[m] for v in bus_metrics.values()])) for m in ("mae", "rmse", "wape")}; per_model[kind] = {"node_macro": summary, "per_bus": bus_metrics, "feature_dimensions": sorted(set(dims.values()))}
    base = per_model["self_only"]; bbus = base["per_bus"]
    for kind in MODEL_NAMES:
        cur = per_model[kind]; cur["delta_mae_pct"] = 100*(cur["node_macro"]["mae"]-base["node_macro"]["mae"])/base["node_macro"]["mae"]; cur["delta_rmse_pct"] = 100*(cur["node_macro"]["rmse"]-base["node_macro"]["rmse"])/base["node_macro"]["rmse"]; cur["delta_wape_pct"] = 100*(cur["node_macro"]["wape"]-base["node_macro"]["wape"])/base["node_macro"]["wape"]
        cur["improved_bus_count_mae"] = sum(cur["per_bus"][k]["mae"] < bbus[k]["mae"] for k in bbus); cur["improved_bus_count_wape"] = sum(cur["per_bus"][k]["wape"] < bbus[k]["wape"] for k in bbus); cur["total_load_bus_count"] = len(load)
    diag = {"electrical_neighbors": {str(grid.node_ids[i]): [{"bus_id": str(grid.node_ids[j]), "impedance_abs_distance": d} for j,d in vals] for i, vals in elec.items()}, "statistical_neighbors": {str(grid.node_ids[i]): [{"bus_id": str(grid.node_ids[j]), "audit_fit_residual_correlation": c} for j,c in vals] for i, vals in stat.items()}}
    return {"grid_name": grid.grid_name, "node_count": grid.num_nodes, "load_bus_count": len(load), "train_split": {"start_index": train.start_index, "end_index": train.end_index, "sample_count": train.sample_count}, "audit_split": {"fit_target_count": len(fit_idx), "holdout_target_count": len(hold_idx), "fit_target_start": int(fit_idx[0]), "fit_target_end": int(fit_idx[-1]), "holdout_target_start": int(hold_idx[0]), "holdout_target_end": int(hold_idx[-1]), "shuffle": False}, "valid_target_count": len(idx), "models": per_model, "neighbor_diagnostics": diag}

def render_markdown(report):
    lines = ["# V2 Spatial Predictive Value Audit", "", f"time_scope: `{report['time_scope']}`; audit_split: `{report['audit_split']}`.", "", "| Grid | Loads | Fit | Holdout | Model | MAE | RMSE | WAPE (%) | ΔMAE (%) | ΔRMSE (%) | ΔWAPE (%) |", "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g in report["grids"]:
        for k in MODEL_NAMES:
            m = g["models"][k]; lines.append(f"| {g['grid_name']} | {g['load_bus_count']} | {g['audit_split']['fit_target_count']} | {g['audit_split']['holdout_target_count']} | {k} | {m['node_macro']['mae']:.6g} | {m['node_macro']['rmse']:.6g} | {m['node_macro']['wape']:.6g} | {m['delta_mae_pct']:.4g} | {m['delta_rmse_pct']:.4g} | {m['delta_wape_pct']:.4g} |")
    lines += ["", "Neighbor selection uses audit_fit only; formal validation/test are not evaluated."]; return "\n".join(lines)

def build_report(data_root, mapping_path):
    loader = LVGridLoader(data_root, mapping_path); return {"audit": "v2_spatial_predictive_value", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "time_scope": "formal_train_only", "audit_split": "chronological_80_20_within_train", "target": "next_hour_active_power", "lags": list(LAGS), "ridge_alpha": 1.0, "node_scope": "load_buses_only", "electrical_distance_source": "canonical_impedance_abs_distance", "statistical_neighbor_selection": "audit_fit_hour_of_week_residual_correlation", "test_evaluated": False, "validation_evaluated": False, "excluded_inputs": ["validation", "test", "q", "weather", "static_load_metadata", "model_predictions", "neural_networks"], "grids": [audit_grid(loader.load(n)) for n in loader.available_grids()]}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--data-root", type=Path, default=REPO_ROOT.parent/"pa_stfed_data_v2"/"raw"); p.add_argument("--mapping-json", type=Path, default=REPO_ROOT/"results/audits/v2_schema_mapping.json"); p.add_argument("--output-json", type=Path, default=REPO_ROOT/"results/audits/spatial_predictive_value.json"); p.add_argument("--output-md", type=Path, default=REPO_ROOT/"results/audits/spatial_predictive_value.md"); return p.parse_args()
def main():
    a = parse_args(); r = build_report(a.data_root.resolve(), a.mapping_json.resolve()); a.output_json.parent.mkdir(parents=True, exist_ok=True); a.output_md.parent.mkdir(parents=True, exist_ok=True); a.output_json.write_text(json.dumps(r, indent=2, ensure_ascii=True)+"\n", encoding="utf-8"); a.output_md.write_text(render_markdown(r)+"\n", encoding="utf-8"); print(f"grids={len(r['grids'])}")
if __name__ == "__main__": main()
