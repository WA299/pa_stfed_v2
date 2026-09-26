"""Run the train-only directed cross-grid transfer-benefit audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from code.audits.federated_transfer_benefit import *  # noqa: F401,F403,E402
from code.data.lv_grid_loader import LVGridLoader
from code.federated.parameter_groups import parameter_groups
from scripts.run_federated import CLIENT_GRID_NAMES, build_synthetic_clients

def _matrix(rows, names): return {t: {d: (None if t == d else rows[t].get(d)) for d in names} for t in names}
def _grid_from_client(client):
    from types import SimpleNamespace
    n=6200; nodes=client.model.num_nodes; rng=np.random.default_rng(100+nodes); dynamic=rng.normal(size=(n,nodes,7)).astype(np.float32); return SimpleNamespace(grid_name=client.grid_name,num_nodes=nodes,load_bus_mask=client.load_bus_mask,dynamic_features=dynamic,p=dynamic[:,:,0].copy(),timestamps=np.arange(n),splits={'train':SimpleNamespace(start_index=0,end_index=5964),'validation':SimpleNamespace(start_index=5964,end_index=6082),'test':SimpleNamespace(start_index=6082,end_index=n)})
def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _target_initialized_from_donor(grid, donor_state):
    target = make_proxy(grid)
    names = set(parameter_groups(target)["temporal"])
    state = target.state_dict()
    for name in names:
        state[name] = donor_state[name].clone()
    return state


def _metrics_mean(items):
    return {scope: {metric: float(np.mean([item[scope][metric] for item in items])) for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")}


def _relative(local, selected):
    return float((local["node_macro"]["mae"] - selected["node_macro"]["mae"]) / (local["node_macro"]["mae"] + 1e-12))


def run_audit(data_root, mapping_json, device='cpu', synthetic=False, max_epochs=MAX_EPOCHS):
    grids = ({c.grid_name: _grid_from_client(c) for c in build_synthetic_clients(SEED)}
             if synthetic else {n: LVGridLoader(data_root, mapping_json).load(n) for n in CLIENT_GRID_NAMES})
    names = tuple(CLIENT_GRID_NAMES)
    donor_states, donor_metadata = {}, {}
    # Each full-history donor is trained once, independently of target usage.
    for donor in names:
        grid = grids[donor]; fit, cal = donor_split(grid); scaler = fit_fit_only_scaler(grid, fit)
        model, training = train_proxy(grid, fit, cal, scaler, device=device, max_epochs=max_epochs)
        donor_states[donor] = _cpu_state(model)
        donor_metadata[donor] = {
            "donor_fit_target_count": len(fit), "donor_calibration_target_count": len(cal),
            "scaler_fit_start_index": scaler.fit_start_index, "scaler_fit_end_index": scaler.fit_end_index,
            "scaler_fit_timestamp_end": scaler.fit_timestamp_end,
            "best_calibration_node_mae": training["best_calibration_node_mae"], "epochs_run": training["epochs_run"],
        }
        del model

    local, transfers, scarcity, selected = {}, {}, {}, {}
    cb = {t: {} for t in names}; ab = {t: {} for t in names}; vb = {t: {} for t in names}
    # Stage A: no canonical validation access.
    for target in names:
        grid = grids[target]; split = scarce_split(grid); scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
        first_fit = int(split.fit_indices[0]); first_cal = int(split.calibration_indices[0]); first_audit = int(split.audit_indices[0])
        if not (split.fit_indices[-1] < split.calibration_indices[0] < split.audit_indices[0]): raise AssertionError("target splits are not chronological")
        if not np.all(split.eligible_indices < grid.splits["train"].end_index): raise AssertionError("eligible target enters validation")
        if scaler.fit_end_index > first_cal or scaler.fit_start_index < split.available_start: raise AssertionError("invalid target scaler range")
        scarcity[target] = {
            "canonical_train_raw_hours": split.raw_train_hours, "available_raw_hours": split.available_raw_hours,
            "history_fraction": HISTORY_FRACTION, "available_raw_start_index": split.available_start,
            "available_raw_end_index": split.available_end, "eligible_target_count": len(split.eligible_indices),
            "fit_target_count": len(split.fit_indices), "calibration_target_count": len(split.calibration_indices),
            "audit_target_count": len(split.audit_indices), "first_fit_target_index": first_fit,
            "last_fit_target_index": int(split.fit_indices[-1]), "first_calibration_target_index": first_cal,
            "last_calibration_target_index": int(split.calibration_indices[-1]), "first_audit_target_index": first_audit,
            "last_audit_target_index": int(split.audit_indices[-1]), "target_scaler_fit_start_index": scaler.fit_start_index,
            "target_scaler_fit_end_index": scaler.fit_end_index, "target_scaler_fit_timestamp_end": scaler.fit_timestamp_end,
            "scaler_fit_uses_only_fit": True,
        }
        local_model, local_training = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, history_start_index=split.available_start)
        local_state = _cpu_state(local_model); local[target] = {"state_dict": local_state, "training": local_training, "calibration": _evaluate(local_model, grid, split.calibration_indices, scaler, device, split.available_start), "audit": _evaluate(local_model, grid, split.audit_indices, scaler, device, split.available_start)}; del local_model
        transfers[target] = {}
        for donor in names:
            if donor == target: continue
            adapted, training = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, initial_state=_target_initialized_from_donor(grid, donor_states[donor]), device=device, max_epochs=max_epochs, history_start_index=split.available_start)
            state = _cpu_state(adapted)
            transfers[target][donor] = {"state_dict": state, "training": training, "calibration": _evaluate(adapted, grid, split.calibration_indices, scaler, device, split.available_start), "audit": _evaluate(adapted, grid, split.audit_indices, scaler, device, split.available_start)}
            cb[target][donor] = benefit(local[target]["calibration"]["node_macro"]["mae"], transfers[target][donor]["calibration"]["node_macro"]["mae"])
            ab[target][donor] = benefit(local[target]["audit"]["node_macro"]["mae"], transfers[target][donor]["audit"]["node_macro"]["mae"])
            del adapted
        donor, value = select_donor(cb[target]); selected[target] = {"selected_donor": donor, "selected_calibration_benefit": value, "zero_transfer_fallback": donor is None}
        selected[target]["selected_audit_benefit"] = 0.0 if donor is None else ab[target][donor]
    # Stage B: reconstruct frozen CPU states only after every target's donor selection.
    for target in names:
        grid = grids[target]; split = scarce_split(grid); scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start); val_idx = np.arange(grid.splits["validation"].start_index, grid.splits["validation"].end_index, dtype=np.int64)
        local_eval = make_proxy(grid); local_eval.load_state_dict(local[target]["state_dict"]); local[target]["validation"] = _evaluate(local_eval, grid, val_idx, scaler, device); del local_eval
        for donor in transfers[target]:
            model = make_proxy(grid); model.load_state_dict(transfers[target][donor]["state_dict"]); transfers[target][donor]["validation"] = _evaluate(model, grid, val_idx, scaler, device); vb[target][donor] = benefit(local[target]["validation"]["node_macro"]["mae"], transfers[target][donor]["validation"]["node_macro"]["mae"]); del model
        donor = selected[target]["selected_donor"]; selected[target]["selected_validation_benefit"] = 0.0 if donor is None else vb[target][donor]; selected[target]["local_audit_metrics"] = local[target]["audit"]; selected[target]["selected_audit_metrics"] = local[target]["audit"] if donor is None else transfers[target][donor]["audit"]; selected[target]["local_validation_metrics"] = local[target]["validation"]; selected[target]["selected_validation_metrics"] = local[target]["validation"] if donor is None else transfers[target][donor]["validation"]
        selected[target]["audit_relative_improvement"] = _relative(selected[target]["local_audit_metrics"], selected[target]["selected_audit_metrics"]); selected[target]["validation_relative_improvement"] = _relative(selected[target]["local_validation_metrics"], selected[target]["selected_validation_metrics"])
        if donor is None: selected[target]["selected_audit_benefit"] = selected[target]["selected_validation_benefit"] = 0.0
        local[target].pop("state_dict", None)
        for donor_name in transfers[target]:
            transfers[target][donor_name].pop("state_dict", None)
    audit_local = _metrics_mean([selected[n]["local_audit_metrics"] for n in names]); audit_selected = _metrics_mean([selected[n]["selected_audit_metrics"] for n in names]); val_local = _metrics_mean([selected[n]["local_validation_metrics"] for n in names]); val_selected = _metrics_mean([selected[n]["selected_validation_metrics"] for n in names])
    positive = lambda values: [name for name in names if values.get(name, 0.0) > 0]
    pos_cal, pos_audit, pos_val = positive({n: selected[n]["selected_calibration_benefit"] for n in names}), positive({n: selected[n]["selected_audit_benefit"] for n in names}), positive({n: selected[n]["selected_validation_benefit"] for n in names})
    return {"experiment":"directed_cross_grid_transfer_benefit_audit", "history_fraction":HISTORY_FRACTION, "client_grid_names":list(names), "max_epochs":max_epochs, "patience":PATIENCE, "batch_size":BATCH_SIZE, "learning_rate":LEARNING_RATE, "seed":SEED, "donor_selection_metric":"target_train_internal_calibration_node_macro_mae", "validation_used_for_selection":False, "audit_used_for_selection":False, "validation_used":True, "test_evaluated":False, "donor_proxy_metadata_by_grid":donor_metadata, "scarcity_by_target":scarcity, "local_metrics":local, "transfer_metrics":transfers, "calibration_benefit":_matrix(cb,names), "audit_benefit":_matrix(ab,names), "validation_benefit":_matrix(vb,names), "selected_policy":selected, "four_grid_macro":{"train_audit":{"target_local_proxy":audit_local,"selected_transfer_policy":audit_selected,"relative_improvement":_relative(audit_local,audit_selected)}, "canonical_validation":{"target_local_proxy":val_local,"selected_transfer_policy":val_selected,"relative_improvement":_relative(val_local,val_selected)}}, "targets_with_positive_selected_calibration_benefit":len(pos_cal), "targets_with_positive_selected_audit_benefit":len(pos_audit), "targets_with_positive_selected_validation_benefit":len(pos_val), "targets_using_zero_transfer_fallback":sum(selected[n]["zero_transfer_fallback"] for n in names), "positive_selected_calibration_targets":pos_cal, "positive_selected_audit_targets":pos_audit, "positive_selected_validation_targets":pos_val, "zero_transfer_fallback_targets":[n for n in names if selected[n]["zero_transfer_fallback"]]}


def render_markdown(report):
    names = report["client_grid_names"]
    lines = ["# Directed Cross-Grid Transfer Benefit Audit (25% History)", "", "Validation-only external diagnostic. Canonical test split is never evaluated.", "Donor selection uses train-internal calibration only.", "", "## Scarcity Metadata", "", "| Target | Raw train | Available | Eligible | Fit | Calibration | Audit | Scaler start | Scaler end |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        item = report["scarcity_by_target"][name]
        lines.append(f"| {name} | {item['canonical_train_raw_hours']} | {item['available_raw_hours']} | {item['eligible_target_count']} | {item['fit_target_count']} | {item['calibration_target_count']} | {item['audit_target_count']} | {item['target_scaler_fit_start_index']} | {item['target_scaler_fit_end_index']} |")
    for title, key in (("Calibration Benefit", "calibration_benefit"), ("Audit Benefit", "audit_benefit"), ("Validation Benefit", "validation_benefit")):
        lines.extend(["", f"## Directed {title} Matrix", "", "| Target \\ Donor | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)])
        for target in names:
            lines.append("| " + target + " | " + " | ".join("-" if report[key][target][donor] is None else f"{report[key][target][donor]:.6g}" for donor in names) + " |")
    lines.extend(["", "## Selected Donors", "", "| Target | Donor | Calibration benefit | Audit benefit | Validation benefit | Fallback |", "|---|---|---:|---:|---:|---|"])
    for name in names:
        item = report["selected_policy"][name]
        lines.append(f"| {name} | {item['selected_donor'] or 'none'} | {item['selected_calibration_benefit']:.6g} | {item['selected_audit_benefit']:.6g} | {item['selected_validation_benefit']:.6g} | {item['zero_transfer_fallback']} |")
    for title, key in (("Train Audit", "train_audit"), ("Canonical Validation", "canonical_validation")):
        macro = report["four_grid_macro"][key]
        lines.extend(["", f"## {title} Macro Summary", "", "| Policy | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Grid MAE | Grid RMSE | Grid WAPE (%) | Grid sMAPE (%) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for policy, values in macro.items():
            if policy == "relative_improvement": continue
            node, grid = values["node_macro"], values["grid_aggregate"]
            lines.append(f"| {policy} | {node['mae']:.6g} | {node['rmse']:.6g} | {node['wape_pct']:.6g} | {node['smape_pct']:.6g} | {grid['mae']:.6g} | {grid['rmse']:.6g} | {grid['wape_pct']:.6g} | {grid['smape_pct']:.6g} |")
        lines.append(f"Relative node-MAE improvement: {macro['relative_improvement']:.6g}")
    return "\n".join(lines)
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root',type=Path,default=ROOT.parent/'pa_stfed_data_v2'/'raw'); p.add_argument('--mapping-json',type=Path,default=ROOT/'results'/'audits'/'v2_schema_mapping.json'); p.add_argument('--device',default='cpu'); p.add_argument('--max-epochs',type=int,default=MAX_EPOCHS); p.add_argument('--synthetic-smoke',action='store_true'); p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'audits'); a=p.parse_args(); report=run_audit(a.data_root,a.mapping_json,a.device,a.synthetic_smoke,a.max_epochs); out=a.output_dir/'federated_transfer_benefit_25pct.json'; md=a.output_dir/'federated_transfer_benefit_25pct.md'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x)+'\n',encoding='utf-8'); md.write_text(render_markdown(report)+'\n',encoding='utf-8'); print(f'wrote {out} and {md}')
if __name__=='__main__': main()
