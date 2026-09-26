"""Run the 25%-history BTD full-backbone bridge experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.audits.btd_full_backbone_bridge import (  # noqa: E402
    BATCH_SIZE,
    HISTORY,
    LEARNING_RATE,
    PATIENCE,
    SEED,
    build_scarce_target_graph,
    full_model,
    inject_temporal_state,
    relative_improvement,
    temporal_state_from_model,
    train_full_model,
)
from code.audits.federated_transfer_benefit import (  # noqa: E402
    _evaluate,
    fit_fit_only_scaler,
    make_proxy,
    scarce_split,
    train_proxy,
    transfer_temporal_parameters,
)
from code.data.lv_grid_loader import LVGridLoader  # noqa: E402
from scripts.run_federated import CLIENT_GRID_NAMES, build_synthetic_clients  # noqa: E402

MAX_EPOCHS = 50
VARIANTS = ("full_from_scratch", "local_proxy_warmstart_full", "selected_transfer_proxy_warmstart_full")


def _cpu_state(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _grid_from_client(client: Any) -> Any:
    n, nodes = 6200, client.model.num_nodes
    rng = np.random.default_rng(100 + nodes)
    dynamic = rng.normal(size=(n, nodes, 7)).astype(np.float32)
    edge_count = max(nodes - 1, 0)
    edge_index = np.asarray([np.arange(1, nodes), np.zeros(edge_count, dtype=np.int64)], dtype=np.int64)
    distance = np.zeros((nodes, nodes), dtype=np.float32)
    for index in range(nodes):
        distance[index, :] = np.abs(np.arange(nodes) - index)
    return SimpleNamespace(
        grid_name=client.grid_name, num_nodes=nodes, node_ids=np.arange(nodes),
        load_bus_mask=client.load_bus_mask, dynamic_features=dynamic,
        p=dynamic[:, :, 0].copy(), timestamps=pd.date_range("2020-01-01", periods=n, freq="h"), edge_index=edge_index,
        distance_matrices={"hop_distance": distance, "impedance_abs_distance": distance},
        splits={"train": SimpleNamespace(start_index=0, end_index=5964),
                "validation": SimpleNamespace(start_index=5964, end_index=6082),
                "test": SimpleNamespace(start_index=6082, end_index=n)},
    )


def _load_selection(audit_json: Path) -> dict[str, str | None]:
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    if report.get("validation_used_for_selection") is not False:
        raise ValueError("audit validation selection flag is unsafe")
    if report.get("audit_used_for_selection") is not False:
        raise ValueError("audit audit-selection flag is unsafe")
    if report.get("test_evaluated") is not False:
        raise ValueError("audit includes test evaluation")
    return {name: item.get("selected_donor") for name, item in report["selected_policy"].items()}


def _macro(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {scope: {metric: float(np.mean([item[scope][metric] for item in values])) for metric in ("mae", "rmse", "wape_pct", "smape_pct")} for scope in ("node_macro", "grid_aggregate")}


def run_bridge(data_root: Path, mapping_json: Path, audit_json: Path, device: str = "cpu", synthetic: bool = False, max_epochs: int = MAX_EPOCHS) -> dict[str, Any]:
    grids = ({client.grid_name: _grid_from_client(client) for client in build_synthetic_clients(SEED)}
             if synthetic else {name: LVGridLoader(data_root, mapping_json).load(name) for name in CLIENT_GRID_NAMES})
    selected_donors = _load_selection(audit_json)
    names = tuple(CLIENT_GRID_NAMES)
    donor_states: dict[str, dict[str, Any]] = {}
    donor_metadata: dict[str, Any] = {}
    for donor in sorted({value for value in selected_donors.values() if value is not None}):
        grid = grids[donor]
        train = grid.splits["train"]
        eligible = np.arange(train.start_index + HISTORY, train.end_index, dtype=np.int64)
        cut = int(0.8 * len(eligible)); donor_fit, donor_cal = eligible[:cut], eligible[cut:]
        scaler = fit_fit_only_scaler(grid, donor_fit)
        model, training = train_proxy(grid, donor_fit, donor_cal, scaler, device=device, max_epochs=max_epochs)
        donor_states[donor] = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
        donor_metadata[donor] = {"donor_fit_target_count": len(donor_fit), "donor_calibration_target_count": len(donor_cal), "scaler_fit_start_index": scaler.fit_start_index, "scaler_fit_end_index": scaler.fit_end_index, "scaler_fit_timestamp_end": scaler.fit_timestamp_end, "best_calibration_node_mae": training["best_calibration_node_mae"], "epochs_run": training["epochs_run"]}
        del model

    per_target: dict[str, Any] = {}
    for target in names:
        grid = grids[target]; split = scarce_split(grid); scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start); graph = build_scarce_target_graph(grid, split)
        local_proxy, local_proxy_training = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs, history_start_index=split.available_start)
        local_proxy_state = temporal_state_from_model(local_proxy)
        donor = selected_donors.get(target)
        if donor is None:
            transfer_proxy_state = local_proxy_state
            transfer_proxy_training = {"fallback": True}
        else:
            target_proxy = make_proxy(grid)
            transfer_temporal_parameters(target_proxy, make_proxy(grid))
            target_params = {name: value.detach().cpu().clone() for name, value in target_proxy.named_parameters()}
            for name in local_proxy_state:
                target_params[name] = donor_states[donor][name].clone()
            transfer_proxy, transfer_proxy_training = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, initial_state={**target_proxy.state_dict(), **target_params}, device=device, max_epochs=max_epochs, history_start_index=split.available_start)
            transfer_proxy_state = temporal_state_from_model(transfer_proxy)
            del target_proxy, transfer_proxy
        models = {}
        for variant in VARIANTS:
            model = full_model(grid, graph, device=device)
            if variant == "full_from_scratch":
                pass
            elif variant == "local_proxy_warmstart_full":
                inject_temporal_state(model, local_proxy_state)
            else:
                inject_temporal_state(model, transfer_proxy_state)
            models[variant] = model
        spatial_local = {name: value.detach().cpu().clone() for name, value in models["local_proxy_warmstart_full"].named_parameters() if name not in set(__import__("code.federated.parameter_groups", fromlist=["parameter_groups"]).parameter_groups(models["local_proxy_warmstart_full"])["temporal"])}
        spatial_transfer = {name: value.detach().cpu().clone() for name, value in models["selected_transfer_proxy_warmstart_full"].named_parameters() if name not in set(__import__("code.federated.parameter_groups", fromlist=["parameter_groups"]).parameter_groups(models["selected_transfer_proxy_warmstart_full"])["temporal"])}
        if not all(np.array_equal(spatial_local[name].numpy(), spatial_transfer[name].numpy()) for name in spatial_local):
            raise AssertionError("local and transfer warm starts do not share identical spatial initialization")
        variant_states, variant_training = {}, {}
        for variant, model in models.items():
            state, training = train_full_model(model, grid, split.fit_indices, split.calibration_indices, scaler, device=device, max_epochs=max_epochs)
            variant_states[variant], variant_training[variant] = state, training; del model
        validation_indices = np.arange(grid.splits["validation"].start_index, grid.splits["validation"].end_index, dtype=np.int64)
        evaluations = {}
        for variant in VARIANTS:
            eval_model = full_model(grid, graph, device=device); eval_model.load_state_dict(variant_states[variant]); evaluations[variant] = {"audit": _evaluate(eval_model, grid, split.audit_indices, scaler, device, split.available_start), "validation": _evaluate(eval_model, grid, validation_indices, scaler, device)}; del eval_model
        per_target[target] = {"selected_donor": donor, "graph_metadata": graph.diagnostics, "local_proxy_training": local_proxy_training, "selected_transfer_proxy_training": transfer_proxy_training, "variants": evaluations, "transfer_vs_local_warmstart_relative_improvement": {"audit": relative_improvement(evaluations["local_proxy_warmstart_full"]["audit"], evaluations["selected_transfer_proxy_warmstart_full"]["audit"]), "validation": relative_improvement(evaluations["local_proxy_warmstart_full"]["validation"], evaluations["selected_transfer_proxy_warmstart_full"]["validation"])}}
        del local_proxy
    macro = {}
    for split_name in ("audit", "validation"):
        macro[split_name] = {variant: _macro([per_target[name]["variants"][variant][split_name] for name in names]) for variant in VARIANTS}
        macro[split_name]["transfer_vs_local_relative_improvement"] = relative_improvement(macro[split_name]["local_proxy_warmstart_full"], macro[split_name]["selected_transfer_proxy_warmstart_full"])
    audit_improved = [name for name in names if per_target[name]["transfer_vs_local_warmstart_relative_improvement"]["audit"] > 0]
    validation_improved = [name for name in names if per_target[name]["transfer_vs_local_warmstart_relative_improvement"]["validation"] > 0]
    return {"experiment": "btd_full_backbone_bridge_25pct", "client_grid_names": list(names), "selected_donors": selected_donors, "per_target": per_target, "donor_proxy_metadata_by_grid": donor_metadata, "four_grid_macro": macro, "targets_improved_on_audit": len(audit_improved), "targets_improved_on_validation": len(validation_improved), "audit_improved_targets": audit_improved, "validation_improved_targets": validation_improved, "validation_used_for_selection": False, "audit_used_for_selection": False, "test_evaluated": False, "supportive_criteria": {"audit_macro_transfer_beats_local": macro["audit"]["transfer_vs_local_relative_improvement"] > 0, "validation_macro_transfer_beats_local": macro["validation"]["transfer_vs_local_relative_improvement"] > 0, "at_least_three_audit_targets_improved": len(audit_improved) >= 3}}


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# BTD Full-Backbone Bridge: 25% History", "", "Validation-only diagnostic. Canonical test data is never evaluated.", "Donors are read from the completed calibration-selected transfer audit."]
    lines += ["", "## Selected Donors", "", "| Target | Selected donor |", "|---|---|"] + [f"| {name} | {report['selected_donors'].get(name) or 'none'} |" for name in report["client_grid_names"]]
    for split_name, title in (("audit", "Train Audit"), ("validation", "Canonical Validation")):
        lines += ["", f"## A/B/C {title}", "", "| Target | From scratch MAE | Local warmstart MAE | Transfer warmstart MAE | B vs C improvement |", "|---|---:|---:|---:|---:|"]
        for name in report["client_grid_names"]:
            variants = report["per_target"][name]["variants"]
            lines.append(f"| {name} | {variants['full_from_scratch'][split_name]['node_macro']['mae']:.6g} | {variants['local_proxy_warmstart_full'][split_name]['node_macro']['mae']:.6g} | {variants['selected_transfer_proxy_warmstart_full'][split_name]['node_macro']['mae']:.6g} | {report['per_target'][name]['transfer_vs_local_warmstart_relative_improvement'][split_name]:.6g} |")
    for split_name, title in (("audit", "Train Audit"), ("validation", "Canonical Validation")):
        lines += ["", f"## Four-grid Macro: {title}", "", "| Variant | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Grid MAE | Grid RMSE | Grid WAPE (%) | Grid sMAPE (%) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            values = report["four_grid_macro"][split_name][variant]; node, grid = values["node_macro"], values["grid_aggregate"]; lines.append(f"| {variant} | {node['mae']:.6g} | {node['rmse']:.6g} | {node['wape_pct']:.6g} | {node['smape_pct']:.6g} | {grid['mae']:.6g} | {grid['rmse']:.6g} | {grid['wape_pct']:.6g} | {grid['smape_pct']:.6g} |")
        lines.append(f"B vs C relative improvement: {report['four_grid_macro'][split_name]['transfer_vs_local_relative_improvement']:.6g}")
    lines += ["", f"Targets improved on audit: {report['targets_improved_on_audit']}", f"Targets improved on validation: {report['targets_improved_on_validation']}", "validation_used_for_selection: false", "audit_used_for_selection: false", "test_evaluated: false"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--audit-json", type=Path, default=ROOT / "results" / "audits" / "federated_transfer_benefit_25pct.json")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS); parser.add_argument("--synthetic-smoke", action="store_true"); parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "audits")
    args = parser.parse_args(); report = run_bridge(args.data_root, args.mapping_json, args.audit_json, args.device, args.synthetic_smoke, args.max_epochs)
    output = args.output_dir / "btd_full_backbone_bridge_25pct.json"; markdown = args.output_dir / "btd_full_backbone_bridge_25pct.md"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n", encoding="utf-8"); markdown.write_text(render_markdown(report) + "\n", encoding="utf-8"); print(f"wrote {output} and {markdown}")


if __name__ == "__main__": main()
