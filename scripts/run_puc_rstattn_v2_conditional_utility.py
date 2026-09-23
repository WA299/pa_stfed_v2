"""Validation-only runner for PUC-RSTAttn V2-CU; never evaluates test labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data.forecast_dataset import ForecastWindowDataset
from code.data.lv_grid_loader import LVGridLoader
from code.models.centralized_gru import (
    TORCH_AVAILABLE,
    best_epoch_summary,
    evaluate_validation,
    masked_scaled_mae,
    persistence_1h_predictions,
)
from code.models.puc_rstattn import ANCHOR_LOSS_WEIGHT
from code.models.puc_rstattn_v2_conditional_utility import (
    CONDITIONAL_UTILITY_SEMANTICS,
    UTILITY_AUDIT_SAMPLES,
    UTILITY_FIT_SAMPLES,
    UTILITY_SELECTION_SAMPLES,
    PUCRSTAttnV2ConditionalUtility,
    build_conditional_utility_graph,
)
from scripts.run_puc_rstattn_v2 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    FEATURE_MODE,
    GRID_NAMES,
    GRID_SHORT_NAMES,
    HISTORY_LENGTH,
    INPUT_SIZE,
    HIDDEN_SIZE,
    _batch_indices,
    _set_seed,
    _to_batch,
    build_epoch_record,
)

FORECAST_HORIZON = 1
MODEL_LABEL = "PUC-RSTAttn_V2_conditional_utility"


def parse_grid_name(value: str) -> str:
    if value not in GRID_NAMES:
        raise ValueError(f"unsupported grid: {value}; choose one of {GRID_NAMES}")
    return value


def output_paths(grid_name: str, output_dir: Path) -> tuple[Path, Path]:
    parse_grid_name(grid_name)
    stem = f"puc_rstattn_v2_conditional_{GRID_SHORT_NAMES[grid_name]}_{FEATURE_MODE}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def report_metadata() -> dict[str, Any]:
    return {
        "model": MODEL_LABEL,
        "implementation": "frozen_puc_rstattn_v2_conditional_utility_graph",
        "feature_mode": FEATURE_MODE,
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "gru_anchor_preserved": True,
        "temporal_attention_role": "zero_initialized_residual_correction",
        "utility_attention_prior": True,
        "initial_spatial_attention": "normalized_positive_conditional_predictive_utility",
        "dynamic_attention_scale_initial": 0.0,
        "physical_bias_zero_initialized": True,
        "physical_relation_features": [
            "normalized_impedance_abs_distance",
            "normalized_hop_distance",
            "direct_physical_adjacency",
        ],
        "aggregate_preserving_spatial_residual": True,
        "utility_semantics": CONDITIONAL_UTILITY_SEMANTICS,
        "utility_graph_fit_samples": UTILITY_FIT_SAMPLES,
        "utility_graph_selection_samples": UTILITY_SELECTION_SAMPLES,
        "utility_graph_audit_samples": UTILITY_AUDIT_SAMPLES,
        "utility_graph_uses_audit": False,
        "utility_graph_uses_validation": False,
        "utility_graph_uses_test": False,
        "utility_top_k": 3,
        "utility_positive_only": True,
        "utility_direction": "source_to_target",
        "causal_claim": False,
        "test_evaluated": False,
        "validation_used_for_graph": False,
        "electrical_candidate_expansion": False,
        "distance_monotonic_penalty": False,
        "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
    }


def _evaluate(
    model: Any,
    dataset: ForecastWindowDataset,
    scaler: Any,
    batch_size: int,
    device: Any,
    load_mask: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
    import torch

    final_predictions, temporal_predictions, anchor_predictions, actual = [], [], [], []
    diagnostics: dict[str, list[float]] = {
        "mean_temporal_gate": [],
        "mean_spatial_gate_active": [],
        "mean_abs_temporal_correction": [],
        "mean_abs_centered_spatial_residual": [],
        "mean_spatial_attention_entropy": [],
        "aggregate_preservation_max_abs_error": [],
    }
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size):
            x, y = _to_batch(dataset, scaler, indices)
            final, temporal, details = model(torch.from_numpy(x).to(device), return_details=True)
            anchor = details["y_gru"]
            final_predictions.append(scaler.inverse_transform_target(final.cpu().numpy()))
            temporal_predictions.append(scaler.inverse_transform_target(temporal.cpu().numpy()))
            anchor_predictions.append(scaler.inverse_transform_target(anchor.cpu().numpy()))
            actual.append(scaler.inverse_transform_target(y))
            diagnostics["mean_temporal_gate"].append(float(details["temporal_gate"].mean()))
            active = model.load_bus_mask.to(device) & (details["reliability"][:, 2] > 0)
            active_gate = details["spatial_gate"][:, active]
            diagnostics["mean_spatial_gate_active"].append(float(active_gate.mean()) if active_gate.numel() else 0.0)
            diagnostics["mean_abs_temporal_correction"].append(float(details["temporal_correction"].abs().mean()))
            diagnostics["mean_abs_centered_spatial_residual"].append(float(details["centered_spatial_residual"].abs().mean()))
            attention = details["spatial_attention"]
            entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
            diagnostics["mean_spatial_attention_entropy"].append(float(entropy.mean()) if entropy.numel() else 0.0)
            diagnostics["aggregate_preservation_max_abs_error"].append(float((final[:, model.load_bus_mask.to(device)].sum(-1) - temporal[:, model.load_bus_mask.to(device)].sum(-1)).abs().max()))
    final_array, temporal_array, anchor_array, actual_array = (np.concatenate(values) for values in (final_predictions, temporal_predictions, anchor_predictions, actual))
    return (
        evaluate_validation(actual_array, final_array, load_mask),
        evaluate_validation(actual_array, temporal_array, load_mask),
        evaluate_validation(actual_array, anchor_array, load_mask),
        {key: float(np.mean(values)) for key, values in diagnostics.items()},
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")
    import torch

    parse_grid_name(args.grid)
    _set_seed(args.seed)
    grid = LVGridLoader(args.data_root, args.mapping_json).load(args.grid)
    utility_graph = build_conditional_utility_graph(grid)
    dataset = ForecastWindowDataset.from_grid(grid, FEATURE_MODE)
    scaler = dataset.fit_scaler()
    train_dataset, validation_dataset = dataset.split("train"), dataset.split("validation")
    device = torch.device(args.device)
    model = PUCRSTAttnV2ConditionalUtility(
        grid.num_nodes,
        utility_graph.edge_index,
        utility_graph.relation_features[:, 1:],
        grid.load_bus_mask,
        utility_graph.relation_features[:, 0],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    load_mask = torch.from_numpy(np.asarray(grid.load_bus_mask, dtype=bool)).to(device)
    history, best_state, best_validation, stale_epochs = [], None, float("inf"), 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        final_losses, anchor_losses, total_losses = [], [], []
        for indices in _batch_indices(len(train_dataset), args.batch_size):
            x, y = _to_batch(train_dataset, scaler, indices)
            optimizer.zero_grad(set_to_none=True)
            final, _, details = model(torch.from_numpy(x).to(device), return_details=True)
            target = torch.from_numpy(y).to(device)
            final_loss = masked_scaled_mae(final, target, load_mask)
            anchor_loss = masked_scaled_mae(details["y_gru"], target, load_mask)
            total_loss = final_loss + ANCHOR_LOSS_WEIGHT * anchor_loss
            total_loss.backward()
            optimizer.step()
            final_losses.append(float(final_loss.detach()))
            anchor_losses.append(float(anchor_loss.detach()))
            total_losses.append(float(total_loss.detach()))
        validation, temporal_residual, gru_anchor, _ = _evaluate(model, validation_dataset, scaler, args.batch_size, device, grid.load_bus_mask)
        history.append(build_epoch_record(epoch, final_losses, anchor_losses, total_losses, validation, temporal_residual, gru_anchor))
        current = float(validation["node_macro"]["mae"])
        if current < best_validation:
            best_validation = current
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final, temporal, anchor_metrics, diagnostics = _evaluate(model, validation_dataset, scaler, args.batch_size, device, grid.load_bus_mask)
    persistence = evaluate_validation(grid.p[validation_dataset.target_indices], persistence_1h_predictions(grid.p, validation_dataset.target_indices), grid.load_bus_mask)
    config = {**report_metadata(), "grid_name": args.grid, "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "device": str(device)}
    return {
        "config": config,
        "graph_diagnostics": utility_graph.diagnostics,
        "validation": {"final": final, "temporal_residual": temporal, "gru_anchor": anchor_metrics, "persistence_1h": persistence},
        "validation_diagnostics": diagnostics,
        "summary": best_epoch_summary(history),
        "metric_units": {"mae": "P units", "rmse": "P units", "wape_pct": "%", "smape_pct": "%"},
        "training_history": history,
        "test_evaluated": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {MODEL_LABEL}: {report['config']['grid_name']} / {FEATURE_MODE}", "", "Validation-only metrics; no test labels are accessed.", "", "| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for method in ("final", "temporal_residual", "gru_anchor", "persistence_1h"):
        for scope in ("node_macro", "grid_aggregate"):
            values = report["validation"][method][scope]
            lines.append(f"| {method} | {scope} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
    lines.extend(["", "## Validation diagnostics", "", json.dumps(report["validation_diagnostics"], indent=2, ensure_ascii=True), "", "test_evaluated: False"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=GRID_NAMES, default=GRID_NAMES[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "centralized")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_json, output_md = output_paths(args.grid, args.output_dir)
    report = run_training(args)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {output_json} and {output_md}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    main()
