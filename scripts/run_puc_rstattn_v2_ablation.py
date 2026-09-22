"""Train one formal PUC-RSTAttn V2 ablation variant on validation only."""
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

from code.models.centralized_gru import TORCH_AVAILABLE, best_epoch_summary, evaluate_validation, masked_scaled_mae, persistence_1h_predictions
from code.models.puc_rstattn import ANCHOR_LOSS_WEIGHT, UTILITY_FIT_SAMPLES, UTILITY_SELECTION_SAMPLES, build_utility_graph
from code.models.puc_rstattn_v2_ablation import VARIANT_SPECS, PUCRSTAttnV2Ablation
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
)
from code.data.forecast_dataset import ForecastWindowDataset
from code.data.lv_grid_loader import LVGridLoader

FORECAST_HORIZON = 1
VARIANTS = tuple(VARIANT_SPECS)


def parse_variant(value: str) -> str:
    if value not in VARIANTS:
        raise ValueError(f"unsupported variant: {value}; choose one of {VARIANTS}")
    return value


def output_paths(variant: str, grid_name: str, output_dir: Path) -> tuple[Path, Path]:
    parse_variant(variant)
    if grid_name not in GRID_NAMES:
        raise ValueError(f"unsupported grid: {grid_name}")
    stem = f"puc_rstattn_v2_{variant}_{GRID_SHORT_NAMES[grid_name]}_{FEATURE_MODE}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def report_metadata(variant: str) -> dict[str, Any]:
    spec = VARIANT_SPECS[parse_variant(variant)]
    return {
        "experiment": "puc_rstattn_v2_formal_ablation",
        "variant": variant,
        "model": "PUC-RSTAttn-V2-ablation",
        "independent_training": True,
        "seed": DEFAULT_SEED,
        "feature_mode": FEATURE_MODE,
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "utility_top_k": 3,
        "utility_positive_only": True,
        "utility_fit_samples": UTILITY_FIT_SAMPLES,
        "utility_selection_samples": UTILITY_SELECTION_SAMPLES,
        "temporal_residual_enabled": spec.temporal_residual_enabled,
        "spatial_residual_enabled": spec.spatial_residual_enabled,
        "utility_candidate_selection_enabled": spec.utility_candidate_selection_enabled,
        "utility_magnitude_prior_enabled": spec.utility_magnitude_prior_enabled,
        "dynamic_attention_enabled": spec.dynamic_attention_enabled,
        "physical_relation_bias_enabled": spec.physical_relation_bias_enabled,
        "aggregate_preserving_spatial_residual": spec.aggregate_preserving_spatial_residual,
        "validation_used_for_graph": False,
        "test_evaluated": False,
        "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
    }


def _evaluate_variant(model: Any, dataset: ForecastWindowDataset, scaler: Any, batch_size: int, device: Any, load_mask: Any) -> dict[str, Any]:
    import torch

    final_predictions, temporal_predictions, anchor_predictions, actual = [], [], [], []
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size):
            x, y = _to_batch(dataset, scaler, indices)
            final, temporal, details = model(torch.from_numpy(x).to(device), return_details=True)
            final_predictions.append(scaler.inverse_transform_target(final.cpu().numpy()))
            temporal_predictions.append(scaler.inverse_transform_target(temporal.cpu().numpy()))
            anchor_predictions.append(scaler.inverse_transform_target(details["y_gru"].cpu().numpy()))
            actual.append(scaler.inverse_transform_target(y))
    actual_array = np.concatenate(actual)
    return {
        "final": evaluate_validation(actual_array, np.concatenate(final_predictions), load_mask),
        "temporal_residual": evaluate_validation(actual_array, np.concatenate(temporal_predictions), load_mask),
        "gru_anchor": evaluate_validation(actual_array, np.concatenate(anchor_predictions), load_mask),
    }


def build_epoch_record(epoch: int, final_losses: list[float], anchor_losses: list[float], total_losses: list[float], validation: dict[str, Any]) -> dict[str, Any]:
    if not final_losses or not anchor_losses or not total_losses:
        raise ValueError("epoch loss lists must be non-empty")
    return {
        "epoch": int(epoch),
        "train_scaled_mae": float(np.mean(final_losses)),
        "train_anchor_scaled_mae": float(np.mean(anchor_losses)),
        "train_total_loss": float(np.mean(total_losses)),
        "validation": validation["final"],
        "temporal_residual": validation["temporal_residual"],
        "gru_anchor": validation["gru_anchor"],
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")
    import torch

    parse_variant(args.variant)
    expected = (DEFAULT_SEED, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE)
    actual = (args.seed, args.max_epochs, args.patience, args.batch_size, args.learning_rate)
    if actual != expected:
        raise ValueError("formal ablations require seed=42, max_epochs=50, patience=8, batch_size=32, learning_rate=1e-3")
    _set_seed(args.seed)
    grid = LVGridLoader(args.data_root, args.mapping_json).load(args.grid)
    utility_graph = build_utility_graph(grid)
    dataset = ForecastWindowDataset.from_grid(grid, FEATURE_MODE)
    scaler = dataset.fit_scaler()
    train_dataset, validation_dataset = dataset.split("train"), dataset.split("validation")
    device = torch.device(args.device)
    model = PUCRSTAttnV2Ablation(
        args.variant,
        grid.num_nodes,
        utility_graph.edge_index,
        utility_graph.relation_features,
        grid.load_bus_mask,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    load_mask = torch.from_numpy(np.asarray(grid.load_bus_mask, dtype=bool)).to(device)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_validation = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        final_losses, anchor_losses, total_losses = [], [], []
        for indices in _batch_indices(len(train_dataset), args.batch_size):
            x, y = _to_batch(train_dataset, scaler, indices)
            optimizer.zero_grad(set_to_none=True)
            final, _, details = model(torch.from_numpy(x).to(device), return_details=True)
            target = torch.from_numpy(y).to(device)
            anchor_loss = masked_scaled_mae(details["y_gru"], target, load_mask)
            if args.variant == "gru_anchor_only":
                final_loss = anchor_loss
                total_loss = final_loss
            else:
                final_loss = masked_scaled_mae(final, target, load_mask)
                total_loss = final_loss + ANCHOR_LOSS_WEIGHT * anchor_loss
            total_loss.backward()
            optimizer.step()
            final_losses.append(float(final_loss.detach()))
            anchor_losses.append(float(anchor_loss.detach()))
            total_losses.append(float(total_loss.detach()))
        validation = _evaluate_variant(model, validation_dataset, scaler, args.batch_size, device, grid.load_bus_mask)
        history.append(build_epoch_record(epoch, final_losses, anchor_losses, total_losses, validation))
        current = float(validation["final"]["node_macro"]["mae"])
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
    validation = _evaluate_variant(model, validation_dataset, scaler, args.batch_size, device, grid.load_bus_mask)
    actual = grid.p[validation_dataset.target_indices]
    persistence = evaluate_validation(actual, persistence_1h_predictions(grid.p, validation_dataset.target_indices), grid.load_bus_mask)
    config = {
        **report_metadata(args.variant),
        "grid_name": args.grid,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
    }
    return {
        "config": config,
        "graph_diagnostics": utility_graph.diagnostics,
        "validation": {**validation, "persistence_1h": persistence},
        "summary": best_epoch_summary(history),
        "training_history": history,
        "test_evaluated": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# PUC-RSTAttn V2 ablation: {report['config']['variant']} / {report['config']['grid_name']}", "", "Validation-only metrics; no test labels are accessed.", "", "| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |", "|---|---:|---:|---:|---:|"]
    for scope in ("node_macro", "grid_aggregate"):
        values = report["validation"]["final"][scope]
        lines.append(f"| {scope} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
    lines.extend(["", f"best_epoch: {report['summary']['best_epoch']}", f"epochs_run: {report['summary']['epochs_run']}", "test_evaluated: False"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--grid", choices=GRID_NAMES, default=GRID_NAMES[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1].parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "ablations")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_json, output_md = output_paths(args.variant, args.grid, args.output_dir)
    report = run_training(args)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
