"""Validation-only runner for the centralized PUC-RSTAttn model.

Graph construction uses only formal-train utility fit/selection samples.  This
script intentionally never evaluates the canonical test split.
"""
from __future__ import annotations

import argparse
import json
import random
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
from code.models.puc_rstattn import (
    ANCHOR_LOSS_WEIGHT,
    PUCRSTAttn,
    UTILITY_FIT_SAMPLES,
    UTILITY_SELECTION_SAMPLES,
    build_utility_graph,
)

GRID_NAMES = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
GRID_SHORT_NAMES = {
    "39_bus_semi_urban_reference_grid": "39bus",
    "50_bus_rural_reference_grid": "50bus",
    "56_bus_semi_urban_reference_grid": "56bus",
    "80_bus_rural_reference_grid": "80bus",
}
FEATURE_MODE = "p_calendar"
INPUT_SIZE = 6
HIDDEN_SIZE = 32
HISTORY_LENGTH = 168
FORECAST_HORIZON = 1
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE = 8
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3


def parse_grid_name(value: str) -> str:
    if value not in GRID_NAMES:
        raise ValueError(f"unsupported grid: {value}; choose one of {GRID_NAMES}")
    return value


def output_paths(grid_name: str, output_dir: Path) -> tuple[Path, Path]:
    parse_grid_name(grid_name)
    stem = f"puc_rstattn_{GRID_SHORT_NAMES[grid_name]}_{FEATURE_MODE}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def _batch_indices(size: int, batch_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + batch_size, size), dtype=np.int64) for start in range(0, size, batch_size)]


def _to_batch(dataset: ForecastWindowDataset, scaler: Any, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windows = [dataset[int(index)] for index in indices]
    x = np.stack([scaler.transform(item["x"]) for item in windows]).astype(np.float32)
    y = np.stack([scaler.transform_target(item["y"]) for item in windows]).astype(np.float32)
    return x, y


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def report_metadata() -> dict[str, Any]:
    return {
        "model": "PUC-RSTAttn",
        "implementation": "predictive_utility_calibrated_residual_st_attention",
        "feature_mode": FEATURE_MODE,
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "utility_graph_directed": True,
        "utility_top_k": 3,
        "utility_positive_only": True,
        "utility_fit_samples": UTILITY_FIT_SAMPLES,
        "utility_selection_samples": UTILITY_SELECTION_SAMPLES,
        "utility_metric": "relative_mae_reduction",
        "utility_semantics": "out_of_sample_incremental_predictive_utility",
        "physical_relation_features": ["positive_utility", "normalized_impedance_abs_distance", "normalized_hop_distance", "direct_physical_adjacency"],
        "electrical_candidate_expansion": False,
        "distance_monotonic_penalty": False,
        "residual_spatial_correction": True,
        "zero_initialized_correction": True,
        "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
        "causal_claim": False,
        "validation_used_for_graph": False,
        "test_evaluated": False,
    }


def build_epoch_record(
    epoch: int,
    primary_losses: list[float],
    anchor_losses: list[float],
    total_losses: list[float],
    validation: dict[str, Any],
    temporal_anchor: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable training-history schema consumed by best_epoch_summary."""
    if not primary_losses or not anchor_losses or not total_losses:
        raise ValueError("epoch loss lists must be non-empty")
    return {
        "epoch": int(epoch),
        "train_scaled_mae": float(np.mean(primary_losses)),
        "train_anchor_scaled_mae": float(np.mean(anchor_losses)),
        "train_total_loss": float(np.mean(total_losses)),
        "validation": validation,
        "temporal_anchor": temporal_anchor,
    }


def _evaluate(
    model: Any,
    dataset: ForecastWindowDataset,
    scaler: Any,
    batch_size: int,
    device: Any,
    load_mask: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    final_predictions, anchor_predictions, actual = [], [], []
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size):
            x, y = _to_batch(dataset, scaler, indices)
            final, anchor, _ = model(torch.from_numpy(x).to(device), return_details=True)
            final_predictions.append(scaler.inverse_transform_target(final.cpu().numpy()))
            anchor_predictions.append(scaler.inverse_transform_target(anchor.cpu().numpy()))
            actual.append(scaler.inverse_transform_target(y))
    actual_array = np.concatenate(actual)
    final_array = np.concatenate(final_predictions)
    anchor_array = np.concatenate(anchor_predictions)
    return (
        evaluate_validation(actual_array, final_array, load_mask),
        evaluate_validation(actual_array, anchor_array, load_mask),
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required; use the configured GPU environment")
    import torch

    parse_grid_name(args.grid)
    _set_seed(args.seed)
    grid = LVGridLoader(args.data_root, args.mapping_json).load(args.grid)
    utility_graph = build_utility_graph(grid)
    dataset = ForecastWindowDataset.from_grid(grid, FEATURE_MODE)
    scaler = dataset.fit_scaler()
    train_dataset = dataset.split("train")
    validation_dataset = dataset.split("validation")
    device = torch.device(args.device)
    model = PUCRSTAttn(
        num_nodes=grid.num_nodes,
        utility_edge_index=utility_graph.edge_index,
        relation_features=utility_graph.relation_features,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    load_mask = torch.from_numpy(np.asarray(grid.load_bus_mask, dtype=bool)).to(device)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        primary_losses: list[float] = []
        anchor_losses: list[float] = []
        total_losses: list[float] = []
        for indices in _batch_indices(len(train_dataset), args.batch_size):
            x, y = _to_batch(train_dataset, scaler, indices)
            optimizer.zero_grad(set_to_none=True)
            final, temporal, _ = model(torch.from_numpy(x).to(device), return_details=True)
            target = torch.from_numpy(y).to(device)
            final_loss = masked_scaled_mae(final, target, load_mask)
            anchor_loss = masked_scaled_mae(temporal, target, load_mask)
            total_loss = final_loss + ANCHOR_LOSS_WEIGHT * anchor_loss
            total_loss.backward()
            optimizer.step()
            primary_losses.append(float(final_loss.detach().cpu()))
            anchor_losses.append(float(anchor_loss.detach().cpu()))
            total_losses.append(float(total_loss.detach().cpu()))
        validation, temporal_validation = _evaluate(model, validation_dataset, scaler, args.batch_size, device, grid.load_bus_mask)
        history.append(build_epoch_record(epoch, primary_losses, anchor_losses, total_losses, validation, temporal_validation))
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
    validation, temporal_validation = _evaluate(model, validation_dataset, scaler, args.batch_size, device, grid.load_bus_mask)
    actual = grid.p[validation_dataset.target_indices]
    persistence = evaluate_validation(actual, persistence_1h_predictions(grid.p, validation_dataset.target_indices), grid.load_bus_mask)
    config = {
        **report_metadata(),
        "grid_name": args.grid,
        "feature_mode": FEATURE_MODE,
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "electrical_edge_features_used": False,
    }
    return {
        "config": config,
        "graph_diagnostics": utility_graph.diagnostics,
        "validation": {"final": validation, "temporal_anchor": temporal_validation, "persistence_1h": persistence},
        "summary": best_epoch_summary(history),
        "metric_units": {"mae": "P units", "rmse": "P units", "wape_pct": "%", "smape_pct": "%"},
        "training_history": history,
        "test_evaluated": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# PUC-RSTAttn: {report['config']['grid_name']} / {FEATURE_MODE}", "", "Validation-only metrics; no test labels are accessed.", "", "| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for method, result in (("PUC-RSTAttn final", report["validation"]["final"]), ("Temporal anchor", report["validation"]["temporal_anchor"]), ("Persistence 1h", report["validation"]["persistence_1h"])):
        for scope in ("node_macro", "grid_aggregate"):
            values = result[scope]
            lines.append(f"| {method} | {scope} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
    summary = report["summary"]
    lines.extend(["", "## Training summary", "", f"- best_epoch: {summary['best_epoch']}", f"- epochs_run: {summary['epochs_run']}", f"- best_validation_node_macro_mae: {summary['best_validation_node_macro_mae']:.6g}", "", "## Graph diagnostics", "", json.dumps(report["graph_diagnostics"], indent=2, ensure_ascii=True), "", "test_evaluated: False"])
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
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
