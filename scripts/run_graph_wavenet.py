"""Validation-only fixed-topology STGCN baseline runner for all four LV grids.

This script is a manual training entry point.  It writes validation metrics and
never evaluates test labels.  The feature mode is intentionally fixed to
``p_calendar`` for the V2 9A physical-topology baseline.
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
from code.models.graph_wavenet_baseline import GraphWaveNetBaseline


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
RESIDUAL_CHANNELS = 32
DILATION_CHANNELS = 32
SKIP_CHANNELS = 64
END_CHANNELS = 128
KERNEL_SIZE = 2
BLOCKS = 4
LAYERS_PER_BLOCK = 2
ADAPTIVE_EMBEDDING_DIM = 10
HISTORY_LENGTH = 168
FORECAST_HORIZON = 1
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE = 8
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
REPORT_METRICS = ("mae", "rmse", "wape_pct", "smape_pct")


def parse_grid_name(value: str) -> str:
    if value not in GRID_NAMES:
        raise ValueError(f"unsupported grid: {value}; choose one of {GRID_NAMES}")
    return value


def output_paths(grid_name: str, output_dir: Path) -> tuple[Path, Path]:
    parse_grid_name(grid_name)
    stem = f"graph_wavenet_{GRID_SHORT_NAMES[grid_name]}_{FEATURE_MODE}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def _batch_indices(size: int, batch_size: int) -> list[np.ndarray]:
    return [
        np.arange(start, min(start + batch_size, size), dtype=np.int64)
        for start in range(0, size, batch_size)
    ]


def _to_batch(
    dataset: ForecastWindowDataset,
    scaler: Any,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    windows = [dataset[int(index)] for index in indices]
    x = np.stack([scaler.transform(item["x"]) for item in windows]).astype(np.float32)
    y = np.stack([scaler.transform_target(item["y"]) for item in windows]).astype(np.float32)
    return x, y


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _compare_model_persistence(model_result: dict[str, Any], persistence: dict[str, Any]) -> dict[str, dict[str, str]]:
    comparison: dict[str, dict[str, str]] = {}
    for scope in ("node_macro", "grid_aggregate"):
        comparison[scope] = {}
        for metric in REPORT_METRICS:
            model_value = float(model_result[scope][metric])
            persistence_value = float(persistence[scope][metric])
            if np.isclose(model_value, persistence_value, rtol=1e-12, atol=1e-12):
                relation = "equal"
            elif model_value < persistence_value:
                relation = "graph_wavenet_lower"
            else:
                relation = "graph_wavenet_higher"
            comparison[scope][metric] = relation
    return comparison


def _evaluate_model(
    model: Any,
    dataset: ForecastWindowDataset,
    scaler: Any,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    predictions, actual = [], []
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size):
            x, y = _to_batch(dataset, scaler, indices)
            prediction = model(torch.from_numpy(x).to(device)).cpu().numpy()
            predictions.append(scaler.inverse_transform_target(prediction))
            actual.append(scaler.inverse_transform_target(y))
    return evaluate_validation(
        np.concatenate(actual),
        np.concatenate(predictions),
        dataset.load_bus_mask,
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training; install torch and rerun this script locally")
    parse_grid_name(args.grid)
    import torch

    _set_seed(args.seed)
    grid = LVGridLoader(args.data_root, args.mapping_json).load(args.grid)
    full_dataset = ForecastWindowDataset.from_grid(grid, FEATURE_MODE)
    scaler = full_dataset.fit_scaler()
    train_dataset = full_dataset.split("train")
    validation_dataset = full_dataset.split("validation")
    device = torch.device(args.device)
    model = GraphWaveNetBaseline(
        edge_index=grid.edge_index,
        num_nodes=grid.num_nodes,
        input_size=INPUT_SIZE,
        residual_channels=RESIDUAL_CHANNELS, dilation_channels=DILATION_CHANNELS,
        skip_channels=SKIP_CHANNELS, end_channels=END_CHANNELS, kernel_size=KERNEL_SIZE,
        blocks=BLOCKS, layers_per_block=LAYERS_PER_BLOCK,
        adaptive_embedding_dim=ADAPTIVE_EMBEDDING_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mask = torch.from_numpy(grid.load_bus_mask)
    best_validation, best_state, stale_epochs = float("inf"), None, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        for indices in _batch_indices(len(train_dataset), args.batch_size):
            x, y = _to_batch(train_dataset, scaler, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_scaled_mae(
                model(torch.from_numpy(x).to(device)),
                torch.from_numpy(y).to(device),
                mask,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = _evaluate_model(model, validation_dataset, scaler, args.batch_size, device)
        history.append({"epoch": epoch, "train_scaled_mae": float(np.mean(epoch_losses)), "validation": validation})
        validation_mae = validation["node_macro"]["mae"]
        if validation_mae < best_validation:
            best_validation = validation_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    validation = _evaluate_model(model, validation_dataset, scaler, args.batch_size, device)
    persistence_actual = grid.p[validation_dataset.target_indices]
    persistence_prediction = persistence_1h_predictions(grid.p, validation_dataset.target_indices)
    persistence = evaluate_validation(persistence_actual, persistence_prediction, grid.load_bus_mask)
    return {
        "config": {
            "grid_name": args.grid,
            "feature_mode": FEATURE_MODE,
            "input_size": INPUT_SIZE,
            "history_length": HISTORY_LENGTH,
            "forecast_horizon": FORECAST_HORIZON,
            "implementation": "GraphWaveNet_core_reimplementation",
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "topology_used": True, "graph_type": "physical_plus_adaptive",
            "adaptive_graph_used": True, "adaptive_embedding_dim": ADAPTIVE_EMBEDDING_DIM,
            "residual_channels": RESIDUAL_CHANNELS, "dilation_channels": DILATION_CHANNELS,
            "skip_channels": SKIP_CHANNELS, "end_channels": END_CHANNELS,
            "blocks": BLOCKS, "layers_per_block": LAYERS_PER_BLOCK, "kernel_size": KERNEL_SIZE,
            "receptive_field": int(model.receptive_field),
            "electrical_distance_used": False,
            "electrical_edge_features_used": False,
        },
        "validation": {"graph_wavenet": validation, "persistence_1h": persistence},
        "comparison": _compare_model_persistence(validation, persistence),
        "summary": best_epoch_summary(history),
        "metric_units": {"mae": "P units", "rmse": "P units", "wape_pct": "%", "smape_pct": "%"},
        "training_history": history,
        "test_evaluated": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Fixed-topology STGCN baseline: {report['config']['grid_name']} / {FEATURE_MODE}",
        "",
        "Validation-only metrics. Test labels and test metrics are intentionally not used.",
        "",
        "| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method, result in (("Graph WaveNet", report["validation"]["graph_wavenet"]), ("Persistence 1h", report["validation"]["persistence_1h"])):
        for scope in ("node_macro", "grid_aggregate"):
            values = result[scope]
            lines.append(
                f"| {method} | {scope} | {values['mae']:.6g} | {values['rmse']:.6g} | "
                f"{values['wape_pct']:.6g} | {values['smape_pct']:.6g} |"
            )
    summary = report["summary"]
    lines.extend([
        "",
        "## Best epoch summary",
        "",
        f"- best_epoch: {summary['best_epoch']}",
        f"- best_validation_node_macro_mae: {summary['best_validation_node_macro_mae']:.6g}",
        f"- best_train_scaled_mae: {summary['best_train_scaled_mae']:.6g}",
        f"- epochs_run: {summary['epochs_run']}",
        "",
        "## STGCN vs persistence_1h factual comparison",
        "",
        "| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |",
        "| --- | --- | --- | --- | --- |",
    ])
    for scope in ("node_macro", "grid_aggregate"):
        relation = report["comparison"][scope]
        lines.append(f"| {scope} | {relation['mae']} | {relation['rmse']} | {relation['wape_pct']} | {relation['smape_pct']} |")
    lines.extend([
        "",
        f"topology_used: {report['config']['topology_used']}",
        f"graph_type: {report['config']['graph_type']}",
        f"receptive_field: {report['config']['receptive_field']}",
        "electrical_edge_features_used: False",
        "test_evaluated: False",
        "",
    ])
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
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
