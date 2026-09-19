"""Train the centralized shared-node GRU smoke baseline on the 39-bus grid.

Manual run (requires PyTorch): python scripts/run_gru_smoke_39bus.py
Validation-only metrics are written; no test metric is calculated.
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
    SharedNodeGRU,
    evaluate_validation,
    best_epoch_summary,
    compare_gru_persistence,
    masked_scaled_mae,
    persistence_1h_predictions,
)

GRID_NAME = "39_bus_semi_urban_reference_grid"
FEATURE_MODE = "p_calendar"
HIDDEN_SIZE = 32
NUM_LAYERS = 1
SEED = 42
MAX_EPOCHS = 30
PATIENCE = 5


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


def _evaluate_model(model: Any, dataset: ForecastWindowDataset, scaler: Any, batch_size: int, device: Any) -> dict[str, Any]:
    import torch
    predictions, actual = [], []
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size):
            x, y = _to_batch(dataset, scaler, indices)
            pred_scaled = model(torch.from_numpy(x).to(device)).cpu().numpy()
            predictions.append(scaler.inverse_transform_target(pred_scaled))
            actual.append(scaler.inverse_transform_target(y))
    return evaluate_validation(np.concatenate(actual), np.concatenate(predictions), dataset.load_bus_mask)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training; install torch and rerun this script locally")
    import torch
    _set_seed(args.seed)
    grid = LVGridLoader(args.data_root, args.mapping_json).load(GRID_NAME)
    full_dataset = ForecastWindowDataset.from_grid(grid, FEATURE_MODE)
    scaler = full_dataset.fit_scaler()
    train_dataset = full_dataset.split("train")
    validation_dataset = full_dataset.split("validation")
    device = torch.device(args.device)
    model = SharedNodeGRU(input_size=6, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mask = torch.from_numpy(grid.load_bus_mask)
    best_validation, best_state, stale_epochs = float("inf"), None, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        for indices in _batch_indices(len(train_dataset), args.batch_size):
            x, y = _to_batch(train_dataset, scaler, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(x).to(device))
            loss = masked_scaled_mae(prediction, torch.from_numpy(y).to(device), mask)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = _evaluate_model(model, validation_dataset, scaler, args.batch_size, device)
        validation_mae = validation["node_macro"]["mae"]
        history.append({"epoch": epoch, "train_scaled_mae": float(np.mean(epoch_losses)), "validation": validation})
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
    persistence_prediction = persistence_1h_predictions(grid.p, validation_dataset.target_indices)
    persistence_actual = grid.p[validation_dataset.target_indices]
    persistence = evaluate_validation(persistence_actual, persistence_prediction, grid.load_bus_mask)
    summary = best_epoch_summary(history)
    return {
        "config": {"grid_name": GRID_NAME, "feature_mode": FEATURE_MODE, "history_length": 168,
                   "forecast_horizon": 1, "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
                   "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience,
                   "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                   "device": str(device), "topology_used": False},
        "validation": {"gru": validation, "persistence_1h": persistence},
        "comparison": compare_gru_persistence(validation, persistence),
        "summary": summary,
        "metric_units": {"mae": "P units", "rmse": "P units", "wape_pct": "%", "smape_pct": "%"},
        "training_history": history,
        "test_evaluated": False,
        "test_metric_computed": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Centralized GRU Smoke Baseline: 39-bus", "",
             "Validation-only metrics. Test labels and test metrics are intentionally not used.", "",
             "| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    pairs = (("SharedNodeGRU", report["validation"]["gru"]),
             ("Persistence 1h", report["validation"]["persistence_1h"]))
    for method, result in pairs:
        for scope in ("node_macro", "grid_aggregate"):
            values = result[scope]
            lines.append(f"| {method} | {scope} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
    lines.extend(["", "## Best epoch summary", "",
                  f"- best_epoch: {report['summary']['best_epoch']}",
                  f"- best_validation_node_macro_mae: {report['summary']['best_validation_node_macro_mae']:.6g}",
                  f"- best_train_scaled_mae: {report['summary']['best_train_scaled_mae']:.6g}",
                  f"- epochs_run: {report['summary']['epochs_run']}", "",
                  "## GRU vs persistence_1h factual comparison", "",
                  "| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |",
                  "| --- | --- | --- | --- | --- |"])
    for scope in ("node_macro", "grid_aggregate"):
        relation = report["comparison"][scope]
        lines.append(f"| {scope} | {relation['mae']} | {relation['rmse']} | {relation['wape_pct']} | {relation['smape_pct']} |")
    lines.extend(["", "test_evaluated: False"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "results" / "centralized" / "gru_smoke_39bus.json")
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / "results" / "centralized" / "gru_smoke_39bus.md")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_training(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
