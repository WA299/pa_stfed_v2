"""Validation-only topology-free shared-GRU global attention runner."""

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
from code.models.global_attention_baseline import GlobalAttentionBaseline


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
NUM_LAYERS = 1
HISTORY_LENGTH = 168
FORECAST_HORIZON = 1
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE = 8
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
REPORT_METRICS = ("mae", "rmse", "wape_pct", "smape_pct")
ATTENTION_SOURCES = ("all_nodes", "load_buses")
ATTENTION_SCOPES = ("all_nodes", "physical_hop2", "electrical_load_knn3")


def parse_grid_name(value: str) -> str:
    if value not in GRID_NAMES:
        raise ValueError(f"unsupported grid: {value}; choose one of {GRID_NAMES}")
    return value


def validate_attention_options(attention_scope: str, attention_source: str) -> None:
    if attention_scope not in ATTENTION_SCOPES:
        raise ValueError(f"unsupported attention scope: {attention_scope}")
    if attention_source not in ATTENTION_SOURCES:
        raise ValueError(f"unsupported attention source: {attention_source}")
    if attention_scope in ("physical_hop2", "electrical_load_knn3") and attention_source == "load_buses":
        raise ValueError(f"{attention_scope} cannot be combined with load_buses attention source")


def build_physical_hop2_mask(hop_distance: Any) -> np.ndarray:
    hop = np.asarray(hop_distance)
    if hop.ndim != 2 or hop.shape[0] != hop.shape[1]:
        raise ValueError("hop_distance must be a square matrix")
    return np.asarray(hop <= 2, dtype=bool)


def build_electrical_load_knn_mask(
    impedance_distance: Any, load_bus_mask: Any, k: int = 3
) -> np.ndarray:
    """Build self + nearest load-bus candidate mask from canonical distances."""
    distance = np.asarray(impedance_distance, dtype=float)
    load = np.asarray(load_bus_mask, dtype=bool)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("impedance_abs_distance must be square")
    if load.ndim != 1 or len(load) != distance.shape[0] or not np.any(load):
        raise ValueError("load_bus_mask must align with distance matrix and select a bus")
    if k < 0:
        raise ValueError("k must be non-negative")
    n = distance.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    load_indices = np.flatnonzero(load)
    for i in range(n):
        mask[i, i] = True
        if not load[i]:
            continue
        others = [int(j) for j in load_indices if int(j) != i]
        others.sort(key=lambda j: (float(distance[i, j]), j))
        mask[i, others[:k]] = True
    return mask


def output_paths(
    grid_name: str,
    output_dir: Path,
    attention_source: str = "all_nodes",
    attention_scope: str = "all_nodes",
) -> tuple[Path, Path]:
    parse_grid_name(grid_name)
    validate_attention_options(attention_scope, attention_source)
    suffix = {"physical_hop2": "_hop2", "electrical_load_knn3": "_elecknn3"}.get(attention_scope, "")
    if attention_source == "load_buses":
        suffix += "_loadsrc"
    stem = f"global_attn_{GRID_SHORT_NAMES[grid_name]}_{FEATURE_MODE}{suffix}"
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


def _compare_attention_persistence(attention: dict[str, Any], persistence: dict[str, Any]) -> dict[str, dict[str, str]]:
    comparison: dict[str, dict[str, str]] = {}
    for scope in ("node_macro", "grid_aggregate"):
        comparison[scope] = {}
        for metric in REPORT_METRICS:
            attention_value = float(attention[scope][metric])
            persistence_value = float(persistence[scope][metric])
            if np.isclose(attention_value, persistence_value, rtol=1e-12, atol=1e-12):
                relation = "equal"
            elif attention_value < persistence_value:
                relation = "global_attention_lower"
            else:
                relation = "global_attention_higher"
            comparison[scope][metric] = relation
    return comparison


def _evaluate_model(
    model: Any,
    dataset: ForecastWindowDataset,
    scaler: Any,
    batch_size: int,
    device: Any,
    source_mask: Any = None,
    candidate_mask: Any = None,
) -> dict[str, Any]:
    import torch
    predictions, actual = [], []
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size):
            x, y = _to_batch(dataset, scaler, indices)
            predictions.append(
                scaler.inverse_transform_target(
                    model(
                        torch.from_numpy(x).to(device),
                        source_mask=source_mask,
                        candidate_mask=candidate_mask,
                    ).cpu().numpy()
                )
            )
            actual.append(scaler.inverse_transform_target(y))
    return evaluate_validation(np.concatenate(actual), np.concatenate(predictions), dataset.load_bus_mask)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training; install torch and rerun this script locally")
    parse_grid_name(args.grid)
    validate_attention_options(args.attention_scope, args.attention_source)
    import torch
    _set_seed(args.seed)
    grid = LVGridLoader(args.data_root, args.mapping_json).load(args.grid)
    full_dataset = ForecastWindowDataset.from_grid(grid, FEATURE_MODE)
    scaler = full_dataset.fit_scaler()
    train_dataset = full_dataset.split("train")
    validation_dataset = full_dataset.split("validation")
    device = torch.device(args.device)
    source_mask = (
        torch.from_numpy(grid.load_bus_mask).to(device)
        if args.attention_source == "load_buses"
        else None
    )
    candidate_mask = None
    if args.attention_scope == "physical_hop2":
        candidate_mask = torch.from_numpy(
            build_physical_hop2_mask(grid.distance_matrices["hop_distance"])
        ).to(device)
    elif args.attention_scope == "electrical_load_knn3":
        candidate_mask = torch.from_numpy(
            build_electrical_load_knn_mask(
                grid.distance_matrices["impedance_abs_distance"], grid.load_bus_mask, k=3
            )
        ).to(device)
    model = GlobalAttentionBaseline(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
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
                model(
                    torch.from_numpy(x).to(device),
                    source_mask=source_mask,
                    candidate_mask=candidate_mask,
                ),
                torch.from_numpy(y).to(device),
                mask,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = _evaluate_model(
            model,
            validation_dataset,
            scaler,
            args.batch_size,
            device,
            source_mask,
            candidate_mask,
        )
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
    validation = _evaluate_model(
        model,
        validation_dataset,
        scaler,
        args.batch_size,
        device,
        source_mask,
        candidate_mask,
    )
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
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "temporal_encoder": "shared_gru",
            "spatial_attention": "global_scaled_dot_product",
            "topology_used": args.attention_scope in ("physical_hop2", "electrical_load_knn3"),
            "attention_scope": args.attention_scope,
            "attention_source": "candidate_mask_defined" if args.attention_scope == "electrical_load_knn3" else ("all_nodes" if args.attention_source == "all_nodes" else "load_buses_only"),
            "electrical_distance_used": args.attention_scope == "electrical_load_knn3",
            "electrical_distance_role": "candidate_selection_only" if args.attention_scope == "electrical_load_knn3" else None,
            "electrical_bias_used": False if args.attention_scope == "electrical_load_knn3" else None,
            "knn_k": 3 if args.attention_scope == "electrical_load_knn3" else None,
            "electrical_edge_features_used": False,
        },
        "validation": {"global_attention": validation, "persistence_1h": persistence},
        "comparison": _compare_attention_persistence(validation, persistence),
        "summary": best_epoch_summary(history),
        "metric_units": {"mae": "P units", "rmse": "P units", "wape_pct": "%", "smape_pct": "%"},
        "training_history": history,
        "test_evaluated": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Shared GRU + global spatial attention: {report['config']['grid_name']} / {FEATURE_MODE}",
        "", "Validation-only metrics. Test labels and test metrics are intentionally not used.", "",
        "| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method, result in (("Global attention", report["validation"]["global_attention"]), ("Persistence 1h", report["validation"]["persistence_1h"])):
        for scope in ("node_macro", "grid_aggregate"):
            values = result[scope]
            lines.append(f"| {method} | {scope} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['wape_pct']:.6g} | {values['smape_pct']:.6g} |")
    summary = report["summary"]
    lines.extend([
        "", "## Best epoch summary", "", f"- best_epoch: {summary['best_epoch']}",
        f"- best_validation_node_macro_mae: {summary['best_validation_node_macro_mae']:.6g}",
        f"- best_train_scaled_mae: {summary['best_train_scaled_mae']:.6g}", f"- epochs_run: {summary['epochs_run']}",
        "", "## Global attention vs persistence_1h factual comparison", "",
        "| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |", "| --- | --- | --- | --- | --- |",
    ])
    for scope in ("node_macro", "grid_aggregate"):
        relation = report["comparison"][scope]
        lines.append(f"| {scope} | {relation['mae']} | {relation['rmse']} | {relation['wape_pct']} | {relation['smape_pct']} |")
    config = report["config"]
    lines.extend([
        "", f"temporal_encoder: {config['temporal_encoder']}", f"spatial_attention: {config['spatial_attention']}",
        f"topology_used: {config['topology_used']}", f"attention_scope: {config['attention_scope']}",
        f"attention_source: {config['attention_source']}",
        f"electrical_distance_used: {config.get('electrical_distance_used', False)}",
        f"electrical_distance_role: {config.get('electrical_distance_role')}",
        f"electrical_bias_used: {config.get('electrical_bias_used', False)}",
        f"knn_k: {config.get('knn_k')}",
        "electrical_edge_features_used: False", "test_evaluated: False", "",
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
    parser.add_argument("--attention-source", choices=ATTENTION_SOURCES, default="all_nodes")
    parser.add_argument("--attention-scope", choices=ATTENTION_SCOPES, default="all_nodes")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "centralized")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_json, output_md = output_paths(
        args.grid,
        args.output_dir,
        args.attention_source,
        args.attention_scope,
    )
    report = run_training(args)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
