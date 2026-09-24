"""Run the four-client topology-heterogeneous PUC-RSTAttn V2-CU framework."""
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
from code.federated.aggregation import ALGORITHMS, SHARING_MODES
from code.federated.trainer import CLIENT_GRID_NAMES, FederatedClient, train_federated
from code.models.puc_rstattn_v2_conditional_utility import (
    PUCRSTAttnV2ConditionalUtility,
    build_conditional_utility_graph,
)
from scripts.run_puc_rstattn_v2 import DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_SEED

DEFAULT_ROUNDS = 2
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_HIDDEN_SIZE = 32
FEATURE_MODE = "p_calendar"


def output_paths(mode: str, output_dir: Path, algorithm: str = "standard") -> tuple[Path, Path]:
    if algorithm in ("fedprox", "fedper"):
        output_dir = output_dir / "baselines"
        stem = f"fl_{algorithm}_dev"
        return output_dir / f"{stem}.json", output_dir / f"{stem}.md"
    if algorithm != "standard":
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if mode not in SHARING_MODES:
        raise ValueError(f"unsupported sharing mode {mode!r}")
    stem = f"fl_{mode}_dev"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def build_clients(data_root: Path, mapping_json: Path, seed: int = DEFAULT_SEED) -> list[FederatedClient]:
    """Load four independent grids, train scalers, datasets, and local graphs."""
    loader = LVGridLoader(data_root, mapping_json)
    if tuple(loader.available_grids()) != tuple(sorted(CLIENT_GRID_NAMES)):
        available = set(loader.available_grids())
        missing = sorted(set(CLIENT_GRID_NAMES) - available)
        if missing:
            raise ValueError(f"required FL client grids are missing: {missing}")
    import torch

    torch.manual_seed(seed)
    clients = []
    for grid_name in CLIENT_GRID_NAMES:
        grid = loader.load(grid_name)
        graph = build_conditional_utility_graph(grid)
        dataset = ForecastWindowDataset.from_grid(grid, FEATURE_MODE)
        scaler = dataset.fit_scaler()
        # Every client starts from the same trainable initialization; graph
        # buffers and client-local data remain supplied by this grid.
        torch.manual_seed(seed)
        clients.append(
            FederatedClient(
                grid_name=grid_name,
                model=PUCRSTAttnV2ConditionalUtility(
                    grid.num_nodes,
                    graph.edge_index,
                    graph.relation_features[:, 1:],
                    grid.load_bus_mask,
                    graph.relation_features[:, 0],
                ),
                train_dataset=dataset.split("train"),
                validation_dataset=dataset.split("validation"),
                scaler=scaler,
                load_bus_mask=np.asarray(grid.load_bus_mask, dtype=bool),
                graph_metadata=dict(graph.diagnostics),
            )
        )
    return clients


class _SyntheticScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


class _SyntheticDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray, split_name: str):
        self.x = x
        self.y = y
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {"x": self.x[index], "y": self.y[index]}


def build_synthetic_clients(seed: int = DEFAULT_SEED) -> list[FederatedClient]:
    """Construct small 2/3/4/5-node clients for an isolated smoke test."""
    from code.models.puc_rstattn_v2_conditional_utility import (
        UTILITY_AUDIT_SAMPLES,
        UTILITY_FIT_SAMPLES,
        UTILITY_SELECTION_SAMPLES,
    )

    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    clients = []
    for client_index, grid_name in enumerate(CLIENT_GRID_NAMES):
        nodes = client_index + 2
        load_mask = np.ones(nodes, dtype=bool)
        edge_count = max(nodes - 1, 0)
        edges = np.asarray(
            [np.arange(1, nodes, dtype=np.int64), np.zeros(edge_count, dtype=np.int64)],
            dtype=np.int64,
        )
        physical = np.zeros((edge_count, 3), dtype=np.float32)
        prior = np.linspace(0.2, 1.0, edge_count, dtype=np.float32)
        torch.manual_seed(seed)
        model = PUCRSTAttnV2ConditionalUtility(nodes, edges, physical, load_mask, prior)
        train_x = rng.normal(size=(2, 168, nodes, 6)).astype(np.float32)
        train_y = rng.normal(size=(2, nodes)).astype(np.float32)
        validation_x = rng.normal(size=(1, 168, nodes, 6)).astype(np.float32)
        validation_y = rng.normal(size=(1, nodes)).astype(np.float32)
        clients.append(
            FederatedClient(
                grid_name=grid_name,
                model=model,
                train_dataset=_SyntheticDataset(train_x, train_y, "train"),
                validation_dataset=_SyntheticDataset(validation_x, validation_y, "validation"),
                scaler=_SyntheticScaler(),
                load_bus_mask=load_mask,
                graph_metadata={
                    "utility_semantics": "strong_temporal_conditional_predictive_utility",
                    "utility_graph_fit_samples": UTILITY_FIT_SAMPLES,
                    "utility_graph_selection_samples": UTILITY_SELECTION_SAMPLES,
                    "utility_graph_audit_samples": UTILITY_AUDIT_SAMPLES,
                    "utility_graph_uses_audit": False,
                    "utility_graph_uses_validation": False,
                    "utility_graph_uses_test": False,
                    "synthetic_smoke": True,
                },
            )
        )
    return clients


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Federated PUC-RSTAttn V2-CU: {report['sharing_mode']}",
        "",
        "Validation-only metrics; test labels are not evaluated.",
        "",
        "| Round | Client | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Grid MAE | Grid RMSE | Grid WAPE (%) | Grid sMAPE (%) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for round_item in report["round_history"]:
        for client, metrics in round_item["validation"].items():
            node = metrics["node_macro"]
            grid = metrics["grid_aggregate"]
            lines.append(
                f"| {round_item['round']} | {client} | {node['mae']:.6g} | {node['rmse']:.6g} | {node['wape_pct']:.6g} | {node['smape_pct']:.6g} | {grid['mae']:.6g} | {grid['rmse']:.6g} | {grid['wape_pct']:.6g} | {grid['smape_pct']:.6g} |"
            )
        macro = round_item["four_grid_unweighted_macro"]
        lines.append(
            f"| {round_item['round']} | four-grid unweighted macro | {macro['node_macro']['mae']:.6g} | {macro['node_macro']['rmse']:.6g} | {macro['node_macro']['wape_pct']:.6g} | {macro['node_macro']['smape_pct']:.6g} | {macro['grid_aggregate']['mae']:.6g} | {macro['grid_aggregate']['rmse']:.6g} | {macro['grid_aggregate']['wape_pct']:.6g} | {macro['grid_aggregate']['smape_pct']:.6g} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="standard")
    parser.add_argument("--mode", choices=SHARING_MODES, default="fedavg_all")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "pa_stfed_data_v2" / "raw")
    parser.add_argument("--mapping-json", type=Path, default=REPO_ROOT / "results" / "audits" / "v2_schema_mapping.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "federated")
    parser.add_argument("--synthetic-smoke", action="store_true", help="run tiny synthetic clients without loading grid data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    clients = (
        build_synthetic_clients(args.seed)
        if args.synthetic_smoke
        else build_clients(args.data_root, args.mapping_json, args.seed)
    )
    report = train_federated(
        clients,
        args.mode,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate=DEFAULT_LEARNING_RATE,
        seed=args.seed,
        device=args.device,
        algorithm=args.algorithm,
    )
    if args.synthetic_smoke:
        report["synthetic_smoke"] = True
    output_json, output_md = output_paths(args.mode, args.output_dir, args.algorithm)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
