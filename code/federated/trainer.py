"""Common local trainer for four topology-heterogeneous PUC-RSTAttn clients."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from code.models.centralized_gru import evaluate_validation, masked_scaled_mae
from scripts.run_puc_rstattn_v2 import _batch_indices, _to_batch

from .aggregation import SHARING_MODES, aggregate_named_parameters, normalize_sample_weights, sharing_groups
from .parameter_groups import parameter_groups

CLIENT_GRID_NAMES = (
    "39_bus_semi_urban_reference_grid",
    "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid",
    "80_bus_rural_reference_grid",
)
ANCHOR_LOSS_WEIGHT = 0.2
METRICS = ("mae", "rmse", "wape_pct", "smape_pct")
CLIENT_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass
class FederatedClient:
    grid_name: str
    model: Any
    train_dataset: Any
    validation_dataset: Any
    scaler: Any
    load_bus_mask: np.ndarray
    graph_metadata: dict[str, Any]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _named_parameters(model: Any) -> dict[str, Any]:
    return dict(model.named_parameters())


def _parameter_snapshot(model: Any, names: tuple[str, ...]) -> dict[str, Any]:
    parameters = _named_parameters(model)
    return {name: parameters[name].detach().cpu().clone() for name in names}


def _update_vectors(
    model: Any, before: Mapping[str, Any], groups: Mapping[str, tuple[str, ...]]
) -> dict[str, np.ndarray]:
    parameters = _named_parameters(model)
    result = {}
    for group_name, names in groups.items():
        pieces = [(parameters[name].detach().cpu() - before[name]).reshape(-1).numpy() for name in names]
        result[group_name] = np.concatenate(pieces).astype(np.float64, copy=False)
    return result


def cosine_diagnostics(
    vectors_by_client: Mapping[str, np.ndarray], client_order: tuple[str, ...]
) -> dict[str, Any]:
    """Compute six pairwise cosines; undefined zero-norm comparisons map to 0."""
    pair_values: dict[str, float] = {}
    zero_norm_pairs: list[str] = []
    values = []
    for left, right in CLIENT_PAIRS:
        left_name, right_name = client_order[left], client_order[right]
        a = np.asarray(vectors_by_client[left_name], dtype=np.float64).reshape(-1)
        b = np.asarray(vectors_by_client[right_name], dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("client update vectors must have identical flattened group shapes")
        norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        key = f"{left_name}|{right_name}"
        if norm_a == 0.0 or norm_b == 0.0:
            cosine = 0.0
            zero_norm_pairs.append(key)
        else:
            cosine = float(np.dot(a, b) / (norm_a * norm_b))
            cosine = float(np.clip(cosine, -1.0, 1.0))
        pair_values[key] = cosine
        values.append(cosine)
    array = np.asarray(values, dtype=float)
    return {
        "pairwise_cosine": pair_values,
        "mean_cosine": float(np.mean(array)),
        "median_cosine": float(np.median(array)),
        "minimum_cosine": float(np.min(array)),
        "maximum_cosine": float(np.max(array)),
        "negative_pair_fraction": float(np.mean(array < 0.0)),
        "zero_norm_pairs": zero_norm_pairs,
        "zero_norm_cosine": 0.0,
    }


def _evaluate_client(client: FederatedClient, batch_size: int, device: Any) -> dict[str, Any]:
    import torch

    model = client.model
    model.eval()
    predictions, actual = [], []
    with torch.no_grad():
        for indices in _batch_indices(len(client.validation_dataset), batch_size):
            x, y = _to_batch(client.validation_dataset, client.scaler, indices)
            final = model(torch.from_numpy(x).to(device))
            predictions.append(client.scaler.inverse_transform_target(final.cpu().numpy()))
            actual.append(client.scaler.inverse_transform_target(y))
    if not actual:
        raise ValueError(f"client {client.grid_name} has no validation samples")
    return evaluate_validation(np.concatenate(actual), np.concatenate(predictions), client.load_bus_mask)


def _unweighted_macro(client_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        scope: {
            metric: float(np.mean([client_metrics[name][scope][metric] for name in client_metrics]))
            for metric in METRICS
        }
        for scope in ("node_macro", "grid_aggregate")
    }


class FederatedTrainer:
    def __init__(
        self,
        clients: list[FederatedClient],
        mode: str,
        rounds: int = 2,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if mode not in SHARING_MODES:
            raise ValueError(f"unknown sharing mode: {mode}")
        if len(clients) != 4:
            raise ValueError("the topology-heterogeneous framework requires exactly four clients")
        if rounds < 1 or local_epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("rounds, local_epochs, batch_size, and learning_rate must be positive")
        self.clients = clients
        self.mode = mode
        self.rounds = int(rounds)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.device = device
        self.client_order = tuple(client.grid_name for client in clients)
        self.groups = {client.grid_name: parameter_groups(client.model) for client in clients}
        reference_groups = self.groups[self.client_order[0]]
        if any(self.groups[name] != reference_groups for name in self.client_order[1:]):
            raise ValueError("clients must share identical named parameter groups")
        self.parameter_group_names = reference_groups
        self.local_sample_counts = {client.grid_name: len(client.train_dataset) for client in clients}
        self.weights = normalize_sample_weights(self.local_sample_counts)
        import torch

        for client in clients:
            client.model.to(device)
        self.load_masks = {
            client.grid_name: torch.as_tensor(client.load_bus_mask, dtype=torch.bool, device=device)
            for client in clients
        }
        self.optimizers = {
            client.grid_name: torch.optim.Adam(
                client.model.parameters(), lr=self.learning_rate
            )
            for client in clients
        }

    def _train_local(self, client: FederatedClient) -> dict[str, float]:
        import torch

        model = client.model
        optimizer = self.optimizers[client.grid_name]
        final_losses, anchor_losses, total_losses = [], [], []
        for _epoch in range(self.local_epochs):
            model.train()
            for indices in _batch_indices(len(client.train_dataset), self.batch_size):
                x, y = _to_batch(client.train_dataset, client.scaler, indices)
                optimizer.zero_grad(set_to_none=True)
                final, _, details = model(torch.from_numpy(x).to(self.device), return_details=True)
                target = torch.from_numpy(y).to(self.device)
                load_mask = self.load_masks[client.grid_name]
                final_loss = masked_scaled_mae(final, target, load_mask)
                anchor_loss = masked_scaled_mae(details["y_gru"], target, load_mask)
                total_loss = final_loss + ANCHOR_LOSS_WEIGHT * anchor_loss
                total_loss.backward()
                optimizer.step()
                final_losses.append(float(final_loss.detach()))
                anchor_losses.append(float(anchor_loss.detach()))
                total_losses.append(float(total_loss.detach()))
        if not total_losses:
            raise ValueError(f"client {client.grid_name} has no training samples")
        return {
            "train_scaled_mae": float(np.mean(final_losses)),
            "train_anchor_scaled_mae": float(np.mean(anchor_losses)),
            "train_total_loss": float(np.mean(total_losses)),
        }

    def run(self) -> dict[str, Any]:
        _set_seed(self.seed)
        shared_groups = sharing_groups(self.mode)
        round_history = []
        for round_index in range(1, self.rounds + 1):
            snapshots = {
                client.grid_name: _parameter_snapshot(
                    client.model,
                    tuple(name for group in self.parameter_group_names.values() for name in group),
                )
                for client in self.clients
            }
            local_training = {
                client.grid_name: self._train_local(client) for client in self.clients
            }
            deltas = {
                client.grid_name: _update_vectors(
                    client.model, snapshots[client.grid_name], self.parameter_group_names
                )
                for client in self.clients
            }
            update_cosines = {
                group: cosine_diagnostics(
                    {client.grid_name: deltas[client.grid_name][group] for client in self.clients},
                    self.client_order,
                )
                for group in ("temporal", "spatial")
            }

            aggregated_parameter_names = []
            if shared_groups:
                for group in shared_groups:
                    aggregated_parameter_names.extend(self.parameter_group_names[group])
                aggregate_named_parameters(
                    {client.grid_name: client.model for client in self.clients},
                    aggregated_parameter_names,
                    self.weights,
                )

            client_validation = {
                client.grid_name: _evaluate_client(client, self.batch_size, self.device)
                for client in self.clients
            }
            macro = _unweighted_macro(client_validation)
            round_history.append(
                {
                    "round": round_index,
                    "local_training": local_training,
                    "aggregation_weights": dict(self.weights),
                    "aggregated_parameter_names": aggregated_parameter_names,
                    "validation": client_validation,
                    "four_grid_unweighted_macro": macro,
                    "primary_selection_metric": macro["node_macro"]["mae"],
                    "update_cosines": update_cosines,
                }
            )

        return {
            "experiment": "topology_heterogeneous_puc_rstattn_v2_conditional_utility_federated",
            "sharing_mode": self.mode,
            "client_grid_names": list(self.client_order),
            "parameter_group_names": {
                group: list(names) for group, names in self.parameter_group_names.items()
            },
            "shared_parameter_groups": list(shared_groups),
            "local_topology_buffers": [
                "load_bus_mask",
                "utility_edge_index",
                "physical_relation_features",
                "utility_prior",
                "selected_utility",
            ],
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "seed": self.seed,
            "feature_mode": "p_calendar",
            "history_length": 168,
            "forecast_horizon": 1,
            "hidden_size": 32,
            "batch_size": self.batch_size,
            "optimizer": {"name": "Adam", "learning_rate": self.learning_rate},
            "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
            "local_sample_counts": dict(self.local_sample_counts),
            "validation_used": True,
            "test_evaluated": False,
            "graph_metadata_by_client": {
                client.grid_name: {
                    key: value
                    for key, value in client.graph_metadata.items()
                    if key != "graph_audit_indices"
                }
                for client in self.clients
            },
            "round_history": round_history,
        }


def train_federated(
    clients: list[FederatedClient], mode: str, rounds: int = 2, local_epochs: int = 1,
    batch_size: int = 32, learning_rate: float = 1e-3, seed: int = 42, device: str = "cpu"
) -> dict[str, Any]:
    return FederatedTrainer(
        clients, mode, rounds, local_epochs, batch_size, learning_rate, seed, device
    ).run()
