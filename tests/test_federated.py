from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from code.federated.aggregation import (
    aggregate_named_parameters,
    normalize_sample_weights,
    sharing_groups,
)
from code.federated.parameter_groups import TOPOLOGY_BUFFER_NAMES, parameter_groups
from code.federated.trainer import (
    CLIENT_GRID_NAMES,
    FederatedClient,
    FederatedTrainer,
    _evaluate_client,
    _parameter_snapshot,
    _update_vectors,
    _unweighted_macro,
    cosine_diagnostics,
)
from code.models.puc_rstattn_v2_conditional_utility import PUCRSTAttnV2ConditionalUtility
from scripts.run_federated import build_parser, build_synthetic_clients

REPO_ROOT = Path(__file__).resolve().parents[1]


def _model(nodes: int, graph_seed: int = 0):
    edge_count = max(nodes - 1, 0)
    edges = np.asarray(
        [np.arange(1, nodes, dtype=np.int64), np.zeros(edge_count, dtype=np.int64)],
        dtype=np.int64,
    )
    physical = np.full((edge_count, 3), graph_seed + 0.25, dtype=np.float32)
    prior = np.linspace(0.2, 1.0, edge_count, dtype=np.float32)
    return PUCRSTAttnV2ConditionalUtility(
        nodes, edges, physical, np.ones(nodes, dtype=bool), prior
    )


class _IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=np.float32)

    def transform_target(self, y):
        return np.asarray(y, dtype=np.float32)

    def inverse_transform_target(self, y):
        return np.asarray(y, dtype=np.float32)


class _ArrayDataset:
    def __init__(self, x, y, split):
        self.x = x
        self.y = y
        self.split = split
        self.reads = 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        self.reads += 1
        return {"x": self.x[index], "y": self.y[index]}


def _clients(train_size=1, validation_size=1):
    clients = []
    for index, grid_name in enumerate(CLIENT_GRID_NAMES):
        nodes = index + 2
        model = _model(nodes, graph_seed=index)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float(index))
        train = _ArrayDataset(
            np.ones((train_size, 168, nodes, 6), dtype=np.float32),
            np.ones((train_size, nodes), dtype=np.float32),
            "train",
        )
        validation = _ArrayDataset(
            np.ones((validation_size, 168, nodes, 6), dtype=np.float32),
            np.ones((validation_size, nodes), dtype=np.float32),
            "validation",
        )
        clients.append(
            FederatedClient(
                grid_name,
                model,
                train,
                validation,
                _IdentityScaler(),
                np.ones(nodes, dtype=bool),
                {
                    "utility_semantics": "strong_temporal_conditional_predictive_utility",
                    "utility_graph_uses_validation": False,
                    "utility_graph_uses_test": False,
                    "client_graph_seed": index,
                },
            )
        )
    return clients


def test_parameter_groups_are_disjoint_complete_and_topology_buffers_excluded():
    model = _model(4)
    groups = parameter_groups(model)
    temporal, spatial = set(groups["temporal"]), set(groups["spatial"])
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    buffers = set(dict(model.named_buffers()))
    assert temporal.isdisjoint(spatial)
    assert temporal | spatial == trainable
    assert TOPOLOGY_BUFFER_NAMES <= buffers
    assert not (temporal | spatial) & buffers


def test_sample_count_weighted_aggregation_is_exact_and_broadcasts_only_selected():
    clients = _clients()
    models = {client.grid_name: client.model for client in clients[:2]}
    groups = parameter_groups(clients[0].model)
    temporal_name = groups["temporal"][0]
    spatial_name = groups["spatial"][0]
    weights = normalize_sample_weights({clients[0].grid_name: 2, clients[1].grid_name: 3})
    assert weights == {clients[0].grid_name: 0.4, clients[1].grid_name: 0.6}
    params = {name: dict(model.named_parameters()) for name, model in models.items()}
    params[clients[0].grid_name][temporal_name].data.fill_(2.0)
    params[clients[1].grid_name][temporal_name].data.fill_(7.0)
    params[clients[0].grid_name][spatial_name].data.fill_(10.0)
    params[clients[1].grid_name][spatial_name].data.fill_(20.0)
    before_local = params[clients[0].grid_name][spatial_name].detach().clone()
    before_buffers = {
        name: {key: value.detach().clone() for key, value in model.named_buffers()}
        for name, model in models.items()
    }

    aggregate_named_parameters(models, [temporal_name], weights)

    for model in models.values():
        torch.testing.assert_close(
            dict(model.named_parameters())[temporal_name], torch.full_like(params[clients[0].grid_name][temporal_name], 5.0)
        )
    torch.testing.assert_close(params[clients[0].grid_name][spatial_name], before_local)
    for client_name, model in models.items():
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before_buffers[client_name][name])


@pytest.mark.parametrize(
    "mode,shared_groups,local_groups",
    [
        ("local_only", (), ("temporal", "spatial")),
        ("fedavg_all", ("temporal", "spatial"), ()),
        ("temporal_shared_spatial_local", ("temporal",), ("spatial",)),
        ("temporal_local_spatial_shared", ("spatial",), ("temporal",)),
    ],
)
def test_sharing_modes_broadcast_selected_groups_and_retain_local_groups(
    monkeypatch, mode, shared_groups, local_groups
):
    clients = _clients()
    groups = parameter_groups(clients[0].model)
    initial = {
        client.grid_name: {
            name: parameter.detach().clone()
            for name, parameter in client.model.named_parameters()
        }
        for client in clients
    }
    buffers = {
        client.grid_name: {
            name: value.detach().clone() for name, value in client.model.named_buffers()
        }
        for client in clients
    }
    trainer = FederatedTrainer(clients, mode, rounds=1, local_epochs=1, batch_size=1)

    def local_update(client):
        offset = float(CLIENT_GRID_NAMES.index(client.grid_name) + 1)
        with torch.no_grad():
            for group_names in groups.values():
                for name in group_names:
                    dict(client.model.named_parameters())[name].add_(offset)
        return {"train_total_loss": offset}

    monkeypatch.setattr(trainer, "_train_local", local_update)
    monkeypatch.setattr(
        "code.federated.trainer._evaluate_client",
        lambda client, batch_size, device: {
            "node_macro": {key: float(CLIENT_GRID_NAMES.index(client.grid_name) + 1) for key in ("mae", "rmse", "wape_pct", "smape_pct")},
            "grid_aggregate": {key: float(CLIENT_GRID_NAMES.index(client.grid_name) + 2) for key in ("mae", "rmse", "wape_pct", "smape_pct")},
        },
    )
    report = trainer.run()
    if shared_groups:
        for group in shared_groups:
            for name in groups[group]:
                values = [dict(client.model.named_parameters())[name] for client in clients]
                assert all(torch.equal(values[0], value) for value in values[1:])
    for local_group in local_groups:
        name = groups[local_group][0]
        values = [dict(client.model.named_parameters())[name] for client in clients]
        assert not torch.equal(values[0], values[-1])
        assert torch.equal(values[0], initial[clients[0].grid_name][name] + 1.0)
    for client in clients:
        for name, value in client.model.named_buffers():
            torch.testing.assert_close(value, buffers[client.grid_name][name])
    assert report["round_history"][0]["aggregated_parameter_names"] == [
        name for group in sharing_groups(mode) for name in groups[group]
    ]
    assert report["round_history"][0]["aggregation_weights"] == {
        name: 0.25 for name in CLIENT_GRID_NAMES
    }


def test_different_node_and_graph_shapes_do_not_break_parameter_fedavg():
    clients = _clients()
    assert len({client.model.num_nodes for client in clients}) == 4
    assert len({tuple(client.model.utility_edge_index.shape) for client in clients}) > 1
    groups = parameter_groups(clients[0].model)
    models = {client.grid_name: client.model for client in clients}
    weights = normalize_sample_weights({name: 1 for name in models})
    aggregate_named_parameters(models, groups["temporal"] + groups["spatial"], weights)
    for group in groups.values():
        for name in group:
            params = [dict(client.model.named_parameters())[name] for client in clients]
            assert all(torch.equal(params[0], value) for value in params[1:])


def test_validation_metrics_are_per_client_and_macro_is_unweighted():
    clients = _clients(validation_size=2)
    metrics = {}
    for index, client in enumerate(clients):
        client.validation_dataset.y[:] = float(index + 1)
        metrics[client.grid_name] = _evaluate_client(client, 2, "cpu")
        assert client.validation_dataset.reads == 2
        assert metrics[client.grid_name]["sample_count"] == 2
    macro = _unweighted_macro(metrics)
    for scope in ("node_macro", "grid_aggregate"):
        for metric in ("mae", "rmse", "wape_pct", "smape_pct"):
            expected = np.mean([metrics[name][scope][metric] for name in CLIENT_GRID_NAMES])
            assert macro[scope][metric] == pytest.approx(expected)


def test_update_vectors_and_six_pairwise_cosines_and_zero_norm_policy():
    vectors = {
        CLIENT_GRID_NAMES[0]: np.asarray([1.0, 0.0]),
        CLIENT_GRID_NAMES[1]: np.asarray([0.0, 1.0]),
        CLIENT_GRID_NAMES[2]: np.asarray([-1.0, 0.0]),
        CLIENT_GRID_NAMES[3]: np.asarray([1.0, 1.0]),
    }
    result = cosine_diagnostics(vectors, CLIENT_GRID_NAMES)
    assert len(result["pairwise_cosine"]) == 6
    pair = result["pairwise_cosine"]
    assert pair[f"{CLIENT_GRID_NAMES[0]}|{CLIENT_GRID_NAMES[1]}"] == pytest.approx(0.0)
    assert pair[f"{CLIENT_GRID_NAMES[0]}|{CLIENT_GRID_NAMES[2]}"] == pytest.approx(-1.0)
    assert pair[f"{CLIENT_GRID_NAMES[0]}|{CLIENT_GRID_NAMES[3]}"] == pytest.approx(2**-0.5)
    assert result["minimum_cosine"] == pytest.approx(-1.0)
    assert result["negative_pair_fraction"] == pytest.approx(2 / 6)

    zero_result = cosine_diagnostics(
        {name: np.zeros(2) for name in CLIENT_GRID_NAMES}, CLIENT_GRID_NAMES
    )
    assert set(zero_result["pairwise_cosine"].values()) == {0.0}
    assert len(zero_result["zero_norm_pairs"]) == 6
    assert zero_result["mean_cosine"] == 0.0


def test_round_updates_are_flattened_separately_for_temporal_and_spatial_groups():
    clients = _clients()
    model = clients[0].model
    groups = parameter_groups(model)
    all_names = groups["temporal"] + groups["spatial"]
    before = _parameter_snapshot(model, all_names)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in groups["temporal"]:
            parameters[name].add_(1.0)
        for name in groups["spatial"]:
            parameters[name].add_(2.0)
    updates = _update_vectors(model, before, groups)
    assert set(updates) == {"temporal", "spatial"}
    assert updates["temporal"].size == sum(parameters[name].numel() for name in groups["temporal"])
    assert updates["spatial"].size == sum(parameters[name].numel() for name in groups["spatial"])
    assert np.all(updates["temporal"] == 1.0)
    assert np.all(updates["spatial"] == 2.0)


def test_two_round_synthetic_smoke_is_validation_only():
    clients = build_synthetic_clients(seed=42)
    report = FederatedTrainer(
        clients, "temporal_shared_spatial_local", rounds=2, local_epochs=1,
        batch_size=32, seed=42, device="cpu"
    ).run()
    assert len(report["round_history"]) == 2
    assert report["validation_used"] is True
    assert report["test_evaluated"] is False
    assert all("test" not in client.graph_metadata for client in clients)
    assert all(len(item["validation"]) == 4 for item in report["round_history"])
    assert all("temporal" in item["update_cosines"] and "spatial" in item["update_cosines"] for item in report["round_history"])


def test_runner_defaults_outputs_and_direct_help():
    args = build_parser().parse_args([])
    assert (args.rounds, args.local_epochs, args.seed, args.device) == (2, 1, 42, "cpu")
    from scripts.run_federated import output_paths

    assert str(output_paths("local_only", Path("results/federated"))[0]).endswith("fl_local_only_dev.json")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_federated.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--rounds" in result.stdout and "--local-epochs" in result.stdout
    assert "--batch-size" not in result.stdout and "--learning-rate" not in result.stdout
