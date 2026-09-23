from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pandas as pd

from code.audits.strong_temporal_conditional_utility_audit import (
    PAIR_FEATURE_DIM,
    SELF_FEATURE_DIM,
    SOURCE_LAGS,
    SELF_HISTORY_LAGS,
    chronological_split,
    select_conditional_neighbors,
)
from code.models.puc_rstattn import build_utility_graph as frozen_build_utility_graph
from code.models.puc_rstattn_v2 import PUCRSTAttnV2
from code.models.puc_rstattn_v2_conditional_utility import (
    PUCRSTAttnV2ConditionalUtility,
    UTILITY_AUDIT_SAMPLES,
    UTILITY_FIT_SAMPLES,
    UTILITY_SELECTION_SAMPLES,
    build_conditional_utility_graph,
    conditional_graph_split,
)
from scripts.run_puc_rstattn_v2_conditional_utility import (
    MODEL_LABEL,
    build_parser,
    output_paths,
    report_metadata,
)


def _grid(periods=5964 + 168, nodes=4, train_end=None):
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    time = np.arange(periods, dtype=float)
    p = np.column_stack([time + node * 0.1 for node in range(nodes)])
    return SimpleNamespace(
        grid_name="synthetic",
        num_nodes=nodes,
        node_ids=np.asarray([str(i) for i in range(nodes)]),
        timestamps=timestamps,
        p=p,
        load_bus_mask=np.ones(nodes, dtype=bool),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]]),
        distance_matrices={
            "impedance_abs_distance": np.ones((nodes, nodes)),
            "hop_distance": np.ones((nodes, nodes)),
        },
        splits={"train": SimpleNamespace(start_index=0, end_index=train_end or periods)},
    )


def test_exact_graph_contract_and_features():
    indices = np.arange(168, 168 + 5964)
    fit, selection, audit = chronological_split(indices)
    assert tuple(map(len, (fit, selection, audit))) == (3578, 1193, 1193)
    assert SELF_FEATURE_DIM == 173
    assert len(SELF_HISTORY_LAGS) == 168
    assert SELF_HISTORY_LAGS == tuple(range(1, 169))
    assert SOURCE_LAGS == (1, 24, 168)
    assert PAIR_FEATURE_DIM == 176


def test_positive_top3_direction_and_tie_breaking():
    utilities = {(source, target): -1.0 for target in range(4) for source in range(4) if source != target}
    utilities.update({(1, 0): 0.2, (2, 0): 0.2, (3, 0): -0.1})
    selected = select_conditional_neighbors(utilities, range(4))
    assert [source for source, _ in selected[0]] == [1, 2]
    assert all(value > 0 for values in selected.values() for _, value in values)
    assert all(source != target for target, values in selected.items() for source, _ in values)


def test_conditional_builder_metadata_and_audit_is_not_used():
    grid = _grid()
    graph = build_conditional_utility_graph(grid)
    assert graph.diagnostics["utility_graph_fit_samples"] == UTILITY_FIT_SAMPLES == 3578
    assert graph.diagnostics["utility_graph_selection_samples"] == UTILITY_SELECTION_SAMPLES == 1193
    assert graph.diagnostics["utility_graph_audit_samples"] == UTILITY_AUDIT_SAMPLES == 1193
    assert graph.diagnostics["utility_graph_uses_audit"] is False
    assert graph.diagnostics["utility_direction"] == "source_to_target"
    assert graph.diagnostics["utility_positive_only"] is True
    assert all(source != target for source, target in graph.edge_index.T)
    assert all(float(value) > 0 for value in graph.relation_features[:, 0])
    before = graph.edge_index.copy()
    grid.p[3578 + 1193 + 168 :, :] += 1000000.0
    after = build_conditional_utility_graph(grid).edge_index
    assert np.array_equal(before, after)


def test_runner_contract_and_direct_help():
    args = build_parser().parse_args([])
    json_path, md_path = output_paths(args.grid, __import__("pathlib").Path("results/centralized"))
    assert str(json_path).endswith("puc_rstattn_v2_conditional_39bus_p_calendar.json")
    assert str(md_path).endswith("puc_rstattn_v2_conditional_39bus_p_calendar.md")
    metadata = report_metadata()
    assert metadata["model"] == MODEL_LABEL
    assert metadata["utility_semantics"] == "strong_temporal_conditional_predictive_utility"
    assert metadata["utility_graph_uses_validation"] is False
    assert metadata["utility_graph_uses_test"] is False
    result = subprocess.run([sys.executable, "scripts/run_puc_rstattn_v2_conditional_utility.py", "--help"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--grid" in result.stdout


def test_frozen_graph_builder_identity_is_untouched():
    from code.models import puc_rstattn

    assert frozen_build_utility_graph is puc_rstattn.build_utility_graph


def test_conditional_model_is_exact_frozen_architecture():
    import torch

    edges = np.asarray([[1, 2, 0, 2], [0, 0, 1, 1]], dtype=np.int64)
    physical = np.asarray(
        [[0.5, 1.0, 1.0], [1.5, 2.0, 0.0], [0.75, 1.0, 1.0], [1.25, 2.0, 0.0]],
        dtype=np.float32,
    )
    load_mask = np.asarray([True, True, True, False])
    utility_prior = np.asarray([0.7, 0.3, 0.4, 0.6], dtype=np.float32)

    torch.manual_seed(1234)
    frozen = PUCRSTAttnV2(4, edges, physical, load_mask, utility_prior)
    torch.manual_seed(1234)
    conditional = PUCRSTAttnV2ConditionalUtility(
        4, edges, physical, load_mask, utility_prior
    )

    frozen_state = frozen.state_dict()
    conditional_state = conditional.state_dict()
    assert frozen_state.keys() == conditional_state.keys()
    for key in frozen_state:
        assert frozen_state[key].shape == conditional_state[key].shape
        assert torch.equal(frozen_state[key], conditional_state[key]), key

    torch.manual_seed(4321)
    x = torch.randn(2, 168, 4, 6)
    frozen_output = frozen(x, return_details=True)
    conditional_output = conditional(x, return_details=True)
    assert isinstance(frozen_output, tuple) and isinstance(conditional_output, tuple)
    assert len(frozen_output) == len(conditional_output) == 3
    frozen_final, frozen_temporal, frozen_details = frozen_output
    conditional_final, conditional_temporal, conditional_details = conditional_output
    assert frozen_final.shape == conditional_final.shape == (2, 4)
    assert frozen_temporal.shape == conditional_temporal.shape == (2, 4)
    assert frozen_details.keys() == conditional_details.keys()
    for key in frozen_details:
        assert frozen_details[key].shape == conditional_details[key].shape
        torch.testing.assert_close(frozen_details[key], conditional_details[key])
    torch.testing.assert_close(frozen_final, conditional_final)
    torch.testing.assert_close(frozen_temporal, conditional_temporal)


def test_nonempty_spatial_graph_preserves_load_aggregate():
    import torch

    edges = np.asarray([[1, 2, 0, 2], [0, 0, 1, 2]], dtype=np.int64)
    physical = np.asarray(
        [[0.5, 1.0, 1.0], [1.5, 2.0, 0.0], [0.75, 1.0, 1.0], [1.25, 2.0, 0.0]],
        dtype=np.float32,
    )
    load_mask = np.asarray([True, True, True, False])
    utility_prior = np.asarray([0.7, 0.3, 0.4, 0.6], dtype=np.float32)
    model = PUCRSTAttnV2ConditionalUtility(
        4, edges, physical, load_mask, utility_prior
    )
    with torch.no_grad():
        model.spatial_correction[-1].weight.fill_(0.05)
        model.spatial_correction[-1].bias.fill_(0.2)

    final, temporal, details = model(
        torch.randn(3, 168, 4, 6), return_details=True
    )
    active = model.load_bus_mask & (details["reliability"][:, 2] > 0)
    assert edges.shape[1] > 0
    assert int(active.sum()) >= 1
    assert bool(details["raw_spatial_residual"][:, active].abs().max() > 0)
    torch.testing.assert_close(
        details["centered_spatial_residual"][:, active].sum(dim=1),
        torch.zeros(3),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        final[:, model.load_bus_mask].sum(dim=1),
        temporal[:, model.load_bus_mask].sum(dim=1),
        atol=1e-6,
        rtol=0.0,
    )


def test_validation_and_test_values_cannot_affect_conditional_graph():
    formal_train_end = 5964 + 168
    grid = _grid(periods=formal_train_end + 500, train_end=formal_train_end)
    before = build_conditional_utility_graph(grid)
    before_edges = before.edge_index.copy()
    before_neighbors = {
        target: list(values) for target, values in before.neighbors.items()
    }
    before_utilities = dict(before.utilities)
    before_selected_prior = before.relation_features[:, 0].copy()
    assert before_edges.shape[1] > 0
    assert np.all(before_selected_prior > 0.0)

    grid.p[formal_train_end:, :] = (
        np.arange(500, dtype=float)[:, None] * 1000000.0
        + np.arange(grid.num_nodes, dtype=float)[None, :] * 10000.0
    )
    after = build_conditional_utility_graph(grid)

    assert np.array_equal(before_edges, after.edge_index)
    assert before_neighbors == after.neighbors
    assert before_utilities.keys() == after.utilities.keys()
    for edge in before_utilities:
        assert before_utilities[edge] == after.utilities[edge]
    assert np.array_equal(before_selected_prior, after.relation_features[:, 0])
