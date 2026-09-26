from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from code.audits.btd_full_backbone_bridge import (
    build_scarce_target_graph,
    full_model,
    inject_temporal_state,
    temporal_state_from_model,
)
from code.audits.federated_transfer_benefit import make_proxy, scarce_split
from code.federated.parameter_groups import parameter_groups
from scripts.run_btd_full_backbone_bridge import _load_selection, render_markdown, run_bridge


def _grid(nodes=4):
    n = 6200
    rng = np.random.default_rng(nodes)
    dynamic = rng.normal(size=(n, nodes, 7)).astype(np.float32)
    edge_count = nodes - 1
    edge = np.asarray([np.arange(1, nodes), np.zeros(edge_count, dtype=np.int64)])
    distance = np.abs(np.arange(nodes)[:, None] - np.arange(nodes)[None, :]).astype(np.float32)
    return SimpleNamespace(
        grid_name=f"synthetic_{nodes}", num_nodes=nodes, node_ids=np.arange(nodes),
        load_bus_mask=np.ones(nodes, dtype=bool), dynamic_features=dynamic,
        p=dynamic[:, :, 0].copy(), timestamps=pd.date_range("2020-01-01", periods=n, freq="h"), edge_index=edge,
        distance_matrices={"hop_distance": distance, "impedance_abs_distance": distance},
        splits={"train": SimpleNamespace(start_index=0, end_index=5964), "validation": SimpleNamespace(start_index=5964, end_index=6082), "test": SimpleNamespace(start_index=6082, end_index=n)},
    )


def test_scarce_graph_fit_selection_uses_target_fit_only():
    grid = _grid(); split = scarce_split(grid); graph = build_scarce_target_graph(grid, split)
    assert graph.diagnostics["graph_fit_target_count"] == (2 * len(split.fit_indices)) // 3
    assert graph.diagnostics["graph_selection_target_count"] == len(split.fit_indices) - graph.diagnostics["graph_fit_target_count"]
    assert graph.diagnostics["graph_uses_target_fit_only"] is True
    assert graph.diagnostics["graph_uses_calibration"] is False
    assert graph.diagnostics["graph_uses_audit"] is False
    assert graph.diagnostics["graph_uses_validation"] is False
    assert graph.diagnostics["graph_uses_test"] is False
    assert graph.diagnostics["graph_fit_last_target_index"] < graph.diagnostics["graph_selection_first_target_index"]
    assert graph.diagnostics["graph_selection_last_target_index"] < split.calibration_indices[0]
    assert graph.diagnostics["graph_fit_first_target_index"] - 168 >= split.available_start


def test_full_models_share_spatial_initialization_and_only_temporal_injection():
    grid = _grid(5); split = scarce_split(grid); graph = build_scarce_target_graph(grid, split)
    local = make_proxy(grid); donor = make_proxy(grid)
    with torch.no_grad():
        for name in parameter_groups(donor)["temporal"]:
            dict(donor.named_parameters())[name].add_(1.5)
    a, b = full_model(grid, graph), full_model(grid, graph)
    groups = parameter_groups(a); before = {name: value.detach().clone() for name, value in b.named_parameters()}
    inject_temporal_state(b, temporal_state_from_model(donor))
    assert all(torch.equal(dict(a.named_parameters())[name], before[name]) for name in groups["spatial"])
    assert all(torch.equal(dict(b.named_parameters())[name], dict(donor.named_parameters())[name]) for name in groups["temporal"])
    assert all(torch.equal(dict(a.named_buffers())[name], dict(b.named_buffers())[name]) for name in dict(a.named_buffers()))


def test_audit_mapping_requires_selection_isolation():
    path = Path(".test_btd_audit_mapping.json")
    try:
        path.write_text('{"validation_used_for_selection": false, "audit_used_for_selection": false, "test_evaluated": false, "selected_policy": {"a": {"selected_donor": "b"}}}', encoding="utf-8")
        assert _load_selection(path) == {"a": "b"}
        path.write_text('{"validation_used_for_selection": true, "audit_used_for_selection": false, "test_evaluated": false, "selected_policy": {}}', encoding="utf-8")
        with pytest.raises(ValueError): _load_selection(path)
    finally:
        if path.exists(): path.unlink()


def test_tiny_bridge_report_contains_all_variants_and_markdown():
    report = run_bridge(Path("unused"), Path("unused"), Path("results/audits/federated_transfer_benefit_25pct.json"), synthetic=True, max_epochs=1)
    assert set(report["per_target"]) == set(report["client_grid_names"])
    assert set(report["per_target"][report["client_grid_names"][0]]["variants"]) == {"full_from_scratch", "local_proxy_warmstart_full", "selected_transfer_proxy_warmstart_full"}
    assert report["test_evaluated"] is False
    markdown = render_markdown(report)
    assert "A/B/C Train Audit" in markdown and "A/B/C Canonical Validation" in markdown
