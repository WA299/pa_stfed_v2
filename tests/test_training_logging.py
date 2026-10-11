from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from code.audits.federated_transfer_benefit import fit_fit_only_scaler, make_proxy, scarce_split, train_proxy


def _grid(seed=7):
    rng = np.random.default_rng(seed)
    total, nodes = 430, 2
    dynamic = rng.normal(size=(total, nodes, 7)).astype(np.float32)
    return SimpleNamespace(
        grid_name="logging_synthetic",
        num_nodes=nodes,
        load_bus_mask=np.ones(nodes, dtype=bool),
        dynamic_features=dynamic,
        p=dynamic[:, :, 0].copy(),
        timestamps=np.arange(total),
        splits={
            "train": SimpleNamespace(start_index=0, end_index=360),
            "validation": SimpleNamespace(start_index=360, end_index=400),
            "test": SimpleNamespace(start_index=400, end_index=430),
        },
    )


def test_proxy_logging_is_immediate_and_does_not_change_training(tmp_path, capsys):
    grid = _grid()
    split = scarce_split(grid, 0.5)
    scaler = fit_fit_only_scaler(grid, split.fit_indices, split.available_start)
    torch.manual_seed(11)
    quiet_model, quiet = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, device="cpu", max_epochs=1, seed=11, log_enabled=False)
    quiet_out = capsys.readouterr().out
    torch.manual_seed(11)
    loud_model, loud = train_proxy(grid, split.fit_indices, split.calibration_indices, scaler, device="cpu", max_epochs=1, seed=11, log_enabled=True, log_file=str(tmp_path / "train.log"), log_context={"scenario": "synthetic", "history_fraction": 0.5, "target_grid": grid.grid_name, "method": "proxy"})
    loud_out = capsys.readouterr().out
    assert quiet_out == ""
    assert "stage=proxy_epoch_complete" in loud_out
    assert "epoch=1/1" in loud_out and "calibration_mae=" in loud_out and "best_mae=" in loud_out
    assert (tmp_path / "train.log").read_text(encoding="utf-8") == loud_out
    assert loud["epochs_run"] == quiet["epochs_run"]
    assert loud["selected_epoch"] == quiet["selected_epoch"]
    assert loud["best_calibration_node_mae"] == quiet["best_calibration_node_mae"]
    quiet_state, loud_state = quiet_model.state_dict(), loud_model.state_dict()
    assert quiet_state.keys() == loud_state.keys()
    assert all(torch.equal(quiet_state[name], loud_state[name]) for name in quiet_state)
