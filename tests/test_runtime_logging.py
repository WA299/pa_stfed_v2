import sys

from code.runtime_logging import emit_training_log


def test_runtime_logger_flushes_stdout_and_optional_file(tmp_path, capsys):
    path = tmp_path / "run.log"
    emit_training_log(
        "synthetic_epoch_complete",
        context={"scenario": "synthetic", "seed": 42, "history_fraction": 0.25},
        epoch="1/1",
        calibration_mae="0.25",
        log_file=path,
    )
    output = capsys.readouterr().out
    assert "stage=synthetic_epoch_complete" in output
    assert "seed=42" in output and "history_fraction=25%" in output and "epoch=1/1" in output
    assert path.read_text(encoding="utf-8") == output
    assert output.endswith("\n") and "\r" not in output


def test_runtime_logger_disabled_is_silent(tmp_path, capsys):
    emit_training_log("disabled", enabled=False, log_file=tmp_path / "run.log")
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "run.log").exists()


def test_runtime_logger_flushes_the_active_stdout(monkeypatch):
    writes = []
    flushes = []

    class ImmediateTerminal:
        def write(self, value):
            writes.append(value)

        def flush(self):
            flushes.append(True)

    monkeypatch.setattr(sys, "stdout", ImmediateTerminal())
    emit_training_log("round_complete", round="1/1", train_loss="0.5")
    output = "".join(writes)
    assert output.endswith("\n") and "stage=round_complete" in output
    assert flushes == [True]
