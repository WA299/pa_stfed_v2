"""Strict validation for resumable formal benchmark artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from code.federated.formal_btd import CLIENT_NAMES


def load_valid_result(path: Path, *, seed: int, artifact_type: str, run_mode: str,
                      methods: tuple[str, ...], rounds: int | None = None,
                      local_epochs: int | None = None, batch_size: int = 32,
                      learning_rate: float = 1e-3, max_epochs: int | None = None,
                      patience: int | None = None) -> dict[str, Any]:
    """Load only a complete result matching the requested execution contract."""
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_mode": run_mode, "seed": int(seed), "artifact_type": artifact_type,
        "history_fraction": 0.25, "batch_size": int(batch_size),
        "learning_rate": float(learning_rate), "test_evaluated": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"artifact {path} has {key}={report.get(key)!r}; expected {value!r}")
    for key, value in (("rounds", rounds), ("local_epochs", local_epochs),
                       ("max_epochs", max_epochs), ("patience", patience)):
        if value is not None and report.get(key) != value:
            raise ValueError(f"artifact {path} has {key}={report.get(key)!r}; expected {value!r}")
    if tuple(report.get("client_grid_names", ())) != CLIENT_NAMES:
        raise ValueError(f"artifact {path} does not expose the four scarce targets")
    if set(report.get("methods", ())) != set(methods):
        raise ValueError(f"artifact {path} has method set {report.get('methods')!r}; expected {methods!r}")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(CLIENT_NAMES):
        raise ValueError(f"artifact {path} is missing one or more scarce-target scenarios")
    return report


def resumable_result(path: Path, **contract: Any) -> bool:
    """Return true only for a complete matching result; invalid artifacts are not resumable."""
    try:
        load_valid_result(path, **contract)
    except (FileNotFoundError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True
