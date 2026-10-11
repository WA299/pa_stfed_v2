"""Small, configurable training progress logger for the active terminal.

The logger only formats and emits already available values.  It never performs
model evaluation, synchronization, or state mutation.  ``enabled=False`` is a
true no-op apart from argument construction at the call site.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def emit_training_log(
    stage: str,
    *,
    enabled: bool = True,
    log_file: str | Path | None = None,
    context: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Write one newline-terminated, immediately flushed progress record."""
    if not enabled:
        return
    values: dict[str, Any] = {}
    if context:
        values.update({key: value for key, value in context.items() if value is not None})
    values.update({key: value for key, value in fields.items() if value is not None})
    prefix = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    normalized = []
    for key, value in values.items():
        if key == "history_fraction" and isinstance(value, (float, int)) and not isinstance(value, bool):
            value = f"{float(value):.0%}"
        normalized.append(f"{key}={value}")
    payload = " | ".join(normalized)
    line = f"{prefix} | stage={stage}" + (f" | {payload}" if payload else "")
    print(line, flush=True)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()


__all__ = ["emit_training_log"]
