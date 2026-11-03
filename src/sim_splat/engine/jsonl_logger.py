from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class JsonlLogger:
    """Append scalar records without coupling training to TensorBoard."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, step: int, values: Mapping[str, float]) -> None:
        record = {"step": step, **{key: float(value) for key, value in values.items()}}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
