from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    outputs: Path

def default_paths(root: Path) -> Paths:
    return Paths(
        root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_processed=root / "data" / "processed",
        outputs=root / "outputs",
    )