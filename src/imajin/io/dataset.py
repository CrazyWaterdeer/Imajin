from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Dataset:
    data: Any
    axes: str
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    channel_names: list[str] = field(default_factory=list)
    source_path: Path | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        if "C" in self.axes:
            return int(self.data.shape[self.axes.index("C")])
        return 1

    @property
    def is_3d(self) -> bool:
        return "Z" in self.axes
