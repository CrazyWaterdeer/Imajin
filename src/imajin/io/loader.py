from __future__ import annotations

from pathlib import Path

from imajin.io.dataset import Dataset


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    name = p.name.lower()

    if name.endswith(".lsm"):
        from imajin.io.lsm import load_lsm

        return load_lsm(p)

    if name.endswith(".czi"):
        from imajin.io.czi import load_czi

        return load_czi(p)

    if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
        from imajin.io.ome import load_ome

        return load_ome(p)

    if name.endswith((".tif", ".tiff")):
        from imajin.io.ome import load_ome

        return load_ome(p)

    raise ValueError(f"Unsupported file format for: {p}")
