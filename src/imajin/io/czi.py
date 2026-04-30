from __future__ import annotations

from pathlib import Path

from imajin.io.dataset import Dataset


def load_czi(path: Path | str) -> Dataset:
    from bioio import BioImage

    p = Path(path)
    img = BioImage(str(p))
    data = img.dask_data

    ps = img.physical_pixel_sizes
    voxel_size = (
        float(ps.Z or 1.0),
        float(ps.Y or 1.0),
        float(ps.X or 1.0),
    )

    channel_names = list(img.channel_names) if img.channel_names else []

    raw: dict = {}
    try:
        raw["czi"] = dict(img.metadata) if hasattr(img.metadata, "items") else {
            "_repr": repr(img.metadata)[:1000]
        }
    except Exception:
        raw["czi"] = {}

    return Dataset(
        data=data,
        axes="TCZYX",
        voxel_size=voxel_size,
        channel_names=channel_names,
        source_path=p,
        raw_metadata=raw,
    )
