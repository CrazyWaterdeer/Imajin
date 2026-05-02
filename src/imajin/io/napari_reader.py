from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.io.channel_metadata import pad_channel_metadata

LayerData = tuple[Any, dict, str]
_SUPPORTED = (".lsm", ".czi", ".ome.tif", ".ome.tiff", ".tif", ".tiff")


def _to_layer(ds) -> LayerData:
    base = ds.source_path.stem if ds.source_path else "image"
    base = base.removesuffix(".ome")
    scale_per_axis = {
        "T": 1.0,
        "Z": float(ds.voxel_size[0]),
        "Y": float(ds.voxel_size[1]),
        "X": float(ds.voxel_size[2]),
    }

    metadata = {"voxel_size_um": ds.voxel_size, "axes": ds.axes}
    for key in ("load_mode", "estimated_nbytes", "available_memory_bytes"):
        if key in ds.raw_metadata:
            metadata[key] = ds.raw_metadata[key]

    kwargs: dict = {"metadata": metadata}
    if "C" in ds.axes:
        c_idx = ds.axes.index("C")
        kwargs["channel_axis"] = c_idx
        n_ch = int(ds.data.shape[c_idx])
        if ds.channel_names and len(ds.channel_names) == n_ch:
            kwargs["name"] = list(ds.channel_names)
        else:
            kwargs["name"] = [f"{base}_ch{i}" for i in range(n_ch)]
        kwargs["scale"] = tuple(
            scale_per_axis.get(a, 1.0) for a in ds.axes if a != "C"
        )
        metadata["channel_names"] = list(kwargs["name"])
        metadata["channel_metadata"] = pad_channel_metadata(
            list(getattr(ds, "channel_metadata", []) or []),
            n_channels=n_ch,
            names=list(kwargs["name"]),
        )
    else:
        kwargs["name"] = base
        kwargs["scale"] = tuple(scale_per_axis.get(a, 1.0) for a in ds.axes)
        if getattr(ds, "channel_metadata", None):
            metadata["channel_metadata"] = list(ds.channel_metadata)

    return (ds.data, kwargs, "image")


def _do_read(path) -> list[LayerData]:
    if isinstance(path, (list, tuple)):
        path = path[0]
    from imajin.io.loader import load_dataset

    ds = load_dataset(path)
    return [_to_layer(ds)]


def _matches(path) -> bool:
    if isinstance(path, (list, tuple)):
        if not path:
            return False
        path = path[0]
    name = Path(path).name.lower()
    return any(name.endswith(s) for s in _SUPPORTED)


def read_path(path):
    return _do_read if _matches(path) else None


napari_get_reader = read_path
