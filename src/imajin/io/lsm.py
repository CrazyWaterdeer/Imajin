from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import dask.array as da
import tifffile
import zarr

from imajin.io.dataset import Dataset


def _voxel_size_um(lsm_meta: dict[str, Any]) -> tuple[float, float, float]:
    vz = float(lsm_meta.get("VoxelSizeZ", 1e-6)) * 1e6
    vy = float(lsm_meta.get("VoxelSizeY", 1e-6)) * 1e6
    vx = float(lsm_meta.get("VoxelSizeX", 1e-6)) * 1e6
    return (vz, vy, vx)


def _channel_names(lsm_meta: dict[str, Any]) -> list[str]:
    colors = lsm_meta.get("ChannelColors", {})
    if isinstance(colors, dict):
        names = colors.get("ColorNames")
        if names:
            return [str(n) for n in names]

    si = lsm_meta.get("ScanInformation", {})
    if isinstance(si, dict):
        tracks = si.get("Tracks", [])
        out: list[str] = []
        for t in tracks if isinstance(tracks, list) else []:
            if not isinstance(t, dict):
                continue
            for ch in t.get("DataChannels", []):
                if isinstance(ch, dict) and ch.get("Name"):
                    out.append(str(ch["Name"]))
        if out:
            return out
    return []


def load_lsm(path: Path | str, position_index: int = 0) -> Dataset:
    p = Path(path)
    with tifffile.TiffFile(str(p)) as tf:
        lsm_meta = tf.lsm_metadata or {}
        series = tf.series[0]
        axes = series.axes

    n_positions = int(lsm_meta.get("DimensionP", 1) or 1)
    if n_positions > 1:
        if position_index >= n_positions:
            raise ValueError(
                f"position_index {position_index} >= n_positions {n_positions}"
            )
        warnings.warn(
            f"LSM has {n_positions} positions; loading position {position_index}. "
            "Pass position_index=... to load others.",
            stacklevel=2,
        )

    store = tifffile.imread(str(p), aszarr=True, level=0)
    arr = zarr.open(store, mode="r")
    data = da.from_zarr(arr)

    if "P" in axes:
        p_idx = axes.index("P")
        data = data.take(position_index, axis=p_idx)
        axes = axes.replace("P", "")

    return Dataset(
        data=data,
        axes=axes,
        voxel_size=_voxel_size_um(lsm_meta),
        channel_names=_channel_names(lsm_meta),
        source_path=p,
        raw_metadata={"lsm": dict(lsm_meta)},
    )
