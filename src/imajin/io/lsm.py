from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import tifffile

from imajin.io.channel_metadata import build_channel_info
from imajin.io.dataset import Dataset
from imajin.io.memory import (
    array_nbytes,
    available_memory_bytes,
    should_load_into_memory,
)


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


def _first_present(d: dict[str, Any], *keys: str) -> Any:
    lowered = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        if key in d:
            return d[key]
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _channel_metadata(lsm_meta: dict[str, Any]) -> list[dict[str, Any]]:
    names = _channel_names(lsm_meta)
    out: list[dict[str, Any]] = []

    si = lsm_meta.get("ScanInformation", {})
    if isinstance(si, dict):
        tracks = si.get("Tracks", [])
        for track in tracks if isinstance(tracks, list) else []:
            if not isinstance(track, dict):
                continue
            illumination = track.get("IlluminationChannels", [])
            illum_wavelengths: list[Any] = []
            for illum in illumination if isinstance(illumination, list) else []:
                if isinstance(illum, dict):
                    illum_wavelengths.append(
                        _first_present(
                            illum,
                            "Wavelength",
                            "LaserWavelength",
                            "ExcitationWavelength",
                        )
                    )
            data_channels = track.get("DataChannels", [])
            for i, ch in enumerate(data_channels if isinstance(data_channels, list) else []):
                if not isinstance(ch, dict):
                    continue
                name = (
                    _first_present(ch, "Name", "DyeName", "ChannelName")
                    or (names[len(out)] if len(out) < len(names) else None)
                )
                excitation = _first_present(
                    ch,
                    "ExcitationWavelength",
                    "LaserWavelength",
                    "IlluminationWavelength",
                )
                if excitation is None and i < len(illum_wavelengths):
                    excitation = illum_wavelengths[i]
                emission = _first_present(
                    ch,
                    "EmissionWavelength",
                    "DetectionWavelength",
                    "AcquisitionWavelength",
                )
                out.append(
                    build_channel_info(
                        name=str(name) if name is not None else None,
                        excitation=excitation,
                        emission=emission,
                    )
                )

    if not out:
        out = [build_channel_info(name=name) for name in names]
    return out


def _select_position(data: Any, axes: str, position_index: int) -> tuple[Any, str]:
    if "P" in axes:
        p_idx = axes.index("P")
        slicer = [slice(None)] * data.ndim
        slicer[p_idx] = position_index
        data = data[tuple(slicer)]
        axes = axes.replace("P", "")
    return data, axes


def _memmap_lsm_array(path: Path, axes: str, position_index: int) -> tuple[Any, str]:
    with tifffile.TiffFile(str(path)) as tf:
        data = tf.series[0].asarray(out="memmap")
    return _select_position(data, axes, position_index)


def load_lsm(path: Path | str, position_index: int = 0) -> Dataset:
    p = Path(path)
    with tifffile.TiffFile(str(p)) as tf:
        lsm_meta = tf.lsm_metadata or {}
        series = tf.series[0]
        axes = series.axes
        shape = tuple(int(s) for s in series.shape)
        dtype = series.dtype

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

    estimated_nbytes = array_nbytes(shape, dtype)
    available_bytes = available_memory_bytes()
    load_mode = "memory"

    if should_load_into_memory(estimated_nbytes, available_bytes):
        try:
            with tifffile.TiffFile(str(p)) as tf:
                data = tf.series[0].asarray()
            data, axes = _select_position(data, axes, position_index)
        except MemoryError:
            warnings.warn(
                "Not enough RAM to load LSM fully; falling back to disk-backed "
                "memmap loading.",
                RuntimeWarning,
                stacklevel=2,
            )
            data, axes = _memmap_lsm_array(p, axes, position_index)
            load_mode = "memmap"
    else:
        warnings.warn(
            "Available RAM is too low for eager LSM loading; falling back to "
            "disk-backed memmap loading.",
            RuntimeWarning,
            stacklevel=2,
        )
        data, axes = _memmap_lsm_array(p, axes, position_index)
        load_mode = "memmap"

    return Dataset(
        data=data,
        axes=axes,
        voxel_size=_voxel_size_um(lsm_meta),
        channel_names=_channel_names(lsm_meta),
        channel_metadata=_channel_metadata(lsm_meta),
        source_path=p,
        raw_metadata={
            "lsm": dict(lsm_meta),
            "load_mode": load_mode,
            "estimated_nbytes": estimated_nbytes,
            "available_memory_bytes": available_bytes,
        },
    )
