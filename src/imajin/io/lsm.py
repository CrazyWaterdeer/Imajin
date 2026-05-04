from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import tifffile

from imajin.io.channel_metadata import (
    build_channel_info,
    color_from_name,
    display_color_name,
    rgb_from_8bit_triplet,
)
from imajin.io.dataset import Dataset
from imajin.io.memory import (
    array_nbytes,
    available_memory_bytes,
    should_load_into_memory,
)
from imajin.paths import normalize_user_path


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


def _channel_display_colors(lsm_meta: dict[str, Any]) -> list[dict[str, Any]]:
    channel_colors = lsm_meta.get("ChannelColors", {})
    if not isinstance(channel_colors, dict):
        return []
    raw_colors = channel_colors.get("Colors")
    if not isinstance(raw_colors, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_colors:
        rgb = rgb_from_8bit_triplet(raw)
        if rgb is None:
            out.append({})
            continue
        item: dict[str, Any] = {
            "display_color_rgb": tuple(float(v) for v in rgb),
            "display_color_source": "lsm_channel_colors",
        }
        name = display_color_name(rgb)
        if name:
            item["display_color_name"] = name
        out.append(item)
    return out


def _emission_from_detection_channel(ch: dict[str, Any]) -> float | None:
    explicit = _first_present(
        ch,
        "EmissionWavelength",
        "DetectionWavelength",
        "AcquisitionWavelength",
    )
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            return None

    start = _first_present(ch, "SpiWavelengthStart", "WavelengthStart")
    stop = _first_present(ch, "SpiWavelengthStop", "WavelengthStop")
    try:
        lo = float(start) if start is not None else None
        hi = float(stop) if stop is not None else None
    except (TypeError, ValueError):
        return None
    if hi is None or hi <= 0:
        return None
    if lo is None or lo <= 0:
        return hi
    return (lo + hi) / 2.0


def _channel_info_from_detection_channel(
    ch: dict[str, Any], fallback_name: str | None = None
) -> dict[str, Any]:
    channel_name = _first_present(ch, "ChannelName", "Name") or fallback_name
    dye_name = _first_present(ch, "DyeName", "Dye", "Fluorophore")
    excitation = _first_present(
        ch,
        "ExcitationWavelength",
        "LaserWavelength",
        "IlluminationWavelength",
    )
    emission = _emission_from_detection_channel(ch)
    extra: dict[str, Any] = {}
    if dye_name:
        extra["dye_name"] = str(dye_name)
    for src_key, dst_key in (
        ("FilterName", "filter_name"),
        ("FilterSetName", "filter_set_name"),
        ("PointDetectorName", "detector_name"),
        ("ChannelName", "channel_name"),
    ):
        value = _first_present(ch, src_key)
        if value is not None:
            extra[dst_key] = str(value)

    info = build_channel_info(
        name=str(channel_name) if channel_name is not None else None,
        excitation=excitation,
        emission=emission,
        extra=extra,
    )
    if "color" not in info and dye_name:
        color = color_from_name(str(dye_name))
        if color:
            info["color"] = color
    return info


def _channel_metadata(lsm_meta: dict[str, Any]) -> list[dict[str, Any]]:
    names = _channel_names(lsm_meta)
    out: list[dict[str, Any]] = []
    display_colors = _channel_display_colors(lsm_meta)

    si = lsm_meta.get("ScanInformation", {})
    if isinstance(si, dict):
        tracks = si.get("Tracks", [])
        for track in tracks if isinstance(tracks, list) else []:
            if not isinstance(track, dict):
                continue
            detection = track.get("DetectionChannels", [])
            if isinstance(detection, list) and detection:
                for ch in detection:
                    if not isinstance(ch, dict):
                        continue
                    fallback = names[len(out)] if len(out) < len(names) else None
                    out.append(_channel_info_from_detection_channel(ch, fallback))
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
    for i, display in enumerate(display_colors):
        if i >= len(out):
            break
        out[i].update(display)
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
    p = normalize_user_path(path)
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
