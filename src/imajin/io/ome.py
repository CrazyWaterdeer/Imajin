from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import tifffile

from imajin.io.channel_metadata import build_channel_info
from imajin.io.dataset import Dataset
from imajin.io.memory import (
    array_nbytes,
    available_memory_bytes,
    should_load_into_memory,
)

_OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _parse_ome_xml(
    xml: str,
) -> tuple[tuple[float, float, float], list[str], list[dict[str, Any]]]:
    voxel = (1.0, 1.0, 1.0)
    channels: list[str] = []
    channel_metadata: list[dict[str, Any]] = []
    if not xml:
        return voxel, channels, channel_metadata
    try:
        root = ET.fromstring(xml)
        pixels = root.find(".//ome:Pixels", _OME_NS)
        if pixels is None:
            pixels = root.find(".//Pixels")
        if pixels is not None:
            voxel = (
                float(pixels.get("PhysicalSizeZ", 1.0)),
                float(pixels.get("PhysicalSizeY", 1.0)),
                float(pixels.get("PhysicalSizeX", 1.0)),
            )
            for ch in pixels.findall(".//ome:Channel", _OME_NS) or pixels.findall(
                ".//Channel"
            ):
                name = ch.get("Name") or ch.get("ID") or f"ch{len(channels)}"
                channels.append(name)
                channel_metadata.append(
                    build_channel_info(
                        name=name,
                        excitation=ch.get("ExcitationWavelength"),
                        emission=ch.get("EmissionWavelength"),
                        extra={
                            k: v
                            for k, v in {
                                "excitation_wavelength_unit": ch.get(
                                    "ExcitationWavelengthUnit"
                                ),
                                "emission_wavelength_unit": ch.get(
                                    "EmissionWavelengthUnit"
                                ),
                            }.items()
                            if v is not None
                        },
                    )
                )
    except ET.ParseError:
        pass
    return voxel, channels, channel_metadata


def _memmap_tiff_array(path: Path):
    with tifffile.TiffFile(str(path)) as tf:
        return tf.series[0].asarray(out="memmap")


def load_ome(path: Path | str) -> Dataset:
    p = Path(path)
    with tifffile.TiffFile(str(p)) as tf:
        ome_xml = tf.ome_metadata or ""
        series = tf.series[0]
        axes = series.axes
        shape = tuple(int(s) for s in series.shape)
        dtype = series.dtype

    estimated_nbytes = array_nbytes(shape, dtype)
    available_bytes = available_memory_bytes()
    load_mode = "memory"

    if should_load_into_memory(estimated_nbytes, available_bytes):
        try:
            with tifffile.TiffFile(str(p)) as tf:
                data = tf.series[0].asarray()
        except MemoryError:
            warnings.warn(
                "Not enough RAM to load TIFF fully; falling back to disk-backed "
                "memmap loading.",
                RuntimeWarning,
                stacklevel=2,
            )
            data = _memmap_tiff_array(p)
            load_mode = "memmap"
    else:
        warnings.warn(
            "Available RAM is too low for eager TIFF loading; falling back to "
            "disk-backed memmap loading.",
            RuntimeWarning,
            stacklevel=2,
        )
        data = _memmap_tiff_array(p)
        load_mode = "memmap"

    voxel_size, channel_names, channel_metadata = _parse_ome_xml(ome_xml)

    return Dataset(
        data=data,
        axes=axes,
        voxel_size=voxel_size,
        channel_names=channel_names,
        channel_metadata=channel_metadata,
        source_path=p,
        raw_metadata={
            "ome_xml": ome_xml,
            "load_mode": load_mode,
            "estimated_nbytes": estimated_nbytes,
            "available_memory_bytes": available_bytes,
        },
    )
