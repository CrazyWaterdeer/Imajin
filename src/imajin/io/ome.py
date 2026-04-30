from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import dask.array as da
import tifffile
import zarr

from imajin.io.dataset import Dataset

_OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _parse_ome_xml(xml: str) -> tuple[tuple[float, float, float], list[str]]:
    voxel = (1.0, 1.0, 1.0)
    channels: list[str] = []
    if not xml:
        return voxel, channels
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
    except ET.ParseError:
        pass
    return voxel, channels


def load_ome(path: Path | str) -> Dataset:
    p = Path(path)
    with tifffile.TiffFile(str(p)) as tf:
        ome_xml = tf.ome_metadata or ""
        series = tf.series[0]
        axes = series.axes

    store = tifffile.imread(str(p), aszarr=True, level=0)
    arr = zarr.open(store, mode="r")
    data = da.from_zarr(arr)

    voxel_size, channel_names = _parse_ome_xml(ome_xml)

    return Dataset(
        data=data,
        axes=axes,
        voxel_size=voxel_size,
        channel_names=channel_names,
        source_path=p,
        raw_metadata={"ome_xml": ome_xml},
    )
