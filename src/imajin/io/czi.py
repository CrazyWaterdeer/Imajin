from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from imajin.io.channel_metadata import build_channel_info
from imajin.io.dataset import Dataset


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _channel_metadata_from_xml(metadata: Any) -> list[dict[str, Any]]:
    root = metadata
    if isinstance(metadata, str):
        try:
            root = ET.fromstring(metadata)
        except ET.ParseError:
            return []
    if not hasattr(root, "iter"):
        return []

    channels: list[dict[str, Any]] = []
    for elem in root.iter():
        if _local_name(str(elem.tag)).lower() != "channel":
            continue
        attrs = getattr(elem, "attrib", {}) or {}
        name = attrs.get("Name") or attrs.get("ShortName") or attrs.get("Id")
        excitation = (
            attrs.get("ExcitationWavelength")
            or attrs.get("Excitation")
            or attrs.get("LaserWavelength")
        )
        emission = (
            attrs.get("EmissionWavelength")
            or attrs.get("Emission")
            or attrs.get("DetectionWavelength")
        )
        channels.append(
            build_channel_info(name=name, excitation=excitation, emission=emission)
        )
    return channels


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
    channel_metadata: list[dict[str, Any]] = [
        build_channel_info(name=name) for name in channel_names
    ]

    raw: dict = {}
    try:
        raw["czi"] = dict(img.metadata) if hasattr(img.metadata, "items") else {
            "_repr": repr(img.metadata)[:1000]
        }
        parsed = _channel_metadata_from_xml(img.metadata)
        if parsed:
            channel_metadata = parsed
    except Exception:
        raw["czi"] = {}

    return Dataset(
        data=data,
        axes="TCZYX",
        voxel_size=voxel_size,
        channel_names=channel_names,
        channel_metadata=channel_metadata,
        source_path=p,
        raw_metadata=raw,
    )
