from __future__ import annotations

from pathlib import Path
from typing import Any

import tifffile

from imajin.io.channel_metadata import apply_dtype_bit_depth, pad_channel_metadata
from imajin.paths import normalize_user_path


def _file_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".lsm"):
        return "lsm"
    if name.endswith(".czi"):
        return "czi"
    if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
        return "ome_tiff"
    if name.endswith(".tif") or name.endswith(".tiff"):
        return "tiff"
    raise ValueError(f"Unsupported file format for metadata: {path}")


def _n_channels(axes: str, shape: tuple[int, ...]) -> int:
    if "C" in axes and axes.index("C") < len(shape):
        return int(shape[axes.index("C")])
    return 1


def _summary(
    *,
    path: Path,
    file_type: str,
    axes: str,
    shape: tuple[int, ...],
    dtype: Any,
    voxel_size_um: tuple[float, float, float],
    channel_names: list[str],
    channel_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    n_channels = _n_channels(axes, shape)
    channels = pad_channel_metadata(channel_metadata, n_channels, channel_names)
    apply_dtype_bit_depth(channels, dtype)
    return {
        "path": str(path),
        "file_type": file_type,
        "metadata_read_mode": "metadata_only",
        "axes": axes,
        "shape": shape,
        "dtype": str(dtype),
        "n_channels": n_channels,
        "voxel_size_um": voxel_size_um,
        "channel_names": list(channel_names),
        "channel_metadata": channels,
    }


def _read_lsm_metadata(path: Path) -> dict[str, Any]:
    from imajin.io.lsm import _channel_metadata, _channel_names, _voxel_size_um

    with tifffile.TiffFile(str(path)) as tf:
        lsm_meta = tf.lsm_metadata or {}
        series = tf.series[0]
        axes = str(series.axes)
        shape = tuple(int(s) for s in series.shape)
        dtype = series.dtype

    return _summary(
        path=path,
        file_type="lsm",
        axes=axes,
        shape=shape,
        dtype=dtype,
        voxel_size_um=_voxel_size_um(lsm_meta),
        channel_names=_channel_names(lsm_meta),
        channel_metadata=_channel_metadata(lsm_meta),
    )


def _read_tiff_metadata(path: Path, *, file_type: str) -> dict[str, Any]:
    from imajin.io.ome import _parse_ome_xml

    with tifffile.TiffFile(str(path)) as tf:
        ome_xml = tf.ome_metadata or ""
        series = tf.series[0]
        axes = str(series.axes)
        shape = tuple(int(s) for s in series.shape)
        dtype = series.dtype

    voxel, names, channels = _parse_ome_xml(ome_xml)
    return _summary(
        path=path,
        file_type=file_type,
        axes=axes,
        shape=shape,
        dtype=dtype,
        voxel_size_um=voxel,
        channel_names=names,
        channel_metadata=channels,
    )


def _read_czi_metadata(path: Path) -> dict[str, Any]:
    from bioio import BioImage

    from imajin.io.czi import _channel_metadata_from_xml

    img = BioImage(str(path))
    axes = "".join(str(a) for a in getattr(getattr(img, "dims", None), "order", ""))
    if not axes:
        axes = "TCZYX"
    shape = tuple(int(s) for s in getattr(img, "shape", ()))
    dtype = getattr(img, "dtype", None)
    if dtype is None:
        dtype = getattr(getattr(img, "dask_data", None), "dtype", "")

    ps = img.physical_pixel_sizes
    voxel_size = (
        float(ps.Z or 1.0),
        float(ps.Y or 1.0),
        float(ps.X or 1.0),
    )
    channel_names = list(img.channel_names) if img.channel_names else []
    channel_metadata = []
    try:
        channel_metadata = _channel_metadata_from_xml(img.metadata)
    except Exception:
        channel_metadata = []
    return _summary(
        path=path,
        file_type="czi",
        axes=axes,
        shape=shape,
        dtype=dtype,
        voxel_size_um=voxel_size,
        channel_names=channel_names,
        channel_metadata=channel_metadata,
    )


def read_metadata_summary(path: str | Path) -> dict[str, Any]:
    """Read image metadata without materializing pixel arrays."""

    p = normalize_user_path(path).resolve()
    kind = _file_type(p)
    if kind == "lsm":
        return _read_lsm_metadata(p)
    if kind == "czi":
        return _read_czi_metadata(p)
    if kind == "ome_tiff":
        return _read_tiff_metadata(p, file_type="ome_tiff")
    return _read_tiff_metadata(p, file_type="tiff")
