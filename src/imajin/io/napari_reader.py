from __future__ import annotations

from typing import Any

from imajin.io.channel_metadata import display_color_name, pad_channel_metadata
from imajin.paths import normalize_user_path

LayerData = tuple[Any, dict, str]
_SUPPORTED = (".lsm", ".czi", ".ome.tif", ".ome.tiff", ".tif", ".tiff")


def _fallback_colormap_for_color(color: str | None) -> str | None:
    if color == "green":
        return "green"
    if color == "red":
        return "red"
    if color == "uv":
        return "blue"
    if color == "ir":
        return "magenta"
    return None


def _napari_colormap_from_rgb(
    rgb: tuple[float, float, float], name: str
):  # noqa: ANN202 - napari's Colormap type is optional at import time
    builtin = display_color_name(rgb)
    if builtin:
        return builtin
    try:
        from napari.utils import Colormap

        return Colormap(
            colors=[
                (0.0, 0.0, 0.0, 1.0),
                (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0),
            ],
            name=name,
        )
    except Exception:
        return "gray"


def _colormap_for_channel(info: dict[str, Any], index: int) -> Any:
    rgb_raw = info.get("display_color_rgb")
    if isinstance(rgb_raw, (list, tuple)) and len(rgb_raw) >= 3:
        try:
            rgb = (float(rgb_raw[0]), float(rgb_raw[1]), float(rgb_raw[2]))
        except (TypeError, ValueError):
            rgb = None
        if rgb is not None:
            name = str(info.get("display_color_name") or f"channel_{index + 1}_display")
            return _napari_colormap_from_rgb(rgb, name)
    fallback = _fallback_colormap_for_color(str(info.get("color")) if info.get("color") else None)
    return fallback or "gray"


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
        kwargs["colormap"] = [
            _colormap_for_channel(info, i)
            for i, info in enumerate(metadata["channel_metadata"])
        ]
        kwargs["blending"] = "additive"
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
    name = normalize_user_path(path).name.lower()
    return any(name.endswith(s) for s in _SUPPORTED)


def read_path(path):
    return _do_read if _matches(path) else None


napari_get_reader = read_path
