from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from imajin.analysis.arrays import materialize_array, metadata_axes_without_channel
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import get_layer, get_viewer, list_channel_annotations
from imajin.paths import normalize_user_path
from imajin.tools.napari_ops import add_image_from_worker, snapshot_layer
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return materialize_array(arr)


def _projection_scale(layer, axis_idx: int) -> tuple[float, ...]:
    scale_in = tuple(float(s) for s in layer.scale)
    return tuple(s for i, s in enumerate(scale_in) if i != axis_idx)


def _project(data: np.ndarray, projection: str, axis_idx: int) -> np.ndarray:
    mode = projection.lower().strip()
    if mode in {"max", "mip", "maximum"}:
        return np.max(data, axis=axis_idx)
    if mode in {"mean", "avg", "average"}:
        return np.mean(data, axis=axis_idx)
    if mode in {"none", "off"}:
        if data.ndim != 2:
            raise ValueError("projection='none' requires a 2D layer")
        return data
    raise ValueError("projection must be max, mean, or none")


def _normalize_plane(
    plane: np.ndarray,
    percentile_limits: tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    arr = np.asarray(plane, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, percentile_limits)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _solid_color(name: str) -> tuple[float, float, float] | None:
    key = name.lower().replace("-", "").replace(" ", "")
    table = {
        "green": (0.0, 1.0, 0.0),
        "gfp": (0.0, 1.0, 0.0),
        "red": (1.0, 0.0, 0.0),
        "rfp": (1.0, 0.0, 0.0),
        "uv": (0.0, 0.25, 1.0),
        "blue": (0.0, 0.25, 1.0),
        "dapi": (0.0, 0.25, 1.0),
        "ir": (1.0, 0.0, 1.0),
        "farred": (1.0, 0.0, 1.0),
        "cy5": (1.0, 0.0, 1.0),
        "gray": (1.0, 1.0, 1.0),
        "grey": (1.0, 1.0, 1.0),
        "white": (1.0, 1.0, 1.0),
        "counterstain": (1.0, 1.0, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
    }
    return table.get(key)


def _apply_lut(values: np.ndarray, color: str) -> np.ndarray:
    key = color.lower().replace("-", "").replace(" ", "")
    if key == "inferno":
        try:
            from matplotlib import colormaps

            return colormaps["inferno"](values)[..., :3].astype(np.float32)
        except Exception:
            # Fall back to a simple black-red-yellow ramp if matplotlib is unavailable.
            return np.stack(
                [
                    np.clip(values * 1.8, 0, 1),
                    np.clip((values - 0.35) * 1.7, 0, 1),
                    np.clip((values - 0.75) * 4.0, 0, 1),
                ],
                axis=-1,
            ).astype(np.float32)
    solid = _solid_color(color) or (1.0, 1.0, 1.0)
    return values[..., None] * np.asarray(solid, dtype=np.float32)


def _infer_export_color(layer_name: str) -> str:
    for rec in list_channel_annotations():
        if rec.get("layer_name") != layer_name:
            continue
        if rec.get("role") == "counterstain":
            return "gray"
        if rec.get("color"):
            return str(rec["color"])
    lname = layer_name.lower()
    for token, color in (
        ("calexa", "inferno"),
        ("gcamp", "green"),
        ("gfp", "green"),
        ("fitc", "green"),
        ("mcherry", "red"),
        ("rfp", "red"),
        ("tritc", "red"),
        ("dapi", "uv"),
        ("hoechst", "uv"),
        ("cy5", "ir"),
        ("647", "ir"),
        ("farred", "ir"),
        ("far red", "ir"),
    ):
        if token in lname:
            return color
    return "gray"


def _draw_scale_bar(
    rgb: np.ndarray,
    *,
    scale: tuple[float, ...],
    scale_bar_um: float,
    thickness_px: int | None,
    margin_px: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if scale_bar_um <= 0:
        return rgb, None
    if not scale:
        return rgb, None
    x_um_per_px = float(scale[-1])
    if x_um_per_px <= 0:
        return rgb, None

    from PIL import Image, ImageDraw

    h, w = rgb.shape[:2]
    length_px = int(round(float(scale_bar_um) / x_um_per_px))
    if length_px <= 0:
        return rgb, None
    length_px = min(length_px, max(1, w - 2 * margin_px))
    if thickness_px is None:
        thickness_px = max(2, int(round(8 * (w / 2048.0))))
    thickness_px = max(1, int(thickness_px))

    x1 = max(0, w - margin_px - length_px)
    x2 = min(w - 1, x1 + length_px)
    y2 = max(0, h - margin_px)
    y1 = max(0, y2 - thickness_px)

    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    return np.asarray(img).astype(np.float32) / 255.0, {
        "scale_bar_um": float(scale_bar_um),
        "length_px": int(length_px),
        "thickness_px": int(thickness_px),
        "x_um_per_px": x_um_per_px,
    }


def _resolve_axis(layer, axis: int | str) -> int:
    if isinstance(axis, int):
        return axis
    if not isinstance(axis, str):
        raise TypeError(f"axis must be int or str, got {type(axis).__name__}")

    code = axis.upper()
    md = getattr(layer, "metadata", {}) or {}
    layer_axes = metadata_axes_without_channel(md if isinstance(md, dict) else None, layer.data.ndim)
    if layer_axes and code in layer_axes:
        return layer_axes.index(code)

    ndim = layer.data.ndim
    fallback = {"T": 0, "Z": 0 if ndim == 3 else (ndim - 3), "Y": ndim - 2, "X": ndim - 1}
    if code in fallback:
        idx = fallback[code]
        if idx < 0 or idx >= ndim:
            raise ValueError(
                f"cannot resolve axis {axis!r} for {ndim}-D layer; specify integer axis."
            )
        return idx
    raise ValueError(f"unknown axis name {axis!r}")


@tool(
    description="Switch viewer between 2D and 3D display, and optionally set camera "
    "angles (degrees, Euler), zoom, and center. Pass ndisplay=3 to enter 3D volume "
    "rendering. angles is (alpha, beta, gamma); azimuth sweep usually varies the "
    "second component.",
    phase="5",
)
def set_view(
    ndisplay: int = 2,
    angles: tuple[float, float, float] | list[float] | None = None,
    zoom: float | None = None,
    center: tuple[float, ...] | list[float] | None = None,
) -> dict[str, Any]:
    if ndisplay not in (2, 3):
        raise ValueError(f"ndisplay must be 2 or 3, got {ndisplay}")
    viewer = get_viewer()
    viewer.dims.ndisplay = ndisplay

    if angles is not None:
        viewer.camera.angles = tuple(float(a) for a in angles)
    if zoom is not None:
        viewer.camera.zoom = float(zoom)
    if center is not None:
        viewer.camera.center = tuple(float(c) for c in center)

    return {
        "ndisplay": int(viewer.dims.ndisplay),
        "angles": tuple(float(a) for a in viewer.camera.angles),
        "zoom": float(viewer.camera.zoom),
        "center": tuple(float(c) for c in viewer.camera.center),
    }


@tool(
    description="Set the colormap (LUT) for an image layer. Common choices: "
    "gray, viridis, inferno, magma, red, green, blue, cyan, magenta, yellow.",
    phase="5",
)
def set_colormap(layer: str, colormap: str) -> dict[str, Any]:
    L = get_layer(layer)
    if not hasattr(L, "colormap"):
        raise ValueError(f"layer {layer!r} ({type(L).__name__}) does not support colormaps.")
    L.colormap = colormap
    return {"layer": layer, "colormap": colormap}


@tool(
    description="Extract a single timepoint from a time-series image layer and add it "
    "as a new image layer. Use this to create a reference frame for segmentation or "
    "manual ROI drawing before measuring intensity over time.",
    phase="2",
    worker=True,
)
def extract_timepoint(
    layer: str,
    t: int = 0,
    time_axis: int | str = "t",
) -> dict[str, Any]:
    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    idx = _resolve_axis(L, time_axis)
    if idx < 0 or idx >= data.ndim:
        raise ValueError(f"time axis index {idx} out of range for {data.ndim}-D layer")
    if t < 0 or t >= data.shape[idx]:
        raise ValueError(f"timepoint {t} out of range for axis size {data.shape[idx]}")

    frame = np.take(data, t, axis=idx)
    scale_in = tuple(float(s) for s in L.scale)
    new_scale = tuple(s for i, s in enumerate(scale_in) if i != idx)
    new = call_on_main(
        add_image_from_worker,
        frame,
        name=f"{L.name}_t{t}",
        scale=new_scale,
        metadata={"source_layer": L.name, "op": "extract_timepoint", "timepoint": t},
    )
    return {
        "new_layer": new.name,
        "shape": tuple(int(s) for s in frame.shape),
        "timepoint": int(t),
        "time_axis": idx,
    }


@tool(
    description="Capture a screenshot of the napari canvas. Saves to path if given. "
    "Always returns a base64-encoded PNG thumbnail (max 256 px on the long side).",
    phase="5",
)
def screenshot(path: str | None = None) -> dict[str, Any]:
    viewer = get_viewer()
    arr = viewer.screenshot(path=None, canvas_only=True)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.uint8)

    saved_path: str | None = None
    if path:
        from PIL import Image

        out = normalize_user_path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out)
        saved_path = str(out)

    from PIL import Image

    img = Image.fromarray(arr)
    h, w = arr.shape[:2]
    scale = 256.0 / max(h, w)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    thumb = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "path": saved_path,
        "size": (int(arr.shape[1]), int(arr.shape[0])),
        "thumb_base64": thumb,
    }


@tool(
    description="Maximum-intensity projection (MIP) along an axis. axis accepts 'z'/"
    "'y'/'x'/'t' (resolved via layer's recorded axes) or an integer index. Adds a "
    "new image layer with the reduced shape.",
    phase="5",
    worker=True,
)
def max_projection(layer: str, axis: int | str = "z") -> dict[str, Any]:
    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    idx = _resolve_axis(L, axis)
    if idx < 0 or idx >= data.ndim:
        raise ValueError(f"axis index {idx} out of range for {data.ndim}-D layer")

    proj = _project(data, "max", idx)
    new_scale = _projection_scale(L, idx)

    suffix = axis if isinstance(axis, str) else f"ax{idx}"
    new_name = f"{L.name}_mip_{suffix}"
    new = call_on_main(
        add_image_from_worker,
        proj,
        name=new_name,
        scale=new_scale,
        metadata={"source_layer": L.name, "op": "max_projection", "axis": idx},
    )
    return {
        "new_layer": new.name,
        "shape": tuple(int(s) for s in proj.shape),
        "axis": idx,
    }


@tool(
    description="Average-intensity projection along an axis. Use this for intensity "
    "comparison workflows where mean signal across the z-stack matters. axis accepts "
    "'z'/'y'/'x'/'t' or an integer index.",
    phase="5",
    worker=True,
)
def average_projection(layer: str, axis: int | str = "z") -> dict[str, Any]:
    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    idx = _resolve_axis(L, axis)
    if idx < 0 or idx >= data.ndim:
        raise ValueError(f"axis index {idx} out of range for {data.ndim}-D layer")

    proj = _project(data, "mean", idx).astype(np.float32)
    new_scale = _projection_scale(L, idx)
    suffix = axis if isinstance(axis, str) else f"ax{idx}"
    new = call_on_main(
        add_image_from_worker,
        proj,
        name=f"{L.name}_avg_{suffix}",
        scale=new_scale,
        metadata={"source_layer": L.name, "op": "average_projection", "axis": idx},
    )
    return {
        "new_layer": new.name,
        "shape": tuple(int(s) for s in proj.shape),
        "axis": idx,
    }


@tool(
    description="Export a publication-style RGB PNG from multiple channels. Each channel "
    "can be max- or average-projected before merge. Counterstain-annotated channels "
    "default to gray; CaLexA-like layers default to inferno. Adds a 50 um scale bar "
    "by default.",
    phase="5",
    worker=True,
)
def export_channel_composite_png(
    layers: list[str],
    path: str,
    projection: str = "max",
    axis: int | str = "z",
    colors: list[str] | None = None,
    scale_bar_um: float = 50.0,
    scale_bar_thickness_px: int | None = None,
    scale_bar_margin_px: int = 32,
    percentile_limits: tuple[float, float] = (0.5, 99.5),
    add_layer: bool = True,
) -> dict[str, Any]:
    if not layers:
        raise ValueError("layers must be a non-empty list")
    if colors is not None and len(colors) != len(layers):
        raise ValueError("colors must have the same length as layers")

    planes: list[np.ndarray] = []
    used_colors: list[str] = []
    output_scale: tuple[float, ...] | None = None
    for i, layer_name in enumerate(layers):
        L = call_on_main(snapshot_layer, layer_name)
        data = _materialize(L.data)
        if data.ndim == 2:
            plane = data
            scale = tuple(float(s) for s in L.scale)
        else:
            idx = _resolve_axis(L, axis)
            plane = _project(data, projection, idx)
            scale = _projection_scale(L, idx)
        if plane.ndim != 2:
            raise ValueError(f"layer {layer_name!r} did not reduce to 2D; got {plane.shape}")
        if planes and plane.shape != planes[0].shape:
            raise ValueError(
                f"all layers must reduce to the same YX shape; got {plane.shape} "
                f"for {layer_name!r} vs {planes[0].shape}"
            )
        planes.append(_normalize_plane(plane, percentile_limits))
        used_colors.append(colors[i] if colors is not None else _infer_export_color(layer_name))
        if output_scale is None:
            output_scale = scale

    rgb = np.zeros((*planes[0].shape, 3), dtype=np.float32)
    for plane, color in zip(planes, used_colors, strict=False):
        rgb = np.clip(rgb + _apply_lut(plane, color), 0.0, 1.0)

    scale_bar: dict[str, Any] | None = None
    if output_scale is not None:
        rgb, scale_bar = _draw_scale_bar(
            rgb,
            scale=output_scale,
            scale_bar_um=scale_bar_um,
            thickness_px=scale_bar_thickness_px,
            margin_px=scale_bar_margin_px,
        )

    from PIL import Image

    out = normalize_user_path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out)

    layer_name: str | None = None
    if add_layer:
        layer = call_on_main(
            add_image_from_worker,
            rgb,
            name=f"{Path(out).stem}_composite",
            scale=output_scale or (),
            metadata={
                "op": "export_channel_composite_png",
                "source_layers": list(layers),
                "projection": projection,
                "axis": axis,
                "colors": used_colors,
                "path": str(out),
                "scale_bar": scale_bar,
            },
            rgb=True,
        )
        layer_name = layer.name

    return {
        "path": str(out),
        "layer": layer_name,
        "shape": tuple(int(s) for s in rgb.shape),
        "source_layers": list(layers),
        "projection": projection,
        "colors": used_colors,
        "scale_bar": scale_bar,
    }


@tool(
    description="Add XZ and YZ orthogonal max-projection views as companion layers. "
    "For a 3D z-stack, this gives top-down (XY, the original), side (XZ), and front "
    "(YZ) views. Returns the new layer names.",
    phase="5",
    worker=True,
)
def orthogonal_views(layer: str) -> dict[str, Any]:
    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    if data.ndim != 3:
        raise ValueError(
            f"orthogonal_views expects a 3D layer (Z, Y, X); got shape {data.shape}"
        )
    z_idx = _resolve_axis(L, "z")
    y_idx = _resolve_axis(L, "y")
    x_idx = _resolve_axis(L, "x")

    xz = np.max(data, axis=y_idx)
    yz = np.max(data, axis=x_idx)

    scale = tuple(float(s) for s in L.scale)
    scale_xz = tuple(s for i, s in enumerate(scale) if i != y_idx)
    scale_yz = tuple(s for i, s in enumerate(scale) if i != x_idx)

    xz_layer = call_on_main(
        add_image_from_worker,
        xz,
        name=f"{L.name}_XZ",
        scale=scale_xz,
        metadata={"source_layer": L.name, "op": "orthogonal_view", "view": "XZ"},
    )
    yz_layer = call_on_main(
        add_image_from_worker,
        yz,
        name=f"{L.name}_YZ",
        scale=scale_yz,
        metadata={"source_layer": L.name, "op": "orthogonal_view", "view": "YZ"},
    )
    return {
        "xz_layer": xz_layer.name,
        "yz_layer": yz_layer.name,
        "xz_shape": tuple(int(s) for s in xz.shape),
        "yz_shape": tuple(int(s) for s in yz.shape),
    }


@tool(
    description="Render a 360-degree rotation animation of the current 3D scene as a "
    "GIF or MP4. frames is the number of rendered frames (smoothness vs. file size). "
    "axis selects which Euler component to sweep (default 1 = azimuth).",
    phase="5",
)
def animate_z_rotation(
    path: str,
    frames: int = 60,
    axis: int = 1,
    fps: int = 24,
) -> dict[str, Any]:
    if frames < 4:
        raise ValueError("frames must be >= 4")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2 (Euler angle component)")

    viewer = get_viewer()
    viewer.dims.ndisplay = 3
    out_path = normalize_user_path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_angles = list(viewer.camera.angles)
    images: list[np.ndarray] = []
    try:
        for i in range(frames):
            theta = i * 360.0 / frames
            new_angles = list(base_angles)
            new_angles[axis] = base_angles[axis] + theta
            viewer.camera.angles = tuple(new_angles)
            arr = np.asarray(viewer.screenshot(path=None, canvas_only=True))
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            images.append(arr.astype(np.uint8))
    finally:
        viewer.camera.angles = tuple(base_angles)

    import imageio.v3 as iio

    suffix = out_path.suffix.lower()
    if suffix in (".mp4", ".mov", ".webm"):
        iio.imwrite(out_path, images, fps=fps, codec="libx264")
    else:
        iio.imwrite(out_path, images, duration=int(1000 / fps), loop=0)

    return {
        "path": str(out_path),
        "frames": frames,
        "size": (int(images[0].shape[1]), int(images[0].shape[0])),
    }
