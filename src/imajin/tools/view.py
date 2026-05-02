from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import get_layer, get_viewer
from imajin.tools.napari_ops import add_image_from_worker, snapshot_layer
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


def _resolve_axis(layer, axis: int | str) -> int:
    if isinstance(axis, int):
        return axis
    if not isinstance(axis, str):
        raise TypeError(f"axis must be int or str, got {type(axis).__name__}")

    code = axis.upper()
    md = getattr(layer, "metadata", {}) or {}
    axes = md.get("axes")
    if isinstance(axes, str):
        # axes string was the *original* dataset axes (e.g., TCZYX). In our reader,
        # channel_axis splits C out, so the per-layer axes is axes minus 'C'.
        layer_axes = axes.replace("C", "")
        if code in layer_axes:
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

        out = Path(path).expanduser().resolve()
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

    proj = np.max(data, axis=idx)

    scale_in = tuple(float(s) for s in L.scale)
    new_scale = tuple(s for i, s in enumerate(scale_in) if i != idx)

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
    out_path = Path(path).expanduser().resolve()
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
