from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from imajin.analysis.arrays import materialize_array
from imajin.agent.qt_dispatch import call_on_main
from imajin.tools.napari_ops import add_image_from_worker, snapshot_layer
from imajin.tools.registry import tool

# Fast-mode background estimation downsamples each plane by this factor before
# running rolling_ball, then upsamples the (low-frequency) background back.
_RB_DOWNSAMPLE = 4


def _materialize(arr) -> np.ndarray:
    return materialize_array(arr)


def _run_over_planes(fn, n: int) -> None:
    """Apply ``fn(z)`` for z in range(n), across threads when it pays off.

    Independent Z-planes with disjoint output slices — safe to parallelise, and
    ``skimage.restoration.rolling_ball`` releases the GIL, so this is a real
    speedup (measured ~7x on a multi-core box) with byte-identical output.
    """
    if n <= 1:
        for z in range(n):
            fn(z)
        return
    workers = min(n, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, range(n)))


def _subtract_rolling_ball(
    plane: np.ndarray, radius: float, *, fast: bool, out_dtype: Any | None
) -> np.ndarray:
    """Background-subtract one 2D plane. ``fast`` estimates the background on a
    downsampled copy for a large speedup — accurate when the background is smooth
    (the usual case for uneven illumination), approximate otherwise.
    ``out_dtype=None`` keeps the float result (the original 2D behaviour); a dtype
    casts back (the original 3D behaviour)."""
    from skimage.restoration import rolling_ball

    if not fast:
        result = plane - rolling_ball(plane, radius=radius)
        return result if out_dtype is None else result.astype(out_dtype)

    from skimage.transform import resize

    factor = _RB_DOWNSAMPLE
    p = plane.astype(np.float32, copy=False)
    small_shape = (
        max(1, plane.shape[0] // factor),
        max(1, plane.shape[1] // factor),
    )
    small = resize(p, small_shape, order=1, anti_aliasing=True, preserve_range=True)
    bg_small = rolling_ball(small, radius=max(1.0, radius / factor))
    bg = resize(bg_small, plane.shape, order=1, preserve_range=True)
    result = p - bg
    if out_dtype is None:
        return result
    return np.clip(result, 0, None).astype(out_dtype)


def _add_image(
    base_layer, data: np.ndarray, suffix: str, **kwargs: Any
) -> dict[str, Any]:
    name = f"{base_layer.name}_{suffix}"
    new = call_on_main(
        add_image_from_worker,
        data,
        name=name,
        scale=tuple(base_layer.scale),
        metadata={**dict(getattr(base_layer, "metadata", {}) or {}), **kwargs},
    )
    return {
        "new_layer": new.name,
        "shape": tuple(int(s) for s in new.data.shape),
        "dtype": str(new.data.dtype),
    }


@tool(
    description="Subtract rolling-ball background per Z-slice. Reduces uneven "
    "illumination before segmentation. Larger radius for larger structures. "
    "Set fast=True to estimate the background on a downsampled copy — much faster, "
    "accurate when the background is smooth (typical uneven illumination), "
    "approximate for high-frequency backgrounds.",
    phase="2",
    worker=True,
)
def rolling_ball_background(
    layer: str, radius: float = 50.0, fast: bool = False
) -> dict[str, Any]:
    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)

    if data.ndim == 2:
        out = _subtract_rolling_ball(data, radius, fast=fast, out_dtype=None)
    elif data.ndim == 3:
        out = np.empty_like(data)
        _run_over_planes(
            lambda z: out.__setitem__(
                z, _subtract_rolling_ball(data[z], radius, fast=fast, out_dtype=data.dtype)
            ),
            data.shape[0],
        )
    else:
        raise ValueError(f"Expected 2D or 3D layer, got shape {data.shape}")

    return _add_image(L, out, "rb", op="rolling_ball", radius=radius, fast=fast)


@tool(
    description="Rescale intensity to (low_pct, high_pct) percentiles → [0, 1] float. "
    "Improves contrast and normalizes across acquisitions.",
    phase="2",
    worker=True,
)
def auto_contrast(
    layer: str, low_pct: float = 1.0, high_pct: float = 99.0
) -> dict[str, Any]:
    from skimage.exposure import rescale_intensity

    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    lo, hi = np.percentile(data, (low_pct, high_pct))
    out = rescale_intensity(data, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(
        np.float32
    )
    return _add_image(L, out, "ac", op="auto_contrast", percentiles=(low_pct, high_pct))


@tool(
    description="Apply Gaussian smoothing. Reduces noise for cleaner segmentation. "
    "Sigma in pixels (use ~1-2 for fine structures, ~3-5 for cells).",
    phase="2",
    worker=True,
)
def gaussian_denoise(layer: str, sigma: float = 1.0) -> dict[str, Any]:
    from skimage.filters import gaussian

    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    out = gaussian(data, sigma=sigma, preserve_range=True).astype(data.dtype)
    return _add_image(L, out, "gauss", op="gaussian_denoise", sigma=sigma)
