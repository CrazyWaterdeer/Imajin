from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.state import get_layer, get_viewer
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


def _add_image(
    base_layer, data: np.ndarray, suffix: str, **kwargs: Any
) -> dict[str, Any]:
    viewer = get_viewer()
    name = f"{base_layer.name}_{suffix}"
    new = viewer.add_image(
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
    "illumination before segmentation. Larger radius for larger structures.",
    phase="2",
)
def rolling_ball_background(layer: str, radius: float = 50.0) -> dict[str, Any]:
    from skimage.restoration import rolling_ball

    L = get_layer(layer)
    data = _materialize(L.data)

    if data.ndim == 2:
        bg = rolling_ball(data, radius=radius)
        out = data - bg
    elif data.ndim == 3:
        out = np.empty_like(data)
        for z in range(data.shape[0]):
            bg = rolling_ball(data[z], radius=radius)
            out[z] = data[z] - bg
    else:
        raise ValueError(f"Expected 2D or 3D layer, got shape {data.shape}")

    return _add_image(L, out, "rb", op="rolling_ball", radius=radius)


@tool(
    description="Rescale intensity to (low_pct, high_pct) percentiles → [0, 1] float. "
    "Improves contrast and normalizes across acquisitions.",
    phase="2",
)
def auto_contrast(
    layer: str, low_pct: float = 1.0, high_pct: float = 99.0
) -> dict[str, Any]:
    from skimage.exposure import rescale_intensity

    L = get_layer(layer)
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
)
def gaussian_denoise(layer: str, sigma: float = 1.0) -> dict[str, Any]:
    from skimage.filters import gaussian

    L = get_layer(layer)
    data = _materialize(L.data)
    out = gaussian(data, sigma=sigma, preserve_range=True).astype(data.dtype)
    return _add_image(L, out, "gauss", op="gaussian_denoise", sigma=sigma)
