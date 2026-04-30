from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.state import get_layer
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


def _resolve_threshold(arr: np.ndarray, threshold: float | str) -> float:
    if isinstance(threshold, (int, float)):
        return float(threshold)
    if isinstance(threshold, str):
        if threshold == "otsu":
            from skimage.filters import threshold_otsu

            return float(threshold_otsu(arr))
        if threshold == "zero":
            return 0.0
    raise ValueError(f"unsupported threshold spec: {threshold!r}")


@tool(
    description="Manders' colocalization coefficients M1/M2 between two image layers. "
    "M1 = fraction of channel A intensity that overlaps non-zero channel B. M2 is the "
    "reciprocal. Optionally restrict to a Labels layer mask. Threshold accepts a scalar "
    "or 'otsu' / 'zero'.",
    phase="4",
)
def manders_coefficients(
    image_a: str,
    image_b: str,
    mask: str | None = None,
    threshold_a: float | str = "otsu",
    threshold_b: float | str = "otsu",
) -> dict[str, Any]:
    a = _materialize(get_layer(image_a).data).astype(np.float64)
    b = _materialize(get_layer(image_b).data).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {image_a} {a.shape} vs {image_b} {b.shape}")

    if mask:
        m = _materialize(get_layer(mask).data) > 0
        if m.shape != a.shape:
            raise ValueError(f"mask shape mismatch: {mask} {m.shape} vs {a.shape}")
    else:
        m = np.ones_like(a, dtype=bool)

    ta = _resolve_threshold(a[m], threshold_a)
    tb = _resolve_threshold(b[m], threshold_b)

    a_in = a[m]
    b_in = b[m]
    a_above = a_in > ta
    b_above = b_in > tb

    sum_a = a_in.sum()
    sum_b = b_in.sum()
    m1 = float(a_in[b_above].sum() / sum_a) if sum_a > 0 else 0.0
    m2 = float(b_in[a_above].sum() / sum_b) if sum_b > 0 else 0.0

    return {
        "M1": m1,
        "M2": m2,
        "threshold_a": ta,
        "threshold_b": tb,
        "n_pixels": int(m.sum()),
        "image_a": image_a,
        "image_b": image_b,
    }


@tool(
    description="Pearson correlation r between two image layers, optionally restricted "
    "to a Labels layer mask. Use when both channels have continuous intensity "
    "distributions (vs Manders for thresholded colocalization).",
    phase="4",
)
def pearson_correlation(
    image_a: str, image_b: str, mask: str | None = None
) -> dict[str, Any]:
    a = _materialize(get_layer(image_a).data).astype(np.float64)
    b = _materialize(get_layer(image_b).data).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {image_a} {a.shape} vs {image_b} {b.shape}")

    if mask:
        m = _materialize(get_layer(mask).data) > 0
        if m.shape != a.shape:
            raise ValueError(f"mask shape mismatch: {mask} {m.shape} vs {a.shape}")
        a = a[m]
        b = b[m]
    else:
        a = a.ravel()
        b = b.ravel()

    if a.size < 2:
        return {"r": 0.0, "n_pixels": int(a.size), "image_a": image_a, "image_b": image_b}

    if a.std() == 0 or b.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(a, b)[0, 1])

    return {
        "r": r,
        "n_pixels": int(a.size),
        "image_a": image_a,
        "image_b": image_b,
    }
