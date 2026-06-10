from __future__ import annotations

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.session import get_layer


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from skimage.measure import label

    labeled = label(mask.astype(bool), connectivity=1)
    return labeled.astype(np.int32), int(labeled.max())


def _layer_kind(layer_name: str) -> str:
    try:
        layer = call_on_main(get_layer, layer_name)
    except Exception:
        return ""
    kind = getattr(layer, "kind", None)
    if isinstance(kind, str):
        return kind.lower()
    return type(layer).__name__.lower()


def _binary_from_layer_data(
    data: np.ndarray,
    *,
    layer_name: str,
    threshold: float | None,
) -> np.ndarray:
    kind = _layer_kind(layer_name)
    if "label" in kind:
        return data > 0
    if threshold is not None:
        return data > float(threshold)
    finite = data[np.isfinite(data)]
    unique = np.unique(finite)
    if unique.size <= 2 and set(unique.tolist()).issubset({0, 1, False, True}):
        return data.astype(bool)
    raise ValueError(
        "skeletonize expects a binary/Labels layer. For continuous image data, "
        "run segment_neural_processes first or pass an explicit threshold."
    )


def _normalize_image(data: np.ndarray) -> np.ndarray:
    from skimage.exposure import rescale_intensity

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return rescale_intensity(data, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def _rolling_ball_subtract(data: np.ndarray, radius: float = 50.0) -> np.ndarray:
    from skimage.restoration import rolling_ball

    if data.ndim == 2:
        return data - rolling_ball(data, radius=radius)
    if data.ndim == 3:
        out = np.empty_like(data)
        for z in range(data.shape[0]):
            out[z] = data[z] - rolling_ball(data[z], radius=radius)
        return out
    raise ValueError(f"Expected 2D or 3D layer, got shape {data.shape}")
