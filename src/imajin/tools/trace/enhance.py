from __future__ import annotations

from typing import Any

import numpy as np

from imajin import session as state
from imajin.agent.qt_dispatch import call_on_main
from imajin.tools._trace_image import (
    _component_labels,
    _materialize,
    _normalize_image,
    _rolling_ball_subtract,
)
from imajin.tools._trace_tables import _scale_is_physical, _scale_tuple
from imajin.tools.napari_ops import (
    add_image_from_worker,
    add_labels_from_worker,
    snapshot_layer,
)
from imajin.tools.registry import tool


@tool(
    description="Enhance a 2D/3D neural process image before segmentation. Methods: "
    "tubeness/sato, frangi, gaussian, or none. Optionally subtract rolling-ball "
    "background and percentile-normalize. Adds a new image layer without mutating raw data.",
    phase="6B",
    subagent="neural_tracer",
    worker=True,
)
def enhance_neural_processes(
    layer: str,
    method: str = "tubeness",
    sigma: float | tuple[float, ...] | None = None,
    background: str | None = "rolling_ball",
    normalize: bool = True,
) -> dict[str, Any]:
    from skimage.filters import frangi, gaussian, sato

    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data).astype(np.float32, copy=False)
    if data.ndim not in (2, 3):
        raise ValueError(f"enhance_neural_processes expects 2D or 3D data, got {data.shape}")

    out = data
    if background in {"rolling_ball", "rolling-ball"}:
        out = _rolling_ball_subtract(out, radius=50.0)
    elif background not in {None, "none", ""}:
        raise ValueError("background must be rolling_ball, none, or None")

    method_key = method.lower().strip()
    if sigma is None:
        sigmas = (1.0, 2.0, 3.0)
        gaussian_sigma: float | tuple[float, ...] = 1.0
    elif isinstance(sigma, (list, tuple)):
        sigmas = tuple(float(s) for s in sigma)
        gaussian_sigma = tuple(float(s) for s in sigma)
    else:
        sigmas = (float(sigma),)
        gaussian_sigma = float(sigma)

    if method_key in {"none", "raw"}:
        enhanced = out
    elif method_key in {"gaussian", "denoise"}:
        enhanced = gaussian(out, sigma=gaussian_sigma, preserve_range=True)
    elif method_key in {"tubeness", "sato"}:
        enhanced = sato(out, sigmas=sigmas, black_ridges=False)
    elif method_key in {"frangi", "vesselness"}:
        enhanced = frangi(out, sigmas=sigmas, black_ridges=False)
    else:
        raise ValueError("method must be tubeness, sato, frangi, vesselness, gaussian, or none")

    enhanced = enhanced.astype(np.float32, copy=False)
    if normalize:
        enhanced = _normalize_image(enhanced)

    new = call_on_main(
        add_image_from_worker,
        enhanced,
        name=f"{L.name}_neural_enhanced",
        scale=tuple(L.scale),
        metadata={
            **dict(L.metadata or {}),
            "source_layer": L.name,
            "op": "enhance_neural_processes",
            "method": method_key,
            "sigma": sigma,
            "background": background,
            "normalize": normalize,
        },
        colormap="gray",
    )
    return {
        "new_layer": new.name,
        "shape": tuple(int(s) for s in enhanced.shape),
        "dtype": str(enhanced.dtype),
        "scale": tuple(float(s) for s in L.scale),
        "method": method_key,
    }


@tool(
    description="Threshold a 2D/3D enhanced neural process image into connected process "
    "labels. Threshold may be otsu, yen, triangle, local/adaptive, or a numeric scalar.",
    phase="6B",
    subagent="neural_tracer",
    worker=True,
)
def segment_neural_processes(
    layer: str,
    threshold: str | float = "otsu",
    min_size_um3: float | None = None,
    fill_holes: bool = False,
    keep_largest: bool = False,
) -> dict[str, Any]:
    from scipy import ndimage as ndi
    from skimage.filters import threshold_local, threshold_otsu, threshold_triangle, threshold_yen
    from skimage.morphology import remove_small_objects

    L = call_on_main(snapshot_layer, layer)
    data = _materialize(L.data)
    if data.ndim not in (2, 3):
        raise ValueError(f"segment_neural_processes expects 2D or 3D data, got {data.shape}")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("cannot segment an image with no finite pixels")

    mode = str(threshold).lower().strip()
    if isinstance(threshold, (int, float)):
        thr = float(threshold)
        mask = data > thr
        threshold_value: float | str = thr
    elif mode == "otsu":
        thr = float(threshold_otsu(finite))
        mask = data > thr
        threshold_value = thr
    elif mode == "yen":
        thr = float(threshold_yen(finite))
        mask = data > thr
        threshold_value = thr
    elif mode == "triangle":
        thr = float(threshold_triangle(finite))
        mask = data > thr
        threshold_value = thr
    elif mode in {"local", "adaptive"}:
        block = max(3, min(31, *(s for s in data.shape[-2:])))
        if block % 2 == 0:
            block -= 1
        if data.ndim == 2:
            local = threshold_local(data, block_size=block)
            mask = data > local
        else:
            mask = np.stack(
                [plane > threshold_local(plane, block_size=block) for plane in data],
                axis=0,
            )
        threshold_value = f"local_block_{block}"
    else:
        try:
            thr = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "threshold must be otsu, yen, triangle, local/adaptive, or numeric"
            ) from exc
        mask = data > thr
        threshold_value = thr

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    spacing = _scale_tuple(tuple(L.scale), data.ndim)
    min_size_pixels = 0
    if min_size_um3 is not None and min_size_um3 > 0:
        voxel_measure = float(np.prod(spacing)) if _scale_is_physical(spacing) else 1.0
        min_size_pixels = max(1, int(np.ceil(float(min_size_um3) / voxel_measure)))
        try:
            mask = remove_small_objects(
                mask.astype(bool), max_size=max(0, min_size_pixels - 1)
            )
        except TypeError:
            mask = remove_small_objects(mask.astype(bool), min_size=min_size_pixels)

    labels, component_count = _component_labels(mask)
    if keep_largest and component_count > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))
        labels = (labels == largest).astype(np.int32)
        component_count = 1
    else:
        labels = labels.astype(np.int32)

    foreground_fraction = float(np.count_nonzero(labels) / labels.size)
    warnings: list[str] = []
    if foreground_fraction < 0.0005:
        warnings.append("Foreground fraction is very low; threshold may be too stringent.")
    if foreground_fraction > 0.35:
        warnings.append("Foreground fraction is high for sparse neural process tracing.")
    if component_count == 0:
        warnings.append("No connected process components were found.")

    new = call_on_main(
        add_labels_from_worker,
        labels,
        name=f"{L.name}_process_mask",
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "op": "segment_neural_processes",
            "threshold": threshold,
            "threshold_value": threshold_value,
            "min_size_pixels": min_size_pixels,
            "foreground_fraction": foreground_fraction,
            "component_count": component_count,
            "warnings": warnings,
        },
    )
    status = "fail" if component_count == 0 else ("warning" if warnings else "pass")
    state.put_qc_record(
        new.name,
        status=status,  # type: ignore[arg-type]
        warnings=warnings,
        metrics={
            "kind": "neural_process_segmentation",
            "foreground_fraction": foreground_fraction,
            "component_count": component_count,
            "threshold_value": threshold_value,
        },
    )
    return {
        "mask_layer": new.name,
        "shape": tuple(int(s) for s in labels.shape),
        "foreground_fraction": foreground_fraction,
        "component_count": component_count,
        "threshold_value": threshold_value,
        "min_size_pixels": min_size_pixels,
        "warnings": warnings,
        "qc_status": status,
    }


