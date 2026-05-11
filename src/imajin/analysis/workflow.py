from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np


_VALID_PREPROCESS = {
    "rolling_ball": "rolling_ball",
    "rb": "rolling_ball",
    "background": "rolling_ball",
    "auto_contrast": "auto_contrast",
    "ac": "auto_contrast",
    "contrast": "auto_contrast",
    "gaussian": "gaussian_denoise",
    "gauss": "gaussian_denoise",
    "denoise": "gaussian_denoise",
}


_TARGET_OBJECT_ALIASES = {
    "target",
    "target_object",
    "target_objects",
    "segment_target_object",
    "segment_target_objects",
    "objects",
    "rois",
}
_CELLPOSE_SAM_ALIASES = {"cellpose", "cpsam", "cellpose_sam"}
_INTENSITY_REGION_ALIASES = {
    "intensity",
    "intensity_region",
    "intensity_regions",
    "segment_intensity_region",
    "segment_intensity_regions",
    "roi",
}
_NOISE_FLOOR_ALIASES = {
    "noise_floor",
    "noisefloor",
    "expression_domain",
    "expression",
    "expression_region",
}


def normalize_preprocess(name: str | None) -> str | None:
    if name is None:
        return None
    key = name.strip().lower().replace("-", "_")
    if not key or key in {"none", "off"}:
        return None
    if key not in _VALID_PREPROCESS:
        raise ValueError(
            f"unknown preprocess step {name!r}. Use one of: rolling_ball, "
            "auto_contrast, gaussian_denoise, or None."
        )
    return _VALID_PREPROCESS[key]


def normalize_segmentation_method(method: str) -> str:
    """Map user-facing segmentation method names to the canonical Tier-2 name."""

    key = str(method).strip().lower().replace("-", "_")
    if key in _TARGET_OBJECT_ALIASES:
        return "target_objects"
    if key in _CELLPOSE_SAM_ALIASES:
        return "cellpose_sam"
    if key in _INTENSITY_REGION_ALIASES:
        return "intensity_regions"
    raise ValueError(
        "segmentation_method must be 'target_objects', 'cellpose_sam', "
        f"or 'intensity_regions' (got {method!r}). "
        "Tier-1 domain steps like 'expression_domain' belong in the "
        "`domain` slot, not `segmentation`."
    )


def normalize_domain_spec(
    domain: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Translate a recipe `domain` dict to (domain_strategy, domain_options)."""

    if not domain:
        return None, None
    raw = domain.get("strategy") or domain.get("method")
    if raw is None:
        raise ValueError(
            "recipe.domain must include 'strategy' (e.g. 'noise_floor') "
            "or 'method' (e.g. 'expression_domain')."
        )
    key = str(raw).strip().lower().replace("-", "_")
    if key not in _NOISE_FLOOR_ALIASES:
        raise ValueError(
            f"recipe.domain strategy must be 'noise_floor' (got {raw!r})."
        )
    options = {k: v for k, v in domain.items() if k not in {"strategy", "method"}}
    return "noise_floor", options


def decide_3d(do_3d: bool | None, layer_axes: str | None, ndim: int) -> bool:
    if do_3d is True:
        return True
    if do_3d is False:
        return False
    if layer_axes and "Z" in layer_axes and "T" not in layer_axes:
        return True
    return ndim == 3


def rss_mb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="utf-8") as fh:
            fields = fh.read().split()
        if len(fields) >= 2:
            return int(fields[1]) * int(page_size) / 1024**2
    except Exception:
        return None
    return None


def release_worker_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def array_nbytes(data: Any) -> int | None:
    nbytes = getattr(data, "nbytes", None)
    if isinstance(nbytes, (int, np.integer)):
        return int(nbytes)
    shape = getattr(data, "shape", None)
    dtype = getattr(data, "dtype", None)
    if shape is None or dtype is None:
        return None
    try:
        return int(np.prod(tuple(int(s) for s in shape), dtype=np.int64)) * int(
            np.dtype(dtype).itemsize
        )
    except Exception:
        return None


def check_analysis_memory_budget(
    layer_name: str,
    *,
    data: Any,
    method: str,
) -> None:
    """Fail before known high-peak 3D paths can push the process into OOM kill."""

    nbytes = array_nbytes(data)
    if nbytes is None:
        return
    try:
        from imajin.io.memory import available_memory_bytes

        available = available_memory_bytes()
    except Exception:
        available = None
    if available is None:
        return

    minimal_multiplier = {
        "target_objects": 4.0,
        "intensity_regions": 2.5,
        "cellpose_sam": 4.0,
    }.get(method, 4.0)
    minimal_required = int(nbytes * minimal_multiplier) + 256 * 1024**2
    if minimal_required <= available:
        return
    raise MemoryError(
        f"analysis of layer {layer_name!r} is likely to exceed available RAM "
        f"(input ~{nbytes / 1024**2:.0f} MiB, minimum working set "
        f"~{minimal_required / 1024**2:.0f} MiB, available "
        f"~{available / 1024**2:.0f} MiB). Use a mean/max projection recipe, "
        "crop/downsample first, or increase WSL memory before running 3D analysis."
    )


def normalize_match_text(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def projection_from_step(step: dict[str, Any]) -> tuple[str | None, Any]:
    raw = step.get("step") or step.get("op") or step.get("tool") or step.get("name")
    if raw is None:
        return None, None
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in {"average_projection", "avg_projection", "mean_projection"}:
        return "mean", step.get("axis", "z")
    if mode in {"max_projection", "mip", "maximum_projection"}:
        return "max", step.get("axis", "z")
    return None, None


def projection_request(
    measurement: dict[str, Any],
    preprocessing: list[dict[str, Any]],
) -> tuple[str | None, Any]:
    raw = measurement.get("projection")
    if raw is not None:
        mode = str(raw).strip().lower().replace("-", "_")
        if mode in {"", "none", "off", "false"}:
            return None, None
        if mode in {"mean", "avg", "average", "average_projection"}:
            return "mean", measurement.get("axis", "z")
        if mode in {"max", "mip", "maximum", "max_projection"}:
            return "max", measurement.get("axis", "z")
        raise ValueError("measurement.projection must be mean, max, or none")

    for step in preprocessing:
        projection, axis = projection_from_step(step)
        if projection is not None:
            return projection, axis
    return None, None


def first_analysis_preprocess(preprocessing: list[dict[str, Any]]) -> str | None:
    for step in preprocessing:
        projection, _axis = projection_from_step(step)
        if projection is not None:
            continue
        raw = step.get("step") or step.get("op") or step.get("tool") or step.get("name")
        if raw:
            return str(raw)
    return None


def derive_size_params(
    cell_diameter_um: float | None,
    voxel_spacing: tuple[float, ...] | None,
) -> dict[str, float]:
    if cell_diameter_um is None or cell_diameter_um <= 0:
        return {}
    out: dict[str, float] = {
        "min_distance_um": float(cell_diameter_um) * 0.7,
        "min_area_um2": float(np.pi * (cell_diameter_um / 4.0) ** 2),
    }
    if voxel_spacing is not None:
        xy = voxel_spacing[-1]
        if xy and xy > 0:
            out["cellpose_diameter_px"] = float(cell_diameter_um) / float(xy)
    return out


def build_sample_summary(
    *,
    sample_name: str,
    status: str,
    error: str | None = None,
    n_cells: int | None = None,
    n_domain_components: int | None = None,
    domain_label_count: int | None = None,
    domain_area_um2: float | None = None,
    domain_volume_um3: float | None = None,
    domain_voxels: int | None = None,
    qc_warnings: list[str] | None = None,
    outputs: dict[str, str | None] | None = None,
    group: str | None = None,
    file_id: str | None = None,
    source_file: str | None = None,
    source_layer: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_name": sample_name,
        "group": group,
        "file_id": file_id,
        "source_file": source_file,
        "source_layer": source_layer,
        "status": status,
        "error": error,
        "outputs": outputs or {"labels_cells": None, "labels_domain": None, "qc_png": None},
        "summary": {
            "n_cells": n_cells,
            "n_domain_components": n_domain_components,
            "domain_label_count": domain_label_count,
            "domain_area_um2": domain_area_um2,
            "domain_volume_um3": domain_volume_um3,
            "domain_voxels": domain_voxels,
            "qc_warnings": list(qc_warnings or []),
        },
    }
