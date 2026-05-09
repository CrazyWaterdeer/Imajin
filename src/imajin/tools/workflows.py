from __future__ import annotations

import gc
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np

from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import (
    AmbiguousChannelError,
    resolve_target_channel,
)
from imajin.tools import measure as _measure
from imajin.tools import preprocess as _preprocess
from imajin.tools import segment as _segment
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool
from imajin.workers.qt_worker import CancelledError


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


def _normalize_preprocess(name: str | None) -> str | None:
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


def _normalize_segmentation_method(method: str) -> str:
    """Map user-facing segmentation method names to the canonical Tier-2 name.

    Raises ValueError if the name is not a Tier-2 segmentation method (e.g.
    'expression_domain', which is a Tier-1 domain step).
    """
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


def _normalize_domain_spec(
    domain: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Translate a recipe `domain` dict to (domain_strategy, domain_options).

    Accepts either {"strategy": "noise_floor", ...} or
    {"method": "expression_domain", ...}; both map to the canonical
    domain_strategy='noise_floor'. Remaining keys flow through as
    domain_options.
    """
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


def _decide_3d(do_3D: bool | None, layer_axes: str | None, ndim: int) -> bool:
    if do_3D is True:
        return True
    if do_3D is False:
        return False
    if layer_axes and "Z" in layer_axes and "T" not in layer_axes:
        return True
    return ndim == 3


def _layer_axes(snapshot: Any) -> str | None:
    md = snapshot.metadata or {}
    axes = md.get("axes") if isinstance(md, dict) else None
    if isinstance(axes, str):
        return axes.replace("C", "")
    return None


def _filtered_kwargs(func: Any, options: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(func).parameters
    return {key: value for key, value in options.items() if key in params}


def _rss_mb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="utf-8") as fh:
            fields = fh.read().split()
        if len(fields) >= 2:
            return int(fields[1]) * int(page_size) / 1024**2
    except Exception:
        return None
    return None


def _release_worker_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _array_nbytes(data: Any) -> int | None:
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


def _check_analysis_memory_budget(
    layer_name: str,
    *,
    data: Any,
    method: str,
) -> None:
    """Fail before known high-peak 3D paths can push the process into OOM kill."""
    nbytes = _array_nbytes(data)
    if nbytes is None:
        return
    try:
        from imajin.io.memory import available_memory_bytes

        available = available_memory_bytes()
    except Exception:
        available = None
    if available is None:
        return

    # This is only a hard guard for obviously impossible cases. It must not block
    # a single z-stack that the user could previously process; the batch cleanup
    # path handles the accumulation problem separately.
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


def _viewer_layer_names() -> list[str]:
    from imajin.agent.state import get_viewer

    viewer = get_viewer()
    return [str(layer.name) for layer in viewer.layers]


def _remove_layers_by_name(layer_names: list[str]) -> list[str]:
    from imajin.agent.state import get_viewer

    viewer = get_viewer()
    removed: list[str] = []
    for name in reversed(layer_names):
        try:
            layer = viewer.layers[name]
        except Exception:
            continue
        try:
            viewer.layers.remove(layer)
            removed.append(name)
        except Exception:
            try:
                viewer.layers.remove(name)
                removed.append(name)
            except Exception:
                continue
    return removed


def _cleanup_new_layers(base_layer_names: set[str]) -> list[str]:
    return _cleanup_sample_layers(base_layer_names, [])


def _cleanup_sample_layers(
    base_layer_names: set[str],
    managed_layer_names: list[str],
) -> list[str]:
    current = _viewer_layer_names()
    current_set = set(current)
    created = [name for name in current if name not in base_layer_names]
    managed = [name for name in managed_layer_names if name in current_set]
    to_remove = list(dict.fromkeys([*created, *managed]))
    return call_on_main(_remove_layers_by_name, to_remove)


def _normalize_match_text(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def _loaded_layer_metadata_text(layer: Any) -> str:
    md = getattr(layer, "metadata", {}) or {}
    parts = [getattr(layer, "name", "")]
    if isinstance(md, dict):
        for key in ("name", "channel_name", "marker", "color"):
            if key in md and md[key] is not None:
                parts.append(str(md[key]))
    try:
        from imajin.agent import state as _state

        channel_info = _state._layer_channel_metadata(layer)
    except Exception:
        channel_info = {}
    if isinstance(channel_info, dict):
        for key in (
            "name",
            "channel_name",
            "marker",
            "color",
            "display_color_name",
            "dye_name",
            "excitation_wavelength_nm",
            "emission_wavelength_nm",
        ):
            if key in channel_info and channel_info[key] is not None:
                parts.append(str(channel_info[key]))
    return " ".join(parts)


def _resolve_target_within_loaded_layers(
    target: str | None,
    loaded_layers: list[str],
) -> str | None:
    """Resolve a recipe target only against layers loaded for the current sample.

    Batch runs often keep an already-loaded representative image in napari while
    auto-loading each sample. A recipe target like "green" must bind to the
    current sample's loaded green layer, not to an older layer in the viewer.
    """
    if not loaded_layers:
        return target
    current = list(dict.fromkeys(str(name) for name in loaded_layers))
    if target is None:
        return current[0] if len(current) == 1 else None
    if target in current:
        return target

    from imajin.agent.state import canonical_channel_color, get_viewer

    query = _normalize_match_text(target)
    target_color = canonical_channel_color(target)
    viewer = get_viewer()
    matches: list[str] = []
    for layer_name in current:
        try:
            layer = viewer.layers[layer_name]
        except Exception:
            continue
        text = _normalize_match_text(_loaded_layer_metadata_text(layer))
        layer_color = canonical_channel_color(text)
        if query and (query == _normalize_match_text(layer_name) or query in text):
            matches.append(layer_name)
        elif target_color is not None and layer_color == target_color:
            matches.append(layer_name)

    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(current) == 1:
        return current[0]
    return target


def _load_file_for_sample_if_needed(
    info: dict[str, Any],
    *,
    auto_load_files: bool,
) -> dict[str, Any] | None:
    if not auto_load_files or not info.get("file_path"):
        return None
    sample_layers = list(getattr(info["sample"], "layers", []) or [])
    existing = set(call_on_main(_viewer_layer_names))
    if sample_layers and any(layer_name in existing for layer_name in sample_layers):
        return None
    from imajin.tools import files as _files

    return call_on_main(_files.load_file, str(info["file_path"]))


def _projection_from_step(step: dict[str, Any]) -> tuple[str | None, Any]:
    raw = step.get("step") or step.get("op") or step.get("tool") or step.get("name")
    if raw is None:
        return None, None
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in {"average_projection", "avg_projection", "mean_projection"}:
        return "mean", step.get("axis", "z")
    if mode in {"max_projection", "mip", "maximum_projection"}:
        return "max", step.get("axis", "z")
    return None, None


def _projection_request(
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
        projection, axis = _projection_from_step(step)
        if projection is not None:
            return projection, axis
    return None, None


def _first_analysis_preprocess(preprocessing: list[dict[str, Any]]) -> str | None:
    for step in preprocessing:
        projection, _axis = _projection_from_step(step)
        if projection is not None:
            continue
        raw = step.get("step") or step.get("op") or step.get("tool") or step.get("name")
        if raw:
            return str(raw)
    return None


def _derive_size_params(
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


def _project_layer_for_recipe(
    layer_name: str,
    *,
    projection: str | None,
    axis: Any,
) -> dict[str, Any] | None:
    if projection is None:
        return None
    from imajin.tools import view as _view

    if projection == "mean":
        return _view.average_projection(layer_name, axis=axis)
    if projection == "max":
        return _view.max_projection(layer_name, axis=axis)
    raise ValueError("projection must be mean or max")


def _build_sample_summary(
    *,
    sample_name: str,
    status: str,
    error: str | None = None,
    n_cells: int | None = None,
    n_domain_components: int | None = None,
    domain_area_um2: float | None = None,
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
            "domain_area_um2": domain_area_um2,
            "qc_warnings": list(qc_warnings or []),
        },
    }


@tool(
    description="High-level target object analysis workflow. Resolves the target "
    "channel, optionally preprocesses it, segments target-positive objects/ROIs, "
    "and measures per-object intensity and size on the same target channel. Pass "
    "target as a layer name, color (green/red/UV/IR), or marker (GFP/DAPI). Pass preprocess="
    "'rolling_ball' / 'auto_contrast' / 'gaussian_denoise' to apply one preprocessing "
    "step before segmentation. Returns labels layer, measurement table, object count, "
    "and QC metrics. Counterstain channels are not auto-selected — annotate them as "
    "'counterstain' first if you only have one target channel.",
    phase="2",
    worker=True,
)
def analyze_target_cells(
    target: str | None = None,
    do_3D: bool | None = None,
    diameter: float | None = None,
    preprocess: str | None = None,
    segmentation_method: str = "target_objects",
    segmentation_options: dict[str, Any] | None = None,
    domain_strategy: str | None = None,
    domain_options: dict[str, Any] | None = None,
    counterstain_layer: str | None = None,
    cell_diameter_um: float | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    report_progress(stage="resolve_target", message="Resolving target channel.")
    try:
        resolution = resolve_target_channel(target)
    except AmbiguousChannelError as e:
        return {
            "ok": False,
            "error": str(e),
            "candidates": list(e.candidates),
            "stage": "resolve_target",
        }

    target_layer = resolution.layer
    if resolution.source == "inference":
        warnings.append(
            f"single image layer ({target_layer}) was assumed as target — "
            "confirm by annotating the channel."
        )

    seg_input_layer = target_layer
    pre_step = _normalize_preprocess(preprocess)
    pre_record: dict[str, Any] | None = None
    if pre_step == "rolling_ball":
        report_progress(stage="preprocess", message=f"Preprocessing {target_layer}.")
        raise_if_cancelled()
        pre_record = _preprocess.rolling_ball_background(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]
    elif pre_step == "auto_contrast":
        report_progress(stage="preprocess", message=f"Preprocessing {target_layer}.")
        raise_if_cancelled()
        pre_record = _preprocess.auto_contrast(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]
    elif pre_step == "gaussian_denoise":
        report_progress(stage="preprocess", message=f"Preprocessing {target_layer}.")
        raise_if_cancelled()
        pre_record = _preprocess.gaussian_denoise(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]

    snapshot = call_on_main(snapshot_layer, seg_input_layer)
    axes = _layer_axes(snapshot)
    use_3d = _decide_3d(do_3D, axes, getattr(snapshot.data, "ndim", 2))

    # Tier-1: pre-compute expression domain before Tier-2 segmentation.
    pre_computed_domain_layer: str | None = None
    if domain_strategy is not None:
        from imajin.tools import channels as _channels_pre
        from imajin.tools.segment import segment_expression_domain as _seg_dom
        cs_layer_pre = counterstain_layer
        cs_is_nuclear_pre: bool | None = None
        if cs_layer_pre is None:
            cs_info_pre = _channels_pre.detect_counterstain_channel()
            if cs_info_pre["confidence"] == "annotated":
                cs_layer_pre = cs_info_pre["counterstain_layer"]
                cs_is_nuclear_pre = cs_info_pre["is_nuclear"]
        d_opts = dict(domain_options or {})
        if cs_layer_pre:
            d_opts.setdefault("counterstain_layer", cs_layer_pre)
            d_opts.setdefault("is_nuclear", cs_is_nuclear_pre)
        derived_pre = _derive_size_params(
            cell_diameter_um,
            _segment._voxel_spacing(tuple(snapshot.scale), getattr(snapshot.data, "ndim", 2)),
        )
        if "min_area_um2" not in d_opts and "min_area_um2" in derived_pre:
            d_opts["min_area_um2"] = derived_pre["min_area_um2"]
        domain_pre = _seg_dom(
            image_layer=target_layer,
            **_filtered_kwargs(_seg_dom, d_opts),
        )
        pre_computed_domain_layer = domain_pre["labels_layer"]

    method = _normalize_segmentation_method(segmentation_method)

    _check_analysis_memory_budget(seg_input_layer, data=snapshot.data, method=method)

    report_progress(stage="segmentation", message=f"Segmenting {seg_input_layer}.")
    raise_if_cancelled()
    seg_options = dict(segmentation_options or {})
    seg_options.pop("tool", None)
    seg_options.pop("do_3D", None)
    seg_options.pop("diameter", None)
    if pre_computed_domain_layer is not None:
        seg_options.setdefault("boundary_mask", pre_computed_domain_layer)
    if method == "target_objects":
        seg_result = _segment.segment_target_objects(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.segment_target_objects, seg_options),
        )
    elif method == "intensity_regions":
        seg_result = _segment.segment_intensity_regions(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.segment_intensity_regions, seg_options),
        )
    else:
        cellpose_options = {
            **seg_options,
            "do_3D": use_3d,
            "diameter": diameter,
        }
        seg_result = _segment.cellpose_sam(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.cellpose_sam, cellpose_options),
        )
    if seg_result.get("empty_mask", False):
        return {
            "ok": False,
            "stage": "segment",
            "error": f"{method} produced zero objects on the target channel; "
            "no measurements were taken. Try a different channel, a preprocess "
            "step (rolling_ball / auto_contrast / gaussian_denoise), or set a "
            "manual diameter.",
            "target_channel": target_layer,
            "target_source": resolution.source,
            "labels_layer": seg_result["labels_layer"],
            "qc_png_path": seg_result.get("qc_png_path"),
            "qc_png_error": seg_result.get("qc_png_error"),
            "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
            "preprocess": pre_step,
            "segmentation_method": method,
            "warnings": warnings,
        }

    report_progress(
        stage="measurement",
        message=f"Measuring {seg_result['labels_layer']}.",
    )
    raise_if_cancelled()
    measure_result = _measure.measure_intensity(
        labels_layer=seg_result["labels_layer"],
        image_layers=[seg_input_layer],
    )
    if not measure_result.get("has_physical_units"):
        warnings.append(
            "no voxel size on the target layer — physical-unit columns were not "
            "added. Annotate or reload with scale information for area_um2 / "
            "volume_um3."
        )

    voxel = measure_result.get("voxel_scale")
    if voxel and len(voxel) == 3 and voxel[0] != voxel[1]:
        warnings.append(
            f"anisotropic voxel spacing (z={voxel[0]:.3g}, y={voxel[1]:.3g}, "
            f"x={voxel[2]:.3g}); 3D segmentation/measurement may be biased."
        )

    bundle_path: Path | None = None
    bundle_outputs: dict[str, str | None] = {
        "labels_cells": None,
        "labels_domain": None,
        "qc_png": None,
    }
    if domain_strategy is None:
        from imajin.results import create_result_bundle, slugify_result_name as _slug
        from imajin.tools.results import (
            current_bundle,
            populate_sample_outputs,
            write_combined_csv,
            finalize_bundle_metadata,
        )

        sample_slug = _slug(target_layer)
        parent = current_bundle()
        own_bundle = parent is None
        if own_bundle:
            bundle_path = create_result_bundle(
                name=target_layer,
                kind="single",
                tier="single_tier",
                metadata={
                    "recipe": None,
                    "target_channel": target_layer,
                    "target_source": resolution.source,
                    "segmentation_method": method,
                    "analysis_dim": "3d" if use_3d else "2d",
                },
            )
        else:
            bundle_path = parent

        try:
            bundle_outputs = populate_sample_outputs(
                bundle_path,
                sample_slug=sample_slug,
                labels_cells=seg_result["labels_layer"],
                qc_png=seg_result.get("qc_png_path"),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"bundle outputs could not be written: {type(exc).__name__}: {exc}"
            )

        if own_bundle:
            sample_summary = _build_sample_summary(
                sample_name=target_layer,
                status="complete",
                n_cells=int(seg_result.get("n_objects", seg_result.get("n_cells", 0))),
                qc_warnings=list(seg_result.get("qc_warnings", [])),
                outputs=bundle_outputs,
                source_layer=target_layer,
            )
            try:
                write_combined_csv(bundle_path, [measure_result["table_name"]])
                finalize_bundle_metadata(
                    bundle_path,
                    samples=[sample_summary],
                    status="complete",
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"bundle could not be finalized: {type(exc).__name__}: {exc}"
                )

    if domain_strategy is not None:
        if domain_strategy != "noise_floor":
            raise ValueError(
                f"domain_strategy must be 'noise_floor' (got {domain_strategy!r})"
            )

        import pandas as _pd
        from imajin.agent.state import get_layer as _get_layer
        domain_layer = pre_computed_domain_layer
        domain_layer_md = dict(getattr(_get_layer(domain_layer), "metadata", {}) or {})
        domain_result = {
            "labels_layer": domain_layer,
            "n_components": int(domain_layer_md.get("n_components", 0)),
            "domain_area_um2": float(domain_layer_md.get("domain_area_um2", 0.0)),
            "counterstain_warnings": list(domain_layer_md.get("counterstain_warnings", [])),
        }

        domain_measure = _measure.measure_intensity(
            labels_layer=domain_layer,
            image_layers=[seg_input_layer],
        )
        domain_table_name = domain_measure["table_name"]
        cells_table_name = measure_result["table_name"]

        from imajin.agent.state import get_table, put_table

        domain_df = get_table(domain_table_name).copy()
        cells_df = get_table(cells_table_name).copy()
        domain_df["tier"] = "domain"
        cells_df["tier"] = "cells"
        combined = _pd.concat([domain_df, cells_df], ignore_index=True, sort=False)

        tier_table_name = put_table(
            f"{target_layer}_two_tier",
            combined,
            spec={
                "tool": "analyze_target_cells",
                "mode": "two_tier",
                "target_channel": target_layer,
                "domain_layer": domain_layer,
                "cells_layer": seg_result["labels_layer"],
            },
        )

        from imajin.results import create_result_bundle, slugify_result_name as _slug
        from imajin.tools.results import (
            current_bundle,
            populate_sample_outputs,
            write_combined_csv,
            finalize_bundle_metadata,
        )

        sample_slug = _slug(target_layer)
        parent = current_bundle()
        own_bundle = parent is None
        if own_bundle:
            bundle_path = create_result_bundle(
                name=target_layer,
                kind="single",
                tier="two_tier",
                metadata={
                    "recipe": None,
                    "target_channel": target_layer,
                    "target_source": resolution.source,
                    "segmentation_method": method,
                    "analysis_dim": "3d" if use_3d else "2d",
                },
            )
        else:
            bundle_path = parent

        bundle_outputs = {
            "labels_cells": None,
            "labels_domain": None,
            "qc_png": None,
        }
        try:
            bundle_outputs = populate_sample_outputs(
                bundle_path,
                sample_slug=sample_slug,
                labels_cells=seg_result["labels_layer"],
                labels_domain=domain_layer,
                qc_png=seg_result.get("qc_png_path"),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"bundle outputs could not be written: {type(exc).__name__}: {exc}"
            )

        if own_bundle:
            sample_summary = _build_sample_summary(
                sample_name=target_layer,
                status="complete",
                n_cells=int(seg_result.get("n_objects", seg_result.get("n_cells", 0))),
                n_domain_components=domain_result["n_components"],
                domain_area_um2=domain_result["domain_area_um2"],
                qc_warnings=(
                    list(seg_result.get("qc_warnings", []))
                    + list(domain_result.get("counterstain_warnings", []))
                ),
                outputs=bundle_outputs,
                source_layer=target_layer,
            )
            try:
                write_combined_csv(bundle_path, [tier_table_name])
                finalize_bundle_metadata(
                    bundle_path,
                    samples=[sample_summary],
                    status="complete",
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"bundle could not be finalized: {type(exc).__name__}: {exc}"
                )

        return {
            "ok": True,
            "target_channel": target_layer,
            "target_source": resolution.source,
            "preprocess": pre_step,
            "preprocessed_layer": pre_record["new_layer"] if pre_record else None,
            "segmentation_method": method,
            "analysis_dim": "3d" if use_3d else "2d",
            "labels_layer": seg_result["labels_layer"],
            "cells_layer": seg_result["labels_layer"],
            "domain_layer": domain_layer,
            "n_domain_components": domain_result["n_components"],
            "domain_area_um2": domain_result["domain_area_um2"],
            "n_cells": int(seg_result.get("n_objects", 0)),
            "tier_table_name": tier_table_name,
            "primary_table_name": tier_table_name,
            "table_name": measure_result["table_name"],
            "table_columns": measure_result["columns"],
            "qc_png_path": seg_result.get("qc_png_path"),
            "qc_png_error": seg_result.get("qc_png_error"),
            "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
            "result_bundle_path": str(bundle_path) if own_bundle else None,
            "result_files": dict(bundle_outputs),
            "voxel_scale": voxel,
            "warnings": warnings + list(domain_result.get("counterstain_warnings", [])),
        }

    return {
        "ok": True,
        "target_channel": target_layer,
        "target_source": resolution.source,
        "preprocess": pre_step,
        "preprocessed_layer": pre_record["new_layer"] if pre_record else None,
        "segmentation_method": method,
        "analysis_dim": "3d" if use_3d else "2d",
        "labels_layer": seg_result["labels_layer"],
        "qc_png_path": seg_result.get("qc_png_path"),
        "qc_png_error": seg_result.get("qc_png_error"),
        "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
        "object_unit": seg_result.get("object_unit", "object_or_roi"),
        "n_objects": int(seg_result.get("n_objects", seg_result.get("n_cells", 0))),
        "do_3D": bool(use_3d),
        "object_area_min": seg_result.get("object_area_min"),
        "object_area_median": seg_result.get("object_area_median"),
        "object_area_max": seg_result.get("object_area_max"),
        "top_bright_outside_fraction": seg_result.get("top_bright_outside_fraction"),
        "mask_fraction": seg_result.get("mask_fraction"),
        "segmentation_warnings": seg_result.get("qc_warnings", []),
        "table_name": measure_result["table_name"],
        "primary_table_name": measure_result["table_name"],
        "table_columns": measure_result["columns"],
        "result_bundle_path": str(bundle_path) if own_bundle else None,
        "result_files": dict(bundle_outputs),
        "voxel_scale": voxel,
        "has_physical_units": bool(measure_result.get("has_physical_units")),
        "warnings": warnings,
    }


def _resolve_sample_inputs(sample_name: str) -> dict[str, Any]:
    """Pick the layer name + file path the recipe should operate on for one sample."""
    from imajin.agent.state import _FILES, get_sample

    s = get_sample(sample_name)
    layer_name = s.layers[0] if s.layers else None
    file_path: str | None = None
    file_id: str | None = None
    if s.file_ids:
        file_id = s.file_ids[0]
        rec = _FILES.get(file_id)
        if rec is not None:
            file_path = rec.path
    elif s.files:
        file_path = s.files[0]
    return {
        "sample": s,
        "layer_name": layer_name,
        "file_path": file_path,
        "file_id": file_id,
    }


@tool(
    description="Apply a stored analysis recipe to one or more annotated samples. "
    "Iterates samples one by one: resolves the target channel/layer, runs the "
    "Phase-2 analyze_target_cells pipeline, attaches sample/group/file columns to "
    "the resulting measurement table, records a per-sample AnalysisRun, and by "
    "default removes layers created for each sample so large batches do not retain "
    "all image volumes in memory. A failure on one sample never aborts the batch.",
    phase="3",
    worker=True,
)
def run_recipe_on_samples(
    recipe_name: str,
    sample_names: list[str] | None = None,
    execution_mode: str = "serial_cleanup",
    auto_load_files: bool = True,
    keep_layers: bool = False,
    keep_failed_layers: bool = False,
) -> dict[str, Any]:
    from imajin.agent.state import (
        attach_sample_columns_to_table,
        get_recipe,
        list_samples,
        put_run,
    )

    recipe = get_recipe(recipe_name)
    if sample_names is None:
        sample_names = [s["sample_name"] for s in list_samples()]
    if not sample_names:
        return {
            "recipe": recipe_name,
            "n_samples": 0,
            "n_complete": 0,
            "n_failed": 0,
            "runs": [],
        }

    mode = execution_mode.strip().lower().replace("-", "_")
    if mode == "parallel_headless":
        raise ValueError(
            "parallel_headless is planned for headless/CLI workers but is not "
            "implemented in the napari GUI runner yet. Use serial_cleanup."
        )
    if mode not in {"serial_cleanup", "cleanup", "serial"}:
        raise ValueError(
            "execution_mode must be 'serial_cleanup', 'serial', or 'parallel_headless'"
        )
    cleanup_enabled = mode in {"serial_cleanup", "cleanup"} and not keep_layers

    seg = recipe.segmentation or {}
    measurement = recipe.measurement or {}
    pre_steps = recipe.preprocessing or []
    pre_choice = _first_analysis_preprocess(pre_steps)
    projection, projection_axis = _projection_request(measurement, pre_steps)
    domain_strategy, domain_options = _normalize_domain_spec(recipe.domain)

    runs: list[dict[str, Any]] = []
    n_complete = 0
    n_failed = 0
    total = len(sample_names)

    from imajin.results import create_result_bundle
    from imajin.tools.results import with_active_bundle

    parent_bundle = create_result_bundle(
        name=recipe.name,
        kind="batch",
        tier="two_tier" if domain_strategy is not None else "single_tier",
        metadata={
            "recipe": {
                "name": recipe.name,
                "target_channel": recipe.target_channel,
                "preprocessing": list(recipe.preprocessing or []),
                "segmentation": dict(recipe.segmentation or {}),
                "measurement": dict(recipe.measurement or {}),
                "domain": dict(recipe.domain) if recipe.domain else None,
                "cell_diameter_um": recipe.cell_diameter_um,
            },
        },
    )

    with with_active_bundle(parent_bundle):
        for index, name in enumerate(sample_names):
            raise_if_cancelled()
            info = _resolve_sample_inputs(name)
            s = info["sample"]
            current_file = info["file_path"] or info["layer_name"] or s.sample_name
            base_layer_names = set(call_on_main(_viewer_layer_names))
            mem_before = _rss_mb()
            failed_sample = False
            managed_layer_names: list[str] = []
            report_progress(
                progress=index / total,
                stage="sample",
                current_file=current_file,
                file_index=index + 1,
                total_files=total,
                completed=n_complete,
                failed=n_failed,
                skipped=0,
                show_in_chat=True,
                message=f"Processing {s.sample_name} ({index + 1}/{total}).",
                detail={
                    "rss_mb": mem_before,
                    "layer_count": len(base_layer_names),
                    "execution_mode": mode,
                },
            )
            try:
                load_result = _load_file_for_sample_if_needed(
                    info,
                    auto_load_files=auto_load_files,
                )
                raise_if_cancelled()
                target = recipe.target_channel or info["layer_name"]
                loaded_layers = list((load_result or {}).get("layer_names") or [])
                managed_layer_names.extend(str(name) for name in loaded_layers)
                target = call_on_main(
                    _resolve_target_within_loaded_layers,
                    target,
                    loaded_layers,
                )
                if target is None and len(loaded_layers) == 1:
                    target = loaded_layers[0]
                if projection is not None and target is None:
                    raise ValueError(
                        "measurement.projection requires a resolved target layer. "
                        "Set recipe.target_channel to a layer name/color that resolves "
                        "within each loaded sample."
                    )
                projection_record = _project_layer_for_recipe(
                    target,
                    projection=projection,
                    axis=projection_axis,
                )
                raise_if_cancelled()
                analysis_target = (
                    projection_record["new_layer"] if projection_record else target
                )
                if projection_record:
                    managed_layer_names.append(str(projection_record["new_layer"]))
                result = analyze_target_cells(
                    target=analysis_target,
                    do_3D=False if projection_record else seg.get("do_3D"),
                    diameter=seg.get("diameter"),
                    preprocess=pre_choice,
                    segmentation_method=seg.get("tool")
                    or seg.get("method", "target_objects"),
                    segmentation_options=seg,
                    domain_strategy=domain_strategy,
                    domain_options=domain_options,
                )
            except CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                failed_sample = True
                run_id = put_run(
                    sample_id=s.sample_id,
                    file_id=info["file_id"] or "",
                    recipe_id=recipe.recipe_id,
                    status="failed",
                    error=str(exc),
                )
                runs.append({"run_id": run_id, "status": "failed", "error": str(exc)})
                n_failed += 1
                report_progress(
                    progress=(index + 1) / total,
                    stage="failed",
                    current_file=current_file,
                    file_index=index + 1,
                    total_files=total,
                    completed=n_complete,
                    failed=n_failed,
                    skipped=0,
                    show_in_chat=True,
                    message=f"Failed {s.sample_name}; continuing.",
                )
            else:
                if not result.get("ok"):
                    failed_sample = True
                    run_id = put_run(
                        sample_id=s.sample_id,
                        file_id=info["file_id"] or "",
                        recipe_id=recipe.recipe_id,
                        status="failed",
                        error=result.get("error", "analysis returned ok=false"),
                        summary=result,
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "status": "failed",
                            "error": result.get("error"),
                        }
                    )
                    n_failed += 1
                    report_progress(
                        progress=(index + 1) / total,
                        stage="failed",
                        current_file=current_file,
                        file_index=index + 1,
                        total_files=total,
                        completed=n_complete,
                        failed=n_failed,
                        skipped=0,
                        show_in_chat=True,
                        message=f"Failed {s.sample_name}; continuing.",
                    )
                else:
                    table_name = result.get("table_name")
                    if table_name:
                        attach_sample_columns_to_table(
                            table_name=table_name,
                            sample_id=s.sample_id,
                            sample_name=s.sample_name,
                            group=s.group,
                            file_id=info["file_id"],
                            source_file=info["file_path"],
                            source_layer=result.get("target_channel"),
                        )

                    run_id = put_run(
                        sample_id=s.sample_id,
                        file_id=info["file_id"] or "",
                        recipe_id=recipe.recipe_id,
                        status="complete",
                        table_names=[table_name] if table_name else [],
                        layer_names=[
                            ln
                            for ln in (
                                result.get("labels_layer"),
                                result.get("preprocessed_layer"),
                            )
                            if ln
                        ],
                        summary={
                            "n_objects": result.get("n_objects"),
                            "object_unit": result.get("object_unit"),
                            "segmentation_method": result.get("segmentation_method"),
                            "analysis_dim": result.get("analysis_dim"),
                            "target_channel": result.get("target_channel"),
                            "source_target_channel": target,
                            "projection": projection,
                            "projection_axis": projection_axis,
                            "warnings": result.get("warnings", []),
                            "qc_png_skipped_reason": result.get("qc_png_skipped_reason"),
                        },
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "status": "complete",
                            "sample_name": s.sample_name,
                            "table_names": [table_name] if table_name else [],
                        }
                    )
                    n_complete += 1
                    report_progress(
                        progress=(index + 1) / total,
                        stage="complete_sample",
                        current_file=current_file,
                        file_index=index + 1,
                        total_files=total,
                        completed=n_complete,
                        failed=n_failed,
                        skipped=0,
                        show_in_chat=True,
                        message=f"Completed {s.sample_name} ({index + 1}/{total}).",
                    )
            finally:
                removed_layers: list[str] = []
                if cleanup_enabled and not (failed_sample and keep_failed_layers):
                    try:
                        removed_layers = _cleanup_sample_layers(
                            base_layer_names,
                            managed_layer_names,
                        )
                    except Exception:
                        removed_layers = []
                _release_worker_memory()
                mem_after = _rss_mb()
                if runs:
                    runs[-1]["cleanup_removed_layers"] = removed_layers
                    runs[-1]["rss_mb_before"] = mem_before
                    runs[-1]["rss_mb_after"] = mem_after
                report_progress(
                    progress=(index + 1) / total,
                    stage="cleanup",
                    current_file=current_file,
                    file_index=index + 1,
                    total_files=total,
                    completed=n_complete,
                    failed=n_failed,
                    skipped=0,
                    show_in_chat=True,
                    message=f"Cleaned up {len(removed_layers)} layers for {s.sample_name}.",
                    detail={
                        "rss_mb": mem_after,
                        "cleanup_removed_layers": len(removed_layers),
                        "layer_count": len(call_on_main(_viewer_layer_names)),
                    },
                )

    return {
        "recipe": recipe_name,
        "n_samples": len(sample_names),
        "n_complete": n_complete,
        "n_failed": n_failed,
        "execution_mode": mode,
        "cleanup_enabled": cleanup_enabled,
        "runs": runs,
        "bundle_path": str(parent_bundle),
    }
