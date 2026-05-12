from __future__ import annotations

import inspect
from typing import Any

from imajin.analysis.arrays import metadata_axes_without_channel
from imajin.analysis.workflow import (
    derive_size_params,
    normalize_preprocess,
)
from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.tools import preprocess as _preprocess
from imajin.tools import segment as _segment


def _layer_axes(snapshot: Any) -> str | None:
    return metadata_axes_without_channel(
        snapshot.metadata if isinstance(snapshot.metadata, dict) else None,
        getattr(snapshot.data, "ndim", 0),
    )


def _filtered_kwargs(func: Any, options: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(func).parameters
    return {key: value for key, value in options.items() if key in params}


def _run_preprocess_step(
    target_layer: str,
    preprocess: str | None,
) -> tuple[str, str | None, dict[str, Any] | None]:
    seg_input_layer = target_layer
    pre_step = normalize_preprocess(preprocess)
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
    return seg_input_layer, pre_step, pre_record


def _precompute_domain_layer(
    *,
    target_layer: str,
    snapshot: Any,
    domain_options: dict[str, Any] | None,
    counterstain_layer: str | None,
    cell_diameter_um: float | None,
) -> str:
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
    derived_pre = derive_size_params(
        cell_diameter_um,
        _segment._voxel_spacing(tuple(snapshot.scale), getattr(snapshot.data, "ndim", 2)),
    )
    if "min_area_um2" not in d_opts and "min_area_um2" in derived_pre:
        d_opts["min_area_um2"] = derived_pre["min_area_um2"]
    d_opts.setdefault("k_mad", 6.25)
    d_opts.setdefault("dark_percentile", 10.0)
    d_opts.setdefault("smooth_sigma_um", 0.75)
    d_opts.setdefault("max_components", 128)
    d_opts.setdefault("save_qc_png", False)
    domain_pre = _seg_dom(
        image_layer=target_layer,
        **_filtered_kwargs(_seg_dom, d_opts),
    )
    return str(domain_pre["labels_layer"])


def _run_segmentation_step(
    *,
    method: str,
    seg_input_layer: str,
    seg_options: dict[str, Any],
    use_3d: bool,
    diameter: float | None,
) -> dict[str, Any]:
    report_progress(stage="segmentation", message=f"Segmenting {seg_input_layer}.")
    raise_if_cancelled()
    if method == "target_objects":
        return _segment.segment_target_objects(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.segment_target_objects, seg_options),
        )
    if method == "intensity_regions":
        return _segment.segment_intensity_regions(
            image_layer=seg_input_layer,
            **_filtered_kwargs(_segment.segment_intensity_regions, seg_options),
        )
    cellpose_options = {
        **seg_options,
        "do_3D": use_3d,
        "diameter": diameter,
    }
    return _segment.cellpose_sam(
        image_layer=seg_input_layer,
        **_filtered_kwargs(_segment.cellpose_sam, cellpose_options),
    )
