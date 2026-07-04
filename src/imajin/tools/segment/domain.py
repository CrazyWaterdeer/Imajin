from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import materialize_array
from imajin.analysis.domain_segmentation import (
    domain_min_size_from_physical as _domain_min_size_from_physical,
    domain_physical_sizes as _domain_physical_sizes,
    filter_domain_components as _filter_domain_components,
    smooth_domain_image as _smooth_domain_image,
)
from imajin.analysis.segmentation import (
    dilate_binary_um as _dilate_binary_um,
    remove_small_binary_objects as _remove_small_binary_objects,
    threshold_noise_floor as _threshold_noise_floor,
    voxel_spacing as _voxel_spacing,
)
from imajin.tools._segmentation_io import (
    finalize_qc_png,
    load_and_guard,
    project_boundary_outline_2d,
    resolve_boundary,
)
from imajin.tools._segmentation_outputs import (
    _saturation_warnings,
    _source_metadata_from_layer,
)
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
from imajin.tools.registry import tool


@tool(
    description="Segment a permissive expression domain on a reporter channel using "
    "a noise-floor threshold (median + k*MAD of the dark percentile). No background "
    "subtraction, so cluster interiors are preserved. Use as Tier 1 of two-tier "
    "expression analyses where baseline expression must be captured alongside "
    "active sub-objects. By default, connected components are cleaned and merged "
    "into one binary domain label for compact domain-level intensity measurement.",
    phase="2",
    worker=True,
)
def segment_expression_domain(
    image_layer: str,
    threshold_strategy: str = "noise_floor",
    k_mad: float = 5.25,
    dark_percentile: float = 10.0,
    counterstain_layer: str | None = None,
    counterstain_dilation_um: float = 0.0,
    is_nuclear: bool | None = None,
    min_area_um2: float = 5.0,
    min_volume_um3: float | None = None,
    smooth_sigma_um: float = 0.5,
    max_components: int = 256,
    min_component_fraction: float = 0.0,
    merge_components: bool = True,
    dilation_um: float = 0.0,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
    boundary_mask: str | None = None,
) -> dict[str, Any]:
    if threshold_strategy != "noise_floor":
        raise ValueError(
            f"threshold_strategy must be 'noise_floor' (got {threshold_strategy!r})"
        )

    L, data, axes = load_and_guard(
        image_layer,
        tool_name="segment_expression_domain",
        dims="2d_or_3d_terse",
    )
    saturation_warnings = _saturation_warnings(data, layer_name=L.name)
    raw = np.asarray(data, dtype=np.float32)

    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    boundary_bool, _bnd_raw = resolve_boundary(boundary_mask, raw.shape)
    boundary_outline_2d: np.ndarray | None = None
    if _bnd_raw is not None:
        boundary_outline_2d = project_boundary_outline_2d(_bnd_raw > 0)

    threshold_image = _smooth_domain_image(
        raw,
        spacing=spacing,
        smooth_sigma_um=smooth_sigma_um,
    )
    if boundary_bool is not None:
        # ROI-local noise floor: estimate it from the smoothed values *inside* the ROI
        # only (finite raw + finite smoothed), so signal outside the drawn region can't
        # shift the threshold, and clip the domain to the ROI from the start.
        inside = boundary_bool & np.isfinite(raw) & np.isfinite(threshold_image)
        threshold = _threshold_noise_floor(
            threshold_image[inside], k_mad=k_mad, dark_percentile=dark_percentile
        )
        binary = inside & (threshold_image > threshold)
    else:
        threshold = _threshold_noise_floor(
            threshold_image, k_mad=k_mad, dark_percentile=dark_percentile
        )
        binary = np.isfinite(raw) & np.isfinite(threshold_image) & (
            threshold_image > threshold
        )

    counterstain_used = False
    counterstain_warnings: list[str] = []
    if counterstain_layer is not None:
        if not is_nuclear:
            counterstain_warnings.append(
                "counterstain marker is non-nuclear or unknown; reporter-only "
                "domain used"
            )
        else:
            cs_layer = call_on_main(snapshot_layer, counterstain_layer)
            cs_data = materialize_array(cs_layer.data, dtype=np.float32)
            if cs_data.shape != raw.shape:
                counterstain_warnings.append(
                    f"counterstain shape {cs_data.shape} differs from reporter "
                    f"shape {raw.shape}; counterstain ignored"
                )
            else:
                from skimage import filters as _filters
                cs_finite = cs_data[np.isfinite(cs_data)]
                if cs_finite.size and float(cs_finite.max()) > float(cs_finite.min()):
                    cs_threshold = float(_filters.threshold_otsu(cs_finite))
                    cs_binary = np.isfinite(cs_data) & (cs_data > cs_threshold)
                    if counterstain_dilation_um > 0 and spacing is not None:
                        cs_binary = _dilate_binary_um(
                            cs_binary,
                            spacing=spacing,
                            radius_um=counterstain_dilation_um,
                        )
                    binary = binary & cs_binary
                    counterstain_used = True
                else:
                    counterstain_warnings.append(
                        "counterstain has no usable signal; reporter-only domain used"
                    )

    physical_min_size = _domain_min_size_from_physical(
        min_area_um2=min_area_um2,
        min_volume_um3=min_volume_um3,
        spacing=spacing,
        ndim=raw.ndim,
    )
    if physical_min_size:
        binary = _remove_small_binary_objects(binary, physical_min_size)

    if dilation_um > 0 and spacing is not None:
        binary = _dilate_binary_um(binary, spacing=spacing, radius_um=dilation_um)
        if boundary_bool is not None:
            # Dilation must not grow the domain back outside the ROI.
            binary = binary & boundary_bool

    labels, component_stats, component_warnings = _filter_domain_components(
        binary,
        max_components=max_components,
        min_component_fraction=min_component_fraction,
        merge_components=merge_components,
    )
    domain_warnings = saturation_warnings + component_warnings
    size_stats = _domain_physical_sizes(labels > 0, spacing)
    n_components = int(component_stats["n_components_retained"])
    domain_label_count = int(component_stats["domain_label_count"])

    if domain_label_count == 0:
        empty = np.zeros(raw.shape, dtype=np.int32)
        out_name = f"{L.name}_domain"
        layer = call_on_main(
            add_labels_from_worker,
            empty,
            name=out_name,
            scale=tuple(L.scale),
            metadata={
                "source_layer": L.name,
                **_source_metadata_from_layer(L),
                "segmentation_method": "expression_domain",
                "noise_floor_threshold": float(threshold),
                "threshold_image": "smoothed" if smooth_sigma_um > 0 else "raw",
                "smooth_sigma_um": float(smooth_sigma_um),
                "k_mad": float(k_mad),
                "dark_percentile": float(dark_percentile),
                "counterstain_used": counterstain_used,
                "counterstain_warnings": counterstain_warnings,
                "domain_warnings": domain_warnings,
                "min_area_um2": float(min_area_um2),
                "min_volume_um3": min_volume_um3,
                "min_size_voxels": physical_min_size,
                "max_components": max_components,
                "min_component_fraction": float(min_component_fraction),
                "merge_components": bool(merge_components),
                **component_stats,
                **size_stats,
                "empty_mask": True,
            },
        )
        return {
            "labels_layer": layer.name,
            "n_components": 0,
            "domain_label_count": 0,
            "domain_area_um2": 0.0,
            "domain_volume_um3": 0.0 if raw.ndim == 3 else None,
            "domain_voxels": 0,
            "noise_floor_threshold": float(threshold),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "domain_warnings": domain_warnings,
            "qc_png_path": None,
            "qc_png_error": None,
            "qc_png_skipped_reason": "empty mask",
            "empty_mask": True,
        }

    domain_area_um2 = float(size_stats["domain_area_um2"])
    domain_volume_um3 = size_stats["domain_volume_um3"]

    out_name = f"{L.name}_domain"
    layer = call_on_main(
        add_labels_from_worker,
        labels,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            **_source_metadata_from_layer(L),
            "segmentation_method": "expression_domain",
            "noise_floor_threshold": float(threshold),
            "threshold_image": "smoothed" if smooth_sigma_um > 0 else "raw",
            "smooth_sigma_um": float(smooth_sigma_um),
            "k_mad": float(k_mad),
            "dark_percentile": float(dark_percentile),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "domain_warnings": domain_warnings,
            "n_components": n_components,
            "domain_label_count": domain_label_count,
            "domain_area_um2": domain_area_um2,
            "domain_volume_um3": domain_volume_um3,
            "domain_voxels": int(size_stats["domain_voxels"]),
            "min_area_um2": float(min_area_um2),
            "min_volume_um3": min_volume_um3,
            "min_size_voxels": physical_min_size,
            "max_components": max_components,
            "min_component_fraction": float(min_component_fraction),
            "merge_components": bool(merge_components),
            **component_stats,
            "boundary_mask": boundary_mask,
            "threshold_scope": "boundary_mask" if boundary_bool is not None else "global",
            "empty_mask": False,
        },
    )

    saved_qc_png, qc_png_error, qc_png_skipped_reason = finalize_qc_png(
        raw,
        labels,
        layer,
        L,
        method="expression_domain",
        save_qc_png=save_qc_png,
        qc_png_path=qc_png_path,
        secondary_outline_mask=boundary_outline_2d,
    )

    return {
        "labels_layer": layer.name,
        "n_components": n_components,
        "domain_label_count": domain_label_count,
        "domain_area_um2": domain_area_um2,
        "domain_volume_um3": domain_volume_um3,
        "domain_voxels": int(size_stats["domain_voxels"]),
        "noise_floor_threshold": float(threshold),
        "counterstain_used": counterstain_used,
        "counterstain_warnings": counterstain_warnings,
        "domain_warnings": domain_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
        "boundary_mask": boundary_mask,
        "threshold_scope": "boundary_mask" if boundary_bool is not None else "global",
        "empty_mask": False,
    }
