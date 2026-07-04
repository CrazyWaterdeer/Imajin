from __future__ import annotations

from typing import Any

import numpy as np

from imajin.session import get_layer, get_viewer
from imajin.analysis.arrays import materialize_array
from imajin.analysis.domain_segmentation import (
    domain_min_size_from_physical as _domain_min_size_from_physical,
    domain_physical_sizes as _domain_physical_sizes,
    filter_domain_components as _filter_domain_components,
    smooth_domain_image as _smooth_domain_image,
)
from imajin.analysis.segmentation import (
    boundary_bbox_slices as _boundary_bbox_slices,
    dilate_binary_um as _dilate_binary_um,
    intersect_labels_with_mask as _intersect_labels_with_mask,
    label_qc as _label_qc,
    scatter_labels_to_full as _scatter_labels_to_full,
    label_qc_warnings as _label_qc_warnings,
    min_size_from_physical as _min_size_from_physical,
    remove_small_binary_objects as _remove_small_binary_objects,
    segment_connected_regions as _segment_connected_regions,
    threshold_noise_floor as _threshold_noise_floor,
    voxel_spacing as _voxel_spacing,
)
from imajin.analysis.target_pipeline import (
    auto_correct_target as _auto_correct_target,
    prepare_corrected as _prepare_corrected,
    threshold_and_label as _threshold_and_label,
)
from imajin.analysis.roi_quality import assess_roi as _assess_roi
from imajin.analysis.segmentation_auto3d import (
    SegmentationCandidate as _SegmentationCandidate,
    build_auto3d_candidates as _build_auto3d_candidates,
    filter_labels_by_z_extent as _filter_labels_by_z_extent,
    rank_segmentation_labels as _rank_segmentation_labels,
    selection_confidence as _selection_confidence,
)
from imajin.agent.qt_dispatch import call_on_main
from imajin.tools import _segmentation_io as _seg_io
from imajin.tools._segmentation_io import (
    boundary_broadcast_warning,
    effective_target_min_size,
    finalize_qc_png,
    load_and_guard,
    project_boundary_outline_2d,
    resolve_boundary,
)
from imajin.tools._segmentation_outputs import (
    _saturation_warnings,
    _source_metadata_from_layer,
    _write_segmentation_qc_png,
)
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
from imajin.tools.registry import tool

# Tool families live in submodules; import them for @tool registration and to
# re-export the public tool names on ``imajin.tools.segment``.
from imajin.tools.segment.cellpose import cellpose_sam  # noqa: F401
from imajin.tools.segment.auto3d import segment_3d_cells_auto  # noqa: F401
from imajin.tools.segment.intensity import segment_intensity_regions  # noqa: F401
from imajin.tools.segment.target import (  # noqa: F401
    auto_segment_target,
    correct_roi,
    segment_target_objects,
)
from imajin.tools.segment.domain import segment_expression_domain  # noqa: F401
from imajin.tools.segment.review import review_target_roi  # noqa: F401

# The Cellpose model cache lives in _segmentation_io. cellpose_sam /
# segment_3d_cells_auto call it module-qualified (_seg_io._get_cellpose_model) so a
# test patching imajin.tools._segmentation_io._get_cellpose_model intercepts the
# call after the Phase 2 package split. The bare alias below is readable only.
_get_cellpose_model = _seg_io._get_cellpose_model
