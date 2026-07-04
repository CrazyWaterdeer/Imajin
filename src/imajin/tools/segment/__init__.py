"""Segmentation tools, split by family into submodules (GitHub issue #3).

The eight ``@tool`` functions live in the family submodules (``cellpose``,
``auto3d``, ``intensity``, ``target``, ``domain``, ``review``); the shared
tool-wrapper scaffolding lives in ``_segmentation_io`` / ``_segmentation_outputs``.
Importing the submodules here registers the tools (via ``@tool``) and re-exports
their names on ``imajin.tools.segment`` so the public import path is unchanged.
"""
from __future__ import annotations

# Public tools -- importing each submodule registers it via @tool and re-exports
# the tool name on ``imajin.tools.segment``.
from imajin.tools.segment.cellpose import cellpose_sam
from imajin.tools.segment.auto3d import segment_3d_cells_auto
from imajin.tools.segment.intensity import segment_intensity_regions
from imajin.tools.segment.target import (
    auto_segment_target,
    correct_roi,
    segment_target_objects,
)
from imajin.tools.segment.domain import segment_expression_domain
from imajin.tools.segment.review import review_target_roi

# Readable private aliases preserved for tests / _workflow_steps that read them via
# ``imajin.tools.segment.<name>``. These are read-only references; the *patch*
# targets for the ones tests monkeypatch live in the module that actually calls
# them (_segmentation_io for _get_cellpose_model, segment.target for
# _prepare_corrected / _boundary_bbox_slices).
from imajin.analysis.segmentation import (
    boundary_bbox_slices as _boundary_bbox_slices,  # noqa: F401
    threshold_noise_floor as _threshold_noise_floor,  # noqa: F401
    voxel_spacing as _voxel_spacing,  # noqa: F401
)
from imajin.analysis.target_pipeline import prepare_corrected as _prepare_corrected  # noqa: F401
from imajin.tools._segmentation_io import _get_cellpose_model  # noqa: F401
from imajin.tools._segmentation_outputs import _write_segmentation_qc_png  # noqa: F401

__all__ = [
    "cellpose_sam",
    "segment_3d_cells_auto",
    "segment_intensity_regions",
    "segment_target_objects",
    "auto_segment_target",
    "segment_expression_domain",
    "correct_roi",
    "review_target_roi",
]
