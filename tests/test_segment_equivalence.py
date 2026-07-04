"""Characterization / equivalence guard for the segment tool-module split.

This is the safety net (C0) for the refactor tracked in GitHub issue #3. It pins
the *observable* contract of the eight segment tools so the Phase 1 dedup and the
Phase 2 package split can be proven byte-equivalent commit by commit:

* the registry entry (flags), the ``inspect.signature`` string, and the pydantic
  json-schema property/required sets for all 8 tools (the LLM tool contract);
* the public + readable-private import surface on ``imajin.tools.segment``;
* that ``call_tool`` dispatches through the registry;
* that the three monkeypatched private helpers still intercept the real call
  (the load-bearing subtlety behind Phase 2);
* the produced **label array** (sha1), result-dict and layer-metadata key sets,
  and object count for each segmentation tool on fixed synthetic inputs.

It asserts observable output only -- never that a particular helper lives at a
particular path (those move in Phase 1/2). Cellpose is stubbed so the guard never
downloads weights or touches a GPU.
"""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from imajin.tools import segment
from imajin.tools.registry import call_tool, get_tool

# --- patch targets (relocated in ONE place by C2 / C11) ----------------------
# monkeypatch rebinds a name in a module namespace; it only intercepts a call
# that looks the name up in *that* namespace. These strings point at where each
# helper is actually called from, so the interception survives the package split
# by editing only these constants.
CELLPOSE_MODEL_TARGET = "imajin.tools.segment._get_cellpose_model"
PREPARE_CORRECTED_TARGET = "imajin.tools.segment._prepare_corrected"
BOUNDARY_BBOX_TARGET = "imajin.tools.segment._boundary_bbox_slices"

TOOLS = [
    "cellpose_sam",
    "segment_3d_cells_auto",
    "segment_intensity_regions",
    "segment_target_objects",
    "auto_segment_target",
    "segment_expression_domain",
    "correct_roi",
    "review_target_roi",
]


# --- Cellpose stub -----------------------------------------------------------
class _FakeCellpose:
    """Deterministic stand-in for a Cellpose model: labels bright connected
    components of the input, so the wrapper output is fully determined."""

    def eval(self, data, **kwargs):
        from skimage import measure

        masks = measure.label(np.asarray(data) > 0).astype(np.int32)
        return masks, None, None


# =============================================================================
# 1. Registry / signature / schema golden (all 8 tools)
# =============================================================================
REGISTRY_GOLDEN: dict[str, dict] = {
    "cellpose_sam": {
        "flags": {"phase": "2", "vision_hint": True, "worker": True, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', do_3D: 'bool' = False, diameter: 'float | None' = None, model: 'str' = 'cpsam', flow_threshold: 'float' = 0.4, cellprob_threshold: 'float' = 0.0, min_size: 'int' = 15, max_size_fraction: 'float' = 0.4, save_qc_png: 'bool' = True, qc_png_path: 'str | None' = None) -> 'dict[str, Any]'",
        "props": ["image_layer", "do_3D", "diameter", "model", "flow_threshold", "cellprob_threshold", "min_size", "max_size_fraction", "save_qc_png", "qc_png_path"],
        "required": ["image_layer"],
    },
    "segment_3d_cells_auto": {
        "flags": {"phase": "2", "vision_hint": True, "worker": True, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', background_radius: 'int' = 48, background_method: 'str' = 'opening', background_percentile: 'float' = 20.0, threshold_method: 'str' = 'auto', threshold_percentile: 'float' = 99.0, min_snr: 'float' = 2.0, high_snr: 'float' = 4.0, min_size: 'int | None' = None, min_area_um2: 'float | None' = None, min_volume_um3: 'float | None' = None, smoothing_sigma: 'float' = 1.0, fill_holes: 'bool' = True, split_touching: 'bool' = False, min_distance: 'int' = 20, min_distance_um: 'float | None' = None, boundary_mask: 'str | None' = None, candidate_modes: 'list[str] | None' = None, max_candidates: 'int' = 8, stitch_min_overlap: 'float' = 0.2, stitch_max_centroid_distance: 'float | None' = None, stitch_max_area_ratio: 'float' = 3.0, min_z_planes: 'int | None' = 2, include_cellpose_sam: 'bool' = False, cellpose_model: 'str' = 'cpsam', cellpose_diameter: 'float | None' = None, cellpose_flow_threshold: 'float' = 0.4, cellpose_cellprob_threshold: 'float' = 0.0, cellpose_max_size_fraction: 'float' = 0.4, save_qc_png: 'bool' = True, qc_png_path: 'str | None' = None) -> 'dict[str, Any]'",
        "props": ["image_layer", "background_radius", "background_method", "background_percentile", "threshold_method", "threshold_percentile", "min_snr", "high_snr", "min_size", "min_area_um2", "min_volume_um3", "smoothing_sigma", "fill_holes", "split_touching", "min_distance", "min_distance_um", "boundary_mask", "candidate_modes", "max_candidates", "stitch_min_overlap", "stitch_max_centroid_distance", "stitch_max_area_ratio", "min_z_planes", "include_cellpose_sam", "cellpose_model", "cellpose_diameter", "cellpose_flow_threshold", "cellpose_cellprob_threshold", "cellpose_max_size_fraction", "save_qc_png", "qc_png_path"],
        "required": ["image_layer"],
    },
    "segment_intensity_regions": {
        "flags": {"phase": "2", "vision_hint": True, "worker": True, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', threshold_method: 'str' = 'otsu', percentile: 'float' = 99.0, min_size: 'int' = 128, min_area_um2: 'float | None' = None, min_volume_um3: 'float | None' = None, smoothing_sigma: 'float' = 1.0, fill_holes: 'bool' = True, split_touching: 'bool' = False, min_distance: 'int' = 20, min_distance_um: 'float | None' = None, save_qc_png: 'bool' = True, qc_png_path: 'str | None' = None) -> 'dict[str, Any]'",
        "props": ["image_layer", "threshold_method", "percentile", "min_size", "min_area_um2", "min_volume_um3", "smoothing_sigma", "fill_holes", "split_touching", "min_distance", "min_distance_um", "save_qc_png", "qc_png_path"],
        "required": ["image_layer"],
    },
    "segment_target_objects": {
        "flags": {"phase": "2", "vision_hint": True, "worker": True, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', background_radius: 'int' = 48, background_method: 'str' = 'opening', background_percentile: 'float' = 20.0, threshold_method: 'str' = 'auto', threshold_percentile: 'float' = 99.0, threshold_clip_percentile: 'float | None' = None, auto_mask_hyperbright: 'bool' = False, hyperbright_percentile: 'float' = 99.5, hyperbright_dilate_radius: 'int' = 2, min_snr: 'float' = 2.0, high_snr: 'float' = 4.0, min_size: 'int | None' = None, min_area_um2: 'float | None' = None, min_volume_um3: 'float | None' = None, smoothing_sigma: 'float' = 1.0, fill_holes: 'bool' = True, split_touching: 'bool' = False, min_distance: 'int' = 20, min_distance_um: 'float | None' = None, save_qc_png: 'bool' = True, qc_png_path: 'str | None' = None, boundary_mask: 'str | None' = None) -> 'dict[str, Any]'",
        "props": ["image_layer", "background_radius", "background_method", "background_percentile", "threshold_method", "threshold_percentile", "threshold_clip_percentile", "auto_mask_hyperbright", "hyperbright_percentile", "hyperbright_dilate_radius", "min_snr", "high_snr", "min_size", "min_area_um2", "min_volume_um3", "smoothing_sigma", "fill_holes", "split_touching", "min_distance", "min_distance_um", "save_qc_png", "qc_png_path", "boundary_mask"],
        "required": ["image_layer"],
    },
    "auto_segment_target": {
        "flags": {"phase": "2", "vision_hint": True, "worker": True, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', background_radius: 'int' = 48, background_method: 'str' = 'opening', background_percentile: 'float' = 20.0, threshold_method: 'str' = 'auto', threshold_percentile: 'float' = 99.0, min_snr: 'float' = 2.0, high_snr: 'float' = 4.0, min_size: 'int | None' = None, min_area_um2: 'float | None' = None, min_volume_um3: 'float | None' = None, smoothing_sigma: 'float' = 1.0, fill_holes: 'bool' = True, split_touching: 'bool' = False, min_distance: 'int' = 20, min_distance_um: 'float | None' = None, max_iters: 'int' = 3, save_qc_png: 'bool' = True, qc_png_path: 'str | None' = None, boundary_mask: 'str | None' = None) -> 'dict[str, Any]'",
        "props": ["image_layer", "background_radius", "background_method", "background_percentile", "threshold_method", "threshold_percentile", "min_snr", "high_snr", "min_size", "min_area_um2", "min_volume_um3", "smoothing_sigma", "fill_holes", "split_touching", "min_distance", "min_distance_um", "max_iters", "save_qc_png", "qc_png_path", "boundary_mask"],
        "required": ["image_layer"],
    },
    "segment_expression_domain": {
        "flags": {"phase": "2", "vision_hint": False, "worker": True, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', threshold_strategy: 'str' = 'noise_floor', k_mad: 'float' = 5.25, dark_percentile: 'float' = 10.0, counterstain_layer: 'str | None' = None, counterstain_dilation_um: 'float' = 0.0, is_nuclear: 'bool | None' = None, min_area_um2: 'float' = 5.0, min_volume_um3: 'float | None' = None, smooth_sigma_um: 'float' = 0.5, max_components: 'int' = 256, min_component_fraction: 'float' = 0.0, merge_components: 'bool' = True, dilation_um: 'float' = 0.0, save_qc_png: 'bool' = True, qc_png_path: 'str | None' = None, boundary_mask: 'str | None' = None) -> 'dict[str, Any]'",
        "props": ["image_layer", "threshold_strategy", "k_mad", "dark_percentile", "counterstain_layer", "counterstain_dilation_um", "is_nuclear", "min_area_um2", "min_volume_um3", "smooth_sigma_um", "max_components", "min_component_fraction", "merge_components", "dilation_um", "save_qc_png", "qc_png_path", "boundary_mask"],
        "required": ["image_layer"],
    },
    "correct_roi": {
        "flags": {"phase": "2", "vision_hint": False, "worker": True, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', labels_layer: 'str', min_snr: 'float | None' = None, high_snr: 'float | None' = None, threshold_method: 'str | None' = None, threshold_clip_percentile: 'float | None' = None, auto_mask_hyperbright: 'bool | None' = None, hyperbright_percentile: 'float | None' = None, background_radius: 'int | None' = None, smoothing_sigma: 'float | None' = None, min_size: 'int | None' = None) -> 'dict[str, Any]'",
        "props": ["image_layer", "labels_layer", "min_snr", "high_snr", "threshold_method", "threshold_clip_percentile", "auto_mask_hyperbright", "hyperbright_percentile", "background_radius", "smoothing_sigma", "min_size"],
        "required": ["image_layer", "labels_layer"],
    },
    "review_target_roi": {
        "flags": {"phase": "2", "vision_hint": False, "worker": False, "manual": True, "llm": True, "subagent": None},
        "sig": "(image_layer: 'str', labels_layer: 'str') -> 'dict[str, Any]'",
        "props": ["image_layer", "labels_layer"],
        "required": ["image_layer", "labels_layer"],
    },
}


@pytest.mark.parametrize("name", TOOLS)
def test_registry_signature_and_schema_golden(name: str) -> None:
    golden = REGISTRY_GOLDEN[name]
    entry = get_tool(name)
    assert {
        "phase": entry.phase,
        "vision_hint": entry.vision_hint,
        "worker": entry.worker,
        "manual": entry.manual,
        "llm": entry.llm,
        "subagent": entry.subagent,
    } == golden["flags"]
    assert str(inspect.signature(entry.func)) == golden["sig"]
    schema = entry.input_model.model_json_schema()
    assert list(schema.get("properties", {}).keys()) == golden["props"]
    assert sorted(schema.get("required", [])) == sorted(golden["required"])


def test_public_and_private_surface() -> None:
    # 8 public tools resolve on the package and are the registered wrapped funcs.
    for name in TOOLS:
        assert getattr(segment, name) is get_tool(name).func
    # 6 readable private aliases stay importable via the module.
    for alias in (
        "_threshold_noise_floor",
        "_write_segmentation_qc_png",
        "_voxel_spacing",
        "_prepare_corrected",
        "_boundary_bbox_slices",
        "_get_cellpose_model",
    ):
        assert hasattr(segment, alias), alias


def test_call_tool_dispatches_through_registry(viewer) -> None:
    # review_target_roi against a registered viewer with missing layers returns a
    # typed failure -- exercises the registry validation + dispatch path without
    # heavy compute.
    out = call_tool("review_target_roi", image_layer="nope", labels_layer="nope")
    assert isinstance(out, dict) and out.get("ok") is False


# =============================================================================
# 2. Monkeypatch interception (the Phase 2 load-bearing subtlety)
# =============================================================================
def test_get_cellpose_model_patch_intercepts(viewer, monkeypatch) -> None:
    monkeypatch.setattr(CELLPOSE_MODEL_TARGET, lambda *a, **k: _FakeCellpose())
    img = np.zeros((64, 64), dtype=np.uint16)
    img[10:22, 10:22] = 1000
    img[40:52, 40:52] = 1000
    viewer.add_image(img, name="im")
    res = segment.cellpose_sam("im", diameter=10, save_qc_png=False)
    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    # The fake labels exactly two bright components -> proves the patch was used.
    assert res["n_cells"] == 2
    assert int(labels.max()) == 2


def test_boundary_bbox_patch_intercepts(viewer, monkeypatch) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[30:50, 30:50] = 300.0
    viewer.add_image(img, name="im")
    roi = np.zeros((100, 100), dtype=np.int32)
    roi[0:70, 0:70] = 1
    viewer.add_labels(roi, name="roi")

    calls: list[int] = []
    real = segment._boundary_bbox_slices
    monkeypatch.setattr(
        BOUNDARY_BBOX_TARGET,
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    segment.segment_target_objects(
        "im", boundary_mask="roi", background_radius=16, min_size=30, save_qc_png=False
    )
    assert calls, "patched _boundary_bbox_slices must be reached (crop path)"


def test_review_target_roi_error_paths(viewer) -> None:
    # No layers -> missing image_layer failure.
    out = segment.review_target_roi("missing_img", "missing_labels")
    assert out.get("ok") is False


# =============================================================================
# 3. Output equivalence -- label array + result/metadata key sets + count
# =============================================================================
def _sha1(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sc_intensity_2d(viewer):
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:22, 10:22] = 400.0
    img[40:52, 40:52] = 400.0
    viewer.add_image(img, name="im")
    return segment.segment_intensity_regions("im", min_size=20, save_qc_png=False)


def _sc_target_2d(viewer):
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:22, 10:22] = 400.0
    img[40:52, 40:52] = 400.0
    viewer.add_image(img, name="im")
    return segment.segment_target_objects(
        "im", background_radius=16, min_size=20, smoothing_sigma=1.0, save_qc_png=False
    )


def _sc_target_3d_roi(viewer):
    zyx = np.zeros((4, 80, 80), dtype=np.float32)
    zyx[:, 12:28, 12:28] = 400.0  # inside ROI
    zyx[:, 55:70, 55:70] = 400.0  # outside ROI
    viewer.add_image(zyx, name="im")
    roi = np.zeros((80, 80), dtype=np.int32)
    roi[0:40, 0:40] = 1
    viewer.add_labels(roi, name="roi")
    return segment.segment_target_objects(
        "im", boundary_mask="roi", background_radius=16, min_size=20, save_qc_png=False
    )


def _sc_auto_target_2d(viewer):
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:22, 10:22] = 400.0
    img[40:52, 40:52] = 400.0
    viewer.add_image(img, name="im")
    return segment.auto_segment_target(
        "im", background_radius=16, min_size=20, max_iters=2, save_qc_png=False
    )


def _sc_domain_2d(viewer):
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:30, 10:30] = 500.0
    viewer.add_image(img, name="im")
    return segment.segment_expression_domain("im", min_area_um2=0.0, save_qc_png=False)


def _sc_domain_3d_roi(viewer):
    zyx = np.zeros((3, 60, 60), dtype=np.float32)
    zyx[:, 10:25, 10:25] = 500.0
    zyx[:, 40:55, 40:55] = 500.0
    viewer.add_image(zyx, name="im")
    roi = np.zeros((60, 60), dtype=np.int32)
    roi[0:30, 0:30] = 1
    viewer.add_labels(roi, name="roi")
    return segment.segment_expression_domain(
        "im", boundary_mask="roi", min_area_um2=0.0, save_qc_png=False
    )


def _sc_auto3d(viewer):
    rng = np.random.default_rng(0)
    zyx = rng.normal(30.0, 3.0, size=(4, 48, 48)).astype(np.float32)
    zyx[1:3, 12:28, 12:28] += 400.0  # a blob spanning two planes, over mild noise
    viewer.add_image(zyx, name="im")
    return segment.segment_3d_cells_auto(
        "im", min_size=20, max_candidates=4, save_qc_png=False
    )


def _sc_cellpose_2d(viewer):
    img = np.zeros((64, 64), dtype=np.uint16)
    img[10:22, 10:22] = 1000
    img[40:52, 40:52] = 1000
    viewer.add_image(img, name="im")
    return segment.cellpose_sam("im", diameter=10, save_qc_png=False)


def _sc_correct_roi(viewer):
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:22, 10:22] = 400.0
    img[40:52, 40:52] = 400.0
    viewer.add_image(img, name="im")
    first = segment.segment_target_objects(
        "im", background_radius=16, min_size=20, smoothing_sigma=1.0, save_qc_png=False
    )
    return segment.correct_roi("im", first["labels_layer"], min_snr=3.0)


SCENARIOS = {
    "intensity_2d": _sc_intensity_2d,
    "target_2d": _sc_target_2d,
    "target_3d_roi": _sc_target_3d_roi,
    "auto_target_2d": _sc_auto_target_2d,
    "domain_2d": _sc_domain_2d,
    "domain_3d_roi": _sc_domain_3d_roi,
    "auto3d": _sc_auto3d,
    "cellpose_2d": _sc_cellpose_2d,
    "correct_roi": _sc_correct_roi,
}


def _comparable(viewer, result: dict) -> dict:
    labels_name = result["labels_layer"]
    layer = viewer.layers[labels_name]
    labels = np.asarray(layer.data)
    meta = dict(getattr(layer, "metadata", {}) or {})
    count = next(
        (result[k] for k in ("n_objects", "n_cells", "n_regions", "n_components") if k in result),
        None,
    )
    return {
        "result_keys": sorted(result.keys()),
        "meta_keys": sorted(meta.keys()),
        "count": count,
        "label_sha1": _sha1(labels),
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
    }


# Filled from a capture run against the unchanged (pre-refactor) code.
GOLDEN: dict[str, dict] = {
    "intensity_2d": {"result_keys": ["axes", "dtype", "empty_mask", "fill_holes", "labels_layer", "min_area_um2", "min_distance", "min_distance_um", "min_size", "min_volume_um3", "n_cells", "n_regions", "object_area_max", "object_area_median", "object_area_min", "percentile", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "qc_warnings", "requested_min_size", "shape", "smoothing_sigma", "split_touching", "threshold", "threshold_method", "voxel_spacing"], "meta_keys": ["axes", "dtype", "empty_mask", "fill_holes", "min_area_um2", "min_distance", "min_distance_um", "min_size", "min_volume_um3", "n_objects", "object_area_max", "object_area_median", "object_area_min", "percentile", "qc_warnings", "requested_min_size", "segmentation_method", "shape", "smoothing_sigma", "source_layer", "split_touching", "threshold", "threshold_method", "voxel_spacing"], "count": 2, "label_sha1": "7315f485269547bdd8ccdff009ef224498cc0ca1", "label_shape": [64, 64], "label_dtype": "int32"},
    "target_2d": {"result_keys": ["axes", "background_method", "background_percentile", "background_radius", "boundary_mask", "confidence_drivers", "distribution_flag", "dtype", "empty_mask", "fill_holes", "high_snr", "high_threshold", "inside_corrected_mean", "inside_raw_mean", "labels_layer", "mask_fraction", "min_area_um2", "min_distance", "min_distance_um", "min_size", "min_snr", "min_volume_um3", "n_cells", "n_objects", "noise_sigma", "object_area_max", "object_area_median", "object_area_min", "object_unit", "outside_corrected_mean", "outside_raw_mean", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "qc_warnings", "requested_min_size", "roi_confidence", "roi_score", "shape", "smoothing_sigma", "split_touching", "threshold", "threshold_method", "threshold_percentile", "threshold_scope", "top_bright_outside_fraction", "voxel_spacing"], "meta_keys": ["axes", "background_method", "background_percentile", "background_radius", "boundary_mask", "confidence_drivers", "distribution_flag", "dtype", "empty_mask", "fill_holes", "high_snr", "high_threshold", "inside_corrected_mean", "inside_raw_mean", "mask_fraction", "min_area_um2", "min_distance", "min_distance_um", "min_size", "min_snr", "min_volume_um3", "n_objects", "noise_sigma", "object_area_max", "object_area_median", "object_area_min", "object_unit", "outside_corrected_mean", "outside_raw_mean", "qc_warnings", "requested_min_size", "roi_confidence", "roi_score", "segmentation_method", "shape", "smoothing_sigma", "source_layer", "split_touching", "threshold", "threshold_method", "threshold_percentile", "threshold_scope", "top_bright_outside_fraction", "voxel_spacing"], "count": 2, "label_sha1": "7315f485269547bdd8ccdff009ef224498cc0ca1", "label_shape": [64, 64], "label_dtype": "int32"},
    "target_3d_roi": {"result_keys": ["axes", "background_method", "background_percentile", "background_radius", "boundary_mask", "confidence_drivers", "distribution_flag", "dtype", "empty_mask", "fill_holes", "high_snr", "high_threshold", "inside_corrected_mean", "inside_raw_mean", "labels_layer", "mask_fraction", "min_area_um2", "min_distance", "min_distance_um", "min_size", "min_snr", "min_volume_um3", "n_cells", "n_objects", "noise_sigma", "object_area_max", "object_area_median", "object_area_min", "object_unit", "outside_corrected_mean", "outside_raw_mean", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "qc_warnings", "requested_min_size", "roi_confidence", "roi_score", "shape", "smoothing_sigma", "split_touching", "threshold", "threshold_method", "threshold_percentile", "threshold_scope", "top_bright_outside_fraction", "voxel_spacing"], "meta_keys": ["axes", "background_method", "background_percentile", "background_radius", "boundary_mask", "confidence_drivers", "distribution_flag", "dtype", "empty_mask", "fill_holes", "high_snr", "high_threshold", "inside_corrected_mean", "inside_raw_mean", "mask_fraction", "min_area_um2", "min_distance", "min_distance_um", "min_size", "min_snr", "min_volume_um3", "n_objects", "noise_sigma", "object_area_max", "object_area_median", "object_area_min", "object_unit", "outside_corrected_mean", "outside_raw_mean", "qc_warnings", "requested_min_size", "roi_confidence", "roi_score", "segmentation_method", "shape", "smoothing_sigma", "source_layer", "split_touching", "threshold", "threshold_method", "threshold_percentile", "threshold_scope", "top_bright_outside_fraction", "voxel_spacing"], "count": 1, "label_sha1": "c3a593dcd59621687a1f8f5178a7def6f00f9c87", "label_shape": [4, 80, 80], "label_dtype": "int32"},
    "auto_target_2d": {"result_keys": ["applied_params", "axes", "boundary_mask", "confidence_drivers", "correction_gap", "correction_history", "distribution_flag", "dtype", "inside_corrected_mean", "inside_raw_mean", "labels_layer", "mask_fraction", "max_iters", "min_size", "n_cells", "n_iterations", "n_objects", "noise_sigma", "object_unit", "outside_corrected_mean", "outside_raw_mean", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "qc_warnings", "requested_min_size", "roi_confidence", "roi_score", "shape", "threshold", "threshold_scope", "top_bright_outside_fraction", "voxel_spacing"], "meta_keys": ["auto_mask_hyperbright", "axes", "background_radius", "boundary_mask", "confidence_drivers", "correction_gap", "distribution_flag", "dtype", "empty_mask", "high_snr", "high_threshold", "inside_corrected_mean", "inside_raw_mean", "mask_fraction", "min_size", "min_snr", "n_iterations", "n_objects", "noise_sigma", "object_area_max", "object_area_median", "object_area_min", "object_unit", "outside_corrected_mean", "outside_raw_mean", "qc_warnings", "requested_min_size", "roi_confidence", "roi_score", "segmentation_method", "shape", "smoothing_sigma", "source_layer", "threshold", "threshold_clip_percentile", "threshold_scope", "top_bright_outside_fraction", "voxel_spacing"], "count": 2, "label_sha1": "7315f485269547bdd8ccdff009ef224498cc0ca1", "label_shape": [64, 64], "label_dtype": "int32"},
    "domain_2d": {"result_keys": ["boundary_mask", "counterstain_used", "counterstain_warnings", "domain_area_um2", "domain_label_count", "domain_volume_um3", "domain_voxels", "domain_warnings", "empty_mask", "labels_layer", "n_components", "noise_floor_threshold", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "threshold_scope"], "meta_keys": ["boundary_mask", "counterstain_used", "counterstain_warnings", "dark_percentile", "domain_area_um2", "domain_label_count", "domain_volume_um3", "domain_voxels", "domain_warnings", "empty_mask", "k_mad", "max_components", "merge_components", "min_area_um2", "min_component_fraction", "min_size_voxels", "min_volume_um3", "n_components", "n_components_raw", "n_components_retained", "noise_floor_threshold", "segmentation_method", "smooth_sigma_um", "source_layer", "threshold_image", "threshold_scope"], "count": 1, "label_sha1": "34eed685983bdb420258eb32b59616f405000e63", "label_shape": [64, 64], "label_dtype": "int32"},
    "domain_3d_roi": {"result_keys": ["boundary_mask", "counterstain_used", "counterstain_warnings", "domain_area_um2", "domain_label_count", "domain_volume_um3", "domain_voxels", "domain_warnings", "empty_mask", "labels_layer", "n_components", "noise_floor_threshold", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "threshold_scope"], "meta_keys": ["boundary_mask", "counterstain_used", "counterstain_warnings", "dark_percentile", "domain_area_um2", "domain_label_count", "domain_volume_um3", "domain_voxels", "domain_warnings", "empty_mask", "k_mad", "max_components", "merge_components", "min_area_um2", "min_component_fraction", "min_size_voxels", "min_volume_um3", "n_components", "n_components_raw", "n_components_retained", "noise_floor_threshold", "segmentation_method", "smooth_sigma_um", "source_layer", "threshold_image", "threshold_scope"], "count": 1, "label_sha1": "e2cd8fe9963f3396b844955e724993ac8f58f24b", "label_shape": [3, 60, 60], "label_dtype": "int32"},
    "auto3d": {"result_keys": ["axes", "boundary_mask", "candidate_summaries", "dtype", "empty_mask", "labels_layer", "mask_fraction", "min_z_planes", "n_cells", "n_objects", "object_area_max", "object_area_median", "object_area_min", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "qc_warnings", "roi_confidence", "segmentation_method", "selected_score", "selected_strategy", "selection_confidence", "shape", "single_plane_object_fraction", "top_bright_outside_fraction", "voxel_spacing", "z_gap_object_fraction"], "meta_keys": ["axes", "boundary_mask", "candidate_modes", "candidate_summaries", "dtype", "empty_mask", "include_cellpose_sam", "inside_corrected_mean", "inside_outside_separation_snr", "inside_raw_mean", "largest_to_median_object_ratio", "mask_fraction", "min_z_planes", "n_objects", "noise_sigma", "object_area_max", "object_area_median", "object_area_min", "outside_corrected_mean", "outside_raw_mean", "qc_warnings", "roi_confidence", "segmentation_method", "selected_score", "selected_strategy", "selection_confidence", "shape", "single_plane_object_fraction", "source_layer", "tiny_object_fraction", "top_bright_outside_fraction", "voxel_spacing", "z_gap_object_fraction"], "count": 1, "label_sha1": "6e93f91dbe3db8026a816c8f55278f2db97f115e", "label_shape": [4, 48, 48], "label_dtype": "int32"},
    "cellpose_2d": {"result_keys": ["anisotropy", "axes", "diameter", "do_3D", "dtype", "empty_mask", "labels_layer", "max_size_fraction", "min_size", "model", "n_cells", "object_area_max", "object_area_median", "object_area_min", "qc_png_error", "qc_png_path", "qc_png_skipped_reason", "qc_warnings", "shape"], "meta_keys": ["anisotropy", "axes", "diameter", "do_3D", "dtype", "empty_mask", "max_size_fraction", "min_size", "model", "n_objects", "object_area_max", "object_area_median", "object_area_min", "qc_warnings", "segmentation_method", "shape", "source_layer"], "count": 2, "label_sha1": "19203b1a435721052977b9dc379a0e8458a5d446", "label_shape": [64, 64], "label_dtype": "int32"},
    "correct_roi": {"result_keys": ["applied_params", "labels_layer", "n_objects", "ok", "qc_png_path", "qc_warnings", "replaced_with", "roi_confidence", "roi_score", "threshold", "threshold_scope"], "meta_keys": ["axes", "background_method", "background_percentile", "background_radius", "boundary_mask", "confidence_drivers", "corrected_from", "distribution_flag", "dtype", "empty_mask", "fill_holes", "high_snr", "high_threshold", "inside_corrected_mean", "inside_raw_mean", "mask_fraction", "min_area_um2", "min_distance", "min_distance_um", "min_size", "min_snr", "min_volume_um3", "n_objects", "noise_sigma", "object_area_max", "object_area_median", "object_area_min", "object_unit", "outside_corrected_mean", "outside_raw_mean", "qc_warnings", "requested_min_size", "roi_confidence", "roi_score", "segmentation_method", "shape", "smoothing_sigma", "source_layer", "split_touching", "threshold", "threshold_method", "threshold_percentile", "threshold_scope", "top_bright_outside_fraction", "voxel_spacing"], "count": 2, "label_sha1": "7315f485269547bdd8ccdff009ef224498cc0ca1", "label_shape": [64, 64], "label_dtype": "int32"},
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_output_equivalence(viewer, monkeypatch, scenario: str) -> None:
    # Stub Cellpose everywhere (harmless for non-cellpose scenarios).
    monkeypatch.setattr(CELLPOSE_MODEL_TARGET, lambda *a, **k: _FakeCellpose())
    result = SCENARIOS[scenario](viewer)
    actual = _comparable(viewer, result)
    expected = GOLDEN.get(scenario)
    if expected is None:
        # Capture aid: run once on unchanged code, paste the printed line into GOLDEN.
        print(f"\nGOLDEN[{scenario!r}] = {actual!r}")
        pytest.fail(f"no GOLDEN pinned for {scenario!r} (see printed value)")
    assert actual == expected
