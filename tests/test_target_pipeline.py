from __future__ import annotations

import numpy as np

from imajin.analysis.target_pipeline import (
    auto_correct_target,
    next_correction,
    segment_target_array,
)


# --- next_correction policy (deterministic, no image) ---


def test_next_correction_zero_objects_lowers_min_snr() -> None:
    move = next_correction({"n_objects": 0}, {"min_snr": 4.0})
    assert move == {"min_snr": 2.0}  # halved for fast recovery
    # already at the floor -> no move
    assert next_correction({"n_objects": 0}, {"min_snr": 1.0}) is None


def test_next_correction_too_narrow_lowers_bar() -> None:
    move = next_correction(
        {"n_objects": 5, "mask_fraction": 0.1, "top_bright_outside_fraction": 0.5},
        {"min_snr": 2.0, "high_snr": 4.0},
    )
    assert move["min_snr"] == 1.5
    assert move["high_snr"] == 3.0


def test_next_correction_too_wide_raises_then_masks() -> None:
    metrics = {"n_objects": 8, "mask_fraction": 0.6, "top_bright_outside_fraction": 0.0}
    assert next_correction(metrics, {"min_snr": 2.0}) == {"min_snr": 3.0}
    # once min_snr is maxed, escalate to the hyper-bright mask, then the clip
    assert next_correction(metrics, {"min_snr": 5.0}) == {"auto_mask_hyperbright": True}
    assert next_correction(
        metrics, {"min_snr": 5.0, "auto_mask_hyperbright": True}
    ) == {"threshold_clip_percentile": 99.0}


def test_next_correction_weak_separation_widens_background() -> None:
    move = next_correction(
        {
            "n_objects": 4,
            "mask_fraction": 0.1,
            "top_bright_outside_fraction": 0.0,
            "inside_outside_separation_snr": 0.5,
        },
        {"background_radius": 48},
    )
    assert move == {"background_radius": 96}


def test_next_correction_clean_roi_has_no_move() -> None:
    move = next_correction(
        {
            "n_objects": 6,
            "mask_fraction": 0.08,
            "top_bright_outside_fraction": 0.02,
            "inside_outside_separation_snr": 3.0,
        },
        {"min_snr": 2.0},
    )
    assert move is None


# --- auto_correct_target loop (real images, safety properties) ---


def _two_blob_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.normal(20.0, 1.0, (128, 128)).astype(np.float32)
    image[40:54, 40:54] += 8.0  # ~8-sigma blobs above the background
    image[80:94, 70:84] += 8.0
    return image


def test_auto_correct_recovers_from_too_high_min_snr() -> None:
    # Start with an over-strict threshold so the first pass finds nothing; the
    # loop should lower min_snr and recover the two blobs.
    image = _two_blob_image()
    best, best_params, history = auto_correct_target(
        image,
        spacing=None,
        params={"min_snr": 18.0, "min_size": 30, "smoothing_sigma": 0.0, "background_radius": 16},
        max_iters=3,
    )
    assert history[0]["n_objects"] == 0  # started empty
    assert best.qc["n_objects"] == 2  # recovered
    assert best.roi_score > history[0]["score"]  # strictly improved
    assert best_params["min_snr"] < 10.0  # the bar was lowered
    assert len(history) <= 4  # bounded by max_iters + initial


def test_auto_correct_idempotent_on_a_clean_start() -> None:
    image = _two_blob_image()
    best, best_params, history = auto_correct_target(
        image,
        spacing=None,
        params={"min_snr": 2.0, "min_size": 30, "smoothing_sigma": 0.0, "background_radius": 16},
        max_iters=3,
    )
    # A clean 2-blob start is already confident -> no iteration.
    assert best.qc["n_objects"] == 2
    assert len(history) == 1
    assert best_params["min_snr"] == 2.0


def test_auto_correct_keeps_best_scoring_candidate() -> None:
    image = _two_blob_image()
    best, _params, history = auto_correct_target(
        image,
        spacing=None,
        params={"min_snr": 18.0, "min_size": 30, "smoothing_sigma": 0.0, "background_radius": 16},
        max_iters=3,
    )
    assert best.roi_score == max(h["score"] for h in history)


def test_segment_target_array_and_loop_agree_at_fixed_params() -> None:
    # With max_iters=0 the loop is just a single pass and must match the
    # single-shot helper.
    image = _two_blob_image()
    params = {"min_snr": 2.0, "min_size": 30, "smoothing_sigma": 0.0, "background_radius": 16}
    best, _params, history = auto_correct_target(
        image, spacing=None, params=dict(params), max_iters=0
    )
    single = segment_target_array(image, spacing=None, **params)
    assert len(history) == 1
    assert int(best.masks.max()) == int(single.masks.max())
    assert best.roi_score == single.roi_score
