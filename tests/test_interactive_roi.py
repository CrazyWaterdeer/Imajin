"""Algorithm-level tests for correct_roi_from_markings.

These tests exercise the pure NumPy logic without spinning up napari.
"""

from __future__ import annotations

import numpy as np

from imajin.analysis.interactive_roi import correct_roi_from_markings


def _make_3d_scene(
    *,
    shape: tuple[int, int, int] = (6, 60, 60),
    bright_center: tuple[int, int] = (30, 30),
    bright_radius: int = 8,
    bright_value: float = 200.0,
    dim_center: tuple[int, int] = (30, 50),
    dim_radius: int = 4,
    dim_value: float = 8.0,
    noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (corrected_3d, auto_labels_3d, dim_truth_mask_3d).

    Auto labels only cover the bright blob; the dim blob is the kind of
    signal user markings should recover.
    """
    Z, Y, X = shape
    rng = np.random.default_rng(0)
    corrected = rng.normal(0.0, noise, size=shape).astype(np.float32)

    yy, xx = np.ogrid[:Y, :X]
    by, bx = bright_center
    bright_disk_2d = ((yy - by) ** 2 + (xx - bx) ** 2) <= bright_radius ** 2
    dy, dx = dim_center
    dim_disk_2d = ((yy - dy) ** 2 + (xx - dx) ** 2) <= dim_radius ** 2

    bright_3d = np.broadcast_to(bright_disk_2d[None, :, :], shape).copy()
    dim_3d = np.broadcast_to(dim_disk_2d[None, :, :], shape).copy()
    # Make signal stronger on the middle z slices so argmax_z has a clear peak.
    z_falloff = np.exp(-((np.arange(Z) - Z / 2) ** 2) / (2 * (Z / 4) ** 2)).astype(
        np.float32
    )
    corrected[bright_3d] += (bright_value * z_falloff[:, None, None] * bright_3d)[
        bright_3d
    ]
    corrected[dim_3d] += (dim_value * z_falloff[:, None, None] * dim_3d)[dim_3d]

    # Auto labels only catch the bright blob.
    auto_labels = np.zeros(shape, dtype=np.int32)
    auto_labels[bright_3d] = 1
    return corrected, auto_labels, dim_3d


def test_add_point_recovers_dim_region_next_to_bright_one():
    corrected, auto_labels, dim_truth = _make_3d_scene()
    # The dim blob is centered at (30, 50). Click on it.
    new_labels, info = correct_roi_from_markings(
        auto_labels,
        corrected,
        add_points=[(30, 50)],
        noise_sigma=1.0,
        base_threshold=5.0,
        add_seed_growth_k_snr=1.5,
        min_size=4,
    )
    # Bright region still present.
    assert int((new_labels > 0).sum()) > int((auto_labels > 0).sum())
    # The dim disk is now covered (at least at the central z slice).
    mid_z = corrected.shape[0] // 2
    dim_at_mid = dim_truth[mid_z]
    covered = (new_labels[mid_z] > 0) & dim_at_mid
    assert covered.sum() >= 0.5 * dim_at_mid.sum(), (
        f"dim region poorly recovered: {covered.sum()}/{dim_at_mid.sum()}"
    )
    assert info["add_points_voxels"] > 0


def test_remove_point_erases_connected_component_under_seed():
    corrected, auto_labels, _ = _make_3d_scene()
    # Click on the bright blob to remove it.
    new_labels, info = correct_roi_from_markings(
        auto_labels,
        corrected,
        remove_points=[(30, 30)],
        noise_sigma=1.0,
        base_threshold=5.0,
        min_size=4,
    )
    assert int((new_labels > 0).sum()) == 0
    assert info["remove_points_voxels"] > 0


def test_remove_region_wins_over_add_point_when_overlapping():
    corrected, auto_labels, _ = _make_3d_scene()
    Y, X = corrected.shape[-2:]
    remove_region = np.zeros((Y, X), dtype=bool)
    remove_region[20:40, 20:40] = True  # covers the bright blob
    new_labels, _info = correct_roi_from_markings(
        auto_labels,
        corrected,
        add_points=[(30, 30)],
        remove_regions=[remove_region],
        noise_sigma=1.0,
        base_threshold=5.0,
        min_size=4,
    )
    assert int((new_labels[:, 20:40, 20:40] > 0).sum()) == 0


def test_add_region_with_low_snr_scale_recovers_dim_signal():
    corrected, auto_labels, dim_truth = _make_3d_scene()
    Y, X = corrected.shape[-2:]
    add_region = np.zeros((Y, X), dtype=bool)
    add_region[25:35, 45:55] = True  # rectangle around the dim blob
    new_labels, info = correct_roi_from_markings(
        auto_labels,
        corrected,
        add_regions=[add_region],
        noise_sigma=1.0,
        base_threshold=20.0,  # much higher than dim signal
        region_min_snr_scale=0.25,
        min_size=4,
    )
    mid_z = corrected.shape[0] // 2
    covered = (new_labels[mid_z] > 0) & dim_truth[mid_z]
    assert covered.sum() >= 0.5 * dim_truth[mid_z].sum()
    assert info["add_regions_voxels"] > 0


def test_2d_input_works_without_z_argmax():
    rng = np.random.default_rng(1)
    Y, X = 40, 40
    corrected = rng.normal(0.0, 1.0, size=(Y, X)).astype(np.float32)
    yy, xx = np.ogrid[:Y, :X]
    disk = ((yy - 20) ** 2 + (xx - 20) ** 2) <= 25
    corrected[disk] += 100.0
    auto_labels = np.zeros((Y, X), dtype=np.int32)
    auto_labels[disk] = 1

    new_labels, info = correct_roi_from_markings(
        auto_labels,
        corrected,
        remove_points=[(20, 20)],
        noise_sigma=1.0,
        min_size=4,
    )
    assert new_labels.shape == (Y, X)
    assert int((new_labels > 0).sum()) == 0
    assert info["remove_points_voxels"] > 0


def test_no_markings_returns_relabeled_auto_mask():
    corrected, auto_labels, _ = _make_3d_scene()
    new_labels, info = correct_roi_from_markings(
        auto_labels,
        corrected,
        noise_sigma=1.0,
        base_threshold=5.0,
        min_size=4,
    )
    # Exactly the same voxels labeled (possibly with re-indexed ids).
    assert (new_labels > 0).sum() == (auto_labels > 0).sum()
    assert info["add_points"] == 0
    assert info["remove_points"] == 0


def test_skipped_when_point_outside_image():
    corrected, auto_labels, _ = _make_3d_scene()
    _new_labels, info = correct_roi_from_markings(
        auto_labels,
        corrected,
        add_points=[(-5, 5), (1000, 1000)],
        noise_sigma=1.0,
        min_size=4,
    )
    assert info["skipped_points"] == 2
    assert info["add_points_voxels"] == 0
