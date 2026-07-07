from __future__ import annotations

import numpy as np

from imajin.tools import coloc


def _puncta(shape=(64, 64), n=25, seed=1, amp=800.0):
    rng = np.random.default_rng(seed)
    img = rng.normal(40, 3, size=shape).astype(np.float64)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for _ in range(n):
        cy, cx = rng.integers(4, shape[0] - 4), rng.integers(4, shape[1] - 4)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2))
    return img


def test_block_shuffle_preserves_values_and_shape():
    rng = np.random.default_rng(0)
    img = np.arange(36, dtype=float).reshape(6, 6)
    out = coloc._block_shuffle(img, 2, rng)
    assert out.shape == img.shape
    assert sorted(out.ravel()) == sorted(img.ravel())


def test_costes_threshold_on_correlated_channels(viewer):
    a = _puncta(seed=2)
    # Correlated puncta + independent background noise so the low-intensity
    # (noise) regime decorrelates and Costes finds a threshold above the floor.
    b = 0.8 * a + np.random.default_rng(7).normal(0, 20, a.shape)
    viewer.add_image(a, name="ca")
    viewer.add_image(b, name="cb")
    res = coloc.costes_threshold("ca", "cb")
    assert res["slope"] > 0
    assert res["threshold_a"] > a.min()
    assert res["M1_above"] > 0.5


def test_costes_significance_flags_colocalized_but_not_independent(viewer):
    a = _puncta(seed=3)
    viewer.add_image(a, name="sa")
    viewer.add_image(0.8 * a + np.random.default_rng(9).normal(0, 3, a.shape), name="sb")
    viewer.add_image(_puncta(seed=99), name="sc")  # independent puncta field

    coloc_res = coloc.costes_significance("sa", "sb", n=100)
    indep_res = coloc.costes_significance("sa", "sc", n=100)

    assert coloc_res["observed_r"] > 0.5
    assert coloc_res["significant"]
    assert not indep_res["significant"]


def test_costes_significance_channel_shift_negative_control(viewer):
    a = _puncta(seed=4)
    viewer.add_image(a, name="na")
    viewer.add_image(np.roll(a, 24, axis=1), name="nb")  # decorrelating shift
    res = coloc.costes_significance("na", "nb", n=100)
    assert not res["significant"]


def test_object_colocalization_colocalized_vs_disjoint(viewer):
    viewer.add_labels(np.ones((64, 64), dtype=np.uint8), name="specimen")

    viewer.add_points(
        np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 15]], float),
        name="a_pts",
        scale=(1.0, 1.0),
    )
    viewer.add_points(
        np.array([[10, 11], [20, 21], [30, 31], [40, 41], [50, 16]], float),
        name="b_pts",
        scale=(1.0, 1.0),
    )
    coloc_res = coloc.object_colocalization("a_pts", "b_pts", "specimen", max_distance_um=2.0, n=100)
    assert coloc_res["observed_fraction"] == 1.0
    assert coloc_res["significant"]

    viewer.add_points(np.array([[60, 60], [61, 61]], float), name="far_pts", scale=(1.0, 1.0))
    disjoint = coloc.object_colocalization("a_pts", "far_pts", "specimen", max_distance_um=2.0, n=100)
    assert disjoint["observed_fraction"] == 0.0
    assert not disjoint["significant"]
