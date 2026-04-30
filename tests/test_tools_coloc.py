from __future__ import annotations

import numpy as np
import pytest

from imajin.tools import coloc


def test_pearson_is_one_for_identical_channels(viewer) -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1000, size=(64, 64), dtype=np.uint16)
    viewer.add_image(a, name="ch_a")
    viewer.add_image(a.copy(), name="ch_b")

    res = coloc.pearson_correlation("ch_a", "ch_b")
    assert res["r"] == pytest.approx(1.0)
    assert res["n_pixels"] == 64 * 64


def test_pearson_is_zero_for_independent_channels(viewer) -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(size=(256, 256))
    b = rng.standard_normal(size=(256, 256))
    viewer.add_image(a.astype(np.float32), name="ch_a")
    viewer.add_image(b.astype(np.float32), name="ch_b")

    res = coloc.pearson_correlation("ch_a", "ch_b")
    assert abs(res["r"]) < 0.05


def test_manders_high_when_signals_overlap(viewer) -> None:
    a = np.zeros((40, 40), dtype=np.float32)
    b = np.zeros_like(a)
    a[10:30, 10:30] = 1000.0
    b[10:30, 10:30] = 800.0
    viewer.add_image(a, name="ch_a")
    viewer.add_image(b, name="ch_b")

    res = coloc.manders_coefficients(
        "ch_a", "ch_b", threshold_a=10.0, threshold_b=10.0
    )
    assert res["M1"] == pytest.approx(1.0)
    assert res["M2"] == pytest.approx(1.0)


def test_manders_low_when_signals_disjoint(viewer) -> None:
    a = np.zeros((40, 40), dtype=np.float32)
    b = np.zeros_like(a)
    a[5:15, 5:15] = 1000.0
    b[25:35, 25:35] = 800.0
    viewer.add_image(a, name="ch_a")
    viewer.add_image(b, name="ch_b")

    res = coloc.manders_coefficients(
        "ch_a", "ch_b", threshold_a=10.0, threshold_b=10.0
    )
    assert res["M1"] == pytest.approx(0.0)
    assert res["M2"] == pytest.approx(0.0)


def test_pearson_with_mask(viewer) -> None:
    a = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    b = a.copy()
    mask = np.zeros_like(a, dtype=np.uint8)
    mask[20:40, 20:40] = 1
    viewer.add_image(a, name="ch_a")
    viewer.add_image(b, name="ch_b")
    viewer.add_labels(mask.astype(np.int32), name="mask")

    res = coloc.pearson_correlation("ch_a", "ch_b", mask="mask")
    assert res["r"] == pytest.approx(1.0)
    assert res["n_pixels"] == 20 * 20
