from __future__ import annotations

import numpy as np

from imajin.tools import preprocess


def test_rolling_ball_creates_subtracted_layer(viewer) -> None:
    rng = np.random.default_rng(0)
    data = (
        rng.integers(50, 200, size=(64, 64), dtype=np.uint16).astype(np.int32)
        + 200  # uniform background
    ).astype(np.uint16)
    viewer.add_image(data, name="img")

    res = preprocess.rolling_ball_background("img", radius=15)

    assert res["new_layer"] == "img_rb"
    assert "img_rb" in [L.name for L in viewer.layers]
    out = np.asarray(viewer.layers["img_rb"].data)
    assert out.mean() < data.mean()


def test_rolling_ball_3d_processes_each_z(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 256, size=(4, 32, 32), dtype=np.uint16)
    viewer.add_image(data, name="stack")
    res = preprocess.rolling_ball_background("stack", radius=10)
    assert viewer.layers[res["new_layer"]].data.shape == (4, 32, 32)


def test_auto_contrast_outputs_float_in_zero_to_one(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 4096, size=(64, 64), dtype=np.uint16)
    viewer.add_image(data, name="img")
    res = preprocess.auto_contrast("img", low_pct=2, high_pct=98)
    out = np.asarray(viewer.layers[res["new_layer"]].data)
    assert "float" in str(out.dtype)
    assert out.min() >= -1e-6
    assert out.max() <= 1.0 + 1e-6


def test_gaussian_denoise_preserves_shape_and_dtype(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 4096, size=(64, 64), dtype=np.uint16)
    viewer.add_image(data, name="img")
    res = preprocess.gaussian_denoise("img", sigma=1.5)
    out = np.asarray(viewer.layers[res["new_layer"]].data)
    assert out.shape == data.shape
    assert out.dtype == data.dtype


def test_preprocess_propagates_scale(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 256, size=(8, 32, 32), dtype=np.uint16)
    viewer.add_image(data, name="stack", scale=(0.5, 0.2, 0.2))
    res = preprocess.gaussian_denoise("stack", sigma=1.0)
    new = viewer.layers[res["new_layer"]]
    assert tuple(float(s) for s in new.scale) == (0.5, 0.2, 0.2)
