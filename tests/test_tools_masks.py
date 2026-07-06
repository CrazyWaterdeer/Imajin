from __future__ import annotations

import numpy as np
import pytest

from imajin import session as state
from imajin.tools import get_tool, masks, measure
from imajin.tools.masks import _align, _combine_masks, _MaskError


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


# --- pure core: _combine_masks -------------------------------------------------------

def _blocks():
    a = np.zeros((6, 6), dtype=bool)
    a[1:4, 1:4] = True
    b = np.zeros((6, 6), dtype=bool)
    b[2:5, 2:5] = True
    return a, b


def test_combine_not() -> None:
    a, _ = _blocks()
    assert np.array_equal(_combine_masks("not", a, None, None), ~a)


def test_combine_and_or_subtract() -> None:
    a, b = _blocks()
    assert np.array_equal(_combine_masks("and", a, b, None), a & b)
    assert np.array_equal(_combine_masks("or", a, b, None), a | b)
    assert np.array_equal(_combine_masks("subtract", a, b, None), a & ~b)


def test_combine_within_clips_every_op() -> None:
    a, b = _blocks()
    within = np.zeros((6, 6), dtype=bool)
    within[0:3, 0:3] = True
    for op in ("not", "and", "or", "subtract"):
        out = _combine_masks(op, a, b, within)
        assert not out[~within].any()  # result subset of within


def test_subtract_equals_and_not() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((8, 8)) > 0.5
    b = rng.random((8, 8)) > 0.5
    assert np.array_equal(
        _combine_masks("subtract", a, b, None),
        _combine_masks("and", a, _combine_masks("not", b, None, None), None),
    )


def test_combine_binary_op_requires_b() -> None:
    a, _ = _blocks()
    with pytest.raises(_MaskError):
        _combine_masks("and", a, None, None)


def test_combine_empty_result_is_returned() -> None:
    a, _ = _blocks()
    out = _combine_masks("and", a, ~a, None)
    assert out.sum() == 0


# --- pure core: _align ---------------------------------------------------------------

def test_align_broadcasts_2d_to_3d() -> None:
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:3] = True
    aligned, bz = _align((3, 4, 4), m, broadcast_2d_to_3d=True)
    assert aligned.shape == (3, 4, 4)
    assert bz is True
    assert all(np.array_equal(aligned[0], aligned[z]) for z in range(3))


def test_align_refuses_broadcast_when_disabled() -> None:
    m = np.zeros((4, 4), dtype=bool)
    with pytest.raises(_MaskError):
        _align((3, 4, 4), m, broadcast_2d_to_3d=False)


def test_align_incompatible_shape_raises() -> None:
    m = np.zeros((4, 4), dtype=bool)
    with pytest.raises(_MaskError):
        _align((5, 5), m, broadcast_2d_to_3d=True)


# --- tool ----------------------------------------------------------------------------

def test_mask_logic_is_registered() -> None:
    assert get_tool("mask_logic") is not None


def _add(viewer, arr, name, **kw):
    return viewer.add_image(arr.astype(np.uint8), name=name, **kw)


def test_subtract_equals_not_within(viewer) -> None:
    green = np.zeros((10, 10), dtype=np.uint8)
    green[2:6, 2:6] = 1
    specimen = np.zeros((10, 10), dtype=np.uint8)
    specimen[1:9, 1:9] = 1
    _add(viewer, green, "green")
    _add(viewer, specimen, "specimen")

    r1 = masks.mask_logic("subtract", "specimen", "green")
    r2 = masks.mask_logic("not", "green", within_layer="specimen")
    assert r1["ok"] and r2["ok"]
    m1 = np.asarray(viewer.layers[r1["mask_layer"]].data)
    m2 = np.asarray(viewer.layers[r2["mask_layer"]].data)
    assert np.array_equal(m1 > 0, m2 > 0)


def test_2d_a_3d_b_outputs_3d_with_reference_scale(viewer) -> None:
    a2d = np.ones((8, 8), dtype=np.uint8)
    b3d = np.zeros((4, 8, 8), dtype=np.uint8)
    b3d[:, 2:6, 2:6] = 1
    _add(viewer, a2d, "a2d")
    viewer.add_image(b3d, name="b3d", scale=(0.5, 0.3, 0.3))

    res = masks.mask_logic("and", "a2d", "b3d")
    assert res["ok"] and res["broadcast_z"] is True
    out = viewer.layers[res["mask_layer"]]
    assert np.asarray(out.data).shape == (4, 8, 8)
    assert tuple(out.scale) == (0.5, 0.3, 0.3)


def test_mask_logic_output_feeds_measure_intensity(viewer) -> None:
    green = np.zeros((10, 10), dtype=np.uint8)
    green[2:6, 2:6] = 1
    specimen = np.ones((10, 10), dtype=np.uint8)
    red = np.zeros((10, 10), dtype=np.float32)
    red[6:9, 6:9] = 500.0
    _add(viewer, green, "green")
    _add(viewer, specimen, "specimen")
    viewer.add_image(red, name="red")

    res = masks.mask_logic("subtract", "specimen", "green")
    meas = measure.measure_intensity(labels_layer=res["mask_layer"], image_layers=["red"])
    assert meas["n_rows"] == 1  # one foreground region (label 1)


def test_axes_guard_rejects_time_series(viewer) -> None:
    tyx = np.ones((3, 8, 8), dtype=np.uint8)
    viewer.add_image(tyx, name="movie", metadata={"axes": "TYX"})
    res = masks.mask_logic("not", "movie")
    assert res["ok"] is False
    assert "axes" in res["error"].lower()


def test_scale_mismatch_warns(viewer) -> None:
    a = np.ones((8, 8), dtype=np.uint8)
    b = np.ones((8, 8), dtype=np.uint8)
    viewer.add_image(a, name="a", scale=(1.0, 1.0))
    viewer.add_image(b, name="b", scale=(2.0, 2.0))
    res = masks.mask_logic("and", "a", "b")
    assert res["ok"] and res["scale_mismatch"] is True
    assert any("scale" in w for w in res["warnings"])


def test_not_with_b_layer_warns(viewer) -> None:
    _add(viewer, np.ones((6, 6), dtype=np.uint8), "a")
    _add(viewer, np.ones((6, 6), dtype=np.uint8), "b")
    res = masks.mask_logic("not", "a", b_layer="b")
    assert res["ok"] and any("ignored" in w for w in res["warnings"])


def test_binary_op_missing_b_fails(viewer) -> None:
    _add(viewer, np.ones((6, 6), dtype=np.uint8), "a")
    res = masks.mask_logic("and", "a")
    assert res["ok"] is False


def test_empty_result_ok_and_flagged(viewer) -> None:
    a = np.zeros((6, 6), dtype=np.uint8)
    a[1:3, 1:3] = 1
    _add(viewer, a, "a")
    _add(viewer, a, "b")  # identical
    res = masks.mask_logic("subtract", "a", "b")  # a & ~a == empty
    assert res["ok"] is True and res["empty"] is True
    assert res["voxels"] == 0
