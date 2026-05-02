from __future__ import annotations

import base64
import os

import numpy as np
import pytest

from imajin.tools import view

needs_gl = pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") == "offscreen",
    reason="viewer.screenshot needs a real OpenGL context (vispy fails on offscreen Qt)",
)


def _zstack(viewer, name="vol"):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1024, size=(8, 32, 32), dtype=np.uint16)
    viewer.add_image(
        data, name=name, scale=(0.5, 0.2, 0.2), metadata={"axes": "ZYX"}
    )
    return data


def test_set_view_toggles_2d_3d(viewer) -> None:
    _zstack(viewer)
    res = view.set_view(ndisplay=3)
    assert res["ndisplay"] == 3
    res = view.set_view(ndisplay=2)
    assert res["ndisplay"] == 2


def test_set_view_applies_camera_params(viewer) -> None:
    _zstack(viewer)
    res = view.set_view(ndisplay=3, angles=(0, 45, 0), zoom=1.5)
    assert res["ndisplay"] == 3
    assert res["angles"][1] == pytest.approx(45.0)
    assert res["zoom"] == pytest.approx(1.5)


def test_set_view_rejects_invalid_ndisplay(viewer) -> None:
    with pytest.raises(ValueError, match="ndisplay"):
        view.set_view(ndisplay=4)


def test_set_colormap_changes_layer(viewer) -> None:
    _zstack(viewer, name="vol")
    res = view.set_colormap("vol", "viridis")
    assert res["colormap"] == "viridis"
    assert viewer.layers["vol"].colormap.name == "viridis"


def test_extract_timepoint_adds_reference_frame(viewer) -> None:
    data = np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
    viewer.add_image(data, name="movie", metadata={"axes": "TYX"})

    res = view.extract_timepoint("movie", t=1)

    assert res["shape"] == (16, 16)
    out = np.asarray(viewer.layers[res["new_layer"]].data)
    np.testing.assert_array_equal(out, data[1])


@needs_gl
def test_screenshot_returns_base64(viewer) -> None:
    _zstack(viewer, name="vol")
    res = view.screenshot()
    assert res["path"] is None
    decoded = base64.b64decode(res["thumb_base64"])
    assert decoded.startswith(b"\x89PNG")


@needs_gl
def test_screenshot_saves_file(viewer, tmp_path) -> None:
    _zstack(viewer, name="vol")
    out = tmp_path / "snap.png"
    res = view.screenshot(path=str(out))
    assert out.exists()
    assert res["path"] == str(out.resolve())


def test_max_projection_z_axis(viewer) -> None:
    data = _zstack(viewer, name="vol")
    res = view.max_projection("vol", axis="z")
    assert res["shape"] == (32, 32)
    out = np.asarray(viewer.layers[res["new_layer"]].data)
    np.testing.assert_array_equal(out, data.max(axis=0))


def test_max_projection_integer_axis(viewer) -> None:
    _zstack(viewer, name="vol")
    res = view.max_projection("vol", axis=1)
    assert res["shape"] == (8, 32)
    assert res["axis"] == 1


def test_orthogonal_views_adds_xz_yz(viewer) -> None:
    data = _zstack(viewer, name="vol")
    res = view.orthogonal_views("vol")
    xz = np.asarray(viewer.layers[res["xz_layer"]].data)
    yz = np.asarray(viewer.layers[res["yz_layer"]].data)
    np.testing.assert_array_equal(xz, data.max(axis=1))
    np.testing.assert_array_equal(yz, data.max(axis=2))
    assert res["xz_shape"] == (8, 32)
    assert res["yz_shape"] == (8, 32)


def test_orthogonal_views_rejects_2d_layer(viewer) -> None:
    viewer.add_image(np.zeros((16, 16), dtype=np.uint16), name="flat")
    with pytest.raises(ValueError, match="3D"):
        view.orthogonal_views("flat")


@pytest.mark.slow
@needs_gl
def test_animate_z_rotation_writes_gif(viewer, tmp_path) -> None:
    _zstack(viewer, name="vol")
    out = tmp_path / "rot.gif"
    res = view.animate_z_rotation(path=str(out), frames=6, fps=12)
    assert out.exists()
    assert out.stat().st_size > 0
    assert res["frames"] == 6
