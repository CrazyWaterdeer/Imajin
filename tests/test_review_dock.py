"""Unit tests for ReviewDock state management.

Lock in memory-friendly invariants so future refactors don't reintroduce
unnecessary full-size copies of the labels/image arrays the dock holds
while the user is reviewing a sample.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _make_real_napari_viewer():
    """Return a real `napari.Viewer(show=False)`, or skip if unavailable.

    The default conftest `viewer` fixture swaps in a `_FakeViewer` when
    QT_QPA_PLATFORM=offscreen; that fake does not support add_layer for
    Points/Shapes nor the event API the dock binds to. For dock-specific
    tests we instantiate the real viewer directly.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import napari  # noqa: F401
    except Exception as exc:
        pytest.skip(f"napari unavailable: {exc}")
    try:
        import napari

        return napari.Viewer(show=False)
    except Exception as exc:
        pytest.skip(f"could not construct napari.Viewer: {exc}")


@pytest.fixture
def dock_viewer(qapp):
    v = _make_real_napari_viewer()
    try:
        yield v
    finally:
        try:
            v.layers.clear()
        except Exception:
            pass


def _seed_layers(viewer) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 1.0, size=(4, 32, 32)).astype(np.float32)
    img[1:3, 10:14, 10:14] += 200.0
    labels = np.zeros_like(img, dtype=np.int32)
    labels[1:3, 10:14, 10:14] = 1
    viewer.add_image(img, name="img")
    viewer.add_labels(labels, name="lab")
    return img, labels


def test_load_does_not_eager_copy_current_labels(dock_viewer):
    """Before the user rebuilds, _current_labels must share storage with
    _original_labels. An eager `.copy()` doubles the labels footprint
    (e.g. ~671 MB extra for a 2048x2048x40 int32 stack).
    """
    from imajin.ui.review_dock import ReviewDock

    _seed_layers(dock_viewer)
    dock = ReviewDock(viewer=dock_viewer)
    dock.request_layers("img", "lab")

    assert dock._current_labels is dock._original_labels


def test_rebuild_forks_current_labels_from_original(dock_viewer):
    """`_on_rebuild` must produce a new array so the shared reference is
    broken before any in-place writes. _original_labels stays intact.
    """
    from imajin.ui.review_dock import ReviewDock

    _img, labels = _seed_layers(dock_viewer)
    dock = ReviewDock(viewer=dock_viewer)
    dock.request_layers("img", "lab")

    original_snapshot = dock._original_labels.copy()
    dock._on_rebuild()

    assert dock._current_labels is not dock._original_labels
    np.testing.assert_array_equal(dock._original_labels, original_snapshot)
