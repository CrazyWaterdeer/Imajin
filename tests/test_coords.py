from __future__ import annotations

import numpy as np
import pytest

from imajin.analysis import coords


class _FakeLayer:
    def __init__(self, scale, translate=None):
        self.scale = scale
        if translate is not None:
            self.translate = translate


def test_normalize_vector_pads_and_truncates():
    assert coords.normalize_vector((0.5,), 3, fill=1.0) == (0.5, 1.0, 1.0)
    assert coords.normalize_vector((0.5, 0.25, 0.1, 9.9), 3, fill=1.0) == (0.5, 0.25, 0.1)
    assert coords.normalize_vector(None, 2, fill=0.0) == (0.0, 0.0)
    assert coords.normalize_vector((), 2, fill=1.0) == (1.0, 1.0)


def test_data_to_world_scale_only_and_with_translate():
    out = coords.data_to_world([20.0, 30.0], (0.5, 0.25))
    assert np.allclose(out, [10.0, 7.5])
    out_t = coords.data_to_world([20.0, 30.0], (0.5, 0.25), (2.0, 3.0))
    assert np.allclose(out_t, [12.0, 10.5])


def test_data_to_world_handles_stacks_of_points():
    pts = np.array([[0.0, 0.0], [10.0, 4.0]])
    out = coords.data_to_world(pts, (0.5, 0.25), (1.0, 1.0))
    assert np.allclose(out, [[1.0, 1.0], [6.0, 2.0]])


def test_world_to_data_is_inverse_of_data_to_world():
    scale, translate = (0.5, 0.25, 2.0), (2.0, 3.0, -1.0)
    pts = np.array([[20.0, 30.0, 4.0], [1.0, 2.0, 3.0]])
    world = coords.data_to_world(pts, scale, translate)
    back = coords.world_to_data(world, scale, translate)
    assert np.allclose(back, pts)


def test_point_to_world_uses_layer_scale_and_translate():
    layer = _FakeLayer(scale=(0.5, 0.25), translate=(2.0, 3.0))
    assert np.allclose(coords.point_to_world([20, 30], layer), [12.0, 10.5])
    assert np.allclose(
        coords.point_to_world([20, 30], layer, use_translate=False), [10.0, 7.5]
    )


def test_point_to_world_defaults_missing_transform():
    layer = _FakeLayer(scale=())  # no scale, no translate attribute
    assert np.allclose(coords.point_to_world([5, 7], layer), [5.0, 7.0])


def test_is_physical():
    assert not coords.is_physical((1.0, 1.0))
    assert coords.is_physical((0.5, 0.5))
    assert coords.is_physical((1.0, 1.0, 0.2))
