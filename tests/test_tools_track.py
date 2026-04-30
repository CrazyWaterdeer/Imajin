from __future__ import annotations

import numpy as np
import pytest

from imajin.tools import track


def _moving_blobs_4d(t: int = 5, size: int = 64) -> np.ndarray:
    """4 small disks moving linearly across T frames (TYX labels)."""
    labels = np.zeros((t, size, size), dtype=np.uint16)
    centers = [(15, 15), (15, 45), (45, 15), (45, 45)]
    radius = 4
    for ti in range(t):
        shift = ti
        for cell_id, (cy, cx) in enumerate(centers, start=1):
            yy, xx = np.mgrid[:size, :size]
            mask = (yy - (cy + shift)) ** 2 + (xx - (cx + shift)) ** 2 < radius ** 2
            labels[ti][mask] = cell_id
    return labels


@pytest.mark.slow
def test_track_cells_creates_tracks_layer_and_table(viewer) -> None:
    data = _moving_blobs_4d()
    viewer.add_labels(data, name="moving")

    res = track.track_cells("moving", search_radius=20.0)
    assert res["n_tracks"] >= 1
    assert res["n_detections"] >= len(data)
    assert res["tracks_layer"] in viewer.layers

    from imajin.agent.state import get_table

    df = get_table(res["tracks_table"])
    assert {"track_id", "t", "y", "x"}.issubset(df.columns)


def test_track_cells_rejects_2d(viewer) -> None:
    viewer.add_labels(np.zeros((16, 16), dtype=np.uint16), name="flat")
    with pytest.raises(ValueError, match="3D \\(TYX\\) or 4D"):
        track.track_cells("flat")
