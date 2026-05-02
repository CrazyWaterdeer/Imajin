from __future__ import annotations

import numpy as np
import pytest

from imajin.tools import segment


@pytest.mark.slow
def test_cellpose_sam_finds_blobs_2d(viewer, synthetic_blob_image) -> None:
    viewer.add_image(synthetic_blob_image, name="blobs")

    res = segment.cellpose_sam("blobs", do_3D=False, diameter=10)

    assert res["labels_layer"] == "blobs_masks"
    assert res["n_cells"] >= 3, f"expected >=3 cells, got {res['n_cells']}"
    labels = np.asarray(viewer.layers["blobs_masks"].data)
    assert labels.shape == synthetic_blob_image.shape
    assert labels.max() == res["n_cells"]


@pytest.mark.slow
def test_cellpose_sam_propagates_scale(viewer, synthetic_blob_image) -> None:
    viewer.add_image(synthetic_blob_image, name="blobs", scale=(0.2, 0.2))
    segment.cellpose_sam("blobs", do_3D=False, diameter=10)
    out = viewer.layers["blobs_masks"]
    assert tuple(float(s) for s in out.scale) == (0.2, 0.2)


def test_cellpose_sam_rejects_non_2d_3d(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 256, size=(2, 4, 16, 16), dtype=np.uint16)
    viewer.add_image(data, name="huge", metadata={"axes": "ZYXA"})
    with pytest.raises(ValueError, match="2D \\(YX\\) or 3D \\(ZYX\\)"):
        segment.cellpose_sam("huge")


def test_cellpose_sam_rejects_time_series(viewer) -> None:
    data = np.random.default_rng(0).integers(0, 256, size=(4, 16, 16), dtype=np.uint16)
    viewer.add_image(data, name="movie", metadata={"axes": "TYX"})
    with pytest.raises(ValueError, match="time-series"):
        segment.cellpose_sam("movie")
