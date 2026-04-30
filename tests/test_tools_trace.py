from __future__ import annotations

import numpy as np

from imajin.tools import trace


def _make_branched_mask() -> np.ndarray:
    """Y-shape: a vertical trunk + two diagonal branches in the upper half."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 31:33] = 1
    for i in range(20):
        img[10 + i, 31 - i] = 1
        img[10 + i, 32 + i] = 1
    return img


def test_skeletonize_returns_id_and_layer(viewer) -> None:
    trace.reset_skeletons()
    mask = _make_branched_mask()
    viewer.add_labels(mask, name="ymask")

    res = trace.skeletonize("ymask")
    assert res["skeleton_id"].startswith("skel_")
    assert res["n_paths"] >= 3  # trunk + 2 branches at minimum
    assert f"ymask_skeleton" in viewer.layers


def test_extract_branch_metrics_table(viewer) -> None:
    trace.reset_skeletons()
    viewer.add_labels(_make_branched_mask(), name="ymask")
    skel_res = trace.skeletonize("ymask")

    res = trace.extract_branch_metrics(skel_res["skeleton_id"])
    assert res["n_branches"] >= 3
    assert "branch_type_counts" in res

    from imajin.agent.state import get_table

    df = get_table(res["table_name"])
    assert "branch_length" in df.columns
    assert "branch_type" in df.columns
    assert (df["branch_length"] > 0).all()


def test_morphology_descriptors_shape(viewer) -> None:
    trace.reset_skeletons()
    viewer.add_labels(_make_branched_mask(), name="ymask")
    skel_res = trace.skeletonize("ymask")

    res = trace.compute_morphology_descriptors(skel_res["skeleton_id"])
    assert res["total_length"] > 0
    assert res["n_branches"] >= 3
    assert "note" in res  # stub disclosure


def test_query_connectome_is_stub(viewer) -> None:
    res = trace.query_connectome("any_id", db="neuprint", k=5)
    assert res["status"] == "not_implemented"
    assert res["matches"] == []


def test_classify_neuron_type_is_stub(viewer) -> None:
    res = trace.classify_neuron_type("any_id")
    assert res["status"] == "not_implemented"


def test_skeletonize_rejects_4d(viewer) -> None:
    trace.reset_skeletons()
    viewer.add_labels(np.zeros((2, 2, 16, 16), dtype=np.uint8), name="bad")
    import pytest

    with pytest.raises(ValueError, match="2D or 3D"):
        trace.skeletonize("bad")
