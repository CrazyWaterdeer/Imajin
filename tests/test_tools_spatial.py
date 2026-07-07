from __future__ import annotations

import numpy as np

from imajin import session as state
from imajin.tools import spatial


def _halves_parents(n=100, split=50):
    lab = np.ones((n, n), dtype=np.int32)
    lab[:, split:] = 2
    return lab


def test_assign_points_to_parents(viewer) -> None:
    state.reset_tables()
    viewer.add_labels(_halves_parents(), name="parents")
    pts = np.array([[10, 10], [10, 90], [10, 49]], dtype=float)  # p1, p2, p1
    viewer.add_points(pts, name="spots")

    res = spatial.assign_objects_to_parents("spots", "parents")

    df = state.get_table(res["table_name"])
    assert list(df["parent_id"]) == [1, 2, 1]
    assert res["n_assigned"] == 3
    summ = state.get_table(res["summary_table"])
    assert set(summ["parent_id"]) == {1, 2}
    assert int(summ.loc[summ["parent_id"] == 1, "n_objects"].iloc[0]) == 2


def test_assign_label_objects_flags_ambiguous(viewer) -> None:
    state.reset_tables()
    parents = np.ones((20, 20), dtype=np.int32)
    parents[:, 10:] = 2
    viewer.add_labels(parents, name="par2")
    objs = np.zeros((20, 20), dtype=np.int32)
    objs[2:5, 2:5] = 1          # entirely in parent 1
    objs[2:5, 8:12] = 2         # straddles the x=10 boundary
    viewer.add_labels(objs, name="objs")

    res = spatial.assign_objects_to_parents("objs", "par2")
    df = state.get_table(res["table_name"]).set_index("object_id")

    assert df.loc[1, "parent_id"] == 1
    assert not df.loc[1, "assignment_ambiguous"]
    assert bool(df.loc[2, "assignment_ambiguous"])
    assert res["n_ambiguous"] == 1


def test_distance_points_to_reference_microns(viewer) -> None:
    state.reset_tables()
    ref = np.zeros((64, 64), dtype=np.uint8)
    ref[:, 50] = 1  # a vertical reference line at x=50
    viewer.add_labels(ref, name="membrane", scale=(0.5, 0.5))
    viewer.add_points(np.array([[10, 40]], dtype=float), name="p", scale=(0.5, 0.5))

    res = spatial.measure_distance_to_reference("p", "membrane")
    df = state.get_table(res["table_name"])
    # 10 px in x to the line, at 0.5 µm/px => 5.0 µm.
    assert np.isclose(df["distance_um"].iloc[0], 5.0, atol=1e-6)


def test_distance_signed_is_negative_inside(viewer) -> None:
    state.reset_tables()
    ref = np.zeros((40, 40), dtype=np.uint8)
    ref[10:30, 10:30] = 1
    viewer.add_labels(ref, name="blob", scale=(1.0, 1.0))
    viewer.add_points(np.array([[20, 20]], dtype=float), name="inside_pt")

    res = spatial.measure_distance_to_reference("inside_pt", "blob", signed=True)
    df = state.get_table(res["table_name"])
    assert df["distance_um"].iloc[0] < 0


def test_nearest_neighbor_within_set(viewer) -> None:
    state.reset_tables()
    pts = np.array([[0, 0], [0, 10], [0, 25]], dtype=float)
    viewer.add_points(pts, name="nnpts", scale=(1.0, 1.0))

    res = spatial.nearest_neighbor_distances("nnpts")
    df = state.get_table(res["table_name"]).set_index("object_id")
    assert np.isclose(df.loc[0, "nn_distance_um"], 10.0)
    assert np.isclose(df.loc[1, "nn_distance_um"], 10.0)
    assert np.isclose(df.loc[2, "nn_distance_um"], 15.0)


def test_nearest_neighbor_between_sets(viewer) -> None:
    state.reset_tables()
    viewer.add_points(np.array([[0, 0], [0, 100]], dtype=float), name="src", scale=(1.0, 1.0))
    viewer.add_points(np.array([[0, 5], [0, 90]], dtype=float), name="tgt", scale=(1.0, 1.0))

    res = spatial.nearest_neighbor_distances("src", other_layer="tgt")
    df = state.get_table(res["table_name"]).set_index("object_id")
    assert np.isclose(df.loc[0, "nn_distance_um"], 5.0)
    assert np.isclose(df.loc[1, "nn_distance_um"], 10.0)
