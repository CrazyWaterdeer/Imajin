from __future__ import annotations

import numpy as np

from imajin import session as state
from imajin.tools import trace


def _two_collinear_segments():
    """Horizontal centreline broken by a 5 px gap (cols 18..22)."""
    img = np.zeros((40, 40), dtype=np.uint8)
    img[20, 2:18] = 1
    img[20, 23:38] = 1
    return img


def _support_band(gap=False):
    img = np.zeros((40, 40), dtype=np.float32)
    img[19:22, :] = 1.0  # faint signal along the whole centreline...
    if gap:
        img[:, 18:23] = 0.0  # ...except across the gap
    return img


def _two_parallel_segments():
    img = np.zeros((40, 40), dtype=np.uint8)
    img[18, 10:26] = 1
    img[22, 10:26] = 1
    return img


def _branched():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 31:33] = 1
    for i in range(20):
        img[10 + i, 31 - i] = 1
        img[10 + i, 32 + i] = 1
    return img


def test_propose_bridges_accepts_aligned_supported_gap(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_two_collinear_segments(), name="broken")
    viewer.add_image(_support_band(), name="support")
    sid = trace.skeletonize("broken")["skeleton_id"]

    res = trace.propose_filament_bridges(sid, max_gap_um=8.0, support_layer="support")
    assert res["n_accepted"] == 1
    df = state.get_table(res["table_name"])
    assert bool(df[df["accepted"]].iloc[0]["accepted"])


def test_build_rooted_tree_spans_bridged_components(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_two_collinear_segments(), name="broken2")
    viewer.add_image(_support_band(), name="support2")
    sid = trace.skeletonize("broken2")["skeleton_id"]
    trace.propose_filament_bridges(sid, max_gap_um=8.0, support_layer="support2")

    tree = trace.build_rooted_tree(sid)
    assert tree["n_components_merged"] == 2
    assert tree["n_bridges_applied"] == 1
    assert tree["n_dropped"] == 0
    df = state.get_table(tree["table_name"])
    assert int((df["parent_id"] == -1).sum()) == 1  # exactly one root


def test_propose_bridges_rejects_perpendicular_tangent(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_two_parallel_segments(), name="parallel")
    sid = trace.skeletonize("parallel")["skeleton_id"]

    res = trace.propose_filament_bridges(sid, max_gap_um=8.0)
    assert res["n_accepted"] == 0
    df = state.get_table(res["table_name"])
    assert (df["reason"] == "tangent").any()


def test_propose_bridges_rejects_unsupported_gap(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_two_collinear_segments(), name="broken3")
    viewer.add_image(_support_band(gap=True), name="nosupport")
    sid = trace.skeletonize("broken3")["skeleton_id"]

    res = trace.propose_filament_bridges(
        sid, max_gap_um=8.0, support_layer="nosupport", min_support=0.5
    )
    assert res["n_accepted"] == 0
    df = state.get_table(res["table_name"])
    assert (df["reason"] == "support").any()


def test_build_rooted_tree_connected_and_drops_unbridged(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    # Connected Y: every node reachable, single root, nothing dropped.
    viewer.add_labels(_branched(), name="ytree")
    sid = trace.skeletonize("ytree")["skeleton_id"]
    tree = trace.build_rooted_tree(sid)
    assert tree["n_dropped"] == 0
    assert tree["n_components_merged"] == 1

    # Two disconnected segments with no bridges → only the root component survives.
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_two_collinear_segments(), name="broken4")
    sid2 = trace.skeletonize("broken4")["skeleton_id"]
    tree2 = trace.build_rooted_tree(sid2)  # no propose_filament_bridges called
    assert tree2["n_dropped"] > 0
    assert tree2["n_bridges_applied"] == 0
