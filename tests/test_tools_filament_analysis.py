from __future__ import annotations


import numpy as np
import pytest

from imajin import session as state
from imajin.tools import trace


def _thick_bar():
    """A horizontal bar 5 px thick → centreline skeleton of known width."""
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:15, 5:35] = 1
    return img


def _branched():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 31:33] = 1
    for i in range(20):
        img[10 + i, 31 - i] = 1
        img[10 + i, 32 + i] = 1
    return img


def test_measure_filament_diameter_recovers_width(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_thick_bar(), name="bar")
    sid = trace.skeletonize("bar")["skeleton_id"]

    res = trace.measure_filament_diameter(sid)
    assert 4.0 <= res["median_diameter_um"] <= 8.0  # ~5-6 px bar
    nodes = state.get_table(res["node_table"])
    assert {"radius_um", "diameter_um"} <= set(nodes.columns)
    # Radii are persisted for the SWC export.
    assert trace._entry(sid).record.parameters.get("node_radii_um")


def test_diameter_fills_swc_radius(viewer, tmp_path) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_thick_bar(), name="bar2")
    sid = trace.skeletonize("bar2")["skeleton_id"]
    trace.measure_filament_diameter(sid)

    swc = tmp_path / "t.swc"
    trace.export_neural_trace(sid, str(swc), format="swc")
    body = [ln for ln in swc.read_text().splitlines() if not ln.startswith("#")]
    radii = {ln.split()[5] for ln in body if ln.strip()}
    assert radii != {"0.500000"}  # real measured radii, not the placeholder


def test_compute_tree_topology_on_Y(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_branched(), name="ytop")
    sid = trace.skeletonize("ytop")["skeleton_id"]
    trace.build_rooted_tree(sid)

    res = trace.compute_tree_topology(sid)
    assert res["max_strahler"] >= 2  # a Y bifurcation
    assert res["total_path_length_um"] > 0
    assert res["n_leaves"] >= 2
    df = state.get_table(res["table_name"])
    assert {"branch_order", "strahler", "path_length_to_soma_um"} <= set(df.columns)


def test_topology_requires_rooted_tree(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_branched(), name="ynotree")
    sid = trace.skeletonize("ynotree")["skeleton_id"]
    with pytest.raises(ValueError, match="build_rooted_tree"):
        trace.compute_tree_topology(sid)
