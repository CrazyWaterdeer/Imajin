from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imajin.agent import state
from imajin.tools import report
from imajin.tools import trace


def _make_branched_mask() -> np.ndarray:
    """Y-shape: a vertical trunk + two diagonal branches in the upper half."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 31:33] = 1
    for i in range(20):
        img[10 + i, 31 - i] = 1
        img[10 + i, 32 + i] = 1
    return img


def _make_spur_mask() -> np.ndarray:
    img = np.zeros((40, 40), dtype=np.uint8)
    img[5:35, 20] = 1
    img[10, 20:24] = 1
    return img


def _make_3d_process_image() -> np.ndarray:
    img = np.zeros((9, 32, 32), dtype=np.float32)
    img[2:7, 16, 8:25] = 1.0
    img[4, 8:17, 16] = 1.0
    img[5, 16:25, 18] = 1.0
    rng = np.random.default_rng(123)
    img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
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

    with pytest.raises(ValueError, match="2D or 3D"):
        trace.skeletonize("bad")


def test_enhance_neural_processes_preserves_shape_and_scale(viewer) -> None:
    data = _make_3d_process_image()
    viewer.add_image(data, name="vnc_green", scale=(1.0, 0.5, 0.5))

    res = trace.enhance_neural_processes(
        "vnc_green", method="gaussian", sigma=1.0, background=None
    )

    assert res["new_layer"] == "vnc_green_neural_enhanced"
    assert res["shape"] == data.shape
    assert res["scale"] == (1.0, 0.5, 0.5)
    assert viewer.layers[res["new_layer"]].data.shape == data.shape


def test_segment_neural_processes_creates_labels_and_qc(viewer) -> None:
    data = _make_3d_process_image()
    viewer.add_image(data, name="enhanced", scale=(1.0, 0.5, 0.5))

    res = trace.segment_neural_processes(
        "enhanced", threshold=0.5, min_size_um3=0.5, keep_largest=False
    )

    assert res["mask_layer"] == "enhanced_process_mask"
    assert res["component_count"] >= 1
    assert 0 < res["foreground_fraction"] < 0.2
    assert "enhanced_process_mask" in viewer.layers
    assert state.get_qc_record("enhanced_process_mask").status in {"pass", "warning"}


def test_skeletonize_creates_graph_tables_and_preserves_spacing(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_make_branched_mask(), name="ymask_scaled", scale=(0.5, 0.25))

    res = trace.skeletonize("ymask_scaled")

    assert res["spacing"] == (0.5, 0.25)
    assert {"nodes", "edges", "components"} <= set(res["table_names"])
    nodes = state.get_table(res["table_names"]["nodes"])
    edges = state.get_table(res["table_names"]["edges"])
    components = state.get_table(res["table_names"]["components"])
    assert "coord_0_um" in nodes.columns
    assert "edge_length_um" in edges.columns
    assert len(components) == res["n_components"]


def test_skeletonize_rejects_continuous_image_without_threshold(viewer) -> None:
    trace.reset_skeletons()
    viewer.add_image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8), name="raw")

    with pytest.raises(ValueError, match="binary/Labels"):
        trace.skeletonize("raw")


def test_prune_skeleton_removes_short_branch(viewer) -> None:
    trace.reset_skeletons()
    viewer.add_labels(_make_spur_mask(), name="spur", scale=(1.0, 1.0))
    skel = trace.skeletonize("spur")

    pruned = trace.prune_skeleton(skel["skeleton_id"], min_branch_length_um=5.0)

    assert pruned["n_removed"] >= 1
    assert pruned["new_skeleton_id"] != skel["skeleton_id"]
    assert pruned["n_paths_after"] < pruned["n_paths_before"]


def test_sholl_analysis_outputs_radius_table(viewer) -> None:
    trace.reset_skeletons()
    state.reset_tables()
    viewer.add_labels(_make_branched_mask(), name="ymask")
    skel = trace.skeletonize("ymask")

    res = trace.compute_sholl_analysis(
        skel["skeleton_id"], center="centroid", radius_step_um=5.0
    )

    df = state.get_table(res["table_name"])
    assert {"radius_um", "intersections"} <= set(df.columns)
    assert res["n_radii"] == len(df)


def test_export_neural_trace_writes_swc_csv_and_tiff(viewer, tmp_path) -> None:
    trace.reset_skeletons()
    viewer.add_labels(_make_branched_mask(), name="ymask")
    skel = trace.skeletonize("ymask")

    swc = trace.export_neural_trace(
        skel["skeleton_id"], str(tmp_path / "trace.swc"), format="swc"
    )
    csv = trace.export_neural_trace(
        skel["skeleton_id"], str(tmp_path / "trace_csv"), format="csv"
    )
    tif = trace.export_neural_trace(
        skel["skeleton_id"], str(tmp_path / "trace.tif"), format="tif"
    )

    assert (tmp_path / "trace.swc").exists()
    assert len(csv["paths"]) == 3
    assert all(Path(p).exists() for p in csv["paths"])
    assert (tmp_path / "trace.tif").exists()
    assert swc["note"]
    assert tif["paths"] == [str((tmp_path / "trace.tif").resolve())]


def test_report_includes_neural_trace_summary(viewer, tmp_path) -> None:
    trace.reset_skeletons()
    viewer.add_labels(_make_branched_mask(), name="ymask")
    skel = trace.skeletonize("ymask")
    trace.extract_branch_metrics(skel["skeleton_id"])
    trace.compute_morphology_descriptors(skel["skeleton_id"])

    out = tmp_path / "report.md"
    res = report.generate_report(str(out), format="md")
    body = out.read_text(encoding="utf-8")

    assert "Neural Morphology" in body
    assert skel["skeleton_id"] in body
    assert res["n_neural_traces"] == 1
