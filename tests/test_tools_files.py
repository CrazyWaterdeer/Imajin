from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imajin import session as state
from imajin.tools import call_tool, files


@pytest.fixture
def _clean_runs_files():
    state.reset_runs()
    state.current_session().files.clear()
    yield
    state.reset_runs()
    state.current_session().files.clear()


def _add_A_layers(viewer, src="/d/A.lsm") -> None:
    arr = np.zeros((8, 8), dtype=np.float32)
    lab = arr.astype(np.int32)
    viewer.add_image(arr, name="A_img", metadata={"source_path": src})
    viewer.add_labels(lab, name="A_objects", metadata={"source_path": src})
    viewer.add_image(arr, name="A_mip", metadata={"source_layer": "A_img"})  # MIP, no source_path
    viewer.add_labels(lab, name="A_boundary", metadata={"source_layer": "A_mip"})  # boundary on MIP


def _stub_load(viewer, monkeypatch, name="B_img", src="/d/B.lsm") -> None:
    def _stub(path, *, force_reload):
        viewer.add_image(
            np.zeros((8, 8), dtype=np.float32), name=name, metadata={"source_path": src}
        )
        return {"loaded": True, "layer_names": [name]}

    monkeypatch.setattr(files, "_load_file", _stub)


def test_advance_unloads_whole_analysed_tree(viewer, monkeypatch, _clean_runs_files) -> None:
    _add_A_layers(viewer)
    state.put_run(
        sample_id="A", file_id="/d/A.lsm", recipe_id="interactive:x:single",
        status="complete", table_names=["A_t"],
    )
    _stub_load(viewer, monkeypatch)
    res = files.advance_to_file("/d/B.lsm")
    assert {L.name for L in viewer.layers} == {"B_img"}  # image+labels+MIP+boundary all gone
    assert set(res["unloaded_files"]) == {files._canonical_path_text("/d/A.lsm")}


def test_advance_guard_keeps_foreign_source_path(viewer, monkeypatch, _clean_runs_files) -> None:
    _add_A_layers(viewer)
    viewer.add_labels(
        np.zeros((8, 8), dtype=np.int32), name="foreign",
        metadata={"source_path": "/d/other.lsm", "source_layer": "A_img"},
    )
    state.put_run(sample_id="A", file_id="/d/A.lsm", recipe_id="r", status="complete")
    _stub_load(viewer, monkeypatch)
    files.advance_to_file("/d/B.lsm")
    names = {L.name for L in viewer.layers}
    assert "foreign" in names and "A_img" not in names  # different file -> not swept


def test_advance_keeps_unanalysed_unless_force(viewer, monkeypatch, _clean_runs_files) -> None:
    _add_A_layers(viewer)
    state.put_run(sample_id="A", file_id="/d/A.lsm", recipe_id="r", status="complete")
    viewer.add_image(
        np.zeros((8, 8), dtype=np.float32), name="C_img", metadata={"source_path": "/d/C.lsm"}
    )
    _stub_load(viewer, monkeypatch)
    res = files.advance_to_file("/d/B.lsm")
    names = {L.name for L in viewer.layers}
    assert "C_img" in names and "A_img" not in names  # unanalysed kept, analysed swept
    assert any("not analysed" in w for w in res["warnings"])

    _stub_load(viewer, monkeypatch, name="B2", src="/d/B2.lsm")
    files.advance_to_file("/d/B2.lsm", force_unload=True)
    assert "C_img" not in {L.name for L in viewer.layers}  # force discards it


def test_advance_resolves_registered_file_id(viewer, monkeypatch, _clean_runs_files) -> None:
    fid = state.put_file("/d/D.lsm", "D")
    viewer.add_image(
        np.zeros((8, 8), dtype=np.float32), name="D_img", metadata={"source_path": "/d/D.lsm"}
    )
    state.put_run(sample_id="D", file_id=fid, recipe_id="recipe", status="complete")  # batch-style key
    _stub_load(viewer, monkeypatch)
    files.advance_to_file("/d/B.lsm")
    assert "D_img" not in {L.name for L in viewer.layers}  # fid resolved to path -> swept


def test_advance_ignores_nonpath_run_without_crashing(viewer, monkeypatch, _clean_runs_files) -> None:
    viewer.add_image(
        np.zeros((8, 8), dtype=np.float32), name="E_img", metadata={"source_path": "/d/E.lsm"}
    )
    state.put_run(sample_id="x", file_id="some_layer_name", recipe_id="r", status="complete")
    _stub_load(viewer, monkeypatch)
    res = files.advance_to_file("/d/B.lsm")  # must not raise on the non-path run
    assert "E_img" in {L.name for L in viewer.layers}  # E not complete -> kept
    assert any("not analysed" in w for w in res["warnings"])


def test_load_file_adds_layers(viewer, tiny_ome_tiff: Path) -> None:
    res = files.load_file(str(tiny_ome_tiff))
    assert res["axes"] == "CZYX"
    assert res["shape"] == (3, 5, 64, 64)
    assert len(res["layer_names"]) == 3
    assert len(viewer.layers) == 3
    assert res["already_loaded"] is False
    assert all(L.metadata["source_path"] == str(tiny_ome_tiff.resolve()) for L in viewer.layers)


def test_load_file_reuses_existing_source_layers(viewer, tiny_ome_tiff: Path) -> None:
    first = files.load_file(str(tiny_ome_tiff))
    second = files.load_file(str(tiny_ome_tiff))

    assert second["already_loaded"] is True
    assert second["layer_names"] == first["layer_names"]
    assert len(viewer.layers) == 3


def test_unload_file_layers_removes_by_source_path(viewer, tiny_ome_tiff: Path) -> None:
    files.load_file(str(tiny_ome_tiff))

    res = files.unload_file_layers(str(tiny_ome_tiff))

    assert res["n_removed"] == 3
    assert len(viewer.layers) == 0


def test_list_layers_after_load(viewer, tiny_ome_tiff: Path) -> None:
    files.load_file(str(tiny_ome_tiff))
    items = files.list_layers()
    assert len(items) == 3
    for item in items:
        assert item["kind"] == "image"
        assert item["dtype"] == "uint16"
        assert item["shape"] == (5, 64, 64)


def test_load_file_via_call_tool(viewer, tiny_ome_tiff: Path) -> None:
    res = call_tool("load_file", path=str(tiny_ome_tiff))
    assert "layer_names" in res
    assert len(viewer.layers) == 3
