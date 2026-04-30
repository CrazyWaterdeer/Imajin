from __future__ import annotations

from pathlib import Path

from imajin.tools import call_tool, files


def test_load_file_adds_layers(viewer, tiny_ome_tiff: Path) -> None:
    res = files.load_file(str(tiny_ome_tiff))
    assert res["axes"] == "CZYX"
    assert res["shape"] == (3, 5, 64, 64)
    assert len(res["layer_names"]) == 3
    assert len(viewer.layers) == 3


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
