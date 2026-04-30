from __future__ import annotations

from pathlib import Path

from imajin.io.napari_reader import napari_get_reader, read_path, _do_read


def test_get_reader_accepts_supported_suffixes(tmp_path: Path) -> None:
    for suf in (".lsm", ".czi", ".ome.tif", ".tif", ".tiff", ".ome.tiff"):
        f = tmp_path / f"x{suf}"
        f.touch()
        assert callable(napari_get_reader(str(f)))


def test_get_reader_rejects_other_suffixes(tmp_path: Path) -> None:
    f = tmp_path / "x.png"
    f.touch()
    assert napari_get_reader(str(f)) is None


def test_get_reader_handles_list_of_paths(tmp_path: Path) -> None:
    f = tmp_path / "x.tif"
    f.touch()
    assert callable(napari_get_reader([str(f)]))


def test_read_path_returns_callable_factory(tiny_ome_tiff: Path) -> None:
    reader = read_path(str(tiny_ome_tiff))
    assert callable(reader)
    layers = reader(str(tiny_ome_tiff))
    assert len(layers) == 1


def test_layer_has_channel_axis_and_scale(tiny_ome_tiff: Path) -> None:
    layers = _do_read(str(tiny_ome_tiff))
    data, kwargs, kind = layers[0]
    assert kind == "image"
    assert kwargs["channel_axis"] == 0
    assert kwargs["name"] == ["DAPI", "GFP", "TRITC"]
    assert kwargs["scale"] == (0.5, 0.2, 0.2)
    assert "voxel_size_um" in kwargs["metadata"]
    assert kwargs["metadata"]["axes"] == "CZYX"


def test_layer_names_strip_ome_suffix(tiny_ome_tiff: Path) -> None:
    layers = _do_read(str(tiny_ome_tiff))
    _, kwargs, _ = layers[0]
    for n in kwargs["name"]:
        assert not n.endswith(".ome")
