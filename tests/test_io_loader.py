from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imajin.io import Dataset, load_dataset


def test_load_ome_tiff_round_trip(tiny_ome_tiff: Path) -> None:
    ds = load_dataset(tiny_ome_tiff)
    assert isinstance(ds, Dataset)
    assert ds.axes == "CZYX"
    assert ds.data.shape == (3, 5, 64, 64)
    assert ds.n_channels == 3
    assert ds.is_3d is True
    assert ds.voxel_size == (0.5, 0.2, 0.2)
    assert ds.channel_names == ["DAPI", "GFP", "TRITC"]
    assert ds.source_path == tiny_ome_tiff


def test_load_dataset_unsupported_suffix(tmp_path: Path) -> None:
    f = tmp_path / "nope.bin"
    f.write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(f)


def test_load_plain_tiff_falls_through_to_ome_loader(plain_tiff: Path) -> None:
    ds = load_dataset(plain_tiff)
    assert ds.data.shape == (64, 64)
    # No OME XML -> defaults to (1, 1, 1)
    assert ds.voxel_size == (1.0, 1.0, 1.0)
    assert ds.channel_names == []


def test_dataset_data_is_lazy_dask(tiny_ome_tiff: Path) -> None:
    import dask.array as da

    ds = load_dataset(tiny_ome_tiff)
    assert isinstance(ds.data, da.Array)
    materialized = ds.data.compute()
    assert isinstance(materialized, np.ndarray)
    assert materialized.shape == ds.data.shape
