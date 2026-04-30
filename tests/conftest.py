from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.fixture
def tiny_ome_tiff(tmp_path: Path) -> Path:
    """Multi-channel z-stack OME-TIFF with PhysicalSize metadata."""
    path = tmp_path / "tiny.ome.tif"
    data = np.random.default_rng(0).integers(
        0, 4096, size=(3, 5, 64, 64), dtype=np.uint16
    )
    tifffile.imwrite(
        path,
        data,
        photometric="minisblack",
        ome=True,
        metadata={
            "axes": "CZYX",
            "PhysicalSizeX": 0.2,
            "PhysicalSizeY": 0.2,
            "PhysicalSizeZ": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZUnit": "µm",
            "Channel": {"Name": ["DAPI", "GFP", "TRITC"]},
        },
    )
    return path


@pytest.fixture
def plain_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "plain.tif"
    data = np.random.default_rng(1).integers(0, 1024, size=(64, 64), dtype=np.uint16)
    tifffile.imwrite(path, data)
    return path


@pytest.fixture
def viewer(qapp):
    import napari

    from imajin.agent import state

    v = napari.Viewer(show=False)
    state.set_viewer(v)
    yield v
    state.set_viewer(None)
    v.close()


@pytest.fixture
def synthetic_blob_image() -> np.ndarray:
    rng = np.random.default_rng(123)
    img = rng.integers(50, 100, size=(128, 128), dtype=np.uint16).astype(np.float32)
    yy, xx = np.mgrid[:128, :128]
    centers = [(30, 35), (60, 70), (95, 50), (40, 100), (95, 100)]
    for cy, cx in centers:
        img += 800 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6**2))
    return img.astype(np.uint16)


@pytest.fixture
def random_2d_image() -> np.ndarray:
    return np.random.default_rng(42).integers(0, 65535, size=(128, 128), dtype=np.uint16)
