from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/xdg-config")
os.environ.setdefault("XDG_DATA_HOME", "/tmp/xdg-data")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile


@pytest.fixture(autouse=True)
def _reset_sample_annotations():
    from imajin.agent import state

    state.reset_samples()
    state.reset_channel_annotations()
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()
    yield
    state.reset_samples()
    state.reset_channel_annotations()
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()


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
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        from imajin.agent import state

        v = _FakeViewer()
        state.set_viewer(v)
        yield v
        state.set_viewer(None)
        v.layers.clear()
        return

    import napari

    from imajin.agent import state

    v = napari.Viewer(show=False)
    state.set_viewer(v)
    yield v
    state.set_viewer(None)
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        # napari/Qt 6 can abort in QOpenGLWidget teardown on offscreen backends.
        # Clearing layers is enough isolation for these tests, and avoids making
        # every viewer-based test process crash during fixture teardown.
        v.layers.clear()
    else:
        v.close()


class _FakeColormap:
    def __init__(self, name: str = "gray") -> None:
        self.name = name


class _FakeLayer:
    def __init__(
        self,
        data,
        name: str,
        scale=None,
        metadata=None,
        kind: str = "image",
        **kwargs,
    ) -> None:
        self.data = data
        self.name = name
        self.scale = tuple(scale) if scale is not None else (1.0,) * getattr(data, "ndim", 0)
        self.metadata = dict(metadata or {})
        self.kind = kind
        self.selected_label = 0
        self.show_selected_label = False
        self._colormap = _FakeColormap(kwargs.get("colormap", "gray"))

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        self._colormap = value if hasattr(value, "name") else _FakeColormap(str(value))


class _FakeLayerList(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            for layer in self:
                if layer.name == key:
                    return layer
            raise KeyError(key)
        return super().__getitem__(key)

    def __contains__(self, key) -> bool:
        if isinstance(key, str):
            return any(layer.name == key for layer in self)
        return super().__contains__(key)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _FakeLayerList()
        self.dims = SimpleNamespace(ndisplay=2)
        self.camera = SimpleNamespace(
            angles=(0.0, 0.0, 0.0),
            zoom=1.0,
            center=(0.0, 0.0, 0.0),
        )

    def add_image(self, data, **kwargs):
        channel_axis = kwargs.pop("channel_axis", None)
        name = kwargs.pop("name", "image")
        if channel_axis is not None:
            names = list(name) if isinstance(name, (list, tuple)) else [
                f"{name}_{i}" for i in range(data.shape[channel_axis])
            ]
            layers = []
            for i, layer_name in enumerate(names):
                layer_data = np.take(data, i, axis=channel_axis)
                layer = _FakeLayer(layer_data, layer_name, kind="image", **kwargs)
                self.layers.append(layer)
                layers.append(layer)
            return layers

        layer = _FakeLayer(data, str(name), kind="image", **kwargs)
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        name = kwargs.pop("name", "labels")
        layer = _FakeLayer(data, str(name), kind="labels", **kwargs)
        self.layers.append(layer)
        return layer

    def add_tracks(self, data, **kwargs):
        name = kwargs.pop("name", "tracks")
        layer = _FakeLayer(data, str(name), kind="tracks", **kwargs)
        layer.graph = kwargs.get("graph", {})
        self.layers.append(layer)
        return layer

    def screenshot(self, path=None, canvas_only=True):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        if path:
            from PIL import Image

            Image.fromarray(arr).save(path)
        return arr


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
