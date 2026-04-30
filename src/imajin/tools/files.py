from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.agent.state import get_table, get_viewer
from imajin.tools.registry import tool


@tool(
    description="Open an LSM/CZI/OME-TIFF/TIFF file and add it as napari layers. "
    "Channels split into one layer each. Returns metadata summary.",
    phase="1",
)
def load_file(path: str) -> dict[str, Any]:
    from imajin.io import load_dataset
    from imajin.io.napari_reader import _to_layer

    ds = load_dataset(path)
    viewer = get_viewer()

    data, kwargs, _ = _to_layer(ds)
    layers = viewer.add_image(data, **kwargs)
    if not isinstance(layers, list):
        layers = [layers]

    return {
        "path": str(Path(path).resolve()),
        "axes": ds.axes,
        "shape": tuple(int(s) for s in ds.data.shape),
        "voxel_size_um": tuple(ds.voxel_size),
        "channel_names": list(ds.channel_names),
        "layer_names": [L.name for L in layers],
    }


@tool(
    description="List all layers currently in the napari viewer with shape, dtype, "
    "kind (image/labels/shapes/etc.), and physical scale.",
    phase="1",
)
def list_layers() -> list[dict[str, Any]]:
    viewer = get_viewer()
    out: list[dict[str, Any]] = []
    for L in viewer.layers:
        try:
            shape = tuple(int(s) for s in L.data.shape)
            dtype = str(L.data.dtype)
        except Exception:
            shape, dtype = (), "?"

        md_raw = getattr(L, "metadata", None)
        md = dict(md_raw) if isinstance(md_raw, dict) else {}

        scale_raw = getattr(L, "scale", None)
        scale: tuple[float, ...] = ()
        try:
            if scale_raw is not None:
                scale = tuple(float(s) for s in scale_raw)
        except TypeError:
            scale = ()

        out.append(
            {
                "name": L.name,
                "kind": type(L).__name__.lower(),
                "shape": shape,
                "dtype": dtype,
                "scale": scale,
                "metadata": md,
            }
        )
    return out


@tool(
    description="Export a measurement table to disk as CSV (default) or Parquet. "
    "Returns the resolved absolute path written.",
    phase="4",
)
def export_table(
    table_name: str, path: str, format: str = "csv"
) -> dict[str, Any]:
    df = get_table(table_name)
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower()
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"unsupported format: {format!r} (csv, parquet)")

    return {
        "path": str(out_path),
        "format": fmt,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
    }
