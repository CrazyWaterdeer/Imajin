from __future__ import annotations

from typing import Any

from imajin.session import get_table, get_viewer
from imajin.paths import normalize_user_path
from imajin.tools.layers import remove_layers_by_name as _remove_layers_by_name
from imajin.tools.registry import tool


def _canonical_path_text(path: str) -> str:
    return str(normalize_user_path(path).resolve())


def _layer_source_path(layer: Any) -> str | None:
    md = getattr(layer, "metadata", None)
    if not isinstance(md, dict):
        return None
    raw = md.get("source_path") or md.get("path")
    if not raw:
        return None
    try:
        return _canonical_path_text(str(raw))
    except Exception:
        return str(raw)


def _layer_names_for_source_path(path: str) -> list[str]:
    wanted = _canonical_path_text(path)
    viewer = get_viewer()
    names: list[str] = []
    for layer in viewer.layers:
        if _layer_source_path(layer) == wanted:
            names.append(str(layer.name))
    return names


def _existing_load_result(path: str, layer_names: list[str]) -> dict[str, Any]:
    viewer = get_viewer()
    first = viewer.layers[layer_names[0]]
    md = dict(getattr(first, "metadata", {}) or {})
    shape = md.get("dataset_shape")
    if shape is None:
        shape = tuple(int(s) for s in getattr(first.data, "shape", ()))
    return {
        "path": _canonical_path_text(path),
        "axes": md.get("axes"),
        "shape": tuple(int(s) for s in shape),
        "voxel_size_um": tuple(md.get("voxel_size_um", ())),
        "channel_names": list(md.get("channel_names") or layer_names),
        "channel_metadata": list(md.get("channel_metadata") or []),
        "layer_names": list(layer_names),
        "load_mode": md.get("load_mode"),
        "already_loaded": True,
    }


@tool(
    description="Open an LSM/CZI/OME-TIFF/TIFF file and add it as napari layers. "
    "Channels split into one layer each. Returns metadata summary.",
    phase="1",
)
def load_file(path: str) -> dict[str, Any]:
    return _load_file(path, force_reload=False)


def _load_file(path: str, *, force_reload: bool) -> dict[str, Any]:
    from imajin.io import load_dataset
    from imajin.io.napari_reader import _to_layer

    resolved_path = normalize_user_path(path).resolve()
    existing = _layer_names_for_source_path(str(resolved_path))
    if existing and not force_reload:
        return _existing_load_result(str(resolved_path), existing)

    ds = load_dataset(resolved_path)
    viewer = get_viewer()

    data, kwargs, _ = _to_layer(ds)
    metadata = dict(kwargs.get("metadata") or {})
    metadata.update(
        {
            "source_path": str(resolved_path),
            "dataset_shape": tuple(int(s) for s in ds.data.shape),
        }
    )
    kwargs["metadata"] = metadata
    layers = viewer.add_image(data, **kwargs)
    if not isinstance(layers, list):
        layers = [layers]

    return {
        "path": str(resolved_path),
        "axes": ds.axes,
        "shape": tuple(int(s) for s in ds.data.shape),
        "voxel_size_um": tuple(ds.voxel_size),
        "channel_names": list(ds.channel_names),
        "channel_metadata": list(getattr(ds, "channel_metadata", []) or []),
        "layer_names": [L.name for L in layers],
        "load_mode": ds.raw_metadata.get("load_mode"),
        "already_loaded": False,
    }


@tool(
    description="Force-reload an imaging file even if layers from the same source_path "
    "are already present. Prefer load_file for normal use; this exists for explicit "
    "manual reloads.",
    phase="1",
)
def reload_file(path: str) -> dict[str, Any]:
    return _load_file(path, force_reload=True)


@tool(
    description="Remove one or more napari layers by exact layer name. Use this to "
    "free memory after reviewing intermediate analysis layers.",
    phase="1",
)
def unload_layers(layer_names: list[str], missing_ok: bool = True) -> dict[str, Any]:
    requested = [str(name) for name in layer_names]
    removed = _remove_layers_by_name(requested)
    missing = [name for name in requested if name not in removed]
    if missing and not missing_ok:
        raise KeyError(f"layers not found or not removed: {missing}")
    return {
        "requested": requested,
        "removed_layers": removed,
        "n_removed": len(removed),
        "missing_layers": missing,
    }


@tool(
    description="Remove all currently loaded napari layers that came from one imaging "
    "file path. Layers are matched by metadata.source_path, so this is safer than "
    "guessing names in large batches.",
    phase="1",
)
def unload_file_layers(path: str, missing_ok: bool = True) -> dict[str, Any]:
    resolved = _canonical_path_text(path)
    names = _layer_names_for_source_path(resolved)
    if not names and not missing_ok:
        raise KeyError(f"no loaded layers found for source path: {resolved}")
    removed = _remove_layers_by_name(names)
    return {
        "path": resolved,
        "matched_layers": names,
        "removed_layers": removed,
        "n_removed": len(removed),
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
                "kind": getattr(L, "kind", type(L).__name__.lower()),
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
    out_path = normalize_user_path(path).resolve()
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
