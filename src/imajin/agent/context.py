from __future__ import annotations

import json
from typing import Any

import numpy as np


def _sample_array(data: Any, max_points: int = 65_536) -> tuple[np.ndarray, bool]:
    shape = tuple(int(s) for s in getattr(data, "shape", ()))
    if not shape:
        arr = np.asarray(data.compute() if hasattr(data, "compute") else data)
        return arr, False

    total = int(np.prod(shape, dtype=np.int64))
    if total <= max_points:
        arr = data.compute() if hasattr(data, "compute") else data
        return np.asarray(arr), False

    per_axis = max(1, int(round(max_points ** (1 / max(1, len(shape))))))
    slices = tuple(slice(None, None, max(1, int(np.ceil(s / per_axis)))) for s in shape)
    sample = data[slices]
    sample = sample.compute() if hasattr(sample, "compute") else sample
    return np.asarray(sample), True


def _layer_summary(layer: Any) -> dict[str, Any]:
    data = layer.data
    shape = tuple(int(s) for s in getattr(data, "shape", ()))
    dtype = str(getattr(data, "dtype", "?"))

    md_raw = getattr(layer, "metadata", None)
    md = dict(md_raw) if isinstance(md_raw, dict) else {}

    scale_raw = getattr(layer, "scale", None)
    try:
        scale = tuple(float(s) for s in scale_raw) if scale_raw is not None else ()
    except TypeError:
        scale = ()

    info: dict[str, Any] = {
        "name": layer.name,
        "kind": getattr(layer, "kind", type(layer).__name__.lower()),
        "shape": shape,
        "dtype": dtype,
        "scale": scale,
    }
    if "axes" in md:
        info["axes"] = md["axes"]
    if "voxel_size_um" in md:
        info["voxel_size_um"] = md["voxel_size_um"]

    if info["kind"] == "image" and shape:
        try:
            sample, sampled = _sample_array(data)
            if sample.size > 0:
                info["intensity"] = {
                    "min": float(sample.min()),
                    "max": float(sample.max()),
                    "mean": float(sample.mean()),
                    "p1": float(np.percentile(sample, 1)),
                    "p99": float(np.percentile(sample, 99)),
                    "sampled": sampled,
                }
        except Exception:
            pass
    elif info["kind"] == "labels" and shape:
        try:
            sample, sampled = _sample_array(data)
            info["n_labels_sample"] = int(sample.max())
            info["sampled"] = sampled
        except Exception:
            pass

    return info


def summarize_viewer_state() -> str:
    from imajin.agent.state import (
        list_channel_annotations,
        list_samples,
        list_tables,
        viewer_or_none,
    )

    viewer = viewer_or_none()
    if viewer is None:
        return json.dumps(
            {
                "layers": [],
                "tables": [],
                "samples": list_samples(),
                "channels": list_channel_annotations(),
                "note": "viewer not initialized",
            }
        )

    layers = [_layer_summary(L) for L in viewer.layers]
    tables = list_tables()
    return json.dumps(
        {
            "layers": layers,
            "tables": tables,
            "samples": list_samples(),
            "channels": list_channel_annotations(),
        },
        default=str,
    )
