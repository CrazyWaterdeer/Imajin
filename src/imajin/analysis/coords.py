"""Canonical coordinate contract for Imajin geometry.

Every geometric value in Imajin lives in exactly one of two frames, and this
module is the single place that converts between them:

- **Data / index coordinates** — the frame layer arrays are indexed in. napari
  Points ``data``, skeleton node indices, and ``regionprops`` centroids are all
  here. Geometry attached to a layer is stored in data coordinates and never
  pre-scaled; napari's own ``scale`` / ``translate`` place it in the world for
  rendering. This keeps a Point from being double-scaled (stored in µm *and*
  multiplied again by the layer ``scale``).

- **World / physical coordinates (µm)** — ``world = data * scale + translate``.
  Any value we *store on a record* as a scalar reference (e.g. a trace's soma)
  or *emit into a table* is physical µm; tables additionally keep the raw index
  columns (see ``measure._add_physical_columns``) so both frames stay available.

One rule per surface: layer-attached geometry is in data coordinates; stored or
tabulated geometry is in µm. Convert only through the helpers here.

Note on the trace subsystem: skeleton morphometry (``compute_sholl_analysis``,
SWC export) works in a *scale-only* µm frame (``skel.coordinates * spacing``,
translate assumed 0). Reference points fed into that frame — such as the soma —
must therefore be converted scale-only (``translate=None``) so they land in the
same frame as the skeleton nodes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def normalize_vector(
    values: Sequence[float] | None,
    ndim: int,
    *,
    fill: float,
) -> tuple[float, ...]:
    """Coerce a scale/translate sequence to a length-``ndim`` float tuple.

    Truncates when longer than ``ndim`` and right-pads with ``fill`` when
    shorter (fill 1.0 for scale, 0.0 for translate). An empty/None input yields
    ``(fill,) * ndim``.
    """
    if not values:
        return (float(fill),) * ndim
    vals = tuple(float(v) for v in tuple(values)[:ndim])
    if len(vals) < ndim:
        vals = vals + (float(fill),) * (ndim - len(vals))
    return vals


def layer_scale(layer: Any, ndim: int) -> tuple[float, ...]:
    """Scale (µm per voxel per axis) of ``layer``, normalized to ``ndim``."""
    return normalize_vector(tuple(getattr(layer, "scale", ()) or ()), ndim, fill=1.0)


def layer_translate(layer: Any, ndim: int) -> tuple[float, ...]:
    """World-space translation of ``layer``, normalized to ``ndim``."""
    return normalize_vector(tuple(getattr(layer, "translate", ()) or ()), ndim, fill=0.0)


def data_to_world(
    coords: Any,
    scale: Sequence[float],
    translate: Sequence[float] | None = None,
) -> np.ndarray:
    """Map data/index coordinates to world (µm) coordinates.

    ``world = coords * scale + translate``. ``coords`` is ``(..., ndim)``;
    ``scale`` / ``translate`` are broadcast to that trailing ``ndim``. Passing
    ``translate=None`` gives a scale-only conversion.
    """
    arr = np.asarray(coords, dtype=float)
    ndim = arr.shape[-1]
    s = np.asarray(normalize_vector(scale, ndim, fill=1.0), dtype=float)
    out = arr * s
    if translate is not None:
        t = np.asarray(normalize_vector(translate, ndim, fill=0.0), dtype=float)
        out = out + t
    return out


def world_to_data(
    coords: Any,
    scale: Sequence[float],
    translate: Sequence[float] | None = None,
) -> np.ndarray:
    """Inverse of :func:`data_to_world`: world (µm) → data/index coordinates."""
    arr = np.asarray(coords, dtype=float)
    ndim = arr.shape[-1]
    s = np.asarray(normalize_vector(scale, ndim, fill=1.0), dtype=float)
    if translate is not None:
        t = np.asarray(normalize_vector(translate, ndim, fill=0.0), dtype=float)
        arr = arr - t
    return arr / s


def point_to_world(
    point_data: Any,
    layer: Any,
    *,
    use_translate: bool = True,
) -> tuple[float, ...]:
    """Convert one data-coordinate point on ``layer`` to world (µm) coordinates.

    Uses the layer's own ``scale`` (and ``translate`` unless ``use_translate``
    is False). Set ``use_translate=False`` when the target frame is scale-only,
    as in the skeleton morphometry subsystem.
    """
    point = np.asarray(point_data, dtype=float).ravel()
    ndim = int(point.shape[0])
    scale = layer_scale(layer, ndim)
    translate = layer_translate(layer, ndim) if use_translate else None
    return tuple(float(v) for v in data_to_world(point, scale, translate))


def is_physical(scale: Sequence[float]) -> bool:
    """True if any scale component departs from 1.0 (i.e. the data is µm-calibrated)."""
    return any(abs(float(v) - 1.0) > 1e-9 for v in scale)
