"""Boolean set operations on mask/label layers, and an inside/outside partition.

These tools let a segmented channel *scope* another channel's analysis. ``mask_logic``
is the missing primitive — complement / intersection / union / set-difference of masks —
so "outside the green domain" becomes expressible (``subtract(specimen, green)`` or
``not(green, within=specimen)``). ``partition_inside_outside`` (added alongside) wraps the
common inside-vs-outside case into a single two-label map ready for ``measure_intensity``.

Inputs are treated as **masks**: ``foreground = data > 0`` for any Labels or binary Image
layer. Turning a raw intensity channel into a mask stays the job of the segment/threshold
tools; pass their output here.

napari is never imported at module scope — every viewer touch goes through
``snapshot_layer`` / ``add_labels_from_worker`` on the main thread (``call_on_main``), so
``import imajin.tools`` stays headless-safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array
from imajin.analysis.segmentation import resolve_boundary_mask
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
from imajin.tools.registry import tool

_BINARY_OPS = ("and", "or", "subtract")
_ALL_OPS = ("not", *_BINARY_OPS)


class _MaskError(ValueError):
    """Raised for a rejected input; the tool boundary turns it into {ok: False, error}."""


# --- pure, headless cores (unit-tested without a viewer) -----------------------------

def _foreground(data: Any) -> np.ndarray:
    """Boolean foreground of a mask/label array (``> 0``)."""
    return np.asarray(materialize_array(data)) > 0


def _align(
    target_shape: tuple[int, ...],
    mask_bool: np.ndarray,
    *,
    broadcast_2d_to_3d: bool,
) -> tuple[np.ndarray, bool]:
    """Align a boolean mask to ``target_shape``; return ``(aligned, broadcast_z)``.

    Wraps :func:`resolve_boundary_mask` (exact-shape → writable bool; 2D ``(Y,X)`` vs 3D
    ``(Z,Y,X)`` → read-only broadcast view across Z). Boolean ops downstream allocate fresh
    arrays, so the read-only view is safe. A 2D→3D broadcast is refused when
    ``broadcast_2d_to_3d`` is False; any other incompatible shape raises.
    """
    target_shape = tuple(int(s) for s in target_shape)
    mask_bool = np.asarray(mask_bool)
    if mask_bool.shape == target_shape:
        return mask_bool.astype(bool, copy=False), False
    is_broadcast = (
        mask_bool.ndim == 2
        and len(target_shape) == 3
        and mask_bool.shape == target_shape[-2:]
    )
    if is_broadcast and not broadcast_2d_to_3d:
        raise _MaskError(
            f"mask shape {mask_bool.shape} needs a 2D->3D broadcast to match "
            f"{target_shape}, but broadcast_2d_to_3d=False"
        )
    try:
        aligned = resolve_boundary_mask(mask_bool, target_shape)
    except ValueError as e:
        raise _MaskError(str(e)) from e
    return aligned, bool(is_broadcast)


def _combine_masks(
    op: str,
    a: np.ndarray,
    b: np.ndarray | None,
    within: np.ndarray | None,
) -> np.ndarray:
    """Element-wise boolean set op on aligned masks, optionally clipped to ``within``.

    ``not`` = ``~a``; ``and`` = ``a & b``; ``or`` = ``a | b``; ``subtract`` = ``a & ~b``.
    Binary ops require ``b``. Returns a fresh boolean array.
    """
    if op in _BINARY_OPS and b is None:
        raise _MaskError(f"op={op!r} needs a second mask (b_layer)")
    if op == "not":
        out = ~a
    elif op == "and":
        out = a & b
    elif op == "or":
        out = a | b
    elif op == "subtract":
        out = a & ~b
    else:
        raise _MaskError(f"unknown op {op!r}; expected one of {_ALL_OPS}")
    if within is not None:
        out = out & within
    return out


def _scales_disagree(scale_a: tuple[float, ...], scale_b: tuple[float, ...]) -> bool:
    """True if two layer scales disagree on their overlapping trailing axes.

    Compares the last ``min(len)`` axes so a 2D mask and a 3D stack can still be checked on
    their shared Y/X calibration. Empty/absent scales are treated as agreeing (a hand-built
    mask legitimately carrying no scale is a recoverable case, not a hard error)."""
    if not scale_a or not scale_b:
        return False
    n = min(len(scale_a), len(scale_b))
    a = [float(x) for x in tuple(scale_a)[-n:]]
    b = [float(x) for x in tuple(scale_b)[-n:]]
    return any(abs(x - y) > 1e-6 for x, y in zip(a, b))


# --- viewer-facing helpers -----------------------------------------------------------

def _load_mask_layer(layer_name: str):
    """Snapshot a layer, materialise its array, and guard its axes (2D YX / 3D ZYX only)."""
    snap = call_on_main(snapshot_layer, layer_name)
    data = materialize_array(snap.data)
    axes = layer_axes_from_metadata(snap.metadata or {}, int(data.ndim))
    if axes not in ("YX", "ZYX"):
        raise _MaskError(
            f"layer {layer_name!r} axes {axes!r} unsupported; need a 2D YX image or 3D "
            "ZYX stack (extract a timepoint/slice first)"
        )
    return snap, data, axes


def _source_path(snap) -> str | None:
    md = snap.metadata or {}
    return md.get("source_path") or md.get("path")


# --- tools ---------------------------------------------------------------------------

@tool(
    description="Boolean set operations on mask/label layers, producing a new Labels layer. "
    "op is not | and | or | subtract (foreground = layer > 0). Use to express 'outside a "
    "domain': subtract(specimen, green) or not(green, within_layer=specimen). Optional "
    "within_layer clips the result. A 2D mask is broadcast across a 3D stack's Z. Inputs "
    "are masks — threshold/segment a raw channel first.",
    phase="7",
    worker=True,
)
def mask_logic(
    op: str,
    a_layer: str,
    b_layer: str | None = None,
    within_layer: str | None = None,
    broadcast_2d_to_3d: bool = True,
    name: str | None = None,
) -> dict[str, Any]:
    try:
        if op not in _ALL_OPS:
            raise _MaskError(f"unknown op {op!r}; expected one of {_ALL_OPS}")

        warnings: list[str] = []
        snap_a, data_a, _ = _load_mask_layer(a_layer)

        snap_b = data_b = None
        if op in _BINARY_OPS:
            if not b_layer:
                raise _MaskError(f"op={op!r} needs b_layer")
            snap_b, data_b, _ = _load_mask_layer(b_layer)
        elif b_layer:
            warnings.append(f"b_layer ignored for op={op!r}")

        snap_w = data_w = None
        if within_layer:
            snap_w, data_w, _ = _load_mask_layer(within_layer)

        # Reference = the highest-ndim input so a 2D mask broadcasts into a 3D partner and
        # the output carries the 3D shape/scale.
        candidates = [(snap_a, data_a)]
        if data_b is not None:
            candidates.append((snap_b, data_b))
        if data_w is not None:
            candidates.append((snap_w, data_w))
        ref_snap, ref_data = max(candidates, key=lambda sd: sd[1].ndim)
        target_shape = tuple(int(s) for s in ref_data.shape)
        ref_scale = tuple(ref_snap.scale) if ref_snap.scale else None

        broadcast_z = False
        a_al, bz = _align(target_shape, _foreground(data_a), broadcast_2d_to_3d=broadcast_2d_to_3d)
        broadcast_z = broadcast_z or bz
        b_al = None
        if data_b is not None:
            b_al, bz = _align(target_shape, _foreground(data_b), broadcast_2d_to_3d=broadcast_2d_to_3d)
            broadcast_z = broadcast_z or bz
        w_al = None
        if data_w is not None:
            w_al, bz = _align(target_shape, _foreground(data_w), broadcast_2d_to_3d=broadcast_2d_to_3d)
            broadcast_z = broadcast_z or bz

        scale_mismatch = False
        for snap, _ in candidates[1:]:
            if _scales_disagree(tuple(snap_a.scale or ()), tuple(snap.scale or ())):
                scale_mismatch = True
        if scale_mismatch:
            warnings.append(
                "input layers disagree on voxel scale; the combination may not be "
                "physically aligned (check the layers share calibration)"
            )
        if broadcast_z:
            warnings.append(
                f"a 2D mask was broadcast across all {target_shape[0]} Z planes (an "
                "extrusion, not a true 3D region)"
            )

        mask = _combine_masks(op, a_al, b_al, w_al)
        voxels = int(mask.sum())
        empty = voxels == 0
        if empty:
            warnings.append("result is empty (no foreground voxels)")

        out_name = name or f"{a_layer}_{op}"
        layer = call_on_main(
            add_labels_from_worker,
            mask.astype(np.int32),
            name=out_name,
            scale=ref_scale,
            metadata={
                "source_layer": a_layer,
                "source_path": _source_path(snap_a),
                "mask_op": op,
                "operands": [x for x in (a_layer, b_layer, within_layer) if x],
                "axes": "ZYX" if len(target_shape) == 3 else "YX",
                "broadcast_z": broadcast_z,
                "scale_mismatch": scale_mismatch,
                "mask_voxels": voxels,
                "mask_fraction": float(voxels / mask.size) if mask.size else 0.0,
            },
        )
        return {
            "ok": True,
            "op": op,
            "mask_layer": layer.name,
            "voxels": voxels,
            "fraction": float(voxels / mask.size) if mask.size else 0.0,
            "empty": empty,
            "broadcast_z": broadcast_z,
            "scale_mismatch": scale_mismatch,
            "axes": "ZYX" if len(target_shape) == 3 else "YX",
            "warnings": warnings,
        }
    except _MaskError as e:
        return {"ok": False, "error": str(e)}
