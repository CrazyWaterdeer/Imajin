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
import pandas as pd

from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array
from imajin.analysis.segmentation import (
    dilate_binary_um,
    erode_binary_um,
    resolve_boundary_mask,
    voxel_spacing,
)
from imajin.session import get_layer, put_table
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


def _morph_um(
    mask: np.ndarray,
    kind: str,
    *,
    broadcast_z: bool,
    spacing: tuple[float, ...],
    radius_um: float,
) -> np.ndarray:
    """Erode/dilate a mask by a physical radius, at the right dimensionality.

    When ``broadcast_z`` (the mask is a 2D region broadcast across a 3D stack's Z, i.e. a
    cylinder), the morphology is done on the single YX plane and re-broadcast — otherwise a
    3D erosion would treat the volume's top/bottom borders as background and wipe those
    planes. A natively 3D mask gets full-3D morphology.
    """
    fn = erode_binary_um if kind == "erode" else dilate_binary_um
    mask = np.asarray(mask)
    if broadcast_z and mask.ndim == 3:
        out2d = fn(np.asarray(mask[0]), spacing=tuple(spacing[-2:]), radius_um=radius_um)
        return np.broadcast_to(out2d[None], mask.shape)
    return fn(mask, spacing=spacing, radius_um=radius_um)


def _partition(
    region_aligned: np.ndarray,
    within_aligned: np.ndarray,
    *,
    region_broadcast_z: bool,
    spacing: tuple[float, ...] | None,
    buffer_um: float,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Two-label inside/outside map, disjoint by construction.

    ``inside = region & within`` (label 1); ``outside = within & ~region`` (label 2). A
    positive ``buffer_um`` (with voxel ``spacing``) instead brackets the *region* boundary:
    ``inside = erode(region) & within`` and ``outside = within & ~dilate(region)``, so an
    ambiguous PSF/bleed-through band around the domain edge belongs to neither. The band is
    placed around the region (green) boundary — the specimen (within) edge is not a domain
    boundary, so it is intentionally not eroded. ``region_clipped_fraction`` reports how much
    of ``region`` fell outside ``within`` (a large value flags misregistration / wrong layer).
    """
    warnings: list[str] = []
    do_buffer = bool(buffer_um and buffer_um > 0)
    if do_buffer and spacing is None:
        warnings.append(
            "boundary_buffer_um requested but the reference layer has no voxel scale; "
            "buffer skipped"
        )
        do_buffer = False

    if do_buffer:
        region_er = _morph_um(
            region_aligned, "erode",
            broadcast_z=region_broadcast_z, spacing=spacing, radius_um=buffer_um,
        )
        region_di = _morph_um(
            region_aligned, "dilate",
            broadcast_z=region_broadcast_z, spacing=spacing, radius_um=buffer_um,
        )
        inside = region_er & within_aligned
        outside = within_aligned & ~region_di
    else:
        inside = region_aligned & within_aligned
        outside = within_aligned & ~region_aligned

    outside = outside & ~inside  # defensive; already disjoint
    labels = np.zeros(region_aligned.shape, dtype=np.int32)
    labels[inside] = 1
    labels[outside] = 2

    region_voxels = int(np.asarray(region_aligned).sum())
    stats = {
        "inside_voxels": int(inside.sum()),
        "outside_voxels": int(outside.sum()),
        "region_clipped_fraction": float(
            int((region_aligned & ~within_aligned).sum()) / max(region_voxels, 1)
        ),
    }
    return labels, stats, warnings


def _classify_overlap(
    labels: np.ndarray,
    region_bool: np.ndarray,
    *,
    overlap_threshold: float,
    within_bool: np.ndarray | None = None,
    within_threshold: float = 0.5,
    inside_name: str = "inside",
    outside_name: str = "outside",
    excluded_name: str = "excluded",
) -> tuple[dict[int, str], np.ndarray, np.ndarray | None, dict[str, int]]:
    """Classify each object label by its fractional overlap with ``region_bool``.

    Per label, ``overlap = |object ∩ region| / |object|`` (object-pixel denominator) via
    ``np.bincount`` — exact and O(voxels), no per-object loop. An object is ``inside`` when
    ``overlap >= overlap_threshold`` else ``outside``; if ``within_bool`` is given, an object
    whose within-fraction ``< within_threshold`` is ``excluded`` (off-specimen). Background
    label 0 is never classified. Returns ``(mapping{label: class}, overlap, within_frac,
    counts)``.
    """
    labels = np.asarray(labels)
    max_label = int(labels.max()) if labels.size else 0
    n_unique = int(np.count_nonzero(np.unique(labels)))
    if max_label > 5_000_000 and max_label > 10 * max(n_unique, 1):
        raise _MaskError(
            f"labels max id {max_label} is far larger than the object count ({n_unique}); "
            "relabel the layer sequentially (relabel_sequential) before classifying"
        )
    n = max_label + 1
    total = np.bincount(labels.ravel(), minlength=n).astype(float)
    ins = np.bincount(np.asarray(labels[region_bool]).ravel(), minlength=n).astype(float)
    overlap = np.divide(ins, total, out=np.zeros_like(total), where=total > 0)
    within_frac = None
    if within_bool is not None:
        win = np.bincount(np.asarray(labels[within_bool]).ravel(), minlength=n).astype(float)
        within_frac = np.divide(win, total, out=np.zeros_like(total), where=total > 0)

    mapping: dict[int, str] = {}
    counts = {inside_name: 0, outside_name: 0, excluded_name: 0}
    for lbl in range(1, n):  # skip background (label 0)
        if total[lbl] <= 0:
            continue
        if within_frac is not None and within_frac[lbl] < within_threshold:
            cls = excluded_name
        elif overlap[lbl] >= overlap_threshold:
            cls = inside_name
        else:
            cls = outside_name
        mapping[lbl] = cls
        counts[cls] += 1
    return mapping, overlap, within_frac, counts


def _stamp_classification(
    labels_layer: str, mapping: dict[int, str], provenance: dict[str, Any]
) -> dict[int, str]:
    """Main-thread: stamp the per-object classification onto the cells layer so
    ``measure_intensity`` emits a ``region`` column. Preserves prior ``label_names`` keys not
    re-classified (``{**prev, **mapping}``) and records provenance under a dedicated
    ``classification`` key. Returns the previous ``label_names`` for reporting."""
    layer = get_layer(labels_layer)
    md = layer.metadata
    prev = dict(md.get("label_names") or {})
    md["label_names"] = {**prev, **mapping}
    md["classification"] = provenance
    return prev


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


@tool(
    description="Partition an image into a two-label map for inside-vs-outside comparison: "
    "label 1 = inside a domain region (e.g. a segmented green channel), label 2 = the rest "
    "of the specimen. Pass region_layer (the domain) and within_layer (the specimen/tissue "
    "bound — REQUIRED so 'outside' isn't all image background). Feed the result to "
    "measure_intensity([signal]) to get inside/outside signal in one call (rows carry a "
    "region column). boundary_buffer_um excludes an ambiguous band around the domain edge. "
    "Compare per sample as log2(inside/outside), then test across biological replicates — "
    "the two rows from one image are paired, not independent groups.",
    phase="7",
    worker=True,
)
def partition_inside_outside(
    region_layer: str,
    within_layer: str | None = None,
    boundary_buffer_um: float = 0.0,
    allow_full_frame_outside: bool = False,
    broadcast_2d_to_3d: bool = True,
    name: str | None = None,
) -> dict[str, Any]:
    try:
        warnings: list[str] = []
        if not within_layer and not allow_full_frame_outside:
            raise _MaskError(
                "outside needs a specimen bound: pass within_layer, or set "
                "allow_full_frame_outside=True to measure against the whole frame "
                "(background-dominated, rarely meaningful)"
            )

        snap_r, data_r, _ = _load_mask_layer(region_layer)
        snap_w = data_w = None
        if within_layer:
            snap_w, data_w, _ = _load_mask_layer(within_layer)

        candidates = [(snap_r, data_r)]
        if data_w is not None:
            candidates.append((snap_w, data_w))
        ref_snap, ref_data = max(candidates, key=lambda sd: sd[1].ndim)
        target_shape = tuple(int(s) for s in ref_data.shape)
        ref_scale = tuple(ref_snap.scale) if ref_snap.scale else None
        spacing = voxel_spacing(ref_scale, len(target_shape)) if ref_scale else None

        region_aligned, region_bz = _align(
            target_shape, _foreground(data_r), broadcast_2d_to_3d=broadcast_2d_to_3d
        )
        broadcast_z = region_bz
        if data_w is not None:
            within_aligned, within_bz = _align(
                target_shape, _foreground(data_w), broadcast_2d_to_3d=broadcast_2d_to_3d
            )
            broadcast_z = broadcast_z or within_bz
        else:
            within_aligned = np.ones(target_shape, dtype=bool)
            warnings.append(
                "no within_layer: 'outside' is the whole frame minus the region and "
                "includes all background; intensity means will be near-zero/misleading"
            )

        labels, stats, part_warnings = _partition(
            region_aligned,
            within_aligned,
            region_broadcast_z=region_bz,
            spacing=spacing,
            buffer_um=boundary_buffer_um,
        )
        warnings.extend(part_warnings)

        inside_voxels = stats["inside_voxels"]
        outside_voxels = stats["outside_voxels"]
        if inside_voxels == 0 and outside_voxels == 0:
            raise _MaskError(
                "partition is empty (region and within do not overlap the frame); "
                "check the layers"
            )
        comparable = inside_voxels > 0 and outside_voxels > 0
        if not comparable:
            warnings.append(
                "inside or outside is empty — an inside/outside comparison is not "
                "possible from this image (only one region has voxels)"
            )
        clipped = stats["region_clipped_fraction"]
        if clipped > 0.2:
            warnings.append(
                f"{clipped:.0%} of the region falls outside within_layer — possible "
                "misregistration, wrong layer, or a bad specimen mask"
            )
        if broadcast_z:
            warnings.append(
                f"a 2D mask was broadcast across all {target_shape[0]} Z planes (an "
                "extrusion; volume_um3 is a cylinder, not the true 3D specimen volume)"
            )

        label_names = {1: "inside", 2: "outside"}
        out_name = name or f"{region_layer}_partition"
        layer = call_on_main(
            add_labels_from_worker,
            labels,
            name=out_name,
            scale=ref_scale,
            metadata={
                "source_layer": region_layer,
                "source_path": _source_path(snap_r),
                "label_names": label_names,
                "region_layer": region_layer,
                "within_layer": within_layer,
                "within_used": bool(within_layer),
                "boundary_buffer_um": float(boundary_buffer_um),
                "region_clipped_fraction": clipped,
                "comparable": comparable,
                "broadcast_z": broadcast_z,
                "axes": "ZYX" if len(target_shape) == 3 else "YX",
            },
        )
        return {
            "ok": True,
            "partition_layer": layer.name,
            "inside_voxels": inside_voxels,
            "outside_voxels": outside_voxels,
            "within_used": bool(within_layer),
            "region_clipped_fraction": clipped,
            "boundary_buffer_um": float(boundary_buffer_um),
            "comparable": comparable,
            "broadcast_z": broadcast_z,
            "label_names": label_names,
            "warnings": warnings,
        }
    except _MaskError as e:
        return {"ok": False, "error": str(e)}


@tool(
    description="Classify each segmented object (cell) as inside/outside a domain by its "
    "fractional overlap with a region mask, for a PER-OBJECT inside-vs-outside comparison. "
    "Writes label_names on the cells layer so measure_intensity emits a per-cell region "
    "column; then filter to inside/outside and compare_groups(group_col='region'). Classify "
    "on one channel and measure a DIFFERENT one — measuring the channel that defined the "
    "domain is circular. Per-cell independence holds for a single image / genuinely "
    "independent units; across specimens, aggregate per sample and use the paired mode.",
    phase="7",
    worker=True,
)
def classify_labels_by_mask(
    labels_layer: str,
    region_layer: str,
    overlap_threshold: float = 0.5,
    within_layer: str | None = None,
    within_threshold: float = 0.5,
    inside_name: str = "inside",
    outside_name: str = "outside",
    write_label_names: bool = True,
    broadcast_2d_to_3d: bool = True,
    table_name: str | None = None,
) -> dict[str, Any]:
    try:
        warnings: list[str] = []
        snap_l, data_l, _ = _load_mask_layer(labels_layer)
        labels = np.asarray(data_l).astype(np.int32)

        _snap_r, data_r, _ = _load_mask_layer(region_layer)
        region_bool, broadcast_z = _align(
            labels.shape, _foreground(data_r), broadcast_2d_to_3d=broadcast_2d_to_3d
        )
        within_bool = None
        if within_layer:
            _snap_w, data_w, _ = _load_mask_layer(within_layer)
            within_bool, bz_w = _align(
                labels.shape, _foreground(data_w), broadcast_2d_to_3d=broadcast_2d_to_3d
            )
            broadcast_z = broadcast_z or bz_w

        mapping, overlap, within_frac, counts = _classify_overlap(
            labels,
            region_bool,
            overlap_threshold=overlap_threshold,
            within_bool=within_bool,
            within_threshold=within_threshold,
            inside_name=inside_name,
            outside_name=outside_name,
        )
        n_objects = int(np.count_nonzero(np.unique(labels)))

        table_rows = []
        for lbl in sorted(mapping):
            row = {
                "label": int(lbl),
                "overlap_fraction": float(overlap[lbl]),
                "region": mapping[lbl],
            }
            if within_frac is not None:
                row["within_fraction"] = float(within_frac[lbl])
            table_rows.append(row)
        columns = ["label", "overlap_fraction", "region"] + (
            ["within_fraction"] if within_bool is not None else []
        )
        df = pd.DataFrame(table_rows, columns=columns)
        tname = call_on_main(
            put_table,
            table_name or f"{labels_layer}_classification",
            df,
            spec={
                "tool": "classify_labels_by_mask",
                "labels_layer": labels_layer,
                "region_layer": region_layer,
                "overlap_threshold": float(overlap_threshold),
                "within_layer": within_layer,
                "within_threshold": float(within_threshold),
            },
        )

        prev: dict[int, str] = {}
        if write_label_names:
            provenance = {
                "region": mapping,
                "region_layer": region_layer,
                "overlap_threshold": float(overlap_threshold),
                "within_layer": within_layer,
                "within_threshold": float(within_threshold),
                "tool": "classify_labels_by_mask",
            }
            prev = call_on_main(_stamp_classification, labels_layer, mapping, provenance)
            if prev and any(k in prev and prev[k] != v for k, v in mapping.items()):
                warnings.append(
                    "overwrote existing label_names entries on the cells layer "
                    "(previous values returned in previous_label_names)"
                )

        if broadcast_z:
            warnings.append(
                "a 2D region mask was broadcast across all Z planes (assumes the domain is "
                "constant in Z; a real 3D domain should not be extruded)"
            )
        excluded = counts.get("excluded", 0)
        if excluded > 0:
            warnings.append(
                f"{excluded} object(s) classified 'excluded' (below within_threshold); "
                "filter_table to inside/outside before a two-group compare_groups"
            )

        return {
            "ok": True,
            "labels_layer": labels_layer,
            "table_name": tname,
            "counts": counts,
            "n_objects": n_objects,
            "overlap_threshold": float(overlap_threshold),
            "within_threshold": float(within_threshold),
            "broadcast_z": broadcast_z,
            "previous_label_names": prev,
            "note": "classify on one channel, measure a different one (same-channel is circular)",
            "warnings": warnings,
        }
    except _MaskError as e:
        return {"ok": False, "error": str(e)}
