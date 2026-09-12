"""ROI-drift tools: track or re-detect a hand-drawn ROI across every frame of a
live-imaging movie, instead of analysing it against one fixed frame.

THE BUG THIS FIXES (Korean bug report, verbatim gist): an ROI drawn once is
measured on a single frame; if the sample drifts, later frames are measured at
the WRONG pixels and the trace silently reflects motion, not biology.
measure_intensity_over_time (measure.py) itself is not wrong -- it explicitly
supports either a static OR a dynamic (per-frame) Labels seed, and simply
reuses whichever it is given (its ``static_labels`` branch, measure.py:257-258,
261-264 -- see also the per-frame ``np.take(label_arr, t, axis=time_axis)`` at
measure.py:270 inside the ``dynamic_labels`` branch). The fix is to hand it a
DYNAMIC labels layer instead of a static one; measure.py itself is unchanged.

Two independent producers build that dynamic layer, matching the two ways a
scientist already copes with drift by hand:

* track_roi_over_time (approach A, "draw once, track"): follow the seed's own
  drawn footprint frame to frame via confidence-gated cross-correlation
  (analysis.roi_track.track_rois, itself a thin wrapper over
  analysis.calcium_motion.correct_sparse). Cheap, preserves the exact
  hand-drawn shape, but only within a bounded per-frame search radius -- a
  hard jump loses the lock and reports a gap rather than guessing.
* resegment_roi_over_time (approach B, "draw wide, re-detect"): re-threshold a
  fresh ROI from scratch inside a generously wide, FIXED boundary every frame
  (analysis.roi_redetect.redetect_roi_masks), then link each frame's fresh
  detection back to the seed's stable id by nearest centroid. No step limit,
  so it survives a hard discontinuity approach A would (correctly) refuse, at
  the cost of needing a boundary wide enough to contain wherever the object
  might go.

Both emit a per-frame (T, ...) int32 Labels layer -- SAME full shape as the
movie, T at the SAME axis measure_intensity_over_time will resolve for that
movie -- plus an identical-schema QC table (one row per (label, time_index)
for every frame, built by the one shared ``_write_roi_qc_table`` helper so a
downstream consumer cannot tell which producer made it). A frame neither
producer can confidently place is written as background 0 and gets NO
measurement row out of measure.py's regionprops pass: a gap, never a
fabricated number (see test_gated_frame_produces_no_measurement_row).

set_labels_at_frame is the manual escape hatch for a frame that came back
gated (or simply wrong): splice a hand-corrected frame into either producer's
output and mark its QC rows trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift

from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.agent.qt_dispatch import call_on_main
from imajin.analysis.arrays import (
    layer_axes_from_metadata,
    materialize_array,
    resolve_time_axis,
)
from imajin.analysis.calcium_qc import _centroid_of
from imajin.analysis.roi_redetect import redetect_roi_masks
from imajin.analysis.roi_track import track_rois
from imajin.analysis.segmentation import voxel_spacing
from imajin.result_bundles import register_output
from imajin.session import get_layer, get_table, put_table, update_table
from imajin.tools._segmentation_io import (
    boundary_broadcast_warning,
    effective_target_min_size,
    resolve_boundary,
)
from imajin.tools._segmentation_outputs import _default_qc_png_path
from imajin.tools.napari_ops import add_labels_from_worker, snapshot_layer
from imajin.tools.registry import tool

_materialize = materialize_array  # shared: analysis.arrays.materialize_array

# Pinned QC schema (contract doc), in column order. dtypes are enforced
# explicitly in _write_roi_qc_table so approach A and approach B produce a
# byte-identical schema even when one producer's own qc_rows happen to make a
# column's *inferred* dtype ambiguous (e.g. redetect_roi_masks's "no_candidate"
# rows carry float('nan') y/x -- still float64, but worth pinning rather than
# trusting pandas' per-call inference to agree with track_rois's own rows).
_QC_DTYPES: dict[str, str] = {
    "label": "int64",
    "time_index": "int64",
    "usable": "bool",
    "confidence": "float64",
    "reason": "object",
    "y": "float64",
    "x": "float64",
    "n_pixels": "int64",
    "method": "object",
}


def _layer_axes(layer: Any, ndim: int) -> str:
    return layer_axes_from_metadata(getattr(layer, "metadata", None), ndim, default_3d="ZYX")


def _resolve_time_axis(layer: Any, ndim: int, time_axis: int | str | None) -> int:
    """Thin delegation -- see analysis.arrays.resolve_time_axis for the fail-loud
    body. Each time-series tool module carries this same 3-line adapter locally
    (tools/measure.py:119-122 is the original) rather than importing another
    module's private helper, so every caller here reads ITS OWN layer's axes
    metadata -- never another layer's resolved index. That distinction is the
    whole point of set_labels_at_frame resolving against the LABELS layer
    instead of reusing a movie's index (the measure.py:270 landmine)."""
    axes = _layer_axes(layer, ndim)
    return resolve_time_axis(axes, ndim, time_axis)


def _write_roi_qc_table(
    rows: list[dict[str, Any]], table_name: str, method: str
) -> tuple[str, pd.DataFrame, dict[int, float], list[int]]:
    """Assemble + store the QC table both roi_timeseries producers share.

    One shared helper -- rather than each producer building its own DataFrame
    -- so a downstream consumer (or a test) cannot tell which producer made a
    given QC table: same columns in the same order, same dtypes, same
    50%-coverage rejection rule as qc.py:439-446's calcium tables. ``method``
    is the one column the analysis-layer qc_rows deliberately omit (neither
    track_rois nor redetect_roi_masks knows which tool called it); this is the
    single place it gets stamped on.
    """
    columns = [c for c in _QC_DTYPES if c != "method"]
    df = pd.DataFrame(rows, columns=columns)
    df["method"] = method
    df = df.astype(_QC_DTYPES)

    if df.empty:
        coverage: dict[int, float] = {}
    else:
        coverage = {
            int(label): float(group["usable"].mean()) for label, group in df.groupby("label")
        }
    rejected = sorted(label for label, frac in coverage.items() if frac < 0.5)

    name = call_on_main(
        put_table, table_name, df, spec={"tool": method, "columns": list(df.columns)}
    )
    return name, df, coverage, rejected


def _write_roi_coverage_png(df: pd.DataFrame, path: Path, *, title: str) -> None:
    """Per-label x per-frame confidence strip: viridis while usable, flat grey
    while gated -- so a coverage gap is visible without opening the QC table.
    Short title only (this project's plot-text convention): the layer/channel
    name already lives in the filename, not in the plot itself."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = sorted(int(v) for v in df["label"].unique())
    times = sorted(int(v) for v in df["time_index"].unique())
    row_of = {label: i for i, label in enumerate(labels)}
    col_of = {t: i for i, t in enumerate(times)}

    confidence = np.full((len(labels), len(times)), np.nan, dtype=np.float32)
    usable = np.zeros((len(labels), len(times)), dtype=bool)
    for row in df.itertuples(index=False):
        r, c = row_of[int(row.label)], col_of[int(row.time_index)]
        confidence[r, c] = row.confidence
        usable[r, c] = bool(row.usable)

    fig_h = min(8.0, max(1.5, 0.28 * len(labels) + 0.9))
    fig, ax = plt.subplots(figsize=(6.0, fig_h))
    display = np.where(usable, confidence, np.nan)
    im = ax.imshow(display, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    gated = ~usable
    if gated.any():
        grey = np.zeros((*gated.shape, 4), dtype=np.float32)
        grey[gated] = (0.55, 0.55, 0.55, 1.0)
        ax.imshow(grey, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_xlabel("frame")
    ax.set_ylabel("label")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="confidence", fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_roi_coverage_png(
    df: pd.DataFrame, labels_layer: Any, source_layer: Any, *, method: str
) -> tuple[str | None, str | None, str | None]:
    """Best-effort coverage/confidence strip PNG. Mirrors finalize_qc_png's
    contract (_segmentation_io.py:140-185) -- resolve a default path, save,
    register the output, stamp labels_layer.metadata['qc_png_path'] -- but the
    picture is a coverage strip built from the QC table both producers already
    share, not an image+mask overlay (there is no single raw frame to overlay
    on for a whole movie). Never raises: an agent mid-analysis should not lose
    a tracked/redetected ROI over a diagnostic PNG.
    """
    if df.empty:
        return None, None, "no QC rows to plot"
    try:
        out_path = _default_qc_png_path(labels_layer.name, source_layer)
        _write_roi_coverage_png(df, out_path, title="ROI coverage")
        try:
            register_output(
                "qc_png",
                out_path,
                {
                    "labels_layer": labels_layer.name,
                    "source_layer": getattr(source_layer, "name", None),
                    "method": method,
                },
            )
        except ValueError:
            pass
        saved = str(out_path)
        try:
            labels_layer.metadata["qc_png_path"] = saved
        except Exception:
            pass
        return saved, None, None
    except Exception as exc:  # noqa: BLE001 - a QC picture must never cost the caller their ROI
        return None, f"{type(exc).__name__}: {exc}", None


def _lateral_shift_stack(
    movie_t0: np.ndarray,
    seed_full: np.ndarray,
    labels2d: np.ndarray,
    qc_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Re-apply track_rois' own per-frame lateral (dy, dx) to the FULL,
    un-projected 3D seed mask -- the z_project='max' fallback for a true 3D+T
    seed (roi_track.py is 2D-only by design; see its module docstring).

    ``qc_rows`` is track_rois' own canonical output for the Z-max-projected
    ``labels2d``; ``dy, dx`` per (label, time) is recomputed here from its
    ``y, x`` (track_rois' exact ``pos[t]``) against ``labels2d``'s own centroid
    -- the identical formula rasterize_tracked_labels used to build the 2D
    stack in the first place (roi_track.py: ``base_y, base_x =
    _centroid_of(orig_mask)``; ``dy = round(pos[t,0] - base_y)``) -- so the 3D
    shift applied here is bit-for-bit the same lateral placement the 2D pass
    already validated, just replayed against every Z plane of the real mask.

    Reuses ``usable`` from ``qc_rows`` verbatim as the placement gate --
    INCLUDING its 2D collision arbitration -- rather than re-deriving a
    separate 3D-voxel collision system: shifting only Y/X never changes which
    (y, x) columns a label occupies at some Z, so two labels' shifted 3D masks
    can only newly overlap in voxel space where their shifted 2D projections
    already overlap too -- exactly what the 2D pass already adjudicated. The
    only case this does not cover is two labels sharing a projected footprint
    while occupying disjoint Z ranges (never a real 3D collision); not worth a
    second, 3D-only collision system for a fallback that is already documented
    as lateral-only.
    """
    t_count = int(movie_t0.shape[0])
    frame_shape_full = movie_t0.shape[1:]
    stack = np.zeros((t_count, *frame_shape_full), dtype=np.int32)
    seed_ids = sorted(int(v) for v in np.unique(labels2d) if v != 0)
    base_xy = {label: _centroid_of(labels2d == label) for label in seed_ids}
    by_label_time = {(int(r["label"]), int(r["time_index"])): r for r in qc_rows}

    for label in seed_ids:
        orig_mask = seed_full == label
        if not orig_mask.any():
            continue  # nothing drawn for this id in the ORIGINAL 3D mask
        base_y, base_x = base_xy[label]
        for t in range(t_count):
            row = by_label_time.get((label, t))
            if row is None or not row["usable"]:
                continue
            dy = int(round(float(row["y"]) - base_y))
            dx = int(round(float(row["x"]) - base_x))
            if dy == 0 and dx == 0:
                shifted = orig_mask
            else:
                shift_vec = (0,) * (orig_mask.ndim - 2) + (dy, dx)  # Z (if any) untouched
                shifted = nd_shift(
                    orig_mask.astype(np.float32), shift_vec, order=0, mode="constant"
                ).astype(bool)
            if shifted.any():
                stack[t][shifted] = label
    return stack


# Contrast-driven survivorship bias: correct_sparse (calcium_motion.py) gates a
# frame by the tracked object's own located-confidence, which is itself driven
# by SIGNAL CONTRAST (observability()'s SNR check + the cross-correlation peak
# height) -- never by drift. So on a movie whose signal genuinely dips toward
# baseline (a calcium transient's own trough), the frames GATED are
# systematically the DIM ones and the surviving trace is biased UPWARD. That is
# correct behaviour for the gate itself -- a single tracked ROI has no other
# evidence to fall back on, and correct_sparse's neighbour-interpolation
# recovery is provably inert below 3 co-visible labels (see roi_track.py's own
# module docstring) -- but the tool must not stay silent about the
# consequence. Measured: a single-ROI recording whose signal dipped to
# background kept 51/60 frames (85% coverage, comfortably above the 50%
# `rejected` cutoff below) yet the surviving mean still carried a +15.2%
# upward bias -- so the existing `rejected` warning alone would miss exactly
# the case this exists to catch.
_CONTRAST_GATE_REASONS = frozenset(
    {
        "no_neighbours",
        "degenerate_neighbours",
        "outside_hull",
        "high_residual",
        # "interpolated" counts ONLY because this helper looks at unusable rows
        # exclusively (see the ~usable filter below); a usable interpolated frame
        # is never examined here. An UNUSABLE one means the direct lock already
        # failed on contrast -- that is the only way a frame reaches
        # correct_sparse's interpolation branch at all (calcium_motion.py:134-148)
        # -- and it was then re-gated because interpolation pins confidence to
        # exactly 0.6 (:147) while conf_floor, which track_roi_over_time exposes
        # as a user parameter defaulting to 0.5, was set above that. Leaving it
        # out inverted the whole warning: measured on a 4-ROI recording whose
        # centre ROI dips to baseline, conf_floor=0.5 dropped 7/40 frames (all
        # "no_neighbours") and warned, while conf_floor=0.8 dropped 18/40 -- a
        # WORSE bias -- but 10 of those read "interpolated", so the remaining 8
        # were no longer the majority the check below requires and the warning
        # went silent on the one label that most needed it.
        "interpolated",
    }
)
_BIAS_COVERAGE_FLOOR = 0.9  # below this, enough dim frames were dropped to skew a mean


def _contrast_bias_warnings(
    qc_df: pd.DataFrame, coverage: dict[int, float], t_count: int
) -> list[str]:
    """One warning per label whose gated frames are dominated by a contrast/SNR
    reason (as opposed to rasterize_tracked_labels' own "out_of_bounds" /
    "collision" -- purely spatial outcomes for a frame that WAS located, so they
    carry no brightness signal and must not trip this check). Read-only over
    the QC table the gate already produced; this never changes `usable`.
    """
    warnings: list[str] = []
    for label, frac in sorted(coverage.items()):
        if frac >= _BIAS_COVERAGE_FLOOR:
            continue
        gated = qc_df[(qc_df["label"] == label) & (~qc_df["usable"])]
        if gated.empty:
            continue
        contrast_n = int(gated["reason"].isin(_CONTRAST_GATE_REASONS).sum())
        if contrast_n * 2 <= len(gated):  # not a majority -- e.g. drift/collision-driven
            continue
        # The remedy depends on WHY contrast was low, and the two causes need
        # opposite advice. A signal that dips to baseline is a real brightness
        # problem -> track a structural channel. But a recording whose background
        # is bright, structured tissue rather than black defeats the SNR estimate
        # itself (snr_floor compares local contrast against a whole-movie noise
        # figure that, on such an image, measures anatomy): there the object is
        # perfectly trackable and the floor is simply out of calibration.
        # Measured on a real confocal recording: snr read 1.80 against a floor of
        # 3.0 while template matching found the cell with a median peak of 0.846.
        # Recommending a channel switch there sends the user after a problem they
        # do not have, so name both and let the coverage number discriminate.
        share = len(gated) / t_count if t_count else 0.0
        remedy = (
            "Almost every frame was refused, which usually means snr_floor is out "
            "of calibration for this image rather than that the object is dim -- "
            "on a bright, structured background try snr_floor=0.5 or lower first. "
            if share > 0.75
            else "If the object's signal genuinely dips to baseline, track a "
            "structural/activity-independent channel and measure this Labels layer "
            "against the signal channel instead. "
        )
        warnings.append(
            f"label {label}: {len(gated)}/{t_count} frames were dropped where the "
            "object's contrast fell below the detection floor; the surviving trace "
            f"is biased toward brighter frames. {remedy}"
            "Otherwise use resegment_roi_over_time, which re-detects per frame "
            "instead of following one seed."
        )
    return warnings


@tool(
    description="Track a hand-drawn ROI (a Labels seed drawn on ONE frame) across "
    "every frame of a live-imaging movie by following its own lateral (XY) drift -- "
    "approach A for the ROI-drift problem: draw once, track. 2D+T only; pass "
    "z_project='max' for a 4D TZYX movie (a 3D seed drawn on one Z-stack timepoint is "
    "then corrected for lateral drift only -- Z/focus drift is NOT corrected, and a "
    "warning says so). Emits a per-frame Labels layer (same full shape as the movie) "
    "plus a QC table (coverage/confidence/reason per label per frame); feed the result "
    "into measure_intensity_over_time instead of measuring the single-frame seed "
    "directly against the movie.",
    phase="2",
    worker=True,
)
def track_roi_over_time(
    labels_layer: str,
    image_layer: str,
    time_axis: int | str | None = None,
    search_radius: int = 6,
    snr_floor: float = 3.0,
    conf_floor: float = 0.5,
    z_project: str = "none",
    table_name: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    if z_project not in ("none", "max"):
        raise ValueError(f"z_project must be 'none' or 'max', got {z_project!r}")

    raise_if_cancelled()
    # Deliberately snapshot_layer, NOT _segmentation_io.load_and_guard: that
    # helper's whole reason to exist is its "'T' in axes" guard (:51-55), which
    # rejects exactly the time-series input this tool is FOR.
    image = call_on_main(snapshot_layer, image_layer)
    report_progress(stage="loading_movie")  # before the (possibly multi-GB) materialize
    image_arr = _materialize(image.data)
    labels = call_on_main(snapshot_layer, labels_layer)
    label_arr = _materialize(labels.data).astype(np.int32)

    t_idx = _resolve_time_axis(image, image_arr.ndim, time_axis)
    axes = _layer_axes(image, image_arr.ndim)
    movie_t0 = np.moveaxis(image_arr, t_idx, 0)
    frame_shape_full = movie_t0.shape[1:]
    ndim_frame = len(frame_shape_full)
    t_count = int(movie_t0.shape[0])

    if ndim_frame not in (2, 3):
        raise ValueError(
            f"track_roi_over_time expects a 2D (YX) or 3D (ZYX) movie frame; "
            f"image_layer {image_layer!r} resolved to per-frame shape "
            f"{frame_shape_full} (full shape {image_arr.shape}, axes {axes!r})."
        )
    if ndim_frame == 3 and z_project != "max":
        raise ValueError(
            f"track_roi_over_time is 2D+T only; image_layer {image_layer!r} has a 3D "
            f"per-frame shape {frame_shape_full} (full shape {image_arr.shape}, axes "
            f"{axes!r}). Pass z_project='max' to track lateral drift on a Z-max-"
            "projected copy of the movie (Z/focus drift will NOT be corrected), or "
            "use resegment_roi_over_time for true 3D+T re-detection."
        )
    if label_arr.shape != frame_shape_full:
        raise ValueError(
            f"labels_layer {labels_layer!r} shape {label_arr.shape} does not match "
            f"one frame of image_layer {image_layer!r} ({frame_shape_full}); draw or "
            "segment the seed ROI on a single timepoint of this movie."
        )

    warnings: list[str] = []
    if ndim_frame == 3:
        axes_no_t = axes[:t_idx] + axes[t_idx + 1 :]
        if "Z" not in axes_no_t:
            raise ValueError(
                f"track_roi_over_time cannot z_project image_layer {image_layer!r}: "
                f"axes {axes!r} has no 'Z' to project (time-stripped {axes_no_t!r})."
            )
        z_axis = 1 + axes_no_t.index("Z")
        movie2d = np.max(movie_t0, axis=z_axis)
        # Project the seed the SAME way: ascending label order so a genuine XY
        # overlap between two different Z-ranges (rare) resolves deterministically
        # rather than depending on np.unique's/np.max's own tie-break.
        labels2d = np.zeros(frame_shape_full[-2:], dtype=np.int32)
        for label in sorted(int(v) for v in np.unique(label_arr) if v != 0):
            labels2d[np.any(label_arr == label, axis=0)] = label
        warnings.append(
            "z_project='max' tracked lateral (XY) drift only, on a Z-max-projected "
            "copy of the movie; Z/focus drift in the original 3D stack is NOT "
            "corrected by this pass."
        )
    else:
        movie2d = movie_t0
        labels2d = label_arr

    result = track_rois(
        movie2d, labels2d, snr_floor=snr_floor, max_step=search_radius, conf_floor=conf_floor
    )
    qc_rows = result["qc_rows"]

    if ndim_frame == 3:
        stack = _lateral_shift_stack(movie_t0, label_arr, labels2d, qc_rows)
        # n_pixels re-derived from the actual 3D stack (one np.unique pass per
        # frame, mirroring track_rois' own convention) so it never disagrees with
        # what the LABELS layer actually contains -- the 2D pass's n_pixels would
        # otherwise describe the projected footprint, not the real 3D voxel count.
        pixel_counts = []
        for t in range(t_count):
            vals, counts = np.unique(stack[t], return_counts=True)
            pixel_counts.append(dict(zip(vals.tolist(), counts.tolist())))
        qc_rows = [
            {**row, "n_pixels": int(pixel_counts[row["time_index"]].get(row["label"], 0))}
            for row in qc_rows
        ]
    else:
        stack = result["stack"]

    # Post-hoc checkpoint, kept deliberately even though track_rois is no longer
    # opaque: its rasterization phase now reports its own per-label "rasterizing"
    # progress and raises on a cancel between labels (roi_track.py), but that
    # covers only the mask-shift pass -- correct_sparse's localization before it,
    # and _lateral_shift_stack's z-broadcast above, are still single synchronous
    # calls with no hook. This is the one cancellation point that is guaranteed to
    # exist before the labels layer and QC table get written, i.e. before this
    # tool has any side effect on the session, so a cancel arriving during the
    # un-hooked phases still stops the run instead of silently committing output.
    # Runs over frames (not labels) so the two stages report different, honest
    # cadences rather than one pretending to re-do the other's work.
    for t in range(t_count):
        raise_if_cancelled()
        report_progress(progress=(t + 1) / t_count, stage="tracking")

    label_out = np.moveaxis(stack, 0, t_idx)  # T back to the MOVIE's own resolved axis

    qc_table_name, qc_df, coverage, rejected = _write_roi_qc_table(
        qc_rows, table_name or f"{labels_layer}_{image_layer}_roi_qc", "track_roi_over_time"
    )
    if rejected:
        warnings.append(f"{len(rejected)} label(s) below 50% usable coverage: {rejected}")
    warnings.extend(_contrast_bias_warnings(qc_df, coverage, t_count))

    out_layer = call_on_main(
        add_labels_from_worker,
        label_out,
        name=name or f"{labels_layer}_tracked",
        scale=tuple(float(s) for s in image.scale),
        metadata={
            "axes": axes,
            "time_axis": t_idx,
            "source_layer": labels_layer,
            "image_layer": image_layer,
            "method": "track_roi_over_time",
            "method_params": {
                "search_radius": search_radius,
                "snr_floor": snr_floor,
                "conf_floor": conf_floor,
                "z_project": z_project,
            },
            "qc_table": qc_table_name,
        },
    )
    saved_qc_png, qc_png_error, qc_png_skipped_reason = _save_roi_coverage_png(
        qc_df, out_layer, image, method="track_roi_over_time"
    )

    return {
        "labels_layer": out_layer.name,
        "qc_table": qc_table_name,
        "n_labels": len(coverage),
        "n_timepoints": t_count,
        "coverage": coverage,
        "rejected": rejected,
        "warnings": warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Re-detect an ROI inside a generously wide, fixed boundary on every "
    "frame of a live-imaging movie -- approach B for the ROI-drift problem: draw wide, "
    "re-detect. Works in 2D+T or true 3D+T directly (unlike track_roi_over_time, no "
    "z-projection needed). Give a seed Labels layer (drawn/segmented on one frame, used "
    "only for its centroids + label ids) and a wide boundary_mask (e.g. a Shapes ROI on "
    "a max projection converted via boundary_mask_from_shapes) that the object could "
    "drift anywhere inside of. Emits a per-frame Labels layer (same full shape as the "
    "movie) plus a QC table; feed the result into measure_intensity_over_time instead "
    "of measuring the single-frame seed directly against the movie.",
    phase="2",
    worker=True,
)
def resegment_roi_over_time(
    labels_layer: str,
    boundary_mask: str,
    image_layer: str,
    time_axis: int | str | None = None,
    link_max_distance: float = 25.0,
    link_max_area_ratio: float | None = 3.0,
    auto_correct: bool = True,
    max_iters: int = 3,
    background_radius: int = 48,
    background_method: str = "opening",
    background_percentile: float = 20.0,
    threshold_method: str = "auto",
    threshold_percentile: float = 99.0,
    min_snr: float = 2.0,
    high_snr: float = 4.0,
    smoothing_sigma: float = 1.0,
    min_size: int | None = None,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    table_name: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    raise_if_cancelled()
    image = call_on_main(snapshot_layer, image_layer)
    report_progress(stage="loading_movie")  # before the (possibly multi-GB) materialize
    image_arr = _materialize(image.data)
    labels = call_on_main(snapshot_layer, labels_layer)
    label_arr = _materialize(labels.data).astype(np.int32)

    t_idx = _resolve_time_axis(image, image_arr.ndim, time_axis)
    axes = _layer_axes(image, image_arr.ndim)
    frames = np.moveaxis(image_arr, t_idx, 0)
    frame_shape = frames.shape[1:]
    t_count = int(frames.shape[0])

    # Resolved ONCE against a single frame's shape and never touched again --
    # redetect_roi_masks only ever READS boundary_bool, every frame (it may be a
    # read-only Z-broadcast view; see resolve_boundary_mask).
    boundary_bool, boundary_raw = resolve_boundary(boundary_mask, frame_shape)
    warnings: list[str] = []
    bcast = boundary_broadcast_warning(boundary_bool, boundary_raw)
    if bcast:
        warnings.append(bcast)

    # extract_timepoint's own axis-drop pattern (view.py:260-261), then through
    # voxel_spacing so a trivial/absent layer scale reads as "no physical
    # calibration" here exactly like it does for segment_target_objects
    # (target.py:86) -- the same raw metadata should mean the same thing to
    # effective_target_min_size whichever tool reached it.
    scale_no_t = tuple(float(s) for i, s in enumerate(image.scale) if i != t_idx)
    spacing = voxel_spacing(scale_no_t, len(frame_shape))
    effective_min_size = effective_target_min_size(
        frames[0],
        min_size=min_size,
        min_area_um2=min_area_um2,
        min_volume_um3=min_volume_um3,
        spacing=spacing,
    )
    params = {
        "background_radius": background_radius,
        "background_method": background_method,
        "background_percentile": background_percentile,
        "threshold_method": threshold_method,
        "threshold_percentile": threshold_percentile,
        "min_snr": min_snr,
        "high_snr": high_snr,
        "smoothing_sigma": smoothing_sigma,
        "min_size": effective_min_size,
    }

    stack, qc_rows = redetect_roi_masks(
        frames,
        label_arr,
        boundary_bool,
        spacing=spacing,
        params=params,
        auto_correct=auto_correct,
        max_iters=max_iters,
        link_max_distance=link_max_distance,
        # Forwarded rather than left to redetect_roi_masks' own default so this
        # SIZE gate has an off switch from the tool layer: it rejects a candidate
        # whose area differs from the seed's last confirmed area by more than
        # this factor, which is right for stable objects but will read a real,
        # strongly photobleaching or partially-occluded ROI as "no_candidate".
        # Without a parameter here that failure mode would be unreachable to
        # diagnose or disable -- pass None for the distance-only linking this
        # tool did before the gate existed.
        link_max_area_ratio=link_max_area_ratio,
        # seed_frame intentionally left at redetect_roi_masks' own default (0):
        # this tool's pinned signature carries no parameter for which timepoint
        # the seed was drawn on, and no layer metadata reliably reconstructs it.
        # A seed drawn on frame 0 (extract_timepoint's own default t=0) gets the
        # full outward walk; a seed drawn mid-movie only walks forward from it.
    )

    # Post-hoc per-frame checkpoint -- see track_roi_over_time's identical
    # comment; the per-frame re-segmentation above is the expensive part and
    # offers no callback hook of its own.
    for t in range(t_count):
        raise_if_cancelled()
        report_progress(progress=(t + 1) / t_count, stage="resegmenting")

    label_out = np.moveaxis(stack, 0, t_idx)  # T back to the MOVIE's own resolved axis

    qc_table_name, qc_df, coverage, rejected = _write_roi_qc_table(
        qc_rows,
        table_name or f"{labels_layer}_{image_layer}_roi_qc",
        "resegment_roi_over_time",
    )
    if rejected:
        warnings.append(f"{len(rejected)} label(s) below 50% usable coverage: {rejected}")

    out_layer = call_on_main(
        add_labels_from_worker,
        label_out,
        name=name or f"{labels_layer}_resegmented",
        scale=tuple(float(s) for s in image.scale),
        metadata={
            "axes": axes,
            "time_axis": t_idx,
            "source_layer": labels_layer,
            "image_layer": image_layer,
            "boundary_mask": boundary_mask,
            "method": "resegment_roi_over_time",
            "method_params": {
                "link_max_distance": link_max_distance,
                "link_max_area_ratio": link_max_area_ratio,
                "auto_correct": auto_correct,
                "max_iters": max_iters,
                "background_radius": background_radius,
                "background_method": background_method,
                "background_percentile": background_percentile,
                "threshold_method": threshold_method,
                "threshold_percentile": threshold_percentile,
                "min_snr": min_snr,
                "high_snr": high_snr,
                "smoothing_sigma": smoothing_sigma,
                "min_size": effective_min_size,
                "requested_min_size": min_size,
                "min_area_um2": min_area_um2,
                "min_volume_um3": min_volume_um3,
            },
            "qc_table": qc_table_name,
        },
    )
    saved_qc_png, qc_png_error, qc_png_skipped_reason = _save_roi_coverage_png(
        qc_df, out_layer, image, method="resegment_roi_over_time"
    )

    return {
        "labels_layer": out_layer.name,
        "qc_table": qc_table_name,
        "n_labels": len(coverage),
        "n_timepoints": t_count,
        "coverage": coverage,
        "rejected": rejected,
        "warnings": warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
    }


@tool(
    description="Manually correct one frame of a per-frame tracked/redetected ROI "
    "Labels layer -- e.g. track_roi_over_time or resegment_roi_over_time gated (or "
    "mis-placed) a frame; hand-correct it in a separate layer, then splice it back in "
    "with this tool. Pass label_id to replace only that label's footprint in the frame "
    "(other labels there are left untouched); omit it to replace the whole frame. "
    "Marks the companion QC table's rows for this (label, time_index) trusted "
    "(usable=True, confidence=1.0, reason='manual_correction').",
    phase="4",
    worker=True,
)
def set_labels_at_frame(
    labels_layer: str,
    time_index: int,
    source_layer: str,
    label_id: int | None = None,
    time_axis: int | str | None = None,
) -> dict[str, Any]:
    if label_id == 0:
        raise ValueError(
            "label_id must be a nonzero label id (0 is background; splicing it in "
            "would erase every label wherever source_layer is background)."
        )

    labels = call_on_main(snapshot_layer, labels_layer)
    label_arr = _materialize(labels.data).astype(np.int32)  # .astype() always copies
    # The TARGET labels layer's OWN axes -- never the movie's -- because this
    # tool edits exactly the array measure_intensity_over_time will read next.
    # Reusing a different array's resolved index here is the measure.py:270
    # landmine this whole feature exists to close.
    t_idx = _resolve_time_axis(labels, label_arr.ndim, time_axis)
    if not 0 <= time_index < label_arr.shape[t_idx]:
        raise ValueError(
            f"time_index {time_index} out of range for axis {t_idx} of shape "
            f"{label_arr.shape} (labels_layer {labels_layer!r})"
        )

    source = call_on_main(snapshot_layer, source_layer)
    source_arr = _materialize(source.data).astype(np.int32)
    frame_shape = tuple(s for i, s in enumerate(label_arr.shape) if i != t_idx)
    if source_arr.shape != frame_shape:
        raise ValueError(
            f"shape mismatch: source_layer {source_layer!r} has shape "
            f"{source_arr.shape}; labels_layer {labels_layer!r} has shape "
            f"{label_arr.shape} with time_axis={t_idx}, so one frame is {frame_shape}."
        )

    moved = np.moveaxis(label_arr, t_idx, 0)  # a VIEW into label_arr's own buffer
    if label_id is None:
        moved[time_index] = source_arr
    else:
        updated_slice = moved[time_index].copy()
        vacated = updated_slice == label_id
        claimed = source_arr == label_id
        updated_slice[vacated] = 0
        updated_slice[claimed] = label_id
        moved[time_index] = updated_slice

    def _commit() -> None:
        target = get_layer(labels_layer)
        target.data = label_arr
        meta = dict(getattr(target, "metadata", {}) or {})
        corrections = list(meta.get("manual_corrections") or [])
        corrections.append(
            {
                "time_index": int(time_index),
                "source_layer": source_layer,
                "corrected_by": "manual",
            }
        )
        meta["manual_corrections"] = corrections
        target.metadata = meta

    call_on_main(_commit)

    qc_table_name = (labels.metadata or {}).get("qc_table")
    updated_rows = 0
    if qc_table_name:
        try:
            qc_df = get_table(qc_table_name).copy()
        except KeyError:
            qc_df = None
        if qc_df is not None and {"label", "time_index"}.issubset(qc_df.columns):
            mask = qc_df["time_index"] == int(time_index)
            if label_id is not None:
                mask &= qc_df["label"] == int(label_id)
            updated_rows = int(mask.sum())
            if updated_rows:
                qc_df.loc[mask, "confidence"] = 1.0
                qc_df.loc[mask, "reason"] = "manual_correction"
                # y/x/n_pixels are re-derived from the frame that is now actually
                # in the layer, never left at the values the gated pass recorded:
                # a stale row would have the QC table vouch (usable=True,
                # confidence=1.0) for a footprint that moved, or claim n_pixels=0
                # for a frame the user just filled in. `usable` follows the same
                # evidence -- a label with no pixels left in the corrected frame
                # gets NO measurement row out of regionprops, so calling it usable
                # would break the coverage/measurement agreement this whole
                # feature rests on.
                corrected_frame = moved[int(time_index)]
                for row_idx in qc_df.index[mask]:
                    coords = np.nonzero(corrected_frame == int(qc_df.at[row_idx, "label"]))
                    n_px = int(coords[0].size)
                    qc_df.at[row_idx, "n_pixels"] = n_px
                    qc_df.at[row_idx, "usable"] = n_px > 0
                    qc_df.at[row_idx, "y"] = float(coords[-2].mean()) if n_px else float("nan")
                    qc_df.at[row_idx, "x"] = float(coords[-1].mean()) if n_px else float("nan")
                call_on_main(update_table, qc_table_name, qc_df)

    return {
        "labels_layer": labels_layer,
        "time_index": int(time_index),
        "source_layer": source_layer,
        "label_id": label_id,
        "qc_table": qc_table_name,
        "qc_rows_updated": updated_rows,
    }
