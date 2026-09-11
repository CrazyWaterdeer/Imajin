"""Approach A ("draw once, track it") for the live-imaging drift problem.

A user draws an ROI on frame 1 of a live-imaging movie; if the sample drifts, an
ROI analysed only on that one frame measures the wrong pixels on every later
frame. This module tracks the drawn ROI across time and rasterizes it back into a
per-frame label stack, so measure_intensity_over_time reads the right pixels on
every frame instead of one fixed footprint.

Deliberately does NOT move anything out of analysis.calcium_motion -- that module
is validated by tests/test_calcium_motion.py and tests/test_calcium_v2_validation.py,
and extracting pieces of it would buy nothing here while risking those guarantees.
This module only CALLS calcium_motion.correct_sparse and turns its result into a
label stack + QC rows.

2D+T ONLY. calcium_qc's locate_cell kernel is a fixed 2-level (dy, dx) search loop,
motion_safe_template does `np.nonzero(roi)` expecting a 2D roi, and _patch_at
unpacks a frame's shape as exactly (h, w) -- a 3D (Z, Y, X) frame would either
crash confusingly deep inside one of those or silently misinterpret an axis as Z.
track_rois rejects that shape itself, naming it, instead of letting it fail there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import shift as nd_shift

from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.analysis import calcium_motion
from imajin.analysis.arrays import map_over_axis0
from imajin.analysis.calcium_qc import _centroid_of

# Below this many pixels per frame, threading the per-frame mask shift is a NET
# LOSS, so rasterize_tracked_labels runs it serially instead. Measured on the
# shipped worker policy (map_over_axis0's own min(n_frames, os.cpu_count()) --
# 24 on the box this was measured on), T=100, one label, speedup vs serial:
#     96x96   0.10x | 128x128 0.18x | 256x256 0.56x | 384x384 1.25x
#     448x448 1.91x | 512x512 2.41x | 768x768 3.79x | 1024x1024 4.41x
# Capping the worker count does NOT rescue the small sizes -- 96x96 measured
# 0.32x/0.28x/0.16x/0.15x/0.11x/0.10x/0.09x at 2/3/4/6/8/12/24 workers, i.e. a
# loss at EVERY width, and 256x256 peaks at just 1.14x (4 workers). So the gate
# has to be on frame size, not on how many threads are used: scipy.ndimage.shift
# over a small frame returns faster than a pool costs to dispatch to it, and no
# worker count makes dispatch cheaper. 200_000 is the smallest size actually
# measured to clear a ~1.5x win (448x448 -> 1.91x); 384x384 (147k px) only
# reached 1.25x and is deliberately left serial.
_PARALLEL_SHIFT_MIN_FRAME_PX = 200_000


def rasterize_tracked_labels(
    labels2d: np.ndarray,
    positions: dict[int, np.ndarray],
    usable: dict[int, np.ndarray],
    *,
    confidence: dict[int, np.ndarray],
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], str]]:
    """Per-frame label stack (T, Y, X) int32 built by SHIFTING the original mask.

    positions: {label_id: (T,2) float array of (y,x) ABSOLUTE centroids}, e.g.
        straight from ``calcium_motion.correct_sparse(...).positions``. The
        per-frame shift is this minus the label's OWN seed centroid, recomputed
        here with the same ``_centroid_of`` correct_sparse used internally -- so
        it is bit-identical to the base position `positions` was measured from.
    usable: {label_id: (T,) bool} -- False means "leave this frame background".
        This function trusts the flag as given; it never re-derives or
        second-guesses it, only rasterizes what is already marked usable.
    confidence: {label_id: (T,) float in [0,1]} -- used ONLY to arbitrate a
        same-frame collision between two labels' relocated masks (see below).
        Pass ``correct_sparse(...).confidence`` straight through.
    shape: the full (T, Y, X) output shape; (Y, X) must equal labels2d.shape.

    Translates each label's ORIGINAL mask with scipy.ndimage.shift(mask, (dy, dx),
    order=0) -- the same primitive already used at calcium_qc.py:217 -- rather
    than resynthesizing a disk the way corrected_dff does at
    calcium_motion.py:168-177. That preserves the user's hand-drawn ROI shape
    frame to frame, which is the entire point of tracking instead of re-detecting.

    dy/dx are exactly integral whenever a position came from a direct locate_cell
    lock: locate_cell (calcium_qc.py:127-145) returns integer dy/dx, and
    propagated_locate only ever accumulates those onto the base centroid. So on
    every "located" frame the relocated mask is bit-identical in area to the
    drawn ROI -- verified in tests/test_roi_track.py with a deliberately
    non-round mask (an L-shape), since a disk-synthesizing regression would pass
    an area check by accident on a symmetric shape. A neighbour-interpolated
    position is instead a fitted float; it is rounded to the nearest pixel before
    shifting (rasterizing a mask below pixel resolution is meaningless), so its
    area can drift by a pixel or two at the boundary -- an inherent property of
    interpolation, not a bug in this function.

    Returns ``(stack, notes)``. `stack` is (T, *labels2d.shape) int32, background
    0, each tracked label keeping its own seed id (never renumbered). `notes`
    reports the two ways a frame marked usable still couldn't be placed, keyed by
    (label, time_index):
      - "out_of_bounds": the shifted mask landed entirely off the array (the
        in-bounds part was kept, and it was empty) -- e.g. drift carried the ROI
        past the frame edge.
      - "collision": this label's mask overlapped a higher-confidence label's
        mask on this frame. Resolved by dropping the LOWER-confidence label's
        mask for the WHOLE frame -- never a silently truncated mask, which would
        under-measure area with nothing to signal it happened. Ties (equal
        confidence) go to the smaller label id, so the outcome is deterministic.
    Neither case is guessed or interpolated away: the caller (track_rois) turns
    both into a QC row with usable=False, so this is a reported gap, never a
    fabricated number.
    """
    labels2d = np.asarray(labels2d)
    if labels2d.ndim != 2:
        raise ValueError(
            f"rasterize_tracked_labels needs a 2D labels array (Y, X); got shape "
            f"{labels2d.shape}. This module is 2D+T only -- see roi_track.py header."
        )
    if len(shape) != 3 or tuple(shape[1:]) != labels2d.shape:
        raise ValueError(
            f"shape {shape} must be (T, *labels2d.shape); labels2d is {labels2d.shape}"
        )
    if positions.keys() != usable.keys() or positions.keys() != confidence.keys():
        raise ValueError(
            "rasterize_tracked_labels: positions/usable/confidence must share the same "
            f"label ids; got {sorted(positions)} vs {sorted(usable)} vs {sorted(confidence)}"
        )
    t_count, height, width = shape

    stack = np.zeros((t_count, height, width), dtype=np.int32)
    notes: dict[tuple[int, int], str] = {}

    label_ids = sorted(positions)
    # Per-label shifted mask, keyed by time_index, for frames that placed
    # something. Built once and reused by both the collision pass below and the
    # final stack write, so a mask is never recomputed (and never has a chance to
    # disagree with itself) between the two passes.
    frame_masks: dict[int, dict[int, np.ndarray]] = {lbl: {} for lbl in label_ids}

    # This loop is the DOMINANT cost of track_rois -- measured 57-65% of total
    # wall time at 256x256/512x512 (T=100), bigger than correct_sparse's own
    # localization pass -- because scipy.ndimage.shift runs once per
    # (label, frame) over a full-frame-sized array. Unlike correct_sparse
    # (deliberately left untouched -- see track_rois' docstring for why), the
    # shift IS embarrassingly parallel per frame within a label and measurably
    # threads well (shift releases the GIL): 1.7-1.8x at 256x256 (w4-8),
    # 2.1-5.0x at 512x512 (w4-8, best measured). So each label's own per-frame
    # shifts are dispatched through map_over_axis0 below.
    #
    # Cancellation/progress land BETWEEN labels, in this orchestrating thread
    # -- never inside a map_over_axis0 worker. raise_if_cancelled/
    # report_progress are ContextVar-backed (agent/execution.py), and a
    # ThreadPoolExecutor worker starts with its own fresh, empty Context (see
    # map_over_axis0's docstring for how this was verified) -- a worker
    # checking either would silently see no token/job at all, so it would
    # never actually fire. Per-label is also a real, user-legible unit of
    # progress ("rasterized ROI 3 of 5") and is a strict improvement over the
    # status quo (this function previously offered no interior checkpoint at
    # all). It does NOT subdivide one label's own frame range, so a single
    # label with a very large frame count still runs as one uninterruptible
    # batch -- accepted rather than chunking on an invented, unmeasured batch
    # size; revisit with real numbers if that gap is ever a real complaint.
    for i, lbl in enumerate(label_ids):
        raise_if_cancelled()
        orig_mask = labels2d == lbl
        if orig_mask.any():  # else: nothing drawn for this id -- stays background
            base_y, base_x = _centroid_of(orig_mask)
            pos = np.asarray(positions[lbl], dtype=float)
            usable_t = np.asarray(usable[lbl], dtype=bool)
            n_frames = min(t_count, len(pos), len(usable_t))

            # One slot per frame, written at most once each by whichever
            # thread handles index t -- disjoint list-index writes, safe
            # under the GIL regardless of completion order. The aggregation
            # loop below always walks t in fixed 0..n_frames-1 order, so the
            # result (frame_masks/notes) depends only on t, never on which
            # thread finished first -- bit-identical to the prior serial loop.
            shifted_by_t: list[np.ndarray | None] = [None] * n_frames

            def _shift_frame(
                t: int,
                *,
                pos=pos,
                usable_t=usable_t,
                orig_mask=orig_mask,
                base_y=base_y,
                base_x=base_x,
                shifted_by_t=shifted_by_t,
            ) -> None:
                if not usable_t[t]:
                    return  # stays None; the aggregation loop re-checks usable_t itself
                dy = int(round(float(pos[t, 0]) - base_y))
                dx = int(round(float(pos[t, 1]) - base_x))
                if dy == 0 and dx == 0:
                    shifted_by_t[t] = orig_mask
                else:
                    shifted_by_t[t] = nd_shift(
                        orig_mask.astype(float), (dy, dx), order=0, mode="constant"
                    ).astype(bool)

            # A min_parallel that n_frames can never reach forces map_over_axis0
            # to run _shift_frame inline in this thread -- the size gate above,
            # expressed through its existing threshold rather than by branching
            # around the call, so the serial and parallel paths stay literally
            # the same code and cannot drift apart.
            map_over_axis0(
                _shift_frame,
                n_frames,
                min_parallel=(
                    2
                    if height * width >= _PARALLEL_SHIFT_MIN_FRAME_PX
                    else n_frames + 1
                ),
            )

            for t in range(n_frames):
                if not usable_t[t]:
                    continue
                shifted = shifted_by_t[t]
                if shifted is None or not shifted.any():
                    notes[(lbl, t)] = "out_of_bounds"
                    continue
                frame_masks[lbl][t] = shifted

        report_progress(progress=(i + 1) / len(label_ids), stage="rasterizing")

    # Collision pass: a lower-confidence label loses the ENTIRE frame to a
    # higher-confidence one it overlaps, rather than losing only the overlapping
    # pixels -- a partial mask would under-measure area with nothing to flag it.
    for t in range(t_count):
        # Sort key doubles as the priority order: most-negative confidence (i.e.
        # highest actual confidence) first, ties broken by the smaller label id.
        # Built as plain tuples (not a lambda closing over `t`) so nothing here
        # depends on when the key function happens to run.
        placed = sorted(
            (-float(confidence[lbl][t]), lbl, frame_masks[lbl][t])
            for lbl in label_ids
            if t in frame_masks[lbl]
        )
        if len(placed) < 2:
            continue
        claimed = np.zeros((height, width), dtype=bool)
        for _neg_conf, lbl, mask_t in placed:
            if (mask_t & claimed).any():
                del frame_masks[lbl][t]
                notes[(lbl, t)] = "collision"
            else:
                claimed |= mask_t

    for lbl in label_ids:
        for t, mask_t in frame_masks[lbl].items():
            stack[t][mask_t] = lbl

    return stack, notes


def track_rois(
    movie2d: np.ndarray,
    labels2d: np.ndarray,
    *,
    snr_floor: float = calcium_motion.SNR_FLOOR,
    max_step: int = calcium_motion.MAX_STEP,
    conf_floor: float = calcium_motion.CONF_FLOOR,
) -> dict[str, Any]:
    """Track hand-drawn 2D ROIs across a 2D+T movie (approach A: draw once, track).

    Thin orchestration over calcium_motion.correct_sparse, UNMODIFIED -- this
    function adds no detection/correction logic of its own. It locates/
    interpolates/gates per label per frame via correct_sparse, then
    rasterize_tracked_labels shifts each label's ORIGINAL drawn mask to every
    usable position.

    With a single ROI (or any recording with fewer than 3 labels total),
    correct_sparse's neighbour-interpolation recovery is provably inert:
    `others` (calcium_motion.py:137) collects OTHER labels observable this
    frame, which has fewer than `min_neighbours` (default 3) members whenever
    there are under 3 labels total -- so the interpolation branch never fires.
    A lost lock on a single tracked ROI therefore has NO recovery path: it is
    gated (reason="no_neighbours"), never guessed at.

    correct_sparse itself is deliberately NOT threaded here (measured, not
    assumed): its per-frame search is a sequential recurrence -- frame t+1's
    window is centered on frame t's own result (calcium_motion.propagated_
    locate) -- so it is not an embarrassingly-parallel loop as written, and in
    isolation its locate_cell kernel is a numba @njit(cache=True) WITHOUT
    nogil=True (calcium_qc.py), so it holds the GIL for its whole duration by
    numba's own contract; threading it measured WORSE at every worker count
    tried (0.54-0.64x at 2-8 workers) than running it serially. Recorded here
    so the next person does not re-benchmark this before re-discovering the
    same regression. rasterize_tracked_labels' mask-shift phase below is the
    one that actually threads (see its own comment for the numbers) --
    correct_sparse's result is only ever consumed by it, never modified.

    Returns {"stack": (T,Y,X) int32 label stack, "qc_rows": [...]}. Each qc_rows
    dict carries every PINNED QC column except "method" (the tool layer stamps
    that, since only it knows whether this ran as approach A or B).
    """
    labels2d = np.asarray(labels2d)
    if movie2d.ndim != 3:
        raise ValueError(
            f"track_rois is 2D+T only: movie2d must be (T, Y, X), got shape {movie2d.shape}"
        )
    if labels2d.ndim != 2:
        raise ValueError(
            f"track_rois is 2D+T only: labels2d must be (Y, X), got shape {labels2d.shape}"
        )
    if labels2d.shape != tuple(movie2d.shape[1:]):
        raise ValueError(
            f"labels2d {labels2d.shape} does not match movie2d frame shape "
            f"{tuple(movie2d.shape[1:])}"
        )

    result = calcium_motion.correct_sparse(
        movie2d, labels2d, snr_floor=snr_floor, max_step=max_step, conf_floor=conf_floor
    )

    t_count = int(movie2d.shape[0])
    stack, notes = rasterize_tracked_labels(
        labels2d,
        result.positions,
        result.usable,
        confidence=result.confidence,
        shape=(t_count, *labels2d.shape),
    )

    # One np.unique pass per frame (not one per label-frame pair) to count placed
    # pixels straight off `stack` -- this also makes n_pixels self-consistent
    # with the actual LABELS output by construction, rather than a second
    # bookkeeping path that could silently disagree with it.
    pixel_counts: list[dict[int, int]] = []
    for t in range(t_count):
        vals, counts = np.unique(stack[t], return_counts=True)
        pixel_counts.append(dict(zip(vals.tolist(), counts.tolist())))

    qc_rows: list[dict[str, Any]] = []
    for lbl in sorted(result.positions):
        pos = result.positions[lbl]
        usable_arr = result.usable[lbl]
        conf_arr = result.confidence[lbl]
        reason_arr = result.reason[lbl]
        for t in range(t_count):
            note = notes.get((lbl, t))
            if note == "out_of_bounds":
                usable_row, reason_row = False, "out_of_bounds"
            elif note == "collision":
                # Not one of correct_sparse's own reasons (it has no concept of
                # another label) and not "out_of_bounds" (the mask WAS in bounds;
                # a neighbour just already claimed the pixels) -- a distinct,
                # self-explanatory term, contributed by this rasterization step
                # the same way "out_of_bounds" already is, rather than overloading
                # "gated" with a meaning ("lost lock") this frame doesn't have.
                usable_row, reason_row = False, "collision"
            else:
                usable_row = bool(usable_arr[t])
                reason_row = str(reason_arr[t])
            qc_rows.append({
                "label": int(lbl),
                "time_index": int(t),
                "usable": usable_row,
                "confidence": float(conf_arr[t]),
                "reason": reason_row,
                "y": float(pos[t, 0]),
                "x": float(pos[t, 1]),
                "n_pixels": int(pixel_counts[t].get(int(lbl), 0)),
            })

    return {"stack": stack, "qc_rows": qc_rows}
