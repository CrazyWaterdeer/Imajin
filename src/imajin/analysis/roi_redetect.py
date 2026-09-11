"""Approach B: per-frame re-detection inside a wide, fixed boundary, plus a
stable cross-frame identity for the redetected objects.

Companion to the frame-1-and-track approach in ``calcium_motion.correct_sparse``:
that one holds a single template and searches within a bounded step per frame, so
it degrades gracefully (correctly refuses) when a cell's apparent position jumps
by more than a few pixels between frames -- exactly what a hard stage bump or a
mis-triggered drift correction produces. This module instead re-thresholds the
*whole* boundary from scratch every frame (no step limit at all) and only uses
distance for the separate, much easier problem of which persistent label a fresh
detection belongs to. The two are complementary: ``correct_sparse`` tracks smooth
drift with a real confidence signal; this recovers a hard discontinuity that
``correct_sparse`` would (correctly) gate as unusable -- at the cost of needing a
boundary wide enough to contain wherever the object might jump to, and of losing
the object for any frame where nothing in the boundary crosses threshold at all.

WHY THIS DOES NOT SHARE ITS LINKER WITH THE Z-STACK PLANE STITCHER
--------------------------------------------------------------------
``segmentation_auto3d.stitch_plane_labels`` solves a structurally similar-
looking problem one axis over: score candidate pairs, prefer the best match,
resolve one-to-one, flag ambiguity. It is deliberately NOT reused here (beyond
the one genuinely shared primitive, :func:`~imajin.analysis.segmentation_auto3d.area_ratio`,
imported below), for two concrete reasons a future "helpful" unification would
reintroduce as bugs:

1. MERGE vs SHARE-AN-ID. ``stitch_plane_labels`` union-finds linked nodes into
   ONE permanent node -- correct for Z, where a cell's planes really are one
   3D object, so collapsing them loses nothing and cannot be undone in a
   direction that would matter. :func:`link_nearest` instead keeps every
   frame's detection as its own independent entry and only ever writes it
   under the SEED's stable label id (see ``redetect_roi_masks.process``,
   below) -- nothing is ever unioned. This has to stay one-directional: two
   real objects that transiently cross paths in time must be able to
   separate again on a later frame, which a union-find merge structurally
   cannot do (there is no un-union). Porting the merge step over would fuse a
   moving object's whole trajectory into one node the first time anything
   else came within its gates, and any two objects that ever crossed paths
   would be stuck sharing an identity for the rest of the movie.
2. OVERLAP-FIRST vs DISTANCE-ONLY gating. ``stitch_plane_labels``' primary
   gate is pixel overlap (``min_overlap_fraction``, tried before centroid
   distance and preferred whenever both fire) -- correct for Z, where
   adjacent planes of one cell are expected to overlap heavily. A drifting
   object in TIME can have literally ZERO shared pixels between consecutive
   frames -- that discontinuity is exactly the bug this module exists to
   recover from (see ``test_redetect_recovers_a_discontinuous_jump`` and
   ``test_link_nearest_links_across_zero_pixel_overlap_purely_on_centroid_distance``
   in test_roi_redetect.py). :func:`link_nearest` therefore never looks at
   pixel overlap at all -- it is handed only a ``centroid`` and an
   ``n_pixels`` count per candidate, never a mask to intersect -- so
   overlap-first gating is not just avoided here, it is structurally
   impossible to reintroduce without changing what a "candidate" even is.

What IS shared: :func:`~imajin.analysis.segmentation_auto3d.area_ratio`, a
one-line size-consistency check (``max/min`` of two pixel counts) that was
already independently duplicated inside ``stitch_plane_labels``'s own two
candidate generators before this module ever existed. That extraction is
behaviour-identical for the Z path (same arithmetic, still gated exactly
where it always was) and gives :func:`link_nearest` the one property
"WHAT THE OLD CODE DOES BETTER" review notes it lacked: a candidate that
lands well inside the distance gate but is a wildly different size from the
seed's last confirmed area is probably a different object wandering close by,
not the same one having drifted, and is now excluded from both winning the
match and inflating ``ambiguous`` -- see the two ``*_area_gate_*`` tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imajin.analysis.segmentation_auto3d import area_ratio
from imajin.analysis.target_pipeline import auto_correct_target


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def link_nearest(
    prev_centroids: dict[int, tuple[float, float]],
    candidates: list[dict],
    *,
    max_distance: float,
    prev_areas: dict[int, float] | None = None,
    max_area_ratio: float | None = None,
) -> dict[int, dict]:
    """Greedy nearest-centroid assignment with a hard distance gate.

    Returns per seed label: {"candidate": <candidate or None>, "ambiguous": bool}.
    ambiguous=True when a SECOND candidate also falls inside the gate — the
    nearest is still taken deterministically, but the frame is flagged so the
    scientist can re-check exactly those frames instead of trusting a silent pick.
    A candidate matching no seed is discarded and NEVER given a new label id.

    Each ``candidates`` entry must carry a ``"centroid"`` tuple the same length as
    the ``prev_centroids`` values (2-D ``(y, x)``, or ``(z, y, x)`` for a 3-D
    frame). Any other keys on a candidate just ride along unread and come back
    verbatim inside the winning "candidate" dict for the caller to use.

    ``prev_areas``/``max_area_ratio`` add an OPTIONAL size-consistency gate on
    top of distance, borrowed from the one piece of scoring genuinely shared
    with the Z-stack plane linker (see the module docstring):
    :func:`~imajin.analysis.segmentation_auto3d.area_ratio`. A candidate whose
    ``"n_pixels"`` is more than ``max_area_ratio`` times bigger or smaller than
    ``prev_areas[label]`` is dropped from that seed's pool entirely -- exactly
    like the distance gate, not merely down-ranked -- so it can neither win the
    match nor count toward ``ambiguous``: it was never a plausible candidate,
    just a coincidentally nearby one. Both are opt-in and default to
    disabled (``None``), and a seed absent from ``prev_areas`` is never
    area-gated either, so every distance-only caller (this module's own tests
    among them) keeps its exact original behaviour.

    Assignment is one global greedy pass over every (seed, candidate) pair inside
    the gate, closest first: once a candidate is claimed it is removed from play,
    so a candidate a closer seed wants can never also be handed to a farther seed
    (two real objects drifting toward each other should read as "no_candidate" for
    whichever loses the tie-break, never a mislabelled steal of the other's
    identity). ``ambiguous`` reflects each seed's *own* nearby-candidate count
    computed before that global resolution runs, so it still flags a genuinely
    confusable neighbourhood even for a seed that goes on to win the contested
    candidate.
    """
    pools: dict[int, list[tuple[float, int]]] = {}
    for label, prev in prev_centroids.items():
        prev_area = prev_areas.get(label) if prev_areas else None
        within: list[tuple[float, int]] = []
        for idx, cand in enumerate(candidates):
            d = _distance(prev, cand["centroid"])
            if d > max_distance:
                continue
            if (
                prev_area is not None
                and max_area_ratio is not None
                and area_ratio(prev_area, cand["n_pixels"]) > max_area_ratio
            ):
                continue  # right place, wrong size -- almost certainly a different object
            within.append((d, idx))
        within.sort()
        pools[label] = within

    ordered = sorted((d, label, idx) for label, pool in pools.items() for d, idx in pool)
    claimed: set[int] = set()
    assigned: dict[int, int] = {}
    for _d, label, idx in ordered:
        if label in assigned or idx in claimed:
            continue  # this seed already has a closer match, or another seed got here first
        assigned[label] = idx
        claimed.add(idx)

    return {
        label: {
            "candidate": candidates[assigned[label]] if label in assigned else None,
            "ambiguous": len(pools[label]) >= 2,
        }
        for label in prev_centroids
    }


def _seed_geometry(
    seed_labels: np.ndarray, seed_ids: list[int]
) -> tuple[dict[int, tuple[float, ...]], dict[int, float]]:
    """Each seed's centroid AND pixel count straight from the human-drawn mask --
    the one position (and size baseline) in the whole walk that does not come
    from a redetection call. One regionprops pass for both, since a seed mask
    is drawn once and this never runs per-frame."""
    from skimage.measure import regionprops

    props = {int(r.label): r for r in regionprops(seed_labels)}
    missing = [lbl for lbl in seed_ids if lbl not in props]
    if missing:
        raise ValueError(f"seed_labels has no pixels for label(s) {missing}")
    centroids = {lbl: tuple(float(c) for c in props[lbl].centroid) for lbl in seed_ids}
    areas = {lbl: float(props[lbl].area) for lbl in seed_ids}
    return centroids, areas


def _candidates_from_masks(masks: np.ndarray) -> list[dict[str, Any]]:
    """One candidate per connected component of a single frame's fresh labelling.
    ``label`` is that frame's own id from ``intersect_labels_with_mask(...,
    renumber=True)`` (1..N, independently renumbered every frame) -- never a
    seed's stable id; :func:`link_nearest` is what bridges the two."""
    from skimage.measure import regionprops

    return [
        {
            "label": int(r.label),
            "centroid": tuple(float(c) for c in r.centroid),
            "n_pixels": int(r.area),
        }
        for r in regionprops(masks)
    ]


def _qc_row(
    label: int,
    time_index: int,
    *,
    usable: bool,
    confidence: float,
    reason: str,
    y: float,
    x: float,
    n_pixels: int,
) -> dict[str, Any]:
    """One row of the pinned QC schema, minus ``method`` -- the shared assembly
    helper both producers hand their raw rows to stamps that column on."""
    return {
        "label": int(label),
        "time_index": int(time_index),
        "usable": bool(usable),
        "confidence": float(confidence),
        "reason": reason,
        "y": float(y),
        "x": float(x),
        "n_pixels": int(n_pixels),
    }


def redetect_roi_masks(
    frames: np.ndarray,
    seed_labels: np.ndarray,
    boundary_bool: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    params: dict[str, Any],
    auto_correct: bool = True,
    max_iters: int = 3,
    link_max_distance: float = 25.0,
    link_max_area_ratio: float | None = 3.0,
    seed_frame: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """(stack, qc_rows). frames is (T, ...) with T already at axis 0.

    ``seed_labels`` and ``boundary_bool`` are both single-frame shaped
    (``frames.shape[1:]`` -- a per-timepoint frame's own ndim, 2-D YX or a genuine
    3-D ZYX volume; see the module docstring): one seed mask and one (possibly
    Z-broadcast) boundary shared by every timepoint, exactly what
    :func:`~imajin.analysis.target_pipeline.auto_correct_target` takes per call.
    ``boundary_bool`` may be a read-only broadcast view (the 2D-ROI-on-a-MIP case,
    :func:`~imajin.analysis.segmentation.resolve_boundary_mask`) -- it is only ever
    read here, never written.

    ``seed_frame`` (not part of the pinned call shape, added here because nothing
    else in the given inputs says which timepoint ``seed_labels`` came from) is
    the timepoint the caller actually drew the seed ROI on; it defaults to 0, the
    common "draw on the first frame" case. Identity radiates OUTWARD from it in
    both directions -- the seed frame itself first, then forward to the last
    frame, then backward to frame 0 -- with both directions re-seeded from the
    seed frame's own *redetected* position (not merely the hand-drawn one), so a
    seed drawn mid-movie still recovers the object before it as well as after.

    ``link_max_area_ratio`` (default 3.0, matching
    ``segmentation_auto3d.stitch_plane_labels``'s own default for the same gate
    on the Z axis) rejects a redetected candidate whose size is more than that
    many times bigger or smaller than the seed's own last confirmed area, even
    if it is the nearest thing in the boundary -- a real object drifting keeps
    roughly its own size; something that close but that different in size is
    more likely threshold noise or unrelated debris than the tracked object
    itself. Pass ``None`` to disable it and fall back to distance-only linking
    (:func:`link_nearest`'s original behaviour). This is a SIZE gate, not a
    position-overlap gate -- see the module docstring for why overlap-first
    gating (the Z-stack linker's own primary gate) is never used here.

    ``auto_correct=False`` forces ``max_iters=0`` for every per-frame call, i.e. a
    single segmentation pass with ``params`` taken as given rather than hill-
    climbed (``auto_correct_target(..., max_iters=0)`` is exactly the single-shot
    path -- see ``test_segment_target_array_and_loop_agree_at_fixed_params`` in
    test_target_pipeline.py).

    Every frame contributes exactly one QC row per seed label -- including a
    frame with zero candidates for that seed, written as background (0) in the
    returned stack and 'no_candidate' in the QC row, never carried over from a
    neighbouring frame. The stack and the ``usable`` column agree exactly: a
    frame is painted if and only if its row says usable, so a downstream
    regionprops pass produces a gap (no row at all) for every frame this table
    calls unusable rather than a number nothing vouched for.
    """
    frames = np.asarray(frames)
    seed_labels = np.asarray(seed_labels, dtype=np.int32)
    n_frames = int(frames.shape[0])
    frame_shape = frames.shape[1:]
    if seed_labels.shape != frame_shape:
        raise ValueError(
            f"seed_labels shape {seed_labels.shape} must match one movie frame "
            f"{frame_shape} -- draw the seed ROI on a single timepoint."
        )
    if not 0 <= seed_frame < n_frames:
        raise ValueError(f"seed_frame {seed_frame} out of range for {n_frames} frames")

    seed_ids = sorted(int(v) for v in np.unique(seed_labels) if v != 0)
    if not seed_ids:
        raise ValueError("seed_labels has no labelled pixels (all background)")

    stack = np.zeros(frames.shape, dtype=np.int32)
    rows_by_time: list[list[dict[str, Any]]] = [[] for _ in range(n_frames)]
    iters = max_iters if auto_correct else 0

    def process(
        t: int, prev: dict[int, tuple[float, ...]], prev_areas: dict[int, float]
    ) -> None:
        seg, _params_used, _history = auto_correct_target(
            frames[t],
            spacing=spacing,
            params=params,
            boundary_mask=boundary_bool,
            max_iters=iters,
        )
        candidates = _candidates_from_masks(seg.masks)
        links = link_nearest(
            prev,
            candidates,
            max_distance=link_max_distance,
            prev_areas=prev_areas,
            max_area_ratio=link_max_area_ratio,
        )
        # seg.roi_score / .roi_confidence are already this call's own verdict on
        # THIS frame's segmentation (target_pipeline.py TargetSegmentation) --
        # reused verbatim as the QC confidence rather than scoring anything
        # ourselves, just rescaled into the pinned [0, 1] range (the raw score is
        # unbounded past a 0-floor/100-ceiling design centre; see
        # confidence_from_score's 55/75 tiers in segmentation_auto3d.py).
        confidence = float(min(1.0, max(0.0, seg.roi_score / 100.0)))
        out = np.zeros(seg.masks.shape, dtype=np.int32)
        rows: list[dict[str, Any]] = []
        for label in seed_ids:
            info = links[label]
            candidate = info["candidate"]
            if candidate is None:
                # Never fall back to a neighbouring frame's mask here -- a real
                # absence must read as a real absence, not a fabricated position.
                rows.append(
                    _qc_row(
                        label,
                        t,
                        usable=False,
                        confidence=0.0,
                        reason="no_candidate",
                        y=float("nan"),
                        x=float("nan"),
                        n_pixels=0,
                    )
                )
                continue
            prev[label] = candidate["centroid"]  # only a real detection moves the anchor
            prev_areas[label] = candidate["n_pixels"]  # same rule, same reason, for the size gate
            if info["ambiguous"]:
                reason, usable = "ambiguous_match", False
            elif seg.roi_confidence == "low":
                reason, usable = "low_confidence", False
            else:
                reason, usable = "detected", True
            if usable:
                # ONLY a confidently-placed frame is painted. Writing an
                # ambiguous_match/low_confidence footprint anyway would hand
                # measure_intensity_over_time a full-looking labels stack, and its
                # regionprops pass would emit a measurement row for a frame this
                # very table calls unusable -- a fabricated number where the
                # pinned contract (and approach A's rasterize_tracked_labels,
                # which only ever rasterizes usable frames) promise a gap. The
                # anchor above is still advanced: a real detection is the best
                # place to look next even when it is not trustworthy enough to
                # measure, exactly as correct_sparse keeps its located trajectory
                # through a gated frame.
                out[seg.masks == candidate["label"]] = label
            cy, cx = candidate["centroid"][-2], candidate["centroid"][-1]
            rows.append(
                _qc_row(
                    label,
                    t,
                    usable=usable,
                    confidence=confidence,
                    reason=reason,
                    y=cy,
                    x=cx,
                    # Counted off what was actually written, never off the
                    # candidate -- so n_pixels can never claim pixels the LABELS
                    # layer does not contain (track_rois derives it from its own
                    # output stack for the same reason).
                    n_pixels=candidate["n_pixels"] if usable else 0,
                )
            )
        stack[t] = out
        rows_by_time[t] = rows

    anchor, anchor_areas = _seed_geometry(seed_labels, seed_ids)
    process(seed_frame, anchor, anchor_areas)  # shared jump-off point for both walks

    forward, forward_areas = dict(anchor), dict(anchor_areas)
    for t in range(seed_frame + 1, n_frames):
        process(t, forward, forward_areas)

    backward, backward_areas = dict(anchor), dict(anchor_areas)
    for t in range(seed_frame - 1, -1, -1):
        process(t, backward, backward_areas)

    qc_rows = [row for rows in rows_by_time for row in rows]
    return stack, qc_rows
