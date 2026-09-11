from __future__ import annotations

import numpy as np
import pytest

from imajin.analysis.roi_redetect import link_nearest, redetect_roi_masks

# Gaussian background + an ~8-sigma bump, flat (radius=0) background subtraction,
# no smoothing. background_radius=0 matters, not just convenience: on a canvas
# this small, estimate_local_background's default "opening" (erosion+dilation)
# turns out to have a strong low bias on pure noise -- ndi.grey_opening's erosion
# step is a local-minimum extreme-value statistic, so subtracting it leaves a
# corrected image that is *not* zero-mean, and a signal-free frame then crosses
# threshold as spurious 100+px "objects" almost every time (measured empirically
# while building this suite). Flat background subtraction (a single percentile
# scalar) does not have that bias and reliably reports zero objects for a
# genuinely empty frame while still detecting the real blob just as cleanly.
_PARAMS = {"min_snr": 2.0, "min_size": 30, "smoothing_sigma": 0.0, "background_radius": 0}


def _blob_movie(positions: list[tuple[int, int]], *, size: int = 64, half: int = 5, seed: int = 0):
    """One frame per entry in ``positions``; a bump of the given half-width is
    added at that (cy, cx) (or omitted entirely for a ``None`` entry -- a genuine
    absence, not a dim blob)."""
    rng = np.random.default_rng(seed)
    movie = rng.normal(20.0, 1.0, (len(positions), size, size)).astype(np.float32)
    for t, pos in enumerate(positions):
        if pos is None:
            continue
        cy, cx = pos
        movie[t, cy - half : cy + half, cx - half : cx + half] += 8.0
    return movie


def _seed_at(pos: tuple[int, int], *, size: int = 64, half: int = 5, label: int = 1) -> np.ndarray:
    seed = np.zeros((size, size), dtype=np.int32)
    cy, cx = pos
    seed[cy - half : cy + half, cx - half : cx + half] = label
    return seed


def _rows_for(qc_rows: list[dict], label: int, time_index: int) -> dict:
    matches = [r for r in qc_rows if r["label"] == label and r["time_index"] == time_index]
    assert len(matches) == 1, f"expected exactly one row for label={label} t={time_index}"
    return matches[0]


# --- redetect_roi_masks: the two dedicated end-to-end scenarios ---


def test_redetect_recovers_a_discontinuous_jump() -> None:
    """A hard, instantaneous teleport (no intermediate positions) between two
    far-apart points -- exactly the case correct_sparse's confidence gate
    (calcium_motion.MAX_STEP=6) would correctly REFUSE rather than mistrack,
    since its template search radius is deliberately small. Approach B has no
    step limit at all: it re-thresholds the whole boundary from scratch every
    frame, so a jump is invisible to *detection* and only has to survive the
    much easier nearest-candidate *linking* step (trivial here since each frame
    has exactly one candidate). The two tools are complementary, not redundant.
    """
    p1, p2, jump_at, n = (15, 15), (45, 45), 3, 6
    positions = [p1] * jump_at + [p2] * (n - jump_at)
    movie = _blob_movie(positions)
    seed_labels = _seed_at(p1)
    boundary = np.ones((64, 64), dtype=bool)

    # |p1 - p2| ~= 42.4px, comfortably beyond the 25px default gate -- pass a
    # gate wide enough to span the jump. Nothing else in the frame competes for
    # it, so a generous gate does not risk grabbing the wrong object.
    stack, qc_rows = redetect_roi_masks(
        movie, seed_labels, boundary, spacing=None, params=_PARAMS, link_max_distance=50.0
    )

    assert stack.shape == movie.shape
    for t in range(n):
        want, other = (p1, p2) if t < jump_at else (p2, p1)
        assert stack[t, want[0], want[1]] == 1
        assert stack[t, other[0], other[1]] == 0
        row = _rows_for(qc_rows, 1, t)
        assert row["reason"] == "detected"
        assert row["usable"] is True
        assert row["n_pixels"] > 0


def test_redetect_ignores_an_unseeded_object_in_the_boundary() -> None:
    """Two real blobs, one seed. The boundary is wide enough to threshold both,
    but only the seeded one may ever be written to the output -- the unseeded
    blob's pixels must never appear under any label id (link_nearest discards a
    candidate matching no seed rather than minting a new label for it)."""
    seeded, other, n = (15, 15), (45, 20), 4
    movie = _blob_movie([seeded] * n, seed=1)
    # add the second, unseeded blob to every frame directly (helper only places one)
    for t in range(n):
        movie[t, other[0] - 5 : other[0] + 5, other[1] - 5 : other[1] + 5] += 8.0
    seed_labels = _seed_at(seeded)
    boundary = np.ones((64, 64), dtype=bool)

    stack, qc_rows = redetect_roi_masks(movie, seed_labels, boundary, spacing=None, params=_PARAMS)

    assert set(np.unique(stack).tolist()) == {0, 1}  # never a label 2, never any other id
    for t in range(n):
        assert stack[t, other[0], other[1]] == 0
        assert stack[t, seeded[0], seeded[1]] == 1
        assert _rows_for(qc_rows, 1, t)["reason"] == "detected"


def test_redetect_walks_outward_from_a_mid_movie_seed_frame() -> None:
    """seed_frame need not be 0: a seed drawn in the middle of a movie must still
    recover the object at every earlier timepoint as well as every later one."""
    pos, n = (20, 20), 5
    movie = _blob_movie([pos] * n, seed=2)
    seed_labels = _seed_at(pos)
    boundary = np.ones((64, 64), dtype=bool)

    stack, qc_rows = redetect_roi_masks(
        movie, seed_labels, boundary, spacing=None, params=_PARAMS, seed_frame=2
    )

    assert sorted(r["time_index"] for r in qc_rows) == list(range(n))
    for t in range(n):
        assert stack[t, pos[0], pos[1]] == 1
        assert _rows_for(qc_rows, 1, t)["reason"] == "detected"


def test_redetect_gap_frame_is_background_and_next_frame_still_finds_it() -> None:
    """A frame with no object at all -> background + 'no_candidate', with no
    tracking state to lose: the following frame re-finds the object because
    prev_centroids was never overwritten by the gap.

    auto_correct=False here: auto_correct_target's hill-climb scores n_objects==0
    as a hard -1000 (segmentation_auto3d.score_roi_quality) and never regresses
    once it has *anything* better, so given enough iterations it will happily
    halve min_snr on pure background noise until some noise cluster clears
    min_size -- "confidently" reporting a low-score object rather than admitting
    zero. That is a real property of the shared, not-ours auto-correct loop (built
    for "the cell is dim from photobleaching, keep looking"), not a fabrication in
    this module: a single fixed-params pass (no escalation) is the deterministic
    way to observe a genuinely empty frame here, and is itself a legitimate choice
    of the auto_correct switch this function exposes.
    """
    pos, missing_at, n = (20, 20), 2, 4
    positions = [pos if t != missing_at else None for t in range(n)]
    movie = _blob_movie(positions, seed=3)
    seed_labels = _seed_at(pos)
    boundary = np.ones((64, 64), dtype=bool)

    stack, qc_rows = redetect_roi_masks(
        movie, seed_labels, boundary, spacing=None, params=_PARAMS, auto_correct=False
    )

    assert np.all(stack[missing_at] == 0)
    gap_row = _rows_for(qc_rows, 1, missing_at)
    assert gap_row["reason"] == "no_candidate"
    assert gap_row["usable"] is False
    assert gap_row["n_pixels"] == 0
    assert np.isnan(gap_row["y"]) and np.isnan(gap_row["x"])

    next_row = _rows_for(qc_rows, 1, missing_at + 1)
    assert next_row["reason"] == "detected"
    assert stack[missing_at + 1, pos[0], pos[1]] == 1

    assert len(qc_rows) == n  # exactly one row per (label, time_index), gap included


def test_redetect_roi_masks_rejects_seed_labels_shape_mismatch() -> None:
    movie = np.zeros((3, 32, 32), dtype=np.float32)
    bad_seed = np.zeros((16, 16), dtype=np.int32)
    boundary = np.ones((32, 32), dtype=bool)
    with pytest.raises(ValueError, match="seed_labels shape"):
        redetect_roi_masks(movie, bad_seed, boundary, spacing=None, params={"min_size": 5})


def test_redetect_roi_masks_accepts_link_max_area_ratio_none_to_disable_the_gate() -> None:
    """The area-ratio gate this pass adds is opt-out at the redetect_roi_masks
    level too, not only inside link_nearest -- passing None must reproduce the
    original distance-only behaviour on a plain, single-object movie."""
    pos, n = (20, 20), 4
    movie = _blob_movie([pos] * n, seed=5)
    seed_labels = _seed_at(pos)
    boundary = np.ones((64, 64), dtype=bool)

    stack, qc_rows = redetect_roi_masks(
        movie, seed_labels, boundary, spacing=None, params=_PARAMS, link_max_area_ratio=None
    )

    for t in range(n):
        assert stack[t, pos[0], pos[1]] == 1
        assert _rows_for(qc_rows, 1, t)["reason"] == "detected"


# --- link_nearest: pure geometry, no segmentation involved ---


def test_link_nearest_rejects_beyond_gate() -> None:
    prev = {1: (0.0, 0.0)}
    candidates = [{"label": 1, "centroid": (0.0, 30.0), "n_pixels": 10}]
    result = link_nearest(prev, candidates, max_distance=5.0)
    assert result[1]["candidate"] is None
    assert result[1]["ambiguous"] is False


def test_link_nearest_flags_ambiguous_but_still_picks_the_nearer_one() -> None:
    prev = {1: (0.0, 0.0)}
    candidates = [
        {"label": 1, "centroid": (0.0, 3.0), "n_pixels": 10},
        {"label": 2, "centroid": (0.0, 4.0), "n_pixels": 12},
    ]
    result = link_nearest(prev, candidates, max_distance=5.0)
    assert result[1]["ambiguous"] is True
    assert result[1]["candidate"]["label"] == 1  # the nearer of the two, taken deterministically


def test_link_nearest_deterministic_nearest_pick_among_several() -> None:
    prev = {1: (10.0, 10.0)}
    candidates = [
        {"label": 5, "centroid": (10.0, 12.0), "n_pixels": 5},  # distance 2: nearest, only one in gate
        {"label": 9, "centroid": (10.0, 40.0), "n_pixels": 5},  # distance 30: out of gate
    ]
    result = link_nearest(prev, candidates, max_distance=10.0)
    assert result[1]["candidate"]["label"] == 5
    assert result[1]["ambiguous"] is False


def test_link_nearest_does_not_double_assign_one_candidate_to_two_seeds() -> None:
    """Two seeds both plausible for the SAME lone candidate: the closer seed wins
    it and the farther seed gets None, never a duplicate claim on the same pixels
    under two different label ids. ambiguous stays False for both -- each seed's
    own neighbourhood has exactly one candidate; the contention is between seeds,
    a different thing link_nearest resolves by proximity, not by flagging."""
    prev = {1: (0.0, 0.0), 2: (0.0, 4.0)}
    candidates = [{"label": 7, "centroid": (0.0, 1.0), "n_pixels": 5}]
    result = link_nearest(prev, candidates, max_distance=5.0)
    assert result[1]["candidate"]["label"] == 7  # seed 1 is closer (distance 1 vs 3)
    assert result[2]["candidate"] is None
    assert result[1]["ambiguous"] is False
    assert result[2]["ambiguous"] is False


def test_link_nearest_links_across_zero_pixel_overlap_purely_on_centroid_distance() -> None:
    """The crux of why this linker cannot borrow the Z-stack plane stitcher's
    overlap-first gating (segmentation_auto3d.stitch_plane_labels /
    _overlap_links -- see the module docstring): a drifting object can have
    literally ZERO shared pixels between consecutive frames. Two disjoint 5x5
    boxes, built so they share no pixel (verified from the actual masks below,
    not just asserted from the placement arithmetic), sit only ~7.1px apart
    centroid-to-centroid -- comfortably inside a normal drift-tracking gate.
    link_nearest must still link them: it is handed only a
    ``centroid``/``n_pixels`` per candidate, never a pixel mask, so overlap is
    not merely unused here -- there is nothing in scope to compute it from.
    """
    box_a = np.zeros((30, 30), dtype=bool)
    box_a[5:10, 5:10] = True
    box_b = np.zeros((30, 30), dtype=bool)
    box_b[10:15, 10:15] = True  # starts exactly where box_a ends -- 0 shared row or column
    assert not np.any(box_a & box_b)  # pin the "zero overlap" premise itself

    prev_centroid = tuple(float(c) for c in np.argwhere(box_a).mean(axis=0))
    cand_centroid = tuple(float(c) for c in np.argwhere(box_b).mean(axis=0))
    step = float(np.linalg.norm(np.subtract(prev_centroid, cand_centroid)))
    assert 7.0 < step < 7.2  # a small drift step, not a teleport

    prev = {1: prev_centroid}
    candidates = [{"label": 1, "centroid": cand_centroid, "n_pixels": int(box_b.sum())}]
    result = link_nearest(prev, candidates, max_distance=10.0)

    assert result[1]["candidate"] is not None
    assert result[1]["candidate"]["centroid"] == cand_centroid
    assert result[1]["ambiguous"] is False


def test_link_nearest_area_gate_rejects_a_close_but_wildly_different_sized_candidate() -> None:
    """The area-ratio sanity gate the Z-stack linker already had and this one
    lacked (see the module docstring): a candidate well inside the distance
    gate but a very different size from the seed's last confirmed area is
    almost certainly a different object -- e.g. threshold-noise debris --
    wandering close by, not the same thing having drifted. It must lose the
    match even though it is the nearest point, and must not count toward
    `ambiguous` either: nothing plausible was actually contending for the slot.
    """
    prev = {1: (10.0, 10.0)}
    candidates = [{"label": 1, "centroid": (10.0, 12.0), "n_pixels": 400}]  # 40x the seed's area
    result = link_nearest(
        prev, candidates, max_distance=10.0, prev_areas={1: 10.0}, max_area_ratio=3.0
    )
    assert result[1]["candidate"] is None
    assert result[1]["ambiguous"] is False


def test_link_nearest_area_gate_is_opt_in_and_off_by_default() -> None:
    """No prev_areas/max_area_ratio supplied -> identical to the distance-only
    behaviour every test above this one relies on (back-compat: nothing that
    never measured a seed's area is affected by this gate existing)."""
    prev = {1: (10.0, 10.0)}
    candidates = [{"label": 1, "centroid": (10.0, 12.0), "n_pixels": 400}]
    result = link_nearest(prev, candidates, max_distance=10.0)
    assert result[1]["candidate"]["label"] == 1


def test_link_nearest_area_gate_excludes_candidate_from_ambiguous_count() -> None:
    """A second candidate that falls inside the distance gate but is excluded
    by the area gate must not inflate `ambiguous` -- it was never a plausible
    match, so a scientist re-checking 'ambiguous' frames should not be sent to
    look at this one."""
    prev = {1: (0.0, 0.0)}
    candidates = [
        {"label": 1, "centroid": (0.0, 3.0), "n_pixels": 10},  # real match, similar size
        {"label": 2, "centroid": (0.0, 4.0), "n_pixels": 500},  # close, but wildly bigger
    ]
    result = link_nearest(
        prev, candidates, max_distance=5.0, prev_areas={1: 10.0}, max_area_ratio=3.0
    )
    assert result[1]["candidate"]["label"] == 1
    assert result[1]["ambiguous"] is False  # candidate 2 never entered the pool


def test_redetect_unusable_frame_is_background_so_it_cannot_be_measured() -> None:
    """usable=False must mean the LABELS stack is background at that frame.

    'ambiguous_match' / 'low_confidence' used to paint the winning candidate
    anyway, so measure_intensity_over_time's per-frame regionprops pass found
    pixels and emitted a real measurement row for a frame this very QC table
    calls untrustworthy -- the fabricated number the whole ROI-drift feature
    exists to avoid, and the opposite of approach A, whose
    rasterize_tracked_labels only ever rasterizes usable frames. n_pixels is
    asserted too: it must describe what is actually in the stack, never the
    candidate that was rejected.
    """
    n, size, half = 3, 64, 5
    movie = np.random.default_rng(4).normal(20.0, 1.0, (n, size, size)).astype(np.float32)
    cy, cx = 20, 20
    for t in range(n):
        movie[t, cy - half : cy + half, cx - half : cx + half] += 8.0
        if t >= 1:
            # A second object 14 px away: a distinct connected component, but
            # well inside link_nearest's 25 px identity gate -> ambiguous.
            movie[t, cy - half : cy + half, cx + 9 : cx + 19] += 8.0

    stack, qc_rows = redetect_roi_masks(
        movie,
        _seed_at((cy, cx)),
        np.ones((size, size), dtype=bool),
        spacing=None,
        params=_PARAMS,
        auto_correct=False,
        link_max_distance=25.0,
    )

    clean = _rows_for(qc_rows, 1, 0)
    assert clean["reason"] == "detected"
    assert clean["usable"] is True
    assert stack[0, cy, cx] == 1
    assert clean["n_pixels"] == int((stack[0] == 1).sum())

    for t in (1, 2):
        row = _rows_for(qc_rows, 1, t)
        assert row["reason"] == "ambiguous_match"
        assert row["usable"] is False
        assert np.all(stack[t] == 0), f"frame {t} is flagged unusable but was painted"
        assert row["n_pixels"] == 0
