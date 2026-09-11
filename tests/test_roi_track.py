from __future__ import annotations

import numpy as np
import pytest

from imajin.analysis.calcium_synth import make_recording
from imajin.analysis.roi_track import rasterize_tracked_labels, track_rois

# Same placements/motion as test_calcium_motion.py's POS5 -- central, well-separated,
# stay in-frame under the test drift -- so the accuracy bound this mirrors
# (median error < 1.5 px under 10 px drift) is being asked of the same scenario.
POS5 = [(40, 40), (40, 75), (75, 40), (75, 75), (57, 57)]


def _l_shape_mask(shape=(40, 40), origin=(10, 10)) -> np.ndarray:
    """A deliberately non-round, hand-drawn-looking ROI (an L).

    A disk would preserve its area under any translation "by accident" (it's
    rotationally symmetric); this shape only survives a bit-exact area/shape
    check if the tracker actually shifts the drawn mask instead of resynthesizing
    a disk the way corrected_dff does at calcium_motion.py:168-177.
    """
    y0, x0 = origin
    mask = np.zeros(shape, dtype=bool)
    mask[y0 : y0 + 12, x0 : x0 + 4] = True  # vertical stroke
    mask[y0 + 8 : y0 + 12, x0 : x0 + 10] = True  # foot of the L
    return mask


def _base_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([ys.mean(), xs.mean()])


# --- rasterize_tracked_labels: shape/area preservation --------------------------


def test_rasterize_preserves_area_and_shape_bit_exactly_on_located_frames():
    mask = _l_shape_mask()
    labels2d = np.where(mask, 1, 0).astype(np.int32)
    base = _base_centroid(mask)
    n_pixels_drawn = int(mask.sum())
    yy, xx = np.nonzero(mask)

    # Integral shifts only, as a direct locate_cell lock would produce.
    shifts = [(0, 0), (2, -3), (5, 5), (-4, 1)]
    t_count = len(shifts)
    positions = {1: np.array([base + d for d in shifts])}
    usable = {1: np.ones(t_count, dtype=bool)}
    confidence = {1: np.ones(t_count)}

    stack, notes = rasterize_tracked_labels(
        labels2d, positions, usable, confidence=confidence, shape=(t_count, *labels2d.shape)
    )

    assert stack.dtype == np.int32
    assert notes == {}
    for t, (dy, dx) in enumerate(shifts):
        placed = stack[t] == 1
        assert int(placed.sum()) == n_pixels_drawn, f"area drifted at t={t}"
        expected = np.zeros_like(mask)
        expected[yy + dy, xx + dx] = True
        assert np.array_equal(placed, expected), f"shape not preserved at t={t}"


def test_rasterize_leaves_non_usable_frames_as_background():
    mask = _l_shape_mask()
    labels2d = np.where(mask, 1, 0).astype(np.int32)
    base = _base_centroid(mask)
    t_count = 3
    positions = {1: np.tile(base, (t_count, 1))}
    usable = {1: np.array([True, False, True])}
    confidence = {1: np.array([1.0, 0.0, 1.0])}

    stack, notes = rasterize_tracked_labels(
        labels2d, positions, usable, confidence=confidence, shape=(t_count, *labels2d.shape)
    )

    assert notes == {}
    assert int((stack[0] == 1).sum()) == int(mask.sum())
    assert not np.any(stack[1])  # gated frame: background, never guessed at
    assert int((stack[2] == 1).sum()) == int(mask.sum())


def test_rasterize_reports_out_of_bounds_landing():
    mask = _l_shape_mask(shape=(30, 30), origin=(2, 2))
    labels2d = np.where(mask, 1, 0).astype(np.int32)
    base = _base_centroid(mask)
    # Shift far enough that the whole mask lands off-array.
    positions = {1: np.array([base + (100.0, 100.0)])}
    usable = {1: np.array([True])}
    confidence = {1: np.array([1.0])}

    stack, notes = rasterize_tracked_labels(
        labels2d, positions, usable, confidence=confidence, shape=(1, *labels2d.shape)
    )

    assert notes == {(1, 0): "out_of_bounds"}
    assert not np.any(stack[0])  # background, not a clipped partial mask


def test_rasterize_collision_drops_the_lower_confidence_label_for_the_whole_frame():
    shape = (30, 30)
    mask_a = np.zeros(shape, dtype=bool)
    mask_a[5:10, 5:10] = True
    mask_b = np.zeros(shape, dtype=bool)
    mask_b[5:10, 20:25] = True
    labels2d = np.zeros(shape, dtype=np.int32)
    labels2d[mask_a] = 1
    labels2d[mask_b] = 2

    base_a = _base_centroid(mask_a)
    # Move label 2 (lower confidence) onto label 1's original footprint.
    positions = {1: np.array([base_a]), 2: np.array([base_a])}
    usable = {1: np.array([True]), 2: np.array([True])}
    confidence = {1: np.array([0.9]), 2: np.array([0.6])}

    stack, notes = rasterize_tracked_labels(
        labels2d, positions, usable, confidence=confidence, shape=(1, *shape)
    )

    assert notes == {(2, 0): "collision"}
    assert int((stack[0] == 1).sum()) == int(mask_a.sum())  # winner: untouched
    assert int((stack[0] == 2).sum()) == 0  # loser: whole frame dropped, not truncated


def test_rasterize_rejects_non_2d_labels():
    labels3d = np.zeros((3, 20, 20), dtype=np.int32)
    with pytest.raises(ValueError, match="2D"):
        rasterize_tracked_labels(labels3d, {}, {}, confidence={}, shape=(5, 3, 20, 20))


# --- rasterize_tracked_labels: parallel path (cancellation/progress) ------------
#
# rasterize_tracked_labels threads its per-label mask-shift phase (measured the
# dominant cost of track_rois -- see the function's own comment). These tests
# pin the two behaviours that threading it must not lose: a live cancellation
# request actually stops the loop between labels, and progress is reported at
# that same per-label cadence.


def _three_label_setup(t_count: int = 4):
    """Three well-separated single-block labels, several frames each -- enough
    labels that "cancel/observe after the first one" is a real, checkable
    mid-run stop, not just the trivial only-label case."""
    shape = (30, 30)
    labels2d = np.zeros(shape, dtype=np.int32)
    labels2d[2:6, 2:6] = 1
    labels2d[2:6, 12:16] = 2
    labels2d[2:6, 22:26] = 3
    bases = {lbl: _base_centroid(labels2d == lbl) for lbl in (1, 2, 3)}
    positions = {lbl: np.tile(bases[lbl], (t_count, 1)) for lbl in (1, 2, 3)}
    usable = {lbl: np.ones(t_count, dtype=bool) for lbl in (1, 2, 3)}
    confidence = {lbl: np.ones(t_count) for lbl in (1, 2, 3)}
    return shape, labels2d, positions, usable, confidence


def test_rasterize_stops_when_cancelled_mid_run(monkeypatch):
    """Cancellation must land BETWEEN labels, not only get checked once (if at
    all) after the whole rasterization pass -- and it must do so via the real
    imajin.agent.execution plumbing (a live CancellationToken set as the
    ambient ContextVar, the exact mechanism ToolExecutionService itself uses
    -- see execution.py's own `_CURRENT_TOKEN.set(token)`/`.reset(...)`
    around a job), not a stand-in for it. That matters here specifically
    because the new parallel path could easily check a token that never
    actually observes a real cancellation request if it looked in the wrong
    thread (see map_over_axis0's docstring) -- this proves it does not.
    """
    from imajin.agent import execution
    from imajin.analysis import roi_track
    from imajin.workers.qt_worker import CancellationToken, CancelledError

    shape, labels2d, positions, usable, confidence = _three_label_setup()

    token = CancellationToken()
    real_centroid_of = roi_track._centroid_of
    seen_labels: list[int] = []

    def spy_centroid_of(mask):
        # Called once per label, right after that label's own
        # raise_if_cancelled() check has already passed. Cancelling here
        # simulates a user request arriving WHILE label 1 is being
        # rasterized; the assertion below checks it is honored at the very
        # next checkpoint (the start of label 2), not ignored.
        seen_labels.append(len(seen_labels))
        if len(seen_labels) == 1:
            token.cancel()
        return real_centroid_of(mask)

    monkeypatch.setattr(roi_track, "_centroid_of", spy_centroid_of)

    reset_token = execution._CURRENT_TOKEN.set(token)
    try:
        with pytest.raises(CancelledError):
            roi_track.rasterize_tracked_labels(
                labels2d, positions, usable, confidence=confidence,
                shape=(4, *shape),
            )
    finally:
        execution._CURRENT_TOKEN.reset(reset_token)

    # Only label 1 (sorted first) was ever entered -- labels 2 and 3 never
    # reached _centroid_of, i.e. the loop actually stopped instead of raising
    # only after doing all the work anyway.
    assert seen_labels == [0]


def test_rasterize_uncancelled_run_is_unaffected_by_the_cancellation_plumbing():
    """The mirror image of the test above: with no token ever set (the normal
    case for every existing caller/test in this file), raise_if_cancelled must
    stay a complete no-op and all three labels must be rasterized."""
    shape, labels2d, positions, usable, confidence = _three_label_setup()

    stack, notes = rasterize_tracked_labels(
        labels2d, positions, usable, confidence=confidence, shape=(4, *shape)
    )

    assert notes == {}
    for lbl in (1, 2, 3):
        assert int((stack[0] == lbl).sum()) == 16  # each block is a 4x4 = 16px mask


def test_rasterize_reports_progress_once_per_label(monkeypatch):
    """report_progress lands between labels (a real, user-legible unit --
    "rasterized ROI i of N") rather than zero times (the prior status quo, no
    interior checkpoint existed at all) or once per frame inside the parallel
    pool (which would silently never fire -- see map_over_axis0's docstring)."""
    from imajin.analysis import roi_track

    shape, labels2d, positions, usable, confidence = _three_label_setup(t_count=1)

    calls: list[tuple[float, str]] = []
    monkeypatch.setattr(
        roi_track,
        "report_progress",
        lambda **kw: calls.append((kw["progress"], kw["stage"])),
    )

    roi_track.rasterize_tracked_labels(
        labels2d, positions, usable, confidence=confidence, shape=(1, *shape)
    )

    assert calls == [(1 / 3, "rasterizing"), (2 / 3, "rasterizing"), (1.0, "rasterizing")]


# --- track_rois: end-to-end orchestration ----------------------------------------


def test_track_rois_recovers_positions_under_drift():
    rec = make_recording(
        n_frames=50, shape=(110, 110), n_cells=5, positions=POS5, seed=12,
        motion={"lateral_px": 10.0},
    )
    out = track_rois(rec.movie, rec.labels)

    assert set(out) == {"stack", "qc_rows"}
    assert out["stack"].shape == (50, 110, 110)
    assert out["stack"].dtype == np.int32

    by_label: dict[int, list[dict]] = {}
    for row in out["qc_rows"]:
        for col in ("label", "time_index", "usable", "confidence", "reason", "y", "x", "n_pixels"):
            assert col in row
        assert "method" not in row  # the tool layer stamps this, not track_rois
        by_label.setdefault(row["label"], []).append(row)
    assert set(by_label) == set(rec.true_positions)

    for lbl, true_pos in rec.true_positions.items():
        t_rows = sorted(by_label[lbl], key=lambda r: r["time_index"])
        usable = np.array([r["usable"] for r in t_rows])
        assert usable.mean() > 0.8
        got = np.array([[r["y"], r["x"]] for r in t_rows])
        err = np.hypot(*(got - true_pos).T)
        assert np.median(err[usable]) < 1.5


def test_track_rois_maps_out_of_bounds_and_collision_into_qc_rows(monkeypatch):
    """track_rois's own reason-mapping, isolated from correct_sparse's real
    numerics by monkeypatching calcium_motion.correct_sparse at its point of use
    in roi_track.py (not a private helper another module patches through --
    see memory/agent-guidance/tool-module-split-monkeypatch.md's trap, which does
    not apply here)."""
    from imajin.analysis import calcium_motion

    shape = (20, 20)
    mask_a = np.zeros(shape, dtype=bool)
    mask_a[2:6, 2:6] = True
    mask_b = np.zeros(shape, dtype=bool)
    mask_b[2:6, 12:16] = True
    labels2d = np.zeros(shape, dtype=np.int32)
    labels2d[mask_a] = 1
    labels2d[mask_b] = 2
    movie2d = np.zeros((2, *shape), dtype=np.float32)
    base_a, base_b = _base_centroid(mask_a), _base_centroid(mask_b)

    # t=0: label 2 (lower confidence) collides onto label 1's footprint.
    # t=1: label 1 drifts entirely off-array; label 2 is back at its own base.
    fake_result = calcium_motion.CorrectionResult(
        positions={
            1: np.array([base_a, base_a + 1000.0]),
            2: np.array([base_a, base_b]),
        },
        confidence={1: np.array([0.9, 0.9]), 2: np.array([0.6, 0.6])},
        usable={1: np.array([True, True]), 2: np.array([True, True])},
        reason={
            1: np.array(["located", "located"], dtype=object),
            2: np.array(["located", "located"], dtype=object),
        },
    )
    monkeypatch.setattr(calcium_motion, "correct_sparse", lambda *a, **k: fake_result)

    out = track_rois(movie2d, labels2d)
    rows = {(r["label"], r["time_index"]): r for r in out["qc_rows"]}

    assert rows[(1, 0)]["usable"] is True
    assert rows[(1, 0)]["reason"] == "located"
    assert rows[(1, 0)]["n_pixels"] == int(mask_a.sum())

    assert rows[(2, 0)]["usable"] is False
    assert rows[(2, 0)]["reason"] == "collision"
    assert rows[(2, 0)]["n_pixels"] == 0

    assert rows[(1, 1)]["usable"] is False
    assert rows[(1, 1)]["reason"] == "out_of_bounds"
    assert rows[(1, 1)]["n_pixels"] == 0

    assert rows[(2, 1)]["usable"] is True
    assert rows[(2, 1)]["reason"] == "located"
    assert rows[(2, 1)]["n_pixels"] == int(mask_b.sum())


def test_track_rois_rejects_3d_frames():
    movie4d = np.zeros((5, 3, 20, 20), dtype=np.float32)  # (T, Z, Y, X): 3D frames
    labels2d = np.zeros((20, 20), dtype=np.int32)
    with pytest.raises(ValueError, match="track_rois is 2D") as exc_info:
        track_rois(movie4d, labels2d)
    assert str(movie4d.shape) in str(exc_info.value)


def test_track_rois_rejects_3d_labels():
    movie2d = np.zeros((5, 20, 20), dtype=np.float32)
    labels3d = np.zeros((3, 20, 20), dtype=np.int32)
    with pytest.raises(ValueError, match="track_rois is 2D") as exc_info:
        track_rois(movie2d, labels3d)
    assert str(labels3d.shape) in str(exc_info.value)
