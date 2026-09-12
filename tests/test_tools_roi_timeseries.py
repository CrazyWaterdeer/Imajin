from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imajin import session as state
from imajin.analysis.calcium_synth import make_recording
from imajin.tools import measure, roi_timeseries

# Same placements/motion as test_roi_track.py's own POS5 -- central, well-separated,
# stay in-frame under the test drift -- so this reuses a scenario that module's own
# tests already validate at the analysis layer (median position error < 1.5px,
# usable mean > 0.8). Label 5 (the centre position) is calcium_synth's default
# negative control: flat, no events -- excluded from the "busiest label" pick below.
POS5 = [(40, 40), (40, 75), (75, 40), (75, 75), (57, 57)]


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


def _drifting_recording():
    return make_recording(
        n_frames=50, shape=(110, 110), n_cells=5, positions=POS5, seed=12,
        motion={"lateral_px": 10.0},
    )


def _busiest_label(rec) -> int:
    return max(
        (lbl for lbl in rec.event_frames if lbl != rec.negative_label),
        key=lambda lbl: len(rec.event_frames[lbl]),
    )


def _corr_with_truth(df: pd.DataFrame, label: int, true_dff: np.ndarray) -> float:
    """Pearson r between a measured trace and ground-truth dF/F0, over whatever
    (label, time_index) rows actually exist -- a gated/absent frame contributes
    no row (never a fabricated value), so correlating on the available subset is
    the correct comparison, not a bug in the helper."""
    sub = df[df["label"] == label].sort_values("time_index")
    t_idx = sub["time_index"].to_numpy()
    got = sub["mean_intensity"].to_numpy()
    true = true_dff[t_idx]
    return float(np.corrcoef(got, true)[0, 1])


def _union_bbox_mask(
    true_positions: dict[int, np.ndarray],
    shape: tuple[int, int],
    *,
    radius: float,
    margin: float,
) -> np.ndarray:
    pts = np.concatenate(list(true_positions.values()), axis=0)
    y0 = max(0, int(np.floor(pts[:, 0].min() - radius - margin)))
    x0 = max(0, int(np.floor(pts[:, 1].min() - radius - margin)))
    y1 = min(shape[0], int(np.ceil(pts[:, 0].max() + radius + margin)))
    x1 = min(shape[1], int(np.ceil(pts[:, 1].max() + radius + margin)))
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _tzyx_recording() -> tuple[np.ndarray, np.ndarray]:
    """A tiny, static (no drift) 3D+T blob -- only shape/wiring is under test for
    the z_project cases, never tracking accuracy, so drift is not needed here."""
    rng = np.random.default_rng(0)
    movie = rng.normal(20.0, 1.0, size=(4, 3, 32, 32)).astype(np.float32)
    movie[:, :, 8:16, 8:16] += 40.0
    labels3d = np.zeros((3, 32, 32), dtype=np.int32)
    labels3d[:, 10:14, 10:14] = 1
    return movie, labels3d


# --- decisive end-to-end tests: this is the user's actual bug report ------------


def test_track_roi_over_time_end_to_end_feeds_measure(viewer) -> None:
    rec = _drifting_recording()
    viewer.add_labels(rec.labels, name="roi")
    viewer.add_image(rec.movie, name="movie", metadata={"axes": "TYX"})
    busiest = _busiest_label(rec)

    res = roi_timeseries.track_roi_over_time("roi", "movie")

    assert res["n_timepoints"] == rec.movie.shape[0]
    assert res["coverage"][busiest] > 0.8
    assert busiest not in res["rejected"]

    measured = measure.measure_intensity_over_time(res["labels_layer"], "movie")
    assert measured["n_timepoints"] == rec.movie.shape[0]
    df = state.get_table(measured["table_name"])
    assert _corr_with_truth(df, busiest, rec.true_dff[busiest]) > 0.8


def test_static_labels_would_have_drifted(viewer) -> None:
    """The regression guard: measuring the ORIGINAL single-frame seed directly
    against the whole movie (the old, buggy path) correlates with ground truth
    materially worse than the tracked path -- this is the user's own complaint,
    encoded as a test."""
    rec = _drifting_recording()
    viewer.add_labels(rec.labels, name="roi")
    viewer.add_image(rec.movie, name="movie", metadata={"axes": "TYX"})
    busiest = _busiest_label(rec)

    static_res = measure.measure_intensity_over_time("roi", "movie")
    static_df = state.get_table(static_res["table_name"])
    static_corr = _corr_with_truth(static_df, busiest, rec.true_dff[busiest])

    tracked_res = roi_timeseries.track_roi_over_time("roi", "movie")
    tracked_measured = measure.measure_intensity_over_time(tracked_res["labels_layer"], "movie")
    tracked_df = state.get_table(tracked_measured["table_name"])
    tracked_corr = _corr_with_truth(tracked_df, busiest, rec.true_dff[busiest])

    assert tracked_corr > 0.8
    assert tracked_corr - static_corr > 0.15


def test_resegment_roi_over_time_end_to_end_feeds_measure(viewer) -> None:
    rec = make_recording(
        n_frames=15, shape=(96, 96), n_cells=3,
        positions=[(30, 30), (30, 66), (66, 48)],
        seed=5, motion={"lateral_px": 8.0}, negative_control=False,
    )
    viewer.add_labels(rec.labels, name="roi")
    viewer.add_image(rec.movie, name="movie", metadata={"axes": "TYX"})
    boundary = _union_bbox_mask(rec.true_positions, rec.movie.shape[1:], radius=5.0, margin=15.0)
    viewer.add_labels(boundary.astype(np.int32), name="boundary")

    # Same flat-background / no-smoothing params test_roi_redetect.py's own
    # validated scenarios use -- see that file's _PARAMS comment for why the
    # default "opening" background is unreliable on a canvas this small.
    res = roi_timeseries.resegment_roi_over_time(
        "roi", "boundary", "movie",
        background_radius=0, smoothing_sigma=0.0, min_snr=2.0, min_size=30,
        auto_correct=False,
    )

    assert res["n_timepoints"] == rec.movie.shape[0]
    qc_df = state.get_table(res["qc_table"])
    usable = qc_df[qc_df["usable"]]
    assert len(usable) > 0
    errs = [
        float(np.hypot(row.y - rec.true_positions[row.label][row.time_index][0],
                        row.x - rec.true_positions[row.label][row.time_index][1]))
        for row in usable.itertuples(index=False)
    ]
    assert max(errs) < 3.0

    measured = measure.measure_intensity_over_time(res["labels_layer"], "movie")
    assert measured["n_timepoints"] == rec.movie.shape[0]


def test_both_producers_emit_identical_qc_schema(viewer) -> None:
    rec = make_recording(
        n_frames=12, shape=(48, 48), n_cells=2,
        positions=[(15, 15), (15, 33)],
        seed=1, motion={"lateral_px": 4.0}, negative_control=False,
    )
    viewer.add_labels(rec.labels, name="roi_a")
    viewer.add_labels(rec.labels.copy(), name="roi_b")
    viewer.add_image(rec.movie, name="movie", metadata={"axes": "TYX"})
    viewer.add_labels(np.ones((48, 48), dtype=np.int32), name="boundary")

    res_a = roi_timeseries.track_roi_over_time("roi_a", "movie")
    res_b = roi_timeseries.resegment_roi_over_time(
        "roi_b", "boundary", "movie",
        background_radius=0, smoothing_sigma=0.0, min_snr=2.0, min_size=10,
        auto_correct=False,
    )

    df_a = state.get_table(res_a["qc_table"])
    df_b = state.get_table(res_b["qc_table"])
    assert list(df_a.columns) == list(df_b.columns)
    assert [str(dt) for dt in df_a.dtypes] == [str(dt) for dt in df_b.dtypes]


# --- z_project: 2D-only tracker vs. a real TZYX movie ---------------------------


def test_track_raises_on_tzyx_without_z_project(viewer) -> None:
    movie, labels3d = _tzyx_recording()
    viewer.add_labels(labels3d, name="roi3d")
    viewer.add_image(movie, name="movie4d", metadata={"axes": "TZYX"})

    with pytest.raises(ValueError, match=r"2D\+T only"):
        roi_timeseries.track_roi_over_time("roi3d", "movie4d")


def test_track_roi_over_time_z_project_max_warns(viewer) -> None:
    movie, labels3d = _tzyx_recording()
    viewer.add_labels(labels3d, name="roi3d")
    viewer.add_image(movie, name="movie4d", metadata={"axes": "TZYX"})

    res = roi_timeseries.track_roi_over_time("roi3d", "movie4d", z_project="max")

    assert res["n_timepoints"] == movie.shape[0]
    assert any("lateral" in w.lower() for w in res["warnings"])
    out_layer = viewer.layers[res["labels_layer"]]
    assert out_layer.data.shape == movie.shape  # full TZYX shape, T at the SAME axis
    assert out_layer.metadata["time_axis"] == 0


# --- manual correction -----------------------------------------------------------


def test_set_labels_at_frame_splices_and_refreshes(viewer) -> None:
    stack = np.zeros((3, 16, 16), dtype=np.int32)
    stack[0][2:6, 2:6] = 1
    stack[1][2:6, 2:6] = 1  # deliberately the WRONG (unmoved) footprint at t=1
    stack[2][2:6, 2:6] = 1
    before = stack.copy()
    viewer.add_labels(
        stack, name="tracked",
        metadata={"axes": "TYX", "time_axis": 0, "qc_table": "tracked_qc"},
    )

    corrected_frame = np.zeros((16, 16), dtype=np.int32)
    corrected_frame[9:13, 9:13] = 1
    viewer.add_labels(corrected_frame, name="corrected_frame1")

    state.put_table(
        "tracked_qc",
        pd.DataFrame(
            [
                {"label": 1, "time_index": 0, "usable": True, "confidence": 1.0,
                 "reason": "located", "y": 3.5, "x": 3.5, "n_pixels": 16,
                 "method": "track_roi_over_time"},
                {"label": 1, "time_index": 1, "usable": False, "confidence": 0.0,
                 "reason": "no_neighbours", "y": 3.5, "x": 3.5, "n_pixels": 16,
                 "method": "track_roi_over_time"},
                {"label": 1, "time_index": 2, "usable": True, "confidence": 1.0,
                 "reason": "located", "y": 3.5, "x": 3.5, "n_pixels": 16,
                 "method": "track_roi_over_time"},
            ]
        ),
    )

    res = roi_timeseries.set_labels_at_frame("tracked", 1, "corrected_frame1")

    out = viewer.layers["tracked"].data
    assert np.array_equal(out[0], before[0])
    assert np.array_equal(out[2], before[2])
    assert np.array_equal(out[1], corrected_frame)
    assert res["qc_rows_updated"] == 1
    assert viewer.layers["tracked"].metadata["manual_corrections"] == [
        {"time_index": 1, "source_layer": "corrected_frame1", "corrected_by": "manual"}
    ]

    qc_df = state.get_table("tracked_qc")
    row1 = qc_df[qc_df["time_index"] == 1].iloc[0]
    assert bool(row1["usable"]) is True
    assert row1["confidence"] == pytest.approx(1.0)
    assert row1["reason"] == "manual_correction"
    row0 = qc_df[qc_df["time_index"] == 0].iloc[0]
    assert row0["reason"] == "located"  # untouched rows stay untouched


def test_set_labels_at_frame_with_label_id_only_touches_that_label(viewer) -> None:
    stack = np.zeros((2, 10, 10), dtype=np.int32)
    stack[0][1:4, 1:4] = 1
    stack[0][6:9, 6:9] = 2
    stack[1][1:4, 1:4] = 1  # label 1 needs correcting at t=1
    stack[1][6:9, 6:9] = 2  # label 2 is already right at t=1 -- must survive untouched
    viewer.add_labels(stack, name="tracked", metadata={"axes": "TYX", "time_axis": 0})

    corrected = np.zeros((10, 10), dtype=np.int32)
    corrected[3:6, 3:6] = 1  # label 1's real footprint at t=1
    viewer.add_labels(corrected, name="corrected_frame1")

    roi_timeseries.set_labels_at_frame("tracked", 1, "corrected_frame1", label_id=1)

    out = viewer.layers["tracked"].data[1]
    assert np.array_equal(out == 1, corrected == 1)
    assert np.array_equal(out == 2, stack[1] == 2)  # label 2's footprint untouched


def test_set_labels_at_frame_rejects_label_id_zero(viewer) -> None:
    viewer.add_labels(np.zeros((2, 8, 8), dtype=np.int32), name="tracked",
                       metadata={"axes": "TYX"})
    viewer.add_labels(np.zeros((8, 8), dtype=np.int32), name="frame1")

    with pytest.raises(ValueError, match="label_id"):
        roi_timeseries.set_labels_at_frame("tracked", 1, "frame1", label_id=0)


# --- the contract's most important guarantee: a gap, never a fabricated number --


def test_gated_frame_produces_no_measurement_row(viewer, monkeypatch) -> None:
    movie = np.random.default_rng(0).normal(20.0, 1.0, size=(3, 24, 24)).astype(np.float32)
    labels = np.zeros((24, 24), dtype=np.int32)
    labels[8:12, 8:12] = 1
    viewer.add_labels(labels, name="roi")
    viewer.add_image(movie, name="movie", metadata={"axes": "TYX"})

    fake_stack = np.zeros((3, 24, 24), dtype=np.int32)
    fake_stack[0][labels == 1] = 1
    fake_stack[2][labels == 1] = 1
    # t=1 deliberately left as background -- the gap under test.
    fake_qc_rows = [
        {"label": 1, "time_index": 0, "usable": True, "confidence": 1.0,
         "reason": "located", "y": 9.5, "x": 9.5, "n_pixels": 16},
        {"label": 1, "time_index": 1, "usable": False, "confidence": 0.0,
         "reason": "no_neighbours", "y": 9.5, "x": 9.5, "n_pixels": 0},
        {"label": 1, "time_index": 2, "usable": True, "confidence": 1.0,
         "reason": "located", "y": 9.5, "x": 9.5, "n_pixels": 16},
    ]
    # Patch the name as bound INTO roi_timeseries, not analysis.roi_track's own
    # attribute -- roi_timeseries does `from ... import track_rois`, a direct
    # binding, so patching the source module would not reach this call site
    # (see memory/tool-module-split-monkeypatch.md's trap).
    monkeypatch.setattr(
        roi_timeseries, "track_rois",
        lambda *a, **k: {"stack": fake_stack, "qc_rows": fake_qc_rows},
    )

    res = roi_timeseries.track_roi_over_time("roi", "movie")
    qc_df = state.get_table(res["qc_table"])
    gated = qc_df[(qc_df["label"] == 1) & (qc_df["time_index"] == 1)]
    assert len(gated) == 1
    assert bool(gated.iloc[0]["usable"]) is False

    measured = measure.measure_intensity_over_time(res["labels_layer"], "movie")
    m_df = state.get_table(measured["table_name"])
    assert not ((m_df["label"] == 1) & (m_df["time_index"] == 1)).any()
    assert ((m_df["label"] == 1) & (m_df["time_index"] == 0)).any()
    assert ((m_df["label"] == 1) & (m_df["time_index"] == 2)).any()


def test_set_labels_at_frame_refreshes_position_and_pixel_count(viewer) -> None:
    """A corrected row must describe the mask that is now actually in the layer.

    Only usable/confidence/reason used to be rewritten, so a QC table could
    vouch (usable=True, confidence=1.0) for a footprint at a position the
    spliced mask never occupied, or report n_pixels=0 for the frame the user
    just filled in -- breaking the one invariant both producers otherwise keep,
    that n_pixels equals what the LABELS layer contains for that (label, frame).
    """
    stack = np.zeros((2, 16, 16), dtype=np.int32)
    stack[0][2:6, 2:6] = 1
    viewer.add_labels(
        stack, name="tracked",
        metadata={"axes": "TYX", "time_axis": 0, "qc_table": "tracked_qc"},
    )
    corrected_frame = np.zeros((16, 16), dtype=np.int32)
    corrected_frame[9:13, 9:13] = 1  # 16 px centred on (10.5, 10.5)
    viewer.add_labels(corrected_frame, name="corrected_frame1")

    state.put_table(
        "tracked_qc",
        pd.DataFrame(
            [
                {"label": 1, "time_index": 0, "usable": True, "confidence": 1.0,
                 "reason": "located", "y": 3.5, "x": 3.5, "n_pixels": 16,
                 "method": "track_roi_over_time"},
                {"label": 1, "time_index": 1, "usable": False, "confidence": 0.0,
                 "reason": "no_neighbours", "y": 3.5, "x": 3.5, "n_pixels": 0,
                 "method": "track_roi_over_time"},
            ]
        ),
    )

    roi_timeseries.set_labels_at_frame("tracked", 1, "corrected_frame1", label_id=1)

    row = state.get_table("tracked_qc").query("time_index == 1").iloc[0]
    assert row["reason"] == "manual_correction"
    assert bool(row["usable"]) is True
    assert row["n_pixels"] == int((viewer.layers["tracked"].data[1] == 1).sum()) == 16
    assert row["y"] == pytest.approx(10.5)
    assert row["x"] == pytest.approx(10.5)


def test_set_labels_at_frame_will_not_call_an_empty_frame_usable(viewer) -> None:
    """Splicing in a frame where the label has NO pixels leaves usable=False.

    regionprops emits no row for an absent label, so marking it usable would
    put the QC table's coverage permanently out of step with the measurement
    table -- the same disagreement the producers are gated to avoid.
    """
    stack = np.zeros((2, 12, 12), dtype=np.int32)
    stack[1][2:6, 2:6] = 1
    viewer.add_labels(
        stack, name="tracked",
        metadata={"axes": "TYX", "time_axis": 0, "qc_table": "tracked_qc"},
    )
    viewer.add_labels(np.zeros((12, 12), dtype=np.int32), name="blank_frame")
    state.put_table(
        "tracked_qc",
        pd.DataFrame(
            [{"label": 1, "time_index": 1, "usable": True, "confidence": 0.9,
              "reason": "located", "y": 3.5, "x": 3.5, "n_pixels": 16,
              "method": "track_roi_over_time"}]
        ),
    )

    roi_timeseries.set_labels_at_frame("tracked", 1, "blank_frame", label_id=1)

    row = state.get_table("tracked_qc").iloc[0]
    assert bool(row["usable"]) is False
    assert row["n_pixels"] == 0
    assert np.isnan(row["y"]) and np.isnan(row["x"])


# --- Slice R3: the dim-frame survivorship bias must be REPORTED, not silent ------


def _single_roi_recording(*, silent_windows=None, seed=0) -> object:
    """A single-label recording (n_cells=1, negative_control=False) so
    correct_sparse's neighbour-interpolation recovery is provably inert (see
    roi_track.py's own module docstring: `others` needs >=3 co-visible labels)
    -- any gated frame here is unambiguously a contrast/no_neighbours gate,
    never a rescued/interpolated one. `silent_windows` (calcium_synth's own
    knob) drops the cell fully to background for that window when given --
    the "signal dips to baseline" scenario the bias warning exists to catch --
    and leaves a normal, always-above-baseline transient trace when omitted."""
    return make_recording(
        n_frames=40, shape=(64, 64), n_cells=1, seed=seed,
        negative_control=False, positions=[(32, 32)],
        silent_windows=silent_windows,
    )


def test_dim_frame_bias_warning_fires_when_signal_dips_to_baseline(viewer) -> None:
    """The measured bug this slice closes: track_roi_over_time gates by the
    tracked object's own signal CONTRAST, not by drift, so a signal that dips
    to baseline loses systematically the DIM frames and the surviving trace
    reads high. This must be reported even though coverage (0.700 measured)
    is comfortably above the existing 50% `rejected` cutoff -- a scientist
    reading only `rejected` would otherwise see nothing wrong."""
    rec = _single_roi_recording(silent_windows={1: (10, 22)})
    viewer.add_labels(rec.labels, name="roi")
    viewer.add_image(rec.movie, name="movie", metadata={"axes": "TYX"})

    res = roi_timeseries.track_roi_over_time("roi", "movie")

    assert res["coverage"][1] < 0.9
    assert 1 not in res["rejected"]
    assert any("contrast" in w and "biased" in w for w in res["warnings"])


def test_dim_frame_bias_warning_silent_on_constant_brightness(viewer) -> None:
    """The negative control: a signal that never dips toward baseline tracks
    at full coverage and must NOT trip the bias warning -- it exists to flag a
    real selection effect, not to editorialize on every successful track."""
    rec = _single_roi_recording(silent_windows=None)
    viewer.add_labels(rec.labels, name="roi")
    viewer.add_image(rec.movie, name="movie", metadata={"axes": "TYX"})

    res = roi_timeseries.track_roi_over_time("roi", "movie")

    assert res["coverage"][1] == pytest.approx(1.0)
    assert not any("contrast" in w and "biased" in w for w in res["warnings"])


def _tzyx_drifting_recording() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A true (T, Z, Y, X) movie whose blob sits on ONE Z plane and drifts in X,
    plus a 3D seed and a genuinely 2D boundary.

    Distinct from _tzyx_recording() above, which is deliberately static because
    only shape/wiring is under test there. Here the object actually moves, so a
    regression that silently measured a fixed footprint (the whole bug approach B
    exists to fix) would show up as a centroid that stops following the blob.
    """
    T, Z, Y, X = 6, 3, 40, 40
    rng = np.random.default_rng(3)
    movie = rng.normal(20.0, 1.0, size=(T, Z, Y, X)).astype(np.float32)
    for t in range(T):
        cx = 12 + 3 * t  # +3 px per frame, entirely inside the boundary below
        movie[t, 1, 16:24, cx - 4:cx + 4] += 80.0
    seed3d = np.zeros((Z, Y, X), dtype=np.int32)
    seed3d[1, 16:24, 8:16] = 1  # drawn on t=0's position, on the blob's own plane
    boundary2d = np.zeros((Y, X), dtype=np.int32)  # 2D: the max-projection workflow
    boundary2d[10:30, 4:38] = 1
    return movie, seed3d, boundary2d


def test_resegment_roi_over_time_handles_true_tzyx_with_a_2d_boundary(viewer) -> None:
    """approach B's docstring advertises "2D+T or true 3D+T directly" -- this pins
    that claim, which nothing else covers (the only other TZYX tests here exercise
    track_roi_over_time's z_project path).

    The boundary is drawn as a real 2-D (Y, X) mask, not pre-broadcast to 3-D,
    because that is how a user actually produces one: a Shapes ROI traced on a max
    projection (boundary_mask_from_shapes). resolve_boundary_mask is only ever
    handed ONE timepoint's shape here -- resegment_roi_over_time moves T to axis 0
    and slices it off before resolving -- so the 2D->3D broadcast it supports is
    reached, and its "anything else raises" restriction is never hit by the 4-D
    movie itself.
    """
    movie, seed3d, boundary2d = _tzyx_drifting_recording()
    viewer.add_labels(seed3d, name="roi3d")
    viewer.add_labels(boundary2d, name="boundary2d")
    viewer.add_image(movie, name="movie4d", metadata={"axes": "TZYX"})

    res = roi_timeseries.resegment_roi_over_time(
        "roi3d", "boundary2d", "movie4d",
        background_radius=0, smoothing_sigma=0.0, min_snr=2.0, min_size=20,
        auto_correct=False,
    )

    assert res["n_timepoints"] == movie.shape[0]
    assert any("broadcast" in w.lower() for w in res["warnings"])

    out_layer = viewer.layers[res["labels_layer"]]
    assert out_layer.data.shape == movie.shape  # full TZYX, T left where it was
    assert out_layer.metadata["time_axis"] == 0

    qc_df = state.get_table(res["qc_table"])
    usable = qc_df[qc_df["usable"]]
    assert len(usable) == movie.shape[0]  # every frame recovered
    # The tracked centroid must FOLLOW the +3 px/frame drift, not sit still: a
    # static-footprint regression would hold x roughly constant across frames.
    xs = [float(usable[usable["time_index"] == t]["x"].iloc[0]) for t in range(movie.shape[0])]
    assert xs[-1] - xs[0] > 12.0, xs
    for t in range(movie.shape[0]):
        assert abs(xs[t] - (11.5 + 3 * t)) < 2.0, xs

    measured = measure.measure_intensity_over_time(res["labels_layer"], "movie4d")
    assert measured["n_timepoints"] == movie.shape[0]


def test_contrast_bias_warning_survives_a_conf_floor_above_the_interpolated_score(
    viewer,
) -> None:
    """Raising conf_floor must not SILENCE the bias warning it makes more necessary.

    correct_sparse pins an interpolated frame's confidence to exactly 0.6
    (calcium_motion.py:147). With conf_floor above that, a dim frame whose direct
    lock failed gets rescued by interpolation and then re-gated -- landing in the
    QC table as usable=False with reason "interpolated". Those frames are still
    contrast-driven drops, so if they are not counted as such they dilute the
    majority test in _contrast_bias_warnings and the warning goes silent on the
    label whose trace is the MOST biased. Pins the fix: the centre ROI here loses
    more frames at conf_floor=0.8 than at 0.5 and must be warned about at both.
    """
    T, H, W, sigma = 40, 96, 96, 4.0
    centres = [(46.0, 46.0), (18.0, 18.0), (18.0, 74.0), (74.0, 46.0)]
    t = np.arange(T)
    # Only the centre ROI dips to baseline; the other three stay bright so they
    # remain co-visible neighbours and the interpolation branch can actually fire
    # (it needs min_neighbours=3 usable labels, and the target inside their hull).
    dipping = 250.0 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 12.0))
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[:H, :W]
    movie = np.zeros((T, H, W), dtype=np.float32)
    labels = np.zeros((H, W), dtype=np.int32)
    for k, (cy, cx) in enumerate(centres):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= sigma**2] = k + 1
    for i in range(T):
        frame = np.full((H, W), 100.0)
        for k, (cy, cx) in enumerate(centres):
            amp = dipping[i] if k == 0 else 250.0
            frame += amp * np.exp(
                -((yy - cy - i) ** 2 + (xx - cx - i) ** 2) / (2 * sigma**2)
            )
        movie[i] = frame + rng.normal(0, 3.0, (H, W))

    viewer.add_labels(labels, name="rois")
    viewer.add_image(movie.astype(np.uint16), name="movie", metadata={"axes": "TYX"})

    res = roi_timeseries.track_roi_over_time(
        "rois", "movie", name="tracked", conf_floor=0.8
    )

    qc = state.get_table(res["qc_table"])
    dropped = qc[(qc["label"] == 1) & (~qc["usable"])]
    # The scenario is only meaningful if it actually produced the reason under test.
    assert (dropped["reason"] == "interpolated").any(), dropped["reason"].value_counts()
    assert res["coverage"][1] < 0.9
    assert any(
        w.startswith("label 1:") and "contrast fell below" in w for w in res["warnings"]
    ), res["warnings"]


def _long_thin_recording():
    """Deliberately shaped so infer_time_axis fires: 400 frames of 60x60 is a
    6.7x ratio. The module's usual _drifting_recording (50 frames of 110x110,
    0.45x) is BELOW the threshold on purpose -- 50 planes really could be a
    z-stack -- so it cannot exercise this path."""
    return make_recording(
        n_frames=400, shape=(60, 60), n_cells=3, seed=7, motion={"lateral_px": 6.0},
    )


def test_bare_tiff_axes_are_inferred_and_the_inference_is_reported(viewer) -> None:
    """A no-metadata TIFF must work WITHOUT time_axis, and must say it inferred.

    Every reference recording for this feature is a bare 'IYX' export (tifffile's
    placeholder for a file carrying no axis metadata), so the strict resolver
    refused all of them. Inferring is only acceptable because the tool reports it
    -- a silent guess is exactly the failure mode tools/view.py's _resolve_axis
    has, where 'T' and 'Z' both map to axis 0 and nothing ever raises.
    """
    rec = _long_thin_recording()
    viewer.add_labels(rec.labels, name="roi")
    viewer.add_image(rec.movie, name="movie", metadata={"axes": "IYX"})

    res = roi_timeseries.track_roi_over_time("roi", "movie")  # NO time_axis

    assert res["n_timepoints"] == rec.movie.shape[0]
    note = [w for w in res["warnings"] if "was used as time" in w]
    assert note, f"the inference must be reported; got {res['warnings']}"
    assert "IYX" in note[0] and "time_axis" in note[0]


def test_an_ambiguous_bare_stack_is_still_refused(viewer) -> None:
    """The guard the inference must not dissolve: a leading axis that could
    plausibly be z-planes gets no guess, only the actionable error."""
    viewer.add_image(np.zeros((30, 64, 64), np.float32), name="amb", metadata={"axes": "IYX"})
    viewer.add_labels(np.zeros((64, 64), np.int32), name="amb_roi")

    with pytest.raises(ValueError, match="do not include a time axis"):
        roi_timeseries.track_roi_over_time("amb_roi", "amb")
