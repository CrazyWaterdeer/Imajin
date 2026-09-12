from __future__ import annotations

import threading

import numpy as np
import pytest

from imajin.analysis.arrays import infer_time_axis, map_over_axis0, resolve_time_axis


# --- map_over_axis0 ----------------------------------------------------------
#
# Promoted from tools/preprocess.py's _run_over_planes -- these tests pin the
# same contract that module's own tests already exercised indirectly (via
# rolling_ball_background on a 3D stack): every index gets visited exactly
# once, below-threshold n stays serial, at/above it genuinely dispatches to
# worker threads (not just claims to), and a worker's own min_parallel
# argument is respected as the serial/parallel boundary.


def test_map_over_axis0_visits_every_index_exactly_once():
    for n in (0, 1, 2, 5, 17):
        out = np.full(n, -1, dtype=np.int64)
        map_over_axis0(lambda i, out=out: out.__setitem__(i, i * i), n)
        assert np.array_equal(out, np.arange(n, dtype=np.int64) ** 2), f"n={n}"


def test_map_over_axis0_below_min_parallel_runs_serially_in_the_calling_thread():
    caller_ident = threading.get_ident()
    seen: list[int] = []
    # n=1 < the default min_parallel=2 -- must not spin up a thread pool at all.
    map_over_axis0(lambda i: seen.append(threading.get_ident()), 1)
    assert seen == [caller_ident]


def test_map_over_axis0_at_min_parallel_dispatches_to_worker_threads():
    caller_ident = threading.get_ident()
    seen: list[int] = []
    lock = threading.Lock()

    def record(i: int) -> None:
        with lock:  # list.append is already atomic under the GIL; belt-and-braces
            seen.append(threading.get_ident())

    map_over_axis0(record, 8)  # n=8 >= default min_parallel=2

    assert len(seen) == 8
    # concurrent.futures.ThreadPoolExecutor never runs a submitted callable on
    # the calling thread itself -- it only waits on the pool -- so every call
    # must have landed on a pool thread. Not a timing-dependent assumption.
    assert all(ident != caller_ident for ident in seen)


def test_map_over_axis0_min_parallel_moves_the_serial_parallel_boundary():
    caller_ident = threading.get_ident()
    seen: list[int] = []
    # Without min_parallel, n=4 would go parallel (default threshold is 2).
    # Raising the threshold to 5 must keep it serial.
    map_over_axis0(lambda i: seen.append(threading.get_ident()), 4, min_parallel=5)
    assert seen == [caller_ident] * 4


def test_map_over_axis0_propagates_a_worker_exception():
    def boom(i: int) -> None:
        if i == 3:
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        map_over_axis0(boom, 6)


def test_resolve_time_axis_raises_without_t_axis_or_explicit_time_axis():
    # 'IYX' is tifffile's own placeholder axes string for a bare, no-metadata 3D
    # TIFF -- it must fail loudly, not silently default to axis 0, because a
    # silent guess here would measure/track the wrong dimension with no error to
    # catch it (this is the exact case _resolve_time_axis raised on before, and
    # the whole reason it exists as a fail-loud resolver rather than a fallback
    # dict like view.py's _resolve_axis).
    with pytest.raises(ValueError, match="time axis"):
        resolve_time_axis("IYX", 3, None)


def test_resolve_time_axis_finds_t_in_axes():
    assert resolve_time_axis("TYX", 3, None) == 0
    assert resolve_time_axis("ZTYX", 4, None) == 1


def test_resolve_time_axis_accepts_explicit_int():
    assert resolve_time_axis("IYX", 3, 0) == 0
    assert resolve_time_axis("IYX", 3, -1) == 2  # negative index counts from the end


def test_resolve_time_axis_explicit_int_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        resolve_time_axis("IYX", 3, 5)


def test_resolve_time_axis_accepts_explicit_code_case_insensitive():
    assert resolve_time_axis("ZYXT", 4, "t") == 3


def test_resolve_time_axis_explicit_code_not_in_axes_raises():
    with pytest.raises(ValueError, match="not found"):
        resolve_time_axis("ZYX", 3, "T")


def test_resolve_time_axis_explicit_code_must_be_single_char():
    with pytest.raises(ValueError, match="axis code or integer"):
        resolve_time_axis("TYX", 3, "TT")


class TestInferTimeAxis:
    """The narrow escape hatch for TIFFs that carry no axis metadata at all.

    Every reference recording in this project is a bare 'IYX' export, so the
    strict resolver rejected 100% of real data for a question the shape answers.
    Inferring is only defensible because it is (a) narrow and (b) reported --
    see infer_time_axis's docstring.
    """

    @pytest.mark.parametrize(
        "axes,shape",
        [
            ("IYX", (2882, 250, 251)),  # real file: 11.5x
            ("IYX", (4187, 501, 502)),  # real file: 8.3x
            ("QYX", (3601, 250, 251)),  # tifffile's other placeholder code
        ],
    )
    def test_infers_axis_0_for_an_unlabelled_long_series(self, axes, shape):
        assert infer_time_axis(axes, shape) == 0

    @pytest.mark.parametrize(
        "axes,shape,why",
        [
            ("IYX", (30, 250, 251), "30 planes is a plausible z-stack"),
            ("IYX", (524, 501, 502), "1.04x — under the ratio, genuinely ambiguous"),
            ("ZYX", (2882, 250, 251), "the file SAID z; never override metadata"),
            ("TYX", (2882, 250, 251), "the file said t; the resolver handles it"),
            ("IYX", (250, 251), "2D"),
            ("IZYX", (10, 2882, 250, 251), "4D"),
        ],
    )
    def test_refuses_anything_ambiguous(self, axes, shape, why):
        assert infer_time_axis(axes, shape) is None, why
