from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np


def map_over_axis0(fn: Callable[[int], None], n: int, *, min_parallel: int = 2) -> None:
    """Apply ``fn(i)`` for i in range(n), across threads when it pays off.

    Promoted verbatim (same threshold, same worker formula, same ``ex.map``
    call) from tools/preprocess.py's original ``_run_over_planes``, which is
    now a thin delegation to this function -- see that module for its one
    caller. Generalized name/home because a second, unrelated caller
    (roi_track.py's per-frame mask rasterization) needed the identical
    pattern over a *time* axis, not a Z-plane axis; the safety argument is the
    same for either axis, so duplicating the pool-dispatch logic per caller
    would only be two copies of one fact.

    Safety contract, unchanged from ``_run_over_planes``: this is a
    side-effecting map, not a value-collecting one. ``fn(i)`` must write its
    own result into a slice of the CALLER's preallocated output that no other
    index touches (e.g. ``out[i] = ...`` or ``results[i] = ...``) -- plain
    index/key assignment at disjoint positions is safe to do concurrently
    under the GIL, but this function does not collect or serialize return
    values, so anything ``fn`` returns is discarded.

    Threading only pays when the per-call work actually releases the GIL long
    enough to amortise dispatch overhead (skimage/scipy C code typically does;
    a numba ``@njit`` kernel without ``nogil=True`` does NOT and measurably
    gets WORSE with threads -- confirmed for calcium_qc.locate_cell). Callers
    are responsible for having actually measured their own ``fn`` before
    reaching for this, not assuming it from this docstring.

    ``min_parallel`` (default 2, matching ``_run_over_planes``'s original
    ``n <= 1`` guard) is the smallest ``n`` this bothers spinning a thread
    pool for; below it, ``fn`` just runs serially in the calling thread so a
    trivial-sized caller (e.g. a single-plane stack) never pays pool-creation
    overhead for nothing.

    Callers that must observe cancellation or report progress CANNOT do so
    from inside ``fn`` when it runs in a pool thread: a
    ``ThreadPoolExecutor`` worker starts with its own fresh, empty
    ``contextvars.Context`` (verified empirically -- a value set with
    ``ContextVar.set()`` in the calling thread reads back as that var's
    *default* inside an executor-submitted callable, never the real one), so
    ``imajin.agent.execution``'s ContextVar-backed ``raise_if_cancelled``/
    ``report_progress`` would silently see no token/job at all in a worker.
    Check/report in the orchestrating thread, between calls to this function
    (or between batches of work items), not inside ``fn``.
    """
    if n < min_parallel:
        for i in range(n):
            fn(i)
        return
    workers = min(n, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, range(n)))


def materialize_array(data: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return an in-memory numpy array from numpy-like or dask-like data."""

    if hasattr(data, "compute"):
        data = data.compute()
    if dtype is None:
        return np.asarray(data)
    return np.asarray(data, dtype=dtype)


def metadata_axes_without_channel(
    metadata: dict[str, Any] | None,
    ndim: int,
) -> str | None:
    """Return metadata axes aligned to `ndim`, excluding channel axes."""

    axes = metadata.get("axes") if isinstance(metadata, dict) else None
    if not isinstance(axes, str):
        return None
    layer_axes = axes.replace("C", "")
    if len(layer_axes) != ndim:
        return None
    return layer_axes


def default_axes(ndim: int, *, default_3d: str = "ZYX") -> str:
    if ndim == 4:
        return "TZYX"
    if ndim == 3:
        return default_3d
    if ndim == 2:
        return "YX"
    return "".join(f"A{i}" for i in range(ndim))


def layer_axes_from_metadata(
    metadata: dict[str, Any] | None,
    ndim: int,
    *,
    default_3d: str = "ZYX",
) -> str:
    return metadata_axes_without_channel(metadata, ndim) or default_axes(
        ndim,
        default_3d=default_3d,
    )


def resolve_time_axis(axes: str, ndim: int, time_axis: int | str | None) -> int:
    """Resolve which array axis is time, failing loudly instead of guessing.

    Ported verbatim (body only -- this takes the axes string directly instead of a
    napari layer) from tools/measure.py's original `_resolve_time_axis`, which is
    now a thin delegation to this function. Shared so every time-series consumer
    -- measure_intensity_over_time and roi_track's per-frame tracker alike -- gets
    identical behavior instead of two implementations that could silently drift
    apart. Deliberately does NOT fall back to a positional guess the way
    tools/view.py's `_resolve_axis` does (its fallback dict maps both 'T' and 'Z'
    to axis 0 for a 3D array and never raises): a silently-wrong axis here would
    measure or track the wrong dimension with nothing to signal the mistake.
    """
    if time_axis is None:
        if "T" in axes:
            return axes.index("T")
        raise ValueError(
            f"image layer axes {axes!r} do not include a time axis. Reload with "
            "metadata axes containing 'T' or pass time_axis explicitly."
        )
    if isinstance(time_axis, int):
        idx = time_axis if time_axis >= 0 else ndim + time_axis
        if idx < 0 or idx >= ndim:
            raise ValueError(f"time_axis {time_axis} out of range for {ndim}-D image")
        return idx
    code = time_axis.upper()
    if len(code) != 1:
        raise ValueError(f"time_axis must be an axis code or integer, got {time_axis!r}")
    if code not in axes:
        raise ValueError(f"axis {time_axis!r} not found in image axes {axes!r}")
    return axes.index(code)
