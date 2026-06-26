"""Event detection, negative-control-flat test, and event-preservation metric.

All operate on a detrended trace so a small DC offset (the positive bias a
low-percentile rolling F0 leaves on a flat cell) does not masquerade as signal.
Headless (numpy only).
"""

from __future__ import annotations

import numpy as np


def _detrend(trace) -> np.ndarray:
    x = np.asarray(trace, dtype=float)
    return x - float(np.median(x))


def detect_events(trace, noise_sigma=None, k=3.0, min_len=2) -> list[tuple[int, int]]:
    x = _detrend(trace)
    if noise_sigma is None:
        d = np.diff(x)
        noise_sigma = float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2)) or 1e-9
    above = x > k * noise_sigma
    events: list[tuple[int, int]] = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_len:
                events.append((start, i - 1))
            start = None
    if start is not None and len(above) - start >= min_len:
        events.append((start, len(above) - 1))
    return events


def negative_control_flat(trace, artifact_ceiling=0.05) -> dict:
    x = _detrend(trace)
    ev = detect_events(trace, k=4.0)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    return {"flat": bool(len(ev) == 0 and max_abs < artifact_ceiling),
            "n_events": len(ev), "max_abs": max_abs}


def event_preservation_rate(usable, event_frames) -> float:
    total = kept = 0
    for lbl, frames in event_frames.items():
        mask = usable.get(lbl)
        for f in frames:
            total += 1
            if mask is not None and 0 <= f < len(mask) and mask[f]:
                kept += 1
    return (kept / total) if total else 1.0
