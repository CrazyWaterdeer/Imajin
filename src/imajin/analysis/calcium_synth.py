"""Synthetic calcium-imaging recordings with ground-truth labels.

Headless (numpy/scipy only) generator used as the v1 acceptance gate: it
produces a movie plus the ground truth needed to *score* the QC pipeline —
true ΔF/F0 per cell, event frames, the injected static negative control, and
which frames were blurred / how the field was moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticRecording:
    movie: np.ndarray
    labels: np.ndarray
    true_dff: dict[int, np.ndarray]
    event_frames: dict[int, list[int]]
    f0: dict[int, float]
    negative_label: int | None
    defocus_frames: list[int] = field(default_factory=list)
    motion: dict | None = None
    meta: dict = field(default_factory=dict)


def _disk(cy, cx, radius, shape):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def _transient(n_frames, peaks, amp, tau):
    t = np.arange(n_frames, dtype=np.float64)
    trace = np.zeros(n_frames)
    for p in peaks:
        trace += amp * np.exp(-(t - p) / tau) * (t >= p)
    return trace


def make_recording(*, n_frames=200, shape=(128, 128), n_cells=6, seed=0,
                   bleach_tau=None, noise=2.0, motion=None, defocus=None,
                   negative_control=True) -> SyntheticRecording:
    rng = np.random.default_rng(seed)
    Y, X = shape
    labels = np.zeros(shape, dtype=np.int32)
    f0, true_dff, event_frames = {}, {}, {}
    radius = 5.0
    margin = int(radius) + 3
    base = np.full((n_frames, Y, X), 5.0, dtype=np.float64)

    negative_label = n_cells if negative_control else None
    placed: list[tuple[int, int]] = []
    for lbl in range(1, n_cells + 1):
        # place a non-overlapping disk (so every label survives in `labels`)
        for _ in range(200):
            cy = int(rng.integers(margin, Y - margin))
            cx = int(rng.integers(margin, X - margin))
            if all((cy - py) ** 2 + (cx - px) ** 2 >= (2 * radius + 2) ** 2
                   for py, px in placed):
                break
        placed.append((cy, cx))
        mask = _disk(cy, cx, radius, shape)
        labels[mask] = lbl
        f0[lbl] = float(rng.uniform(40.0, 80.0))
        if negative_label is not None and lbl == negative_label:
            dff, peaks = np.zeros(n_frames), []
        else:
            n_ev = int(rng.integers(2, 5))
            peaks = sorted(int(p) for p in rng.integers(5, n_frames - 5, size=n_ev))
            dff = _transient(n_frames, peaks, amp=rng.uniform(0.4, 1.2), tau=8.0)
        true_dff[lbl] = dff
        event_frames[lbl] = peaks
        intensity = f0[lbl] * (1.0 + dff)
        base[:, mask] += intensity[:, None]

    if bleach_tau:
        base *= np.exp(-np.arange(n_frames) / float(bleach_tau))[:, None, None]

    from scipy.ndimage import gaussian_filter, shift as nd_shift

    movie = base.copy()
    if motion:
        amp = float(motion.get("lateral_px", 0.0))
        for t in range(n_frames):
            frac = t / max(1, n_frames - 1)
            movie[t] = nd_shift(movie[t], (amp * frac, amp * frac * 0.5),
                                order=1, mode="nearest")
    defocus_frames = list(defocus.get("frames", [])) if defocus else []
    if defocus_frames:
        sigma = float(defocus.get("sigma", 3.0))
        for t in defocus_frames:
            movie[t] = gaussian_filter(movie[t], sigma=sigma)

    movie = (movie + rng.normal(0.0, float(noise), size=movie.shape)).astype(np.float32)
    return SyntheticRecording(
        movie=movie, labels=labels, true_dff=true_dff, event_frames=event_frames,
        f0=f0, negative_label=negative_label, defocus_frames=defocus_frames,
        motion=motion, meta={"seed": seed, "noise": noise, "bleach_tau": bleach_tau},
    )
