import numpy as np

from imajin.analysis.calcium_events import (
    detect_events, negative_control_flat, event_preservation_rate)


def test_detect_and_flat_with_offset():
    trace = np.full(200, 0.03)            # small positive DC offset (low-pct F0 bias)
    trace[50:56] += 0.8                   # a real transient on top
    ev = detect_events(trace, k=4.0)
    assert any(s <= 52 <= e for s, e in ev)
    # a flat-but-offset noisy control must read flat after detrending
    flat = negative_control_flat(0.03 + np.random.default_rng(0).normal(0, 0.01, 200))
    assert flat["flat"]
    assert not negative_control_flat(trace)["flat"]


def test_event_preservation():
    usable = np.ones(100, bool)
    usable[40:60] = False
    rate = event_preservation_rate({1: usable}, {1: [10, 50, 80]})
    assert rate == 2 / 3
