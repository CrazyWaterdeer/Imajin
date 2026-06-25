"""v1 acceptance battery: scores the QC pipeline against synthetic ground truth.

Computes the binding v1 criteria — negative-control flatness, event-preservation,
defocus gating recall/precision, F0 bias, artifact magnitude — and a single
``passed`` verdict. Headless (numpy only, plus the calcium_* analysis modules).
"""

from __future__ import annotations

import numpy as np

from imajin.analysis.calcium_qc import gate_traces
from imajin.analysis.calcium_events import negative_control_flat, event_preservation_rate


def _rolling_dff(intensity, window=41, pct=10.0) -> np.ndarray:
    n = len(intensity)
    half = window // 2
    f0 = np.array([np.percentile(intensity[max(0, i - half): i + half + 1], pct)
                   for i in range(n)])
    return (intensity - f0) / np.where(f0 != 0, f0, np.nan)


def run_v1_acceptance(rec) -> dict:
    gate = gate_traces(rec.movie, rec.labels)
    neg = rec.negative_label
    T = rec.movie.shape[0]

    # defocus gating accuracy vs ground truth (union of "defocus" reason over cells)
    truth_def = np.zeros(T, bool)
    truth_def[rec.defocus_frames] = True
    pred_def = np.zeros(T, bool)
    for reason in gate.reason.values():
        pred_def |= (reason == "defocus")
    tp = int(np.sum(pred_def & truth_def))
    recall = tp / max(1, int(truth_def.sum()))
    precision = tp / max(1, int(pred_def.sum())) if pred_def.any() else 1.0

    # negative control: flatness + F0 bias + artifact magnitude on usable frames
    neg_flat, f0_bias, artifact_max = True, 0.0, 0.0
    if neg is not None:
        roi = rec.labels == neg
        inten = rec.movie[:, roi].mean(axis=1)
        dff = _rolling_dff(inten)
        usable = gate.usable[neg]
        trace = np.nan_to_num(dff[usable], nan=0.0)
        nc = negative_control_flat(trace)
        neg_flat = nc["flat"]
        artifact_max = float(nc["max_abs"])
        f0_bias = float(abs(np.nanmedian(dff[usable])))

    signalling = {k: v for k, v in rec.event_frames.items() if k != neg}
    preservation = event_preservation_rate(gate.usable, signalling)

    passed = bool(neg_flat and preservation >= 0.95 and recall >= 0.9
                  and f0_bias < 0.1 and artifact_max < 0.05)
    return {
        "negative_control_flat": bool(neg_flat),
        "event_preservation": float(preservation),
        "defocus_recall": float(recall),
        "defocus_precision": float(precision),
        "f0_bias_negative": float(f0_bias),
        "artifact_max": float(artifact_max),
        "coverage": {int(k): float(v) for k, v in gate.coverage.items()},
        "longest_run": {int(k): int(v) for k, v in gate.longest_run.items()},
        "passed": passed,
    }
