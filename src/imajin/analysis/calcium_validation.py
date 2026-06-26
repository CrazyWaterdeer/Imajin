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


def _safe_corr(a, b):
    a = np.nan_to_num(np.asarray(a, float))
    b = np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_v2_acceptance(rec) -> dict:
    """Score v2a sparse motion correction against synthetic ground truth."""
    from imajin.analysis.calcium_motion import correct_sparse, corrected_dff

    neg = rec.negative_label
    res = correct_sparse(rec.movie, rec.labels)
    dff = corrected_dff(rec.movie, rec.labels, res)
    v1 = gate_traces(rec.movie, rec.labels)

    resids, corrs, amp_ratios, conf_all, act_all = [], [], [], [], []
    cov_fail = False
    for lbl in rec.true_dff:
        u = res.usable[lbl]
        if u.sum() < 5:
            cov_fail = True               # too-low coverage is a failure, not a skip
        else:
            err = np.hypot(*(res.positions[lbl] - rec.true_positions[lbl]).T)
            resids.append(float(np.median(err[u])))
        conf_all.append(res.confidence[lbl])
        act_all.append(np.abs(np.nan_to_num(dff[lbl])))
        if lbl == neg:
            continue                      # exclude negative control from trace/amp
        if u.sum() >= 5:
            c = _safe_corr(dff[lbl][u], rec.true_dff[lbl][u])
            if c is not None:
                corrs.append(c)
        for f in rec.event_frames[lbl]:
            if 0 <= f < len(u) and u[f] and np.isfinite(dff[lbl][f]) and rec.true_dff[lbl][f] > 0:
                amp_ratios.append(float(dff[lbl][f] / rec.true_dff[lbl][f]))

    v1_cov = float(np.mean([v1.coverage[l] for l in rec.true_dff]))
    v2_cov = float(np.mean([res.usable[l].mean() for l in rec.true_dff]))
    coverage_gain_pp = (v2_cov - v1_cov) * 100.0

    moving_neg_flat = True
    if neg is not None:
        u = res.usable[neg]
        moving_neg_flat = bool(u.sum() >= 10 and
                               negative_control_flat(np.nan_to_num(dff[neg][u], nan=0.0))["flat"])

    cd = _safe_corr(np.concatenate(conf_all), np.concatenate(act_all))
    cd = 0.0 if cd is None else cd

    residual_median = float(np.median(resids)) if resids else np.inf
    trace_corr = float(np.median(corrs)) if corrs else 0.0
    amp_ratio = float(np.median(amp_ratios)) if amp_ratios else 0.0
    passed = bool(not cov_fail and residual_median < 1.0 and trace_corr > 0.95
                  and amp_ratio > 0.8 and coverage_gain_pp > 0
                  and moving_neg_flat and abs(cd) < 0.2)
    return {
        "residual_median_px": residual_median,
        "trace_corr_median": trace_corr,
        "event_amp_ratio_median": amp_ratio,
        "coverage_gain_pp": float(coverage_gain_pp),
        "moving_negative_flat": moving_neg_flat,
        "confidence_dynamics_corr": float(cd),
        "passed": passed,
    }
