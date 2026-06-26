# Calcium Imaging Module — v2 (motion correction / tracking) Design

**Status**: Draft (revised after 1 Codex review, 2026-06-25)
**Date**: 2026-06-25
**Type**: Design

Follows and extends `2026-06-17-calcium-imaging-module-design.md` (v1 shipped:
ΔF/F0 + rolling F0 + honest defocus/lateral gating + coverage + synthetic
validation harness). This spec covers the v2 layer the v1 spec deferred.

## Problem

v1 *detects and gates* motion but does not *recover* it. On deforming-gut data
(the acceptance bar) a moving cell's frames fail the lateral-validity gate, so
they are NaN'd and the cell's coverage collapses — honest but lossy. v2 recovers
**in-plane (XY) motion, including non-rigid deformation**, so those frames become
measurable, while **axial (Z) motion stays gated** (still unrecoverable from
single-plane single-channel data — unchanged from v1).

v2 must handle *appear → disappear → reappear* directly: a cell that goes dark
during inactivity while the tissue deforms must keep a usable position inferred
from its still-visible neighbours (EMC2-style), then re-lock on reappearance —
**but only when that inference is provably valid; otherwise the frame is gated,
not faked.**

## Core safety principle

Every v2 correction carries a **confidence**, and a corrected value is emitted
only when confidence passes. v2 must never convert an *honest gap* (v1) into a
*confident-looking artifact*. When landmark observability, interpolation
validity, or warp topology checks fail, v2 **degrades to v1** (gate the frame).

## Goals

- Recover XY (incl. non-rigid) motion so a moving cell's ΔF/F0 is extracted from
  the right pixels every *confidently corrected* frame.
- Follow cumulative drift beyond v1's fixed ±search_radius (frame-to-frame
  propagation), with per-frame confidence.
- Bridge disappearance via *validated* neighbour interpolation; re-acquire on
  reappearance.
- Integrate with v1: re-run v1 gates on the corrected result; coverage on moving
  data rises materially **within a defined motion/SNR regime** vs v1-alone.
- Stay honest: negative control flat after correction even **under motion**;
  event amplitudes preserved; Z and unrecoverable frames still gated.

## Non-goals (YAGNI)

- **Axial (Z) recovery** — unchanged from v1; still gated.
- **Deep-learning non-rigid registration** (BrainAlignNet / StabiFormer) — future.
- **Ratiometric / multi-channel**; **cross-session registration**.

## Decisions

| Decision | Choice | Notes |
|---|---|---|
| Primary approach | Activity-independent-landmark tracking + ROI relocation (sparse) | Acceptance bar = sparse gut EE cells. |
| Secondary approach | Landmark-mesh Delaunay piecewise-affine warp (dense) | Only when density/topology gates pass; else disabled. |
| Registration features | Activity-minimized only (temporal-min/low-pct template; mask high-ΔF pixels) | Carried from v1. |
| Landmark + observability | Dark nucleus / fallback geometry, **gated by a per-frame observability SNR test** | If landmark contrast < threshold → not correctable → gate. |
| Large / cumulative motion | Frame-to-frame propagated locate (seed from previous frame) | Removes v1's single ±window ceiling; each step carries a peak/confidence. |
| Disappearance | Neighbour-deformation interpolation **only when valid** (below); constant-velocity fallback is **low-confidence → gated** | Never an accepted corrected trace on its own. |
| Confidence | Every corrected frame has a confidence in [0,1]; below `conf_floor` → gate | Single knob the QC/coverage path consumes. |
| Post-correction QC | Re-run v1 `gate_traces` on corrected output | Correction trusted only where v1 gates still pass. |

## Confidence & gating rules (operational)

**Landmark observability (per cell, per frame).** Detectable only if the
landmark/cell contrast over local background exceeds an SNR floor (calibrated on
the harness). A silent, low-baseline cell whose cytoplasm falls to background is
**not observable** → that frame is a candidate for neighbour interpolation, else
gated. Never "locate" from pure noise.

**Neighbour-deformation interpolation validity.** Accepted only when ALL hold,
else the frame is gated (low confidence):
- ≥ `min_neighbours` (default 3) tracked neighbours that are themselves
  high-confidence this frame;
- the missing cell lies **inside the convex hull** of those neighbours (no
  extrapolation; rejects one-sided neighbour sets);
- the neighbours' displacements fit a **local affine with low residual**
  (rejects shear/fold/slip/independent motion — high residual ⇒ gate);
- identity uncertainty of the neighbours is low.
Constant-velocity prediction is **only** a seed for re-acquisition, never an
accepted corrected value under deformation.

**Dense warp topology/density gates.** The Delaunay piecewise-affine path is
enabled only when: landmark density ≥ threshold; triangle quality (min angle)
acceptable; per-triangle strain ≤ limit; **no fold/flip** (all transform
Jacobian determinants > 0); defined boundary behaviour (no extrapolation beyond
the hull). If under-constrained → dense warp **disabled → gate**.

## Threshold defaults & freeze protocol

Concrete starting values, **frozen** for the synthetic acceptance run (re-calibrated
only by the protocol below, then re-frozen and recorded in run metadata).

| Threshold | Default |
|---|---|
| `R_max` (regime R1 max lateral drift) | 8 px (~1.5× cell radius) |
| observability SNR floor | cell-vs-local-bg contrast ≥ 3·σ_bg |
| `min_neighbours` | 3 |
| neighbour local-affine residual (valid) | RMS < 1.0 px |
| neighbour identity/link confidence (valid) | ≥ 0.8 |
| `conf_floor` (emit vs gate) | 0.5 |
| confidence-vs-dynamics bound | \|Pearson(confidence, ΔF/F0)\| < 0.2 |
| dense-warp landmark density | ≥ 1 landmark per (50 px)² |
| dense-warp triangle min angle | ≥ 20° |
| dense-warp per-triangle affine singular values | within [0.67, 1.5] |
| dense-warp fold check | all transform Jacobian determinants > 0 |
| coverage-gain (R1, pre-registered) | v2 ≥ v1 + 30 pp **and** v2 ≥ 90% |
| v1 artifact ceiling (reused) | min(5% ΔF/F0, 20%×smallest transient), below event-detector threshold |

**Calibration & freeze protocol:** if a value is re-calibrated, it is done by
sweeping the synthetic harness over regime R1 **only**, choosing the value that
holds reqs 1–3 & 6, then **frozen before** the acceptance battery runs and
recorded in run metadata (pre-registration). No threshold is tuned on the
acceptance run itself.

## Requirements & acceptance criteria

Thresholds are synthetic-calibrated starting points tied to **declared regimes**;
the binding criteria are downstream impact. "Pass" = per-cell unless stated;
report distributions, not just means.

| # | Requirement | Acceptance criterion |
|---|---|---|
| 1 | Motion residual | In regime R1 (lateral drift ≤ `R_max` px, ≥3 surrounding neighbours): per-cell recovered-centroid error median < 1 px **and** corrected-ROI vs true-footprint IoU > 0.8 |
| 2 | Trace recovery | In R1: per-cell Pearson(recovered ΔF/F0, true ΔF/F0) > 0.95 **and** RMSE below the v1 artifact ceiling; recovered-trace RMSE on moving data ≤ 1.5× the still-data RMSE |
| 3 | Event-amplitude preservation | Recovered event peak ΔF/F0 within 20% of true peak (median) — warp must not suppress/spread events |
| 4 | Coverage gain (regime-scoped) | On R1 synthetic data v1 gated heavily: mean usable coverage rises by the pre-registered margin (≥ +30 pp **and** v2 ≥ 90% in R1; see defaults table); reported per regime |
| 5 | Disappearance handling | A cell silent over a window while the field deforms, with valid surrounding neighbours, is relocated within req-1 residual; with invalid neighbours it is **gated** (not mis-corrected) |
| 6 | Honest non-regression (hard) | (a) static negative control under the **same motion field** reads flat after correction (max \|ΔF/F0\| < v1 artifact ceiling); (b) v1 event-preservation ≥ 95% still holds on corrected data; (c) \|Pearson(per-frame confidence, ΔF/F0)\| < 0.2 (confidence must not track activity) |
| 7 | Tracking identity | IDF1 / purity / fragmentation / gap-closure reported; targets dataset-specific (not a universal hard gate) |
| 8 | Z / unrecoverable scope | Out-of-plane and low-confidence frames still gated; warp confidence reported; no recovery claim for them |

## Architecture

```
v1 ROI + activity-minimized template (reuse)
  → per-cell landmark detect + OBSERVABILITY gate (SNR floor)
  → frame-to-frame propagated locate (peak/confidence per step)
  → btrack linking + gap-closing → trajectories
       → gap frames: neighbour-deformation interpolation IF valid (hull+affine
         residual+identity); else gate. Const-velocity only seeds re-acquisition.
  → SPARSE: relocate ROI along trajectory (confident frames) → ΔF/F0
    DENSE:   Delaunay piecewise-affine field IF topology/density gates pass;
             warp movie → fixed ROIs → ΔF/F0; else disabled → gate
  → re-run v1 gate_traces on corrected output
  → ΔF/F0 (rolling F0) + coverage/longest-run + per-frame confidence + motion record
```

## Validation

Extend the synthetic harness with per-frame **ground-truth cell positions/
footprints** (keep true ΔF/F0) and add the failure modes Codex flagged:
- **moving negative control**: a static non-signalling object carried through the
  same motion field — must read flat after correction (hard gate, req 6a);
- **spatial heterogeneity** + **held-out / varied activity patterns** so recovery
  isn't tuned to one activity realisation;
- **event-amplitude** ground truth (req 3);
- **disappear-while-moving** with both *valid* (surrounding) and *invalid*
  (one-sided / shear) neighbour configurations (req 5);
- **under-constrained dense** scenes (sparse/folded) to confirm the warp gates
  disable rather than over-warp.
Metrics reported per declared regime.

**Real-data acceptance gate (release gate, not the code-merge gate).** v2 code may
merge on synthetic GO; *real-data claims* require this. Minimum set (extending
`docs/calcium_manual_reference_labels.md`): **≥ 3 recordings spanning ≥ 2 tissue/
motion regimes** (e.g. gut EE + epithelium), each with hand-tracked identities and
**≥ 10 hand-drawn ROI masks** across representative frames, plus a labelled
moving non-signalling region. **Pass thresholds:** median centroid residual < 2 px
vs hand-tracking; coverage gain ≥ +20 pp vs v1; moving-negative-control
max \|ΔF/F0\| < the v1 artifact ceiling; IDF1 reported (dataset-specific target
agreed before scoring); ≥ 1 labelled Z-gated failure shown to be gated; and an
expert spot-check of recovered-vs-raw traces on ≥ 1 clip. Synthetic GO is
necessary but **not** sufficient for the real-data claim.

## Honest limits

- All cells silent simultaneously while the field deforms → no observable
  landmarks/valid neighbours → frames gated (degrades to v1).
- Sparse + large non-rigid / one-sided neighbours → interpolation invalid → gate.
- Dense + under-constrained → warp disabled → gate.
- Z-motion still unrecoverable.

## Mapping to existing code

| Component | Reuse | New |
|---|---|---|
| Landmark / locate | `calcium_qc.locate_cell` | observability gate + propagated locate (`calcium_motion.py`) |
| Linking + gaps | `track_cells` / btrack | validated neighbour-deformation interpolation (`calcium_motion.py`) |
| Warp | scikit-image `PiecewiseAffineTransform` + scipy `Delaunay` | dense path + topology/density gates (`calcium_motion.py`) |
| Post-correction QC | `calcium_qc.gate_traces` | re-run on corrected output |
| ΔF/F0 | `normalize_timecourse` (rolling) | — |
| Validation | `calcium_synth`, `calcium_validation` | per-frame GT positions, moving neg-control, event-amplitude, valid/invalid-neighbour cases |
| Tool | `assess_calcium_timecourse` | `correct_calcium_motion` (sparse track + gated dense warp) |

## Codex review incorporated (rounds 1-2, 2026-06-25)

Added a Core safety principle (confidence-gated correction; degrade to v1 on
failure) and an operational Confidence & gating section addressing all 6 NO-GO
blockers: (1) landmark-observability SNR gate; (2) neighbour-interpolation
validity conditions (hull containment, local-affine residual, identity, ≥3
neighbours) with constant-velocity demoted to a seed-only/low-confidence role;
(3) hard, regime-scoped acceptance definitions (per-cell, distributions, declared
R1 regime, pre-registered coverage margin); (4) single-channel controls — moving
negative control, event-amplitude preservation, confidence-vs-dynamics bound;
(5) dense-warp density/triangle-quality/strain/fold gates; (6) a defined real-data
acceptance set extending the manual-reference deliverable.

**Round 2:** made the remaining gates operationally binding — a concrete frozen
threshold-defaults table + a calibrate-on-R1/freeze-before-acceptance
(pre-registration) protocol; and a fully-specified real-data acceptance gate
(≥3 recordings / ≥2 regimes, hard pass thresholds, annotation, expert spot-check),
separated as a release gate distinct from the synthetic code-merge gate.

## References

PubMed DOIs: colon GECI [10.3389/fncel.2015.00436](https://doi.org/10.3389/fncel.2015.00436);
BEAS [10.1038/s41598-021-90448-4](https://doi.org/10.1038/s41598-021-90448-4);
EMC2 [10.1371/journal.pcbi.1009432](https://doi.org/10.1371/journal.pcbi.1009432);
C. elegans [10.1371/journal.pcbi.1005517](https://doi.org/10.1371/journal.pcbi.1005517);
BrainAlignNet [10.7554/eLife.108159](https://doi.org/10.7554/eLife.108159);
NoRMCorre / PatchWarp / StabiFormer (web).
