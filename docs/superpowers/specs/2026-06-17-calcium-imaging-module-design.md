# Calcium Imaging Analysis Module

**Status**: Draft (revised after 2 Codex reviews, 2026-06-17)
**Date**: 2026-06-17
**Type**: Design

## Problem

Imajin needs a calcium-imaging analysis module: load a time-series `.tif`/`.lsm`
recording, measure per-ROI fluorescence over time, and produce ΔF/F0 traces and
graphs. Users also need **ROI tracking** because cells move during a recording —
the user's mental model is *appear → assign position, disappear → hold position,
reappear → move position*.

The imaging regime is the hard part and shapes every decision:

- **Live, unstained tissue.** No fixation, no immunostain → **no structural /
  counterstain channel**, and **no co-expressed static (e.g. red) marker**.
  A single functional (GCaMP-type) channel only.
- **Usually a single focal plane over time (2D+T).** TIFF-saving camera systems
  cannot acquire z-stacks; LSM confocal can but rarely does, because temporal
  resolution is prioritized.
- **Multiple tissues.** Not only brain: gut enteroendocrine (EE) cells, epithelia,
  and other Ca-signalling tissue.
- **Motion.** Multi-directional in-plane deformation **plus substantial axial (Z)
  motion from peristalsis** (gut), which moves structures out of the focal plane.

A naive implementation is dangerous, not merely incomplete: it will emit ΔF/F0
traces that *look plausible* but are contaminated by motion/defocus. The bar is
**scientific trustworthiness**, not "the feature runs."

### Already in Imajin (reused, not rebuilt)

`measure_intensity_over_time` (`tools/measure.py`), `normalize_timecourse` with
`f_over_f0` / `delta_f_over_f0` and a fixed-window F0 (`tools/stats.py`),
`plot_timecourse` (`tools/figures.py`), `extract_timecourse_features`,
`compute_timecourse_qc` (`tools/qc.py`), Cellpose segmentation, `track_cells`
via btrack (`tools/track.py`), and `interactive_roi` ROI editing. The ΔF/F0
skeleton therefore already exists. The genuinely new work is **robust F0**,
**honest motion/defocus QC + coverage**, a **validation harness**, and (v2) the
**motion/tracking layer**.

## Goals

- **Staged delivery**, with the acceptance bar set against the **hardest real
  case** (deforming gut, sparse cells), not the easiest — defining "usable"
  against low-motion brain data is the trap that yields an unusable module.
- **v1**: a validated ΔF/F0 + F0 + QC core with **honest gating for BOTH defocus
  and lateral motion**. Handles low-motion data *once its lateral-stability and
  focus QC pass*; on deforming-gut data it **safely rejects** uncorrectable
  frames/cells and reports coverage, rather than emitting contaminated traces.
- **v2**: a motion/tracking/registration layer that **actually recovers**
  deforming-gut (sparse-cell) data and meets the bar above.
- Every emitted output is gated by **quantitative acceptance criteria** and a
  **validation harness**.

## Non-goals (YAGNI)

- **Recovering axial (Z) motion from single-plane single-channel data.** Physically
  impossible (defocus intensity change is confounded with activity). v1 *gates*
  it; it does not "fix" it. Acquisition-side guidance is documented instead.
- **Deep-learning non-rigid registration** (BrainAlignNet/StabiFormer class) in the
  v1 or v2 baseline — kept as an explicit future upgrade path.
- **Spike deconvolution / CNMF source extraction.** Measurement stays ROI-based.
- **Cross-session / multi-day cell registration** (CellReg / Track2p territory).
  Single-recording only.
- **Ratiometric correction** — no second channel exists.
- **New file IO** — reuse existing LSM/TIFF/OME loaders.

## Decisions

| Decision | Choice | Notes |
|---|---|---|
| Delivery | Staged: v1 (core + QC) → v2 (motion) | De-risk; ship a usable, honest core first. |
| Acceptance bar | A = deforming gut, sparse cells | Hardest real case; subsumes brain/epithelium. |
| v1 motion handling | None; honest **defocus QC + lateral-motion/ROI-validity QC** + coverage/reject | Catches both blur AND sharp-but-sliding cells. |
| v1 footprint source | **Detection-only, per frame**: locate cell in a search window around its ROI via normalized cross-correlation of an activity-minimized ROI template + intensity-weighted centroid; footprint via in-window threshold or coarse-cadence Cellpose re-seg; identity = fixed ROI (1:1, no re-linking in v1) | Without a defined footprint source the lateral gate is not operational. |
| Unlocatable frame | Cell not confidently located (weak xcorr peak / empty or failed segmentation) → frame marked **unreliable and gated** (counts against coverage) | Segmentation failure must not silently pass. |
| Defocus discriminator | **Multi-metric**: intensity-normalized Laplacian/Tenengrad + local SNR + ROI area/shape + temporal consistency | Raw Laplacian variance alone is intensity/activity-confounded. |
| Lateral-motion QC | Mask IoU(located footprint, ROI) + centroid drift + boundary-crossing per frame | Sharp slide produces false ΔF/F0; focus QC misses it. |
| v2 motion approach | (a) BEAS-style double-contour + (b) landmark-mesh warp first; (c) DL later | Classical, in-stack, gut-proven. |
| v2 registration features | **Activity-minimized only**: temporal-min / low-percentile template, active-pixel masking, geometry / dark-nucleus — never raw activity | Activity-dependent intensity violates registration assumptions. |
| Landmark source | Dark nucleus (cytoplasmic GECI) **with fallback**: stable-boundary geometry / low-percentile structural image / user landmarks; detection-quality check switches modes | Dark-nucleus is expression/saturation/overlap dependent. |
| F0 estimation | Rolling-percentile (default 10th pct; window ≈ several × **event duration & duty cycle**); keep fixed-window option | Window tied to event duration, not only inter-event interval. |
| Threshold philosophy | Numeric defaults are **calibration starting points validated on the synthetic harness**, calibrated per dataset (ROI size/shape, indicator, noise) — not universal constants | Binding criteria are downstream impact, not the raw constants. |
| Artifact ceiling | `< min(5% ΔF/F0, 20% × smallest accepted transient)` AND below the **event-detector decision threshold** (set from measured per-ROI noise/SNR) | A fixed % is not universally safe. |
| Coverage policy | Report per-cell coverage % **and missingness pattern**; reject when event/stimulus windows are uncovered or no minimum contiguous usable run exists; flat 50% is a coarse secondary floor only; report coverage distribution to expose active/sharp selection bias | 51% coverage that misses the active window is still unusable. |
| Validation | Synthetic GT (full failure-mode set) + positive controls + negative-control-flat + manual reference | Negative control is a hard pass/fail gate. |
| Sparse vs dense | Tracking for sparse cells; whole-field warp for dense sheets | Cell density selects the v2 tool. |

## Imaging regime & constraints (authoritative)

- Single functional channel; no structural/static reference channel available.
- Single focal plane time series (2D+T) is the common case; 3D+T is rare.
- In-plane motion is non-rigid (contraction/elongation), not just translation.
- Axial motion (peristalsis) takes cells out of plane — unrecoverable in software
  from single-plane single-channel data.
- Cytoplasmic GECIs *often* leave the nucleus dark → an activity-independent
  landmark may exist inside the single channel; this is not guaranteed (NLS
  indicators, saturation, overlap) and needs a fallback.

## Requirements & acceptance criteria

Numeric thresholds below are **calibration starting points validated on the
synthetic harness and calibrated per dataset**, not universal constants. The
binding criteria are downstream impact: residual trace error below the artifact
ceiling, event-preservation above target, and event/stimulus windows covered.
Metrics that require labelled data are listed under Validation.

| # | Requirement | Acceptance criterion | Stage |
|---|---|---|---|
| 1 | **Lateral-motion / ROI validity** | Per frame, cell located (per "v1 footprint source"); mask IoU(located footprint, ROI) ≥ threshold (default 0.7, **calibrated so residual trace error stays below the artifact ceiling**) and centroid drift < 0.5 ROI radius; frames that fail, or where the cell cannot be located, are gated | v1 |
| 2 | **Defocus gating** | Multi-metric focus score; out-of-focus detection recall ≥ 0.9 and precision ≥ 0.8 vs labelled frames; **event-preservation ≥ 95% of synthetic ground-truth events survive gating (binding, not just reported)** | v1 |
| 3 | **Coverage + bias** | Per-cell coverage % and missingness pattern reported; reject when event/stimulus windows uncovered or no minimum contiguous usable run (flat 50% is a coarse secondary floor); coverage distribution reported to surface active/sharp selection bias | v1 |
| 4 | **Artifact ceiling** | Residual ΔF/F0 on a non-signalling structure `< min(5% ΔF/F0, 20% × smallest accepted transient)` and below the event-detector decision threshold (from measured noise/SNR) | v1 (gate) / v2 (recover) |
| 5 | **F0 robustness** | On a synthetic battery (plateaus, bursts, long quiescence, drift, bleaching, variable duty cycle): baseline bias < a few % of true F0 and transient-preservation quantified (no clipping of sustained signals) | v1 |
| 6 | **Motion residual** | Post-correction mask IoU / contour error within tolerance and recovered-vs-true trace error small — not centroid jitter alone | v2 |
| 7 | **Tracking identity** | IDF1, track purity, fragmentation, gap-closure accuracy all reported; targets dataset-specific (IDF1 ≥ 0.95 aspirational for low-motion, relaxed for deforming gut/disappearing cells), not a universal hard gate; gap tolerance expressed in time & motion | v2 |
| 8 | **Validation gate (hard)** | Negative-control structure reads flat per the quantitative definition below — zero spurious detected events | v1 |

## Architecture

### v1 pipeline (core + honest QC)

```
load (existing IO)
  → ROI definition (Cellpose / interactive_roi, existing)
  → per-frame intensity over T (measure_intensity_over_time, existing)
  → F0: rolling-percentile (NEW) → ΔF/F0 (normalize_timecourse, existing)
  → QC (NEW), two independent gates per frame per cell:
       (a) lateral-motion/ROI-validity:
             locate cell in a search window (xcorr of activity-minimized template
             + intensity-weighted centroid; footprint via in-window threshold or
             coarse-cadence Cellpose); identity = fixed ROI (1:1)
             → mask IoU(footprint, ROI), centroid drift, boundary-crossing
             → unlocatable (weak xcorr / empty seg) ⇒ unreliable ⇒ gate
       (b) defocus: multi-metric focus score (normalized Laplacian/Tenengrad + SNR + shape + temporal)
       → frame usable only if BOTH pass; else NaN
       → per-cell coverage % + missingness pattern; reject by pattern (event-window
         coverage, min contiguous run); report coverage distribution
  → plot / features (plot_timecourse, extract_timecourse_features, existing)
  + validation harness (NEW)
```

`compute_timecourse_qc` is extended with the lateral-motion and focus gates and
coverage/reject logic; `normalize_timecourse` gains a rolling-percentile F0 option
alongside the existing fixed-window method. v1 lateral QC is detection-only — it
gates, it does not correct (correction is v2).

### v2 motion/tracking layer

Inserted between ROI definition and intensity measurement:

```
landmark detection (dark-nucleus, with fallback to stable geometry / low-pct image)
  → link across frames (btrack, existing) with gap-closing
       → disappeared landmark position interpolated from neighbours' deformation
  → build registration target from ACTIVITY-MINIMIZED features:
       temporal-min / low-percentile template, with active (high-ΔF) pixels masked
  → per-frame deformation field (triangulated piecewise-affine, or optical flow)
  → warp movie to that reference
  → measure on stabilized movie with fixed/ring ROIs → v1 ΔF/F0 + QC path
```

### v2 approaches considered

| Approach | Mechanism | Fit | Cost |
|---|---|---|---|
| (a) BEAS-style double-contour | Track nucleus + cytoplasm contours; contours relocate ROIs | Sparse EE / enteric (gut-proven) | Light; in-stack |
| (b) Landmark-mesh warp | Activity-minimized landmarks → Delaunay piecewise-affine warp | Moderate-density whole-field stabilization | Light; scikit-image + scipy |
| (c) DL non-rigid registration | Learned dense deformation field | Dense / severe deformation | Heavy; GPU, labelled data + validation set required |

**Recommendation**: v2 starts with the (a)/(b) family (classical, in-stack,
validated on this tissue); (c) is a documented upgrade path, gated on having
labelled data and a validation set.

## Validation strategy

The harness is what separates "implemented" from "usable". It must exercise the
single-channel, activity-dependent failure modes explicitly, and it is also where
the numeric thresholds are calibrated.

1. **Synthetic ground truth** — movies with known motion fields and known ΔF/F0
   transients, simulating the full failure-mode set: axial defocus, lateral
   non-rigid deformation, photobleaching, shot noise, background autofluorescence,
   sparse and dense cells, overlapping cells, activity-driven brightness change,
   and disappearing/reappearing cells. Yields reqs 1–2 (gating accuracy +
   event-preservation), 4 (artifact), 5 (F0), 6 (residual), 7 (tracking), and the
   IoU/coverage threshold calibration. A synthetic **injected negative control**
   (static non-signalling object) is always present, so req 8 is always computable
   here.
2. **Negative control (hard gate)** — define "flat" quantitatively: max |ΔF/F0|
   below the artifact ceiling, no events from the event detector, bounded
   low-frequency drift. Acceptable real controls: non-expressing region,
   background annulus, autofluorescent/dead ROI, or pharmacological-silence clip.
   On real recordings lacking any such structure, the gate is best-effort and the
   limitation is reported; the synthetic injected control still enforces it.
3. **Positive controls** — clips with known stimulus timing / pharmacology and
   deliberately motion-corrupted clips, to confirm real transients survive gating
   and that corruption is caught.
4. **Manual reference** — required labels: hand-labelled defocus frames,
   hand-tracked cell identities, and hand-drawn ROI masks/traces on hard gut
   clips. Without these the acceptance metrics (reqs 2, 6, 7) cannot be computed,
   so producing this labelled set is part of v1/v2 scope.

## Honest failure handling (axial / Z motion)

Software cannot recover out-of-plane motion from single-plane single-channel data.
The module's value here is honesty:

- The defocus gate (req 2) flags out-of-focus timepoints; the lateral gate (req 1)
  flags sharp-but-displaced or unlocatable ROIs. Both contribute to coverage.
- Gated timepoints become NaN; per-cell coverage %, missingness pattern, and its
  distribution are reported; cells failing the coverage pattern are rejected.
- Acquisition-side guidance for users: low-NA / large depth-of-field optics
  tolerate axial excursions; higher frame rate reduces per-frame drift; a thin
  z-stack per timepoint (if temporal budget allows) enables best-plane reselection.

## Mapping to existing Imajin code

| Component | Existing | New |
|---|---|---|
| File IO | `io/*` loaders | — |
| ROI definition | Cellpose, `interactive_roi` | per-frame cell-locate (v1 QC); dark-nucleus/landmark detect + fallback (v2) |
| Per-frame intensity | `measure_intensity_over_time` | — |
| F0 / ΔF/F0 | `normalize_timecourse` | rolling-percentile F0 (window tied to event duration) |
| Plotting / features | `plot_timecourse`, `extract_timecourse_features` | ΔF/F0 raster/heatmap (optional) |
| QC | `compute_timecourse_qc` | lateral-motion gate, multi-metric focus gate, coverage pattern/reject + bias report |
| Tracking | `track_cells` (btrack) | landmark linking + neighbour-deformation interpolation (v2) |
| Motion warp | — | activity-minimized template + deformation field + warp (v2) |
| Validation | — | synthetic (full failure modes) + positive/negative controls + manual-ref harness |

## Codex reviews incorporated (2 rounds, 2026-06-17)

**Round 1** (read-only Codex, gpt-5.5): added v1 lateral-motion/ROI-validity QC
(focus gating alone misses sharp-but-sliding cells); multi-metric focus
discriminator; precision/FPR + coverage-bias reporting; relative artifact ceiling;
IoU/contour + trace-error motion residual; IDF1/purity tracking metrics; F0
synthetic battery; activity-minimized registration features (not raw activity);
dark-nucleus fallback; expanded validation harness.

**Round 2** (verification pass, returned NO-GO with blockers; all fixed here):
defined the v1 per-frame **footprint source** and the unlocatable-frame ⇒ gate
rule (lateral QC was otherwise non-operational); made **event-preservation a
binding ≥95% criterion** (was merely "reported"); reframed numeric thresholds as
**synthetic-calibrated, dataset-specific** rather than universal constants;
replaced flat 50% coverage with **missingness-pattern + event-window coverage +
minimum contiguous duration**; made IDF1 dataset-specific; tied the artifact
ceiling to the **event-detector decision threshold from measured noise/SNR**.

## References

Literature retrieved via PubMed (DOIs linked) and web sources.

- Gut/ENS GECI tracking (dark-nucleus + distortion vector maps), intact moving colon —
  [10.3389/fncel.2015.00436](https://doi.org/10.3389/fncel.2015.00436)
- BEAS double-contour (nucleus + cytoplasm) cell tracking in contractile ENS tissue —
  [10.1038/s41598-021-90448-4](https://doi.org/10.1038/s41598-021-90448-4)
- EMC2: elastic motion correction, neighbour-deformation interpolation of disappearing cells; validated on Hydra contractions —
  [10.1371/journal.pcbi.1009432](https://doi.org/10.1371/journal.pcbi.1009432)
- Tracking neurons in a moving/deforming brain (C. elegans), non-rigid point-set registration —
  [10.1371/journal.pcbi.1005517](https://doi.org/10.1371/journal.pcbi.1005517)
- CRASH2p: closed-loop 3D motion correction + ratiometric (hardware ceiling for axial motion) —
  [10.1038/s41467-025-60648-x](https://doi.org/10.1038/s41467-025-60648-x)
- BrainAlignNet: DNN non-rigid registration for moving/deforming nervous systems —
  [10.7554/eLife.108159](https://doi.org/10.7554/eLife.108159)
- NoRMCorre (piecewise-rigid) — https://www.sciencedirect.com/science/article/pii/S0165027017302753 ;
  PatchWarp (piecewise-affine) — https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(22)00063-7 ;
  StabiFormer (transformer optical flow, demonstrated on intestine) — https://doi.org/10.1002/jbio.202500407
- Suite2p — https://github.com/MouseLand/suite2p ; CaImAn — https://elifesciences.org/articles/38173 ;
  Laplacian-variance focus metric — https://link.springer.com/article/10.1007/s12022-025-09893-w
