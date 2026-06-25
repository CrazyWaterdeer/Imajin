# Calcium imaging — manual reference labels (deliverable)

Some v1 acceptance metrics need labelled **real** data (spec reqs 2, 6, 7) and
cannot be computed without a hand-labelled reference set. Until it exists, those
metrics are validated on synthetic data only via
`imajin.analysis.calcium_validation.run_v1_acceptance`.

Produce the following on **≥2 hard (deforming-gut) clips** and store under
`tests/data/calcium_ref/`:

## 1. Per-frame defocus labels (req 2 — defocus recall/precision)
`defocus_labels.csv` with columns:
- `clip` — clip id / filename stem
- `frame` — frame index (int)
- `cell` — cell/ROI label (int)
- `defocus` — `1` if the cell is out of focus in this frame, else `0`

## 2. Hand-tracked cell identities (req 7 — IDF1 / purity / fragmentation, v2)
`identities.csv` with columns:
- `clip`, `frame`, `cell` — the tracked identity
- `y`, `x` — hand-marked centroid (pixels)

## 3. Hand-drawn ROI masks (req 6 — motion residual / recovered-trace error, v2)
Label TIFFs `masks/<clip>_<frame>.tif` (uint16, `0` = background, value = cell id)
for **≥10 cells** across representative frames.

## Scope note
This set lets the harness compute defocus recall/precision (v1) and the tracking
and motion-residual metrics (v2) on real data. **v1 ships with synthetic-only
validation**; producing this labelled set is a prerequisite for any real-data
acceptance claim and for the v2 motion-correction plan.
