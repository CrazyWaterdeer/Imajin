# Imaris → Imajin gap analysis

Research record comparing Imaris (Oxford Instruments) against Imajin, to steer
what to add or improve. Captured 2026-07-08. Sources at the bottom.

## One-line conclusion

Of Imaris's four core detection models — **Spots · Surfaces · Cells ·
Filaments** — Imajin is missing **Spots** (point/puncta detection) entirely and
has no **object-to-object spatial-relationship** layer on top of its
segmentations. Filaments exist only as *skeleton-of-a-mask morphometry*, not as
a connectivity-aware **tracer**. These are the highest-value gaps. Deconvolution
is the best standalone pre-analysis quality lift. Everything else is
incremental.

## Where Imajin already leads Imaris (so we calibrate)

Imaris is ahead on **detection primitives**; Imajin is ahead on **rigor and
reproducibility**. Do not rebuild what we already do better:

- **Honest statistics**: assumption-aware test selection (Shapiro → parametric/
  non-parametric), paired-design detection, pseudoreplication warnings,
  multiplicity-corrected post-hoc. Imaris's stats (Vantage) are basic by
  comparison.
- **Reproducibility**: result bundles, replayable recipes, batch resume.
- **Publication figures**: scientific typography, colorblind-safe palette,
  paired lines, significance brackets.
- **Dual interface**: manual dock ↔ LLM chat sharing one provenance log.

Strategic thread: **import the detection primitives Imaris has that we lack
(spots, distances, deconvolution, a real tracer), and feed them into the
analysis/reporting layer we already do better.**

## Model / module comparison

| Imaris | Imajin today | Verdict |
|---|---|---|
| **Spots** (puncta, FISH, vesicles, viruses) | none (`blob` is a QC class label; `peak_local_max` is watershed-internal) | **core gap** |
| **Surfaces** (arbitrary 3D segmentation) | Cellpose 2D/3D + intensity-region domains | covered |
| **Cells** (parent cell = nucleus + cytoplasm + vesicles) | `partition_inside_outside`, `classify_labels_by_mask` (overlap) | partial (no hierarchy / per-parent counts) |
| **Filaments** (neuron/vessel tracing) | `enhance_neural_processes` (Frangi/Sato) → `segment_neural_processes` → `skeletonize` → branch/Sholl/SWC/classify | **partial: enhancement + morphometry present, but no connectivity-aware tracer, no rooted tree, no diameter, no spines** |
| **Spatial relations** (spots-in-surface, distance, nearest-neighbor) | `mask_logic` only; no distance/NN (`cKDTree`/`cdist` unused) | **core gap** |
| **ClearView deconvolution** | rolling-ball + Gaussian denoise only (`skimage.restoration` already a dep) | gap |
| **ML object classification** (train-by-example on object stats) | morphology neuron-type kNN (traces only) | partial |
| **Tracking** (speed/MSD/division/lineage/editing) | `track_cells` (btrack); little motion stats exposed | partial |
| **Vantage** (interactive N-D gating) | static publication figures | partial |
| **Stitcher** (tile mosaic) | internal plane→3D label stitch only | gap (low fit for chat-driven single-FOV) |

## Ranked backlog

### Tier 1 — highest value, strong fit, tractable

1. **Spot / puncta detection** — the missing fourth model. `detect_spots`
   via `skimage.feature.blob_log`/`blob_dog`, 2D/3D, voxel-scale aware,
   boundary-mask aware. Output: napari Points layer + table (subpixel coords,
   per-channel intensity, size, SNR). Feeds measure → stats → figures → report.
2. **Object-to-object spatial relationships** — the payoff of (1). Per-parent
   child counts ("spots per cell"), distance-to-nearest-surface
   (`distance_transform_edt`), nearest-neighbor distances
   (`scipy.spatial.cKDTree`). Turns two detections into biology.
3. **Deconvolution** — `deconvolve` Richardson-Lucy with a theoretical PSF
   from NA / emission wavelength / refractive index / voxel spacing (all in
   metadata we already read); Gaussian PSF fallback. `skimage.restoration`.
   Independent of 1/2 — good parallel first pick. Cap iterations, warn on noise
   amplification.

### Filament recognition & analysis — elevated to committed scope (user-flagged)

Current path is threshold-and-skeletonize: it depends on the vesselness image
thresholding into one clean connected mask, then morphological skeletonization.
Weaknesses Imaris's tracer specifically solves:

- **No gap bridging / no connectivity model** → fragmented skeletons at faint
  spans and crossings.
- **No rooted directed tree** → no valid branch order / Strahler / path-to-soma,
  and SWC parent pointers are not truly tree-ordered.
- **No dendrite/vessel diameter** (the "radius" in morphometry is the Sholl ring
  radius, not local thickness).
- **No dendritic-spine detection.**

Work splits into (T1) connectivity-correct tracing — skeleton→graph
(`skan`/`sknw`), endpoint gap-bridging (cKDTree within `max_gap_um`), root at
soma, optional minimal-path AutoPath (`skimage.graph` on inverse-vesselness
cost) — and (T2) richer analysis — diameter profile (EDT along nodes), spine
detection (reuse the spot detector on backbone-subtracted residual), tree
topology metrics, SWC/CSV export with radius + spines.

### Tier 2 — sharpen existing strengths

4. **Object-based colocalization + Costes** — extend `coloc.py`: Costes
   automatic threshold + randomization significance (p-value), object-level
   co-occurrence and NN-distance-vs-random. Nearly free once spots + spatial
   relationships exist. Fits our honest-statistics stance.
5. **General object classification** — `classify_objects(features, rules |
   examples)` over any measurement table; write `class` back to labels. High
   synergy with the LLM driver (say the rule instead of clicking).
6. **Track motion analysis** — per-track speed, net vs total displacement,
   directionality, MSD slope, duration; division/lineage from btrack; track
   filtering/QC.

### Tier 3 — bigger or narrower

7. **Vantage-style interactive gating** on scatter (draw gate → subpopulation).
8. **Tile/mosaic stitching** (heavy; weak fit for chat-driven single-FOV).

## Sources

- Imaris overview — https://imaris.oxinst.com/
- Core Facilities feature list — https://imaris.oxinst.com/products/imaris-for-core-facilities
- Spots detection tutorial — https://imaris.oxinst.com/learning/view/article/object-detection-in-imaris-using-spots-basic
- Filter Spots Inside/Outside Surfaces — https://imaris.oxinst.com/learning/view/article/imaris-9-5-filter-spots-inside-or-outside-of-surfaces
- ML classification (9.9) — https://imaris.oxinst.com/versions/9-9
- Filament Tracer (10.0) — https://imaris.oxinst.com/versions/10
- Imaris for Neuroscientists — https://imaris.oxinst.com/products/imaris-for-neuroscientists
