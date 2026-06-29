# ROI Bounding-Box Crop for Boundary-Masked Segmentation — Design

Status: design (revised after one Codex review; ready to plan)
Date: 2026-06-29

## Problem

With `boundary_mask` set, results are confined to the ROI but the **computation runs full-frame and is
clipped only at the end**. In the user's two-tier CaLexA flow:

- **Tier-1 `segment_expression_domain` has no `boundary_mask` at all** -> the expression domain is
  segmented and *measured* over the whole frame. This is the visible "analysis outside my region."
- **Tier-2 `segment_target_objects`**: `prepare_corrected` (opening radius ~48 + smoothing) and
  `labels_from_binary` (watershed) run on the full `raw`, then intersect with the boundary. ~85% of the
  heavy compute is wasted for a 15% ROI in a 63x2048x2048 stack.

The user wants intensity-based ROI detection computed **only inside the drawn region** (efficiency) and
nothing outside it measured (correctness).

## Scope (narrowed after Codex review)

This change covers the two tools in the user's flow: **`segment_target_objects` (single-shot) and
`segment_expression_domain`**, plus two-tier threading. `auto_segment_target` and
`segment_3d_cells_auto` are **out of scope** here: their auto-correction trajectory and candidate
ranking depend on scope-relative QC metrics (`mask_fraction`, `top_bright_outside_fraction`, ROI
score), so cropping changes the *result*, not just runtime. They keep today's full-frame behaviour.

## What crop can and cannot promise (corrected from the first draft)

The first draft's "bit-identical" claim was too strong (Codex). The honest, testable guarantees:

- **`segment_target_objects` is an optimisation, not a behaviour change, only under bounded settings.**
  Its threshold is already computed from inside-boundary pixels (`target_threshold_for_scope`), and the
  background opening/smoothing are local, so cropping the ROI bbox with a sufficient margin yields the
  **same label mask (`labels > 0`) inside the ROI**. This holds only when:
  - `background_radius > 0` (radius<=0 returns a **global** percentile -> skip crop), AND
  - `auto_mask_hyperbright=False` (hyperbright uses a **global** percentile -> skip crop when on), AND
  - `effective_min_size` is derived from the **full-frame** XY area (not the crop), AND
  - the boundary is non-empty and not whole-frame (else bbox is None/full -> no crop).
  When any condition fails, **skip the crop** (correctness over speed). Even when it holds, label *IDs*
  and area-relative QC (`mask_fraction` -> `roi_confidence`) become **ROI-local** and may differ; we
  guarantee the mask, not the IDs/metrics. Tests assert `(labels>0)` equality inside the ROI and equal
  object count.
- **`segment_expression_domain` with a boundary is an intentional behaviour change** (it ignored the
  ROI before): the domain is computed **ROI-locally** and measures ROI ∩ expression only.

## Margin (exactness for the target path)

`margin = 2*background_radius + ceil(truncate*smoothing_sigma) + pad`, with `truncate = 4.0` to match
`skimage.filters.gaussian`/`ndi.gaussian_filter` default kernel radius (Codex: not 3 sigma), `pad = 8`.
Opening influence is up to `2*radius`; percentile filter up to `radius`; `2*radius` covers both. Margin
also exceeds `min_distance` (watershed peak spacing), so seeds/basins for in-ROI objects are unaffected.

## Design

### 1. Pure helpers (`analysis/segmentation.py`)

- `boundary_bbox_slices(yx_mask_2d, raw_shape, margin) -> tuple[slice, ...] | None` — takes a **2D
  (Y,X)** mask (caller passes the original 2D ROI, never `any(axis=0)` over a broadcast view), returns
  the YX bbox expanded by `margin`, clipped to bounds, with full slices for any leading (Z) axes.
  `None` if the mask is empty or already spans the whole frame (caller then runs the normal path).
- `scatter_labels_to_full(cropped_labels, full_shape, slices) -> np.ndarray` — `int32` zeros, assign
  the crop, return. Called **after** the pipeline's own clip+renumber.

### 2. `segment_target_objects` (`tools/segment.py`)

Reorder so the boundary loads **before** `prepare_corrected`. When the boundary is present and the
crop conditions hold, compute `slices` (from the original 2D mask when the ROI was a 2D-on-3D
broadcast), crop `raw` + `boundary_bool`, run the unchanged `prepare_corrected` + `threshold_and_label`
on the crop with `min_size` derived from the **full-frame** XY area, then `scatter_labels_to_full`.
QC PNG / measurement / `add_labels` use full-size `raw` + scattered masks. `threshold`, `noise_sigma`,
`threshold_scope` come from the crop (inside-boundary stats, unchanged). When conditions fail, the
existing full-frame path runs untouched. `boundary_mask=None` path is byte-for-byte unchanged.

### 3. `segment_expression_domain` + `domain_segmentation.py`

Add `boundary_mask: str | None`. When set:
- `resolve_boundary_mask` -> bool (supports 2D-on-3D).
- Crop `raw` + boundary to the bbox (margin = `ceil(4*smooth_sigma_px) + pad`).
- Compute the noise floor on the **smoothed values inside the boundary** (ROI-local; documented change),
  build the binary, **clip to the boundary before** `remove_small_objects` / component filtering /
  dilation (Codex P1 — clip early and after dilation), then scatter labels back.
- Measured area/volume is ROI ∩ expression. QC `secondary_outline` shows the ROI.

### 4. Two-tier threading (`tools/workflows.py`)

`analyze_target_cells`: when `segmentation_options["boundary_mask"]` (the user's ROI) is present,
thread it into `_precompute_domain_layer` so Tier-1 is ROI-constrained, and set Tier-2's boundary to
**ROI ∩ domain** (intersection layer), not just the domain. If only a domain (no user ROI) exists,
behaviour is today's.

## Edge cases / failure modes (Codex)

- **Empty boundary** -> not "no boundary": bbox `None`, segmentation runs but the all-false mask yields
  an empty result (today's empty-mask handling), never a full-frame fallback.
- **2D-ROI-on-3D**: bbox from the original 2D mask; crop YX, keep all Z; no `Z*Y*X` scan.
- **Whole-frame / near-whole boundary**: bbox ~= frame -> skip crop (no benefit, avoids edge math).
- **ROI at image edge**: bbox clipped to bounds; margin may be one-sided -> still correct because the
  image edge is the real support boundary.
- **`split_touching` / `min_distance`**: margin > `min_distance`, and the binary is False in the margin
  (outside boundary), so in-ROI watershed basins are unchanged.

## Files

| File | Change |
| --- | --- |
| `src/imajin/analysis/segmentation.py` | `boundary_bbox_slices`, `scatter_labels_to_full` |
| `src/imajin/tools/segment.py` | crop/scatter in `segment_target_objects`; `boundary_mask` on `segment_expression_domain` |
| `src/imajin/analysis/domain_segmentation.py` | ROI-local noise floor + early boundary clip helpers |
| `src/imajin/tools/workflows.py` | thread ROI to Tier-1; Tier-2 boundary = ROI ∩ domain |
| `tests/test_tools_segment.py`, `tests/test_phase2_workflow.py` | guarantees below |

## Acceptance tests

- **Target mask invariance:** on a synthetic 3D image, `segment_target_objects(boundary_mask=ROI,
  background_radius=48)` gives the **same `labels>0` inside the ROI and the same object count** whether
  the crop runs or not (compare against a `_disable_crop` path or a whole-frame boundary). 
- **Locality:** bright signal placed **outside bbox+margin** cannot change the in-ROI result; assert the
  array passed to `prepare_corrected` has the cropped shape.
- **Skip conditions:** `background_radius=0` and `auto_mask_hyperbright=True` take the full-frame path
  (no crop) and match today's output exactly.
- **Domain:** `segment_expression_domain(boundary_mask=ROI)` has **zero domain voxels outside the ROI**;
  measured area shrinks vs the unconstrained run.
- **Two-tier:** `analyze_target_cells(segmentation_options={"boundary_mask": ROI}, domain_strategy=
  "noise_floor")` -> domain and cells both zero outside the ROI.
- **No-boundary regression:** all existing `boundary_mask=None` tests stay green.

## Non-goals

- Cropping `auto_segment_target` / `segment_3d_cells_auto` (scope-relative metrics change results).
- Per-Z ROIs; oblique/rotated crops. Cellpose internal cropping. Bit-identical label *IDs* or QC.

## Changelog — draft -> revised (accepted Codex findings)

- Dropped the universal "bit-identical" claim; scoped exactness to `segment_target_objects` single-shot
  under bounded settings (radius>0, no hyperbright, full-frame min_size), guaranteeing the **mask**, not
  IDs/QC; skip crop otherwise.
- Removed `auto_segment_target` / `segment_3d_cells_auto` from scope (scope-relative auto-decisions).
- Margin uses `4*sigma` (skimage truncate), `2*radius` for opening.
- Domain: ROI-local noise floor (inside-boundary smoothed pixels) + clip **before** cleanup/after
  dilation, instead of the inconsistent full-frame-threshold + cropped-labeling hybrid.
- bbox from the original 2D mask (no `any(axis=0)` over a broadcast view); explicit empty/whole-frame
  handling.
