# ROI Bounding-Box Crop for Boundary-Masked Segmentation — Plan

Follows `docs/superpowers/specs/2026-06-29-roi-crop-segmentation-design.md` (revised after 1 Codex
review). This plan: pre-Codex-review. Branch `feat/roi-crop-segmentation` off master, commit-by-commit
with `.venv/bin/python -m pytest <files> -q` gates.

Scope: `segment_target_objects` (crop, mask-invariant) + `segment_expression_domain` (boundary_mask,
ROI-local + clip) + two-tier threading. `auto_segment_target` / `segment_3d_cells_auto` unchanged.

## Commit 1 — pure helpers (`analysis/segmentation.py`) + tests

```python
def boundary_bbox_slices(yx_mask_2d, raw_shape, margin):
    """YX bbox of a 2D mask expanded by margin, clipped to raw_shape. Full slice for any
    leading (Z) axes. Returns None if the mask is empty or already spans the whole YX frame."""
    m = np.asarray(yx_mask_2d, dtype=bool)
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    H, W = raw_shape[-2], raw_shape[-1]
    y0 = max(0, int(ys.min()) - margin); y1 = min(H, int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin); x1 = min(W, int(xs.max()) + 1 + margin)
    if (y0, x0, y1, x1) == (0, 0, H, W):
        return None  # whole frame -> no benefit
    lead = (slice(None),) * (len(raw_shape) - 2)
    return (*lead, slice(y0, y1), slice(x0, x1))

def scatter_labels_to_full(cropped_labels, full_shape, slices):
    out = np.zeros(full_shape, dtype=np.int32)
    out[slices] = np.asarray(cropped_labels, dtype=np.int32)
    return out
```

`tests/test_boundary_crop.py` (new, pure): bbox of an off-centre rect mask + margin clips to bounds;
empty mask -> None; whole-frame mask -> None; 3D raw_shape keeps full Z; `scatter` places a crop into
zeros at the right offset and is int32.

**Gate:** `pytest tests/test_boundary_crop.py -q`.

## Commit 2 — `segment_target_objects` crop (`tools/segment.py`)

Reorder: move the boundary-load/resolve block **above** `prepare_corrected`. Keep
`effective_min_size` computed from the **full-frame** `xy_area` (already is). Then:

```python
crop = None
if (boundary_data_bool is not None and background_radius > 0
        and not auto_mask_hyperbright):
    yx2d = (_boundary_raw > 0) if _boundary_raw.ndim == 2 else np.any(_boundary_raw > 0, axis=0)
    margin = 2 * int(background_radius) + int(np.ceil(4.0 * float(smoothing_sigma))) + 8
    crop = _boundary_bbox_slices(yx2d, raw.shape, margin)

if crop is not None:
    raw_w, bnd_w = raw[crop], boundary_data_bool[crop]
else:
    raw_w, bnd_w = raw, boundary_data_bool

corrected_w = _prepare_corrected(raw_w, background_radius=..., ...)
seg = _threshold_and_label(corrected_w, raw_w, spacing=spacing, boundary_mask=bnd_w,
                           min_size=effective_min_size, ...)   # min_size from full frame
masks = _scatter_labels_to_full(seg.masks, raw.shape, crop) if crop is not None else seg.masks
```

Downstream uses `masks` (full) + `seg.threshold/qc/signal_qc/...` (crop-derived; documented ROI-local
for `mask_fraction`/confidence). `bnd_w` uses the original-2D bbox so a broadcast view is never
`any(axis=0)`-scanned. The QC `secondary_mask_array` reload stays (already projects to 2D).
`boundary_mask=None` path: `crop is None`, identical to today.

`tests/test_tools_segment.py`:
- **mask invariance:** synthetic 3D image, bright blobs inside + outside a 2D ROI;
  `segment_target_objects(boundary_mask=ROI, background_radius=48, smoothing_sigma=1)`; compare
  `labels>0` and `n_objects` against the same call forced through the no-crop path (monkeypatch
  `_boundary_bbox_slices`-> None, or a whole-frame boundary). Assert equal inside the ROI.
- **locality:** put a bright blob far outside bbox+margin; assert it never appears and the in-ROI
  result is unchanged vs. without that blob.
- **skip conditions:** `background_radius=0` and `auto_mask_hyperbright=True` -> result equals the
  pre-change output (no crop path).

**Gate:** `pytest tests/test_boundary_crop.py tests/test_tools_segment.py -q`.

## Commit 3 — `segment_expression_domain` boundary_mask (`tools/segment.py`)

Add `boundary_mask: str | None = None`. When set:
- `b = _resolve_boundary_mask(materialize(snapshot), raw.shape)`.
- Compute `threshold_image = _smooth_domain_image(raw, ...)` (full-frame smoothing kept — cheap and
  keeps the smoothed values exact; cropping the domain is out of scope for risk).
- **ROI-local noise floor:** `threshold = threshold_noise_floor(threshold_image[b], k_mad,
  dark_percentile)` (stats from inside-boundary smoothed pixels; documented behaviour change).
- `binary = isfinite(raw) & isfinite(threshold_image) & (threshold_image > threshold) & b` — **clip to
  the boundary up front** (before counterstain/min-size/dilation/components), and re-apply `& b` after
  `dilation_um` so dilation cannot grow outside the ROI.
- Metadata: `boundary_mask`, `threshold_scope="boundary_mask"`; QC `secondary_outline_mask = b` (2D
  projection) so the ROI shows on the QC PNG.

`tests/test_tools_segment.py`: `segment_expression_domain(boundary_mask=ROI)` -> domain has **zero
voxels outside the ROI**; `domain_area_um2` strictly less than the unconstrained run on an image with
signal both in and out of the ROI; `boundary_mask=None` unchanged.

**Gate:** `pytest tests/test_tools_segment.py -q`.

## Commit 4 — two-tier threading (`tools/workflows.py`, `tools/_workflow_steps.py`)

- Add `region_mask: str | None = None` to `analyze_target_cells` (the hand-drawn ROI), distinct from
  the internal domain.
- Pass `region_mask` into `_precompute_domain_layer(..., boundary_mask=region_mask)` so **Tier-1 is
  ROI-constrained**. `_precompute_domain_layer` forwards it to `segment_expression_domain`.
- Tier-2 already uses the domain as `boundary_mask` (the domain is now ⊆ ROI). If `region_mask` is set
  but `domain_strategy` is None (single-tier), set Tier-2 `boundary_mask = region_mask`.

`tests/test_phase2_workflow.py`: `analyze_target_cells(region_mask=ROI, domain_strategy="noise_floor")`
-> both the domain layer and the cells layer have zero labels outside the ROI; existing two-tier test
(no region_mask) unchanged.

**Gate:** `pytest tests/test_phase2_workflow.py tests/test_tools_segment.py tests/test_boundary_crop.py -q`,
then full `pytest -q`.

## Verification before done

1. Full suite green; report counts.
2. Manual sanity: a synthetic 63x256x256-ish stack with a small ROI -> `segment_target_objects` with
   crop touches only the bbox (assert via the shape handed to `_prepare_corrected`, e.g. a spy).

## Risks (carried from spec)

- Exactness is the **mask** inside the ROI for `segment_target_objects` under bounded settings, not IDs
  or `mask_fraction`. Skip-crop when `background_radius<=0` or `auto_mask_hyperbright`.
- Domain threshold is now ROI-local (intentional). No crop for the domain in v1 (clip only).

## Changelog — plan -> rev.1 (accepted Codex plan-review findings)

- **Test rigor (#1,#2,#9):** the invariance test asserts non-empty expected labels, **full-frame
  `labels>0` equality**, full `shape`, zero labels outside the ROI, and **spies `_prepare_corrected`**
  to prove the cropped call received a smaller YX shape. The no-crop oracle is **monkeypatching
  `segment._boundary_bbox_slices -> None`** with the *same* ROI (a whole-frame boundary is NOT
  equivalent — it changes the threshold scope). Import the helper into `tools.segment` as
  `_boundary_bbox_slices` / `_scatter_labels_to_full` so the patch hits the module-local symbol (#8).
- **QC shape (#3):** after scatter, recompute `label_qc(masks_full)` so `shape` / `n_objects` /
  areas are correct for the full labels layer; keep `signal_qc` (mask_fraction, separation) crop-local
  and **document every ROI-local field**, not just `mask_fraction`.
- **Two-tier safety (#5,#6):** `region_mask` is only valid for methods that accept `boundary_mask`
  (`target_objects`, `auto_3d_cells`); reject it (clear error) for `cellpose_sam` / `intensity_regions`
  rather than letting `_filtered_kwargs` silently drop it. If both `region_mask` and a
  `segmentation_options["boundary_mask"]` are given, **error on the conflict** (don't silently pick one).
- **Domain (#10,#11,#12,#13):** noise-floor stats use `threshold_image[b & isfinite(raw) &
  isfinite(threshold_image)]` so smoothed fill values can't pollute it; clip `& b` before size/component
  cleanup **and** after `dilation_um`; clip the counterstain binary to the ROI too; add `boundary_mask`
  + `threshold_scope` to **both** the empty and non-empty return/metadata; QC outline from the 2D
  projection of the original mask (never `any(axis=0)` over a broadcast view). Add tests for a
  component crossing the ROI boundary and a 2D-ROI-on-3D domain.
