# Expression Domain and Boundary-Masked Segmentation

**Status**: Draft
**Date**: 2026-05-05
**Type**: Design

## Problem

Reporter-based fluorescent expression analyses (CaLexA, NFAT, dpERK, fixed-tissue GCaMP, etc.) currently flow through `analyze_target_cells` → `segment_target_objects`, a pipeline designed to detect bright objects above background. For expression quantification this is wrong on three fronts:

1. **Baseline expression is discarded.** Otsu / SNR-floor thresholds capture only highly-expressing cells. In control or low-activity samples almost no ROI is produced. Comparing experimental vs control becomes impossible because the actual biological readout — the baseline shift in expression intensity — is not represented in the output table.
2. **ROI shrinkage on bright cells.** Soft-edged cytosolic signal in cell clusters loses its periphery. QC images for hindgut CaLexA show only the saturated core captured; the visibly clear cytosolic halo is missing. Mean intensity, area, and volume metrics are biased upward (only saturated pixels included) and biologically meaningless.
3. **Background subtraction breaks on clusters.** `morphological_opening(radius=48 px)` treats cluster interiors as background when the cluster is larger than the structuring element, eroding real signal. Naively raising the radius breaks local illumination correction; lowering it does not help bright peripheries.

The fix needs to be generic. The same primitives apply to any reporter expression analysis, not CaLexA specifically. Cell sizes also vary widely across tissues (Drosophila central brain neurons ~5 µm, hindgut epithelia ~15 µm, with smaller cells common), so size parameters must adapt rather than be hard-coded.

## Proposed Solution

Two generic segmentation primitives plus targeted extensions to existing functions, enabling a recipe-level "two-tier" pattern:

- **Tier 1 — Expression Domain.** A permissive mask that captures all reporter-expressing tissue, including baseline. Defined either by reporter signal alone (with a noise-floor threshold) or by intersection with a structural counterstain mask (TOPRO, DAPI, etc.).
- **Tier 2 — Active Cells.** Existing `segment_target_objects` re-applied within the Tier 1 boundary, restoring the strict bright-cell detection but constrained by the domain. Cluster periphery is recovered because the marker-grow naturally extends to the domain boundary rather than stopping at the strict threshold.

A recipe with no `domain` block keeps current single-tier behaviour unchanged. A recipe with a `domain` block runs both tiers and produces a long-format table with a `tier` column.

## Key Decisions and Rationale

- **Two-tier instead of single-tier with looser threshold.** Single-tier loosening loses the active-cell signal in noise. Two-tier preserves both the baseline-tissue measurement and the active-cell detection. This is the standard quantification pattern for fluorescent reporters where baseline matters.
- **Generic primitives, not a CaLexA workflow.** The exact same operation (define expression domain, find bright sub-objects within it) applies to any expression-pattern reporter. Naming primitives generically lets future analyses reuse them without duplication.
- **No background subtraction in Tier 1.** Cluster-safe by construction. Local correction is not needed because the noise-floor threshold uses the global imaging-noise distribution, which is well-defined for confocal data.
- **Marker-grow over Cellpose retraining.** Cellpose retraining requires labelled data the user does not have, especially for the variable sample/condition cases that fail Cellpose now. Marker-grow within a permissive mask is deterministic and works regardless of training corpus. Cellpose remains available via the existing `cellpose_sam` segmentation method.
- **Cell diameter as a recipe parameter.** Tissue-dependent (5–20 µm range observed). Auto-estimation is unreliable and brittle; recipe-level user input is the right interface.

## Architecture

### New public tools

`segment_expression_domain(image_layer, ...)` — Phase-2 segmentation primitive. Threshold-based, cluster-safe, optional counterstain intersection.

`detect_counterstain_channel(sample_name)` — Phase-1 utility. Identifies and returns the counterstain channel for a sample with confidence labelling.

### Modified public tools

`segment_target_objects` — adds `boundary_mask: str | None = None`. When set, output labels are intersected with the mask and any candidate markers outside the mask are dropped before watershed.

`analyze_target_cells` — adds `domain_strategy`, `domain_options`, `counterstain_layer`, `cell_diameter_um`. When `domain_strategy` is set, runs Tier 1 → Tier 2 → two-tier measurement. When `cell_diameter_um` is set, derives `min_distance_um` and `min_area_um2` defaults internally for both tiers.

### Internal helpers

- `_threshold_noise_floor(image, k_mad, dark_percentile)` — `median + k_mad * MAD` over the lowest `dark_percentile`% of finite pixels (default 10%).
- `_intersect_labels_with_mask(labels, mask, *, renumber=False)` — sets labels outside the mask to 0; optional sequential renumber.
- `_derive_size_params(cell_diameter_um, voxel_spacing)` — translates a single user-facing diameter into `min_distance_um` (= 0.7 × diameter), `min_area_um2` (= π × (diameter/4)²), and Cellpose `diameter_px` (= diameter / voxel xy spacing).

### Recipe schema

`AnalysisRecipe` (in `agent/state.py`) gains two optional fields:

```python
@dataclass
class AnalysisRecipe:
    # ... existing fields ...
    cell_diameter_um: float | None = None      # NEW
    domain: dict[str, Any] | None = None       # NEW
```

`domain` block contents (all optional except `strategy`):

```python
{
    "strategy": "noise_floor",          # only "noise_floor" in v1; future: "percentile", "manual"
    "k_mad": 5.0,
    "dark_percentile": 10.0,
    "counterstain_layer": None,
    "counterstain_dilation_um": 0.0,    # set to ~ cell_radius for nuclear counterstain
    "min_area_um2": None,                # if None, derived from cell_diameter_um
    "dilation_um": 0.0,
}
```

`put_recipe` extended to accept these. Project save/load round-trips them.

## Behaviour Specification

### `segment_expression_domain`

Inputs:
- `image_layer: str` — target reporter layer.
- `threshold_strategy: str = "noise_floor"`.
- `k_mad: float = 5.0`.
- `dark_percentile: float = 10.0`.
- `counterstain_layer: str | None = None`. If set, segment counterstain via Otsu, dilate by `counterstain_dilation_um`, intersect with reporter mask.
- `counterstain_dilation_um: float = 0.0`.
- `is_nuclear: bool | None = None` — must be supplied by the caller (typically the workflow, which obtains it from `detect_counterstain_channel`). If `None` or `False`, the counterstain is not used to constrain the mask (see Behaviour).
- `min_area_um2: float = 5.0`.
- `dilation_um: float = 0.0`.

Outputs (returned dict):
- `labels_layer: str` — name `<reporter>_domain`.
- `n_components: int`.
- `domain_area_um2: float`.
- `noise_floor_threshold: float`.
- `counterstain_used: bool`.
- `counterstain_warnings: list[str]`.
- `qc_png_path: str | None`.
- `empty_mask: bool`.

Behaviour:
- No background subtraction. Threshold is applied directly to raw reporter image.
- If `counterstain_layer` provided but `is_nuclear` is `False` or `None`, the counterstain is *not* used to constrain the mask. It appears in QC for reference only and a warning is recorded ("counterstain marker is non-nuclear or unknown; reporter-only domain used").
- If reporter has no finite pixels above the noise floor, returns `empty_mask=True` with `n_components=0`.
- Domain labels are typically 1 or few large connected components. Each gets a unique label ID; component selection is left to downstream tools (Tier 2 just uses the binary union).

### `segment_target_objects` extension

New parameter:
- `boundary_mask: str | None = None`.

When `boundary_mask` is set:
1. Materialise and binarise: any label > 0 is "inside".
2. Existing pipeline runs unchanged (background subtraction, threshold, etc.) — operating on the full image so neighbouring intensities still inform the local threshold.
3. After watershed labelling, multiply by the binary boundary: `labels[~mask_inside] = 0`.
4. Discarded fragments below `min_size` are removed by the existing pipeline.

Default `None` preserves current behaviour exactly. Regression tests must confirm bit-identical results when the parameter is unset.

### `analyze_target_cells` extension

New parameters:
- `domain_strategy: str | None = None`.
- `domain_options: dict | None = None`.
- `counterstain_layer: str | None = None`.
- `cell_diameter_um: float | None = None`.

When `domain_strategy is None`: current behaviour (single-tier), no new code path.

When `domain_strategy` is set:
1. **Counterstain resolution.** If `counterstain_layer` not provided, call `detect_counterstain_channel(current_sample)`. If `confidence == "annotated"`, use it. If `confidence == "inferred"`, surface `needs_user_confirmation=True` in the workflow result and proceed with reporter-only domain unless the agent re-invokes with explicit `counterstain_layer`.
2. **Tier 1.** Call `segment_expression_domain(target_layer, counterstain_layer=resolved, **domain_options)` → `domain_layer`.
3. **Tier 2.** Call `segment_target_objects(target_layer, boundary_mask=domain_layer, min_size=derived, min_distance=derived, ...)`.
4. **Measurement.** Call `measure_intensity` twice — once with `labels_layer=domain_layer`, once with `labels_layer=cells_layer`. Concat results with a new `tier` column ("domain" / "cells"). Existing sample-column attachment runs once on the concat.
5. **QC.** Two QC PNGs:
   - Tier 1: existing `_write_segmentation_qc_png` on (reporter, domain mask).
   - Tier 2: extended `_write_segmentation_qc_png` with `secondary_outline_mask=domain_mask` rendering domain boundary as dashed cyan, primary cell labels with orange boundaries (current style).

Output additions to the workflow return value:
- `domain_layer`, `cells_layer` (alias of `labels_layer`).
- `n_domain_components`, `domain_area_um2`, `domain_mean_intensity`.
- `n_cells` (alias of `n_objects`).
- `tier_table_name` — name of the concatenated long-format table.

When `cell_diameter_um` is set and Tier-2 size params are not explicit, defaults are derived via `_derive_size_params` and used for both Tier-2 watershed (`min_distance_um`) and size filter (`min_area_um2`). If the recipe also sets `domain.min_area_um2`, that value wins for Tier 1.

### `detect_counterstain_channel`

Inputs:
- `sample_name: str`.

Returns:
```python
{
    "counterstain_layer": str | None,
    "counterstain_marker": str | None,        # "topro", "dapi", "hoechst", "nc82", "phalloidin", "other", None
    "is_nuclear": bool | None,
    "confidence": "annotated" | "inferred" | "none",
    "needs_user_confirmation": bool,
    "candidate_layers": list[str],            # alternatives if inferred
}
```

Resolution order:
1. Sample annotation `counterstain_marker` field — `confidence="annotated"`, `needs_user_confirmation=False`.
2. Layer wavelength metadata containing 633 nm or "far_red"/"647" — `confidence="inferred"`, `needs_user_confirmation=True`.
3. None found — `confidence="none"`, `needs_user_confirmation=False`.

Marker → nuclear lookup (initial; extensible):
- Nuclear: `"topro"`, `"to-pro"`, `"to-pro-3"`, `"dapi"`, `"hoechst"`.
- Non-nuclear: `"nc82"`, `"bruchpilot"`, `"phalloidin"`.
- Unknown marker name: `is_nuclear=None` (treated as non-nuclear by `segment_expression_domain`).

The result is informational. The agent or workflow decides whether to use it.

## Files Modified

| File | Changes |
|---|---|
| `src/imajin/tools/segment.py` | + `segment_expression_domain`, + `_threshold_noise_floor`, + `_intersect_labels_with_mask`; modify `segment_target_objects` to accept `boundary_mask`; extend `_write_segmentation_qc_png` with optional `secondary_outline_mask` |
| `src/imajin/tools/channels.py` | + `detect_counterstain_channel` (and supporting marker-name lookup) |
| `src/imajin/tools/workflows.py` | extend `analyze_target_cells` with two-tier branch, + `_derive_size_params` helper |
| `src/imajin/agent/state.py` | `AnalysisRecipe` + `cell_diameter_um`, `domain`; `put_recipe` accepts new args; project save/load round-trip |
| `src/imajin/tools/experiment.py` | `create_analysis_recipe` accepts new fields |
| `tests/test_segment.py` | unit tests for `segment_expression_domain`, `boundary_mask`, `_threshold_noise_floor`, single-tier regression of `segment_target_objects` |
| `tests/test_workflows.py` | end-to-end two-tier workflow with synthetic fixture |
| `tests/test_counterstain.py` (new) | counterstain detection branches |

No new module is introduced. Two new public tools, four function extensions, one schema extension.

## Test Strategy

### Synthetic fixture

A 2D 256×256 image with three regions:
- **Region A**: bright cluster (saturated core ~80 px diameter + soft halo extending to 120 px) — Tier 2 should capture the full extent of bright cells, not just the saturated core.
- **Region B**: scattered isolated cells with intensity gradient (bright, medium, baseline-only).
- **Region C**: pure black (true noise / out-of-tissue).

Optional second-channel counterstain mock — uniform-bright nuclei in regions A and B, none in C.

### Validation

Tier 1 (`segment_expression_domain`):
- Domain mask covers regions A and B; excludes C.
- Cluster-safety: domain area in region A includes interior (no eat-through).
- Counterstain-on path: domain restricted to nuclear-dilated areas; halo periphery beyond nuclear-dilation is excluded if structure says so.
- `is_nuclear=False` path: counterstain ignored, warning recorded.
- `empty_mask=True` returned when reporter has no signal above noise floor.

Tier 2 (`boundary_mask`):
- All output labels lie within the boundary; none in region C.
- With cluster periphery in domain mask but not in strict threshold, marker-grow extends labels to include the halo. Mean intensity and area metrics increase relative to single-tier baseline.
- With `boundary_mask=None`, output is bit-identical to current behaviour (regression).

`_threshold_noise_floor`:
- On synthetic Gaussian-noise + sparse signal image, returns threshold within configured tolerance of `µ + k·σ` of dark percentile.

`detect_counterstain_channel`:
- Annotated path returns `confidence="annotated"`.
- Inferred path (633 layer + no annotation) returns `confidence="inferred"` and `needs_user_confirmation=True`.
- No counterstain returns `confidence="none"`.
- Marker → nuclear lookup correctly identifies TOPRO, DAPI as nuclear; nc82, phalloidin as non-nuclear.

End-to-end workflow:
- Single-tier (no `domain`) yields current behaviour (regression test).
- Two-tier produces a long-format table with `tier` column having both "domain" and "cells" rows.
- Domain row count equals `n_domain_components` per sample; cells row count equals `n_cells`.
- `cell_diameter_um=15` derives `min_distance_um=10.5`, `min_area_um2 ≈ 44`.

Recipe round-trip:
- A recipe with `cell_diameter_um` and `domain` block saves and reloads through `Project.save` / `Project.load` with values intact.
- A recipe without these fields loads as `cell_diameter_um=None`, `domain=None` (single-tier).

## Out of Scope

- Cellpose hybrid markers (deferred; can be added as a separate `marker_source` parameter on `segment_target_objects` if needed).
- Per-cell Cellpose model retraining (out of scope; not warranted for the use cases above).
- Dynamic cell-size auto-estimation (recipe parameter is the chosen interface).
- Process / neuropil-level analysis (cell-body / soma focus only).
- Time-series workflows (this design targets fixed-tissue snapshots).
- Multi-counterstain support (one counterstain layer per analysis).

## Open Questions

None — all design decisions reflect the preceding brainstorming. The recipe-level `cell_diameter_um` parameter handles tissue variability; the two-tier model handles baseline + active dual measurement; counterstain detection has explicit confidence levels with workflow-level fallback.
