# Phase 2 Spec: Core Cell Analysis Workflow

## Goal

Build a reliable single-sample cell analysis workflow on top of the metadata and
channel handling work from Phase 1.

Phase 2 should make the common user request work well:

> "Use this target channel, find cells/ROIs, measure intensity and size, and
> summarize the result."

The target use case is general confocal/live fluorescence imaging of Drosophila
gut, brain, VNC, and fat body samples. Neural process tracing remains outside
this phase.

## Non-Goals

- Do not implement batch/group analysis yet.
- Do not implement connectome/neural process identification.
- Do not make filename parsing rules.
- Do not build a full statistical report system yet.
- Do not make tracking the main path for live imaging; ROI intensity over time
  remains the first priority.

## Assumptions From Phase 1

Phase 2 assumes the app can:

- load LSM/OME/CZI/TIFF files
- preserve axes metadata
- preserve voxel size metadata
- infer channel color from acquisition metadata where possible
- resolve channel phrases like green/red/UV/IR/far red
- store user-confirmed channel annotations
- distinguish metadata facts from biological annotations

If Phase 1 is incomplete, implement only the Phase 2 pieces that do not depend
on missing metadata.

## Core Workflow

### Standard Single-Image/Z-Stack Workflow

1. Resolve target channel.
2. Optionally preprocess target channel.
3. Segment cells/ROIs from target channel.
4. Measure objects on the same target channel.
5. Store result table with pixel and physical-unit measurements.
6. Add output labels layer.
7. Record provenance.
8. Summarize object count and table name.

Default behavior:

- One target channel should drive segmentation and measurement.
- Counterstain channels should not be used for segmentation or measurement
  unless the user explicitly asks.
- If no target is confirmed and channel choice is ambiguous, ask one focused
  question.

### Time-Series / Live Imaging Workflow

1. Resolve target channel.
2. Create or select ROI labels.
3. Measure ROI intensity over time.
4. Store long-format time-course table.
5. Include time metadata if available.

Tracking and drift correction are useful later, but this phase only needs a
solid ROI measurement path.

## Channel Selection Policy

Target channel resolution should follow:

1. Explicit user-specified layer.
2. User-confirmed `target` annotation.
3. User phrase resolved by metadata, e.g. "green".
4. One strong target suggestion, if the agent states the assumption.
5. Ask user if ambiguous.

Examples:

- "green에서 측정해" -> resolve green and use it.
- "target은 red야" -> annotate/resolve red as target, then analyze.
- "분석해줘" with one confirmed target -> proceed.
- "분석해줘" with green and red both plausible -> ask which is target.

Do not automatically select counterstain for measurement.

## Segmentation Requirements

### Initial Supported Method

Use existing Cellpose-SAM path as the default segmentation backend.

Required behavior:

- Accept 2D image layers.
- Accept 3D z-stack image layers.
- Use `do_3D=True` only when appropriate.
- Preserve voxel scale on output label layers.
- Output labels layer should include metadata:
  - source layer
  - segmentation method
  - model
  - diameter
  - 2D/3D mode
  - object count

### 2D vs 3D Policy

Use axis metadata where available.

Recommended defaults:

- `YX`: 2D segmentation.
- `ZYX`: 3D segmentation candidate.
- `TYX`: do not segment whole array as 3D; require timepoint selection or
  per-frame workflow.
- `TZYX`: do not segment whole array as 4D; require timepoint selection or
  future per-time workflow.

If the user says "segment this z-stack", use 3D if data is `ZYX`.

If the user says "segment frame 0" or live imaging data has time axis, use
`extract_timepoint` or equivalent first.

### Segmentation Result Quality

Add lightweight QC fields to segmentation output:

- `n_objects`
- `object_area_min`
- `object_area_median`
- `object_area_max`
- `empty_mask`: boolean
- shape
- dtype

If segmentation returns zero objects, the agent should not continue to
measurement silently. It should report failure and suggest changing channel,
preprocessing, or parameters.

## Measurement Requirements

### Object Measurement

Current `measure_intensity` should become the central object measurement tool.

It should measure:

- label id
- area in pixels
- area in physical units when possible
- centroid in pixels
- centroid in physical units when possible
- mean intensity
- max intensity
- min intensity
- optional integrated intensity

For 3D labels, add when possible:

- volume in voxels
- volume in physical units
- z/y/x centroid

Column naming should be stable and readable.

Recommended physical columns:

- `area_px`
- `area_um2`
- `volume_voxels`
- `volume_um3`
- `centroid_z_um`
- `centroid_y_um`
- `centroid_x_um`

Do not remove existing columns abruptly if tests or reports depend on them.
Prefer additive changes.

### Same-Channel Default

The default common workflow should measure the target channel that was used for
segmentation.

The user should not need to specify separate segmentation and measurement
channels in the common case.

### Multi-Channel Measurement

Still allow measuring multiple channels if explicitly requested.

Use cases:

- colocalization preparation
- reporter plus reference measurement
- comparing target channels

But this should not complicate the default UI/agent path.

## Time-Course Measurement

`measure_intensity_over_time` should support:

- static ROI labels: labels shape `YX` or `ZYX`
- time-varying labels: labels shape `TYX` or `TZYX`
- target image shape `TYX` or `TZYX`
- optional time axis resolution from metadata

Result table should be long-format:

- sample/layer identifiers where available
- `time_index`
- `time_s` if metadata exists
- `label`
- `area`
- `mean_intensity`
- `max_intensity`
- `min_intensity`
- optional integrated intensity

If time interval is missing, use `time_index` and leave `time_s` absent or null.

## Preprocessing Policy

Keep preprocessing optional.

Available preprocessing:

- rolling-ball background subtraction
- percentile auto-contrast
- Gaussian denoise

Default:

- Do not preprocess unless the user asks or segmentation fails.

Agent behavior:

- If segmentation gives zero/few objects and image has high uneven background,
  suggest rolling-ball background subtraction.
- Do not stack preprocessing steps unnecessarily.

## Colocalization Boundary

Colocalization remains supported, but it is not the default cell-analysis path.

Use only when user asks:

- "colocalization"
- "overlap"
- "green and red together?"
- "공국소화"

Colocalization should use explicitly resolved channel pairs.

Counterstain should only be included if user explicitly requests it.

## Tooling Recommendations

### High-Level Tool

Add a high-level tool or workflow wrapper:

```python
analyze_target_cells(
    target: str | None = None,
    do_3D: bool | None = None,
    diameter: float | None = None,
    preprocess: str | None = None,
)
```

Behavior:

1. Resolve target channel.
2. Optionally preprocess.
3. Segment.
4. Measure.
5. Return labels layer, measurement table, object count, and key QC metrics.

This should call existing lower-level tools rather than duplicating logic.

Purpose:

- Give the agent and manual UI a reliable default path.
- Avoid making the LLM chain together too many fragile low-level calls.

### Keep Low-Level Tools

Retain:

- `cellpose_sam`
- `measure_intensity`
- `measure_intensity_over_time`
- `rolling_ball_background`
- `auto_contrast`
- `gaussian_denoise`
- `manders_coefficients`
- `pearson_correlation`

The high-level workflow should be additive.

## UI Direction

Avoid a flat list of all tools as the main user experience.

For Phase 2, a minimal workflow-oriented manual panel would be enough:

- Target channel selector
- Segment button
- Measure button
- Analyze target cells button
- Result summary

This can still be backed by the same registered tools.

Manual execution should eventually use the same background execution path as
LLM tools, but that may be handled in a separate execution/UI phase if too large.

## Report Requirements

For each analysis, record:

- source file/layer
- target channel
- segmentation method and parameters
- object count
- measurement table
- voxel size used for physical measurements
- warnings:
  - no voxel size
  - anisotropic z spacing
  - zero objects
  - ambiguous channel resolution

Reports should describe:

```text
Cells/ROIs were segmented from the user-confirmed target channel using
Cellpose-SAM. Per-object intensity and size measurements were extracted from
the same target channel.
```

Only mention target/counterstain roles if confirmed by user annotation.

## Tests

### Channel-to-Analysis Workflow

- confirmed target channel is used automatically
- green phrase resolves to metadata green channel
- ambiguous target raises useful error or requests clarification path
- counterstain is not used by default

### Segmentation

- 2D input creates labels layer
- 3D input creates labels layer with 3D mode
- 4D input is rejected with a useful message
- zero-object output does not proceed to measurement in high-level workflow

### Measurement

- 2D labels/image produce area/intensity table
- 3D labels/image produce volume/centroid/intensity table
- physical unit columns are correct with scale metadata
- missing scale does not crash
- multiple image layers still work

### Time-Course

- static ROI labels over `TYX` movie
- time-varying labels over `TYX` movie
- `TZYX` time series
- time interval metadata creates `time_s`
- missing time interval still creates `time_index`

### Report/Provenance

- high-level workflow records lower-level operations or clear workflow record
- measurement table name appears
- object count appears
- physical unit metadata appears

## Acceptance Criteria

- A user can load a normal target-channel z-stack and run one command to get:
  - labels layer
  - per-cell measurement table
  - object count summary
- Measurements include physical units when voxel size is available.
- Live/time-series ROI intensity over time works for static ROIs.
- Counterstain is not used for measurement unless requested.
- Ambiguous target selection is handled explicitly.
- Full fast test suite passes:

```bash
uv run python -m compileall -q src tests
uv run pytest -q -m "not slow and not integration"
uv run imajin --doctor
```

## Suggested Implementation Order

1. Add/clean target-channel resolution helper for analysis workflows.
2. Add physical-unit measurement columns.
3. Add high-level `analyze_target_cells` workflow tool.
4. Improve 2D/3D/4D segmentation policy and errors.
5. Improve time-course output columns.
6. Add report/provenance summaries.
7. Add focused tests.

## Open Questions

- Should 3D z-stack segmentation default to true 3D Cellpose or per-slice 2D?
- Should physical 3D volume be computed for all labels or only when labels are
  truly 3D objects?
- Which time-course summary plots should be generated first?
- Should the first manual workflow UI be added now or after execution-service
  cleanup?
