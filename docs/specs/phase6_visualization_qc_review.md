# Phase 6 Spec: Visualization, QC, and Manual Review

## Goal

Make automated analysis results easy to inspect, validate, correct, and trust.

Confocal and live imaging workflows often require human review because:

- segmentation can fail or over/under-split cells
- reporter channels can be noisy or uneven
- z-stacks can contain partial objects
- live tissue can drift or deform
- batch analysis can produce sample-specific failures

Phase 6 should give users clear visual and quantitative feedback before results
are used in final reports.

## Non-Goals

- Do not implement new segmentation algorithms in this phase.
- Do not implement full manual annotation software.
- Do not replace napari's native label editing tools.
- Do not implement advanced statistical outlier modeling.

## Core Principle

Automated results should be reviewable at three levels:

1. image/layer level
2. object/ROI level
3. sample/batch level

The user should be able to see what was measured, identify obvious failures,
make corrections in napari, and refresh downstream measurements.

## Review Workflows

### Single-Sample Segmentation Review

1. Show target image and labels overlay.
2. Display object count and basic object size statistics.
3. Flag suspicious segmentation results.
4. Allow user to click a table row and highlight the corresponding label.
5. Allow user to edit labels in napari.
6. Refresh measurement table after manual edits.

### Z-Stack Review

1. Support z-slice browsing with labels overlay.
2. Provide max projection and orthogonal views.
3. Warn about anisotropic voxel spacing when relevant.
4. Allow object review in 3D or slice-by-slice mode.

### Time-Course Review

1. Show ROI labels over a reference frame.
2. Display intensity traces per ROI.
3. Flag ROIs with missing/flat/saturated traces.
4. Support review of motion/drift qualitatively.
5. Allow re-measurement after ROI edits.

### Batch QC Review

1. Show per-sample object counts.
2. Show per-sample intensity/area summaries.
3. Flag samples with unusual counts or measurements.
4. Track failed samples separately from valid samples.
5. Include QC status in final report.

## Visualization Features

### Segmentation Overlay

Provide an explicit review view or helper for:

- target image
- labels layer
- optional counterstain/reference layer
- adjustable label opacity
- additive or outline overlay

Recommended controls:

- show/hide labels
- label opacity slider
- show selected label only
- toggle target/counterstain channel
- jump to selected object centroid

### Label Outlines

Napari labels can obscure signal when filled. Add an outline view where possible.

Options:

- use napari label contour display if available
- generate binary outline layer
- generate boundary image from labels

This is especially useful for checking whether segmentation matches reporter
signal.

### Object Selection

Table-to-layer interaction should support:

- clicking table row selects label in napari
- selected label is highlighted
- viewer jumps to centroid if available
- object summary is shown in side panel

Layer-to-table interaction can be added later if more difficult.

### Before/After Preprocessing Comparison

When preprocessing was applied, provide a way to compare:

- raw target channel
- preprocessed target channel
- segmentation result

This can be a simple layer toggle workflow rather than a custom viewer.

### Time-Course Plot

Add simple plots for ROI intensity over time:

- selected ROI trace
- all ROI traces with low opacity
- mean trace across ROIs
- optional group/sample trace later

Initial implementation can use matplotlib/Qt or a simple table-linked plot.

## QC Metrics

### Segmentation QC

Compute and store:

- object count
- area/volume min, median, max
- fraction of very small objects
- fraction of very large objects
- fraction touching image border
- empty mask flag
- saturated intensity fraction if available

Warnings:

- zero objects
- too few objects compared to expected or batch median
- too many objects compared to expected or batch median
- many tiny objects
- many border-touching objects
- missing scale metadata
- strong z anisotropy for 3D measurement

### Measurement QC

Compute and store:

- intensity min/median/max
- integrated intensity if available
- number of NaN/null values
- saturation fraction
- object area/volume distribution

Warnings:

- all intensities zero
- many saturated pixels
- extremely broad area distribution
- missing physical-unit columns because voxel size is unavailable

### Time-Course QC

Compute and store:

- trace length
- number of ROIs
- missing timepoints
- flat traces
- saturated traces
- high frame-to-frame jumps

Warnings:

- missing time interval metadata
- possible drift if ROI signal drops sharply across many ROIs
- ROI labels do not match movie frame shape

## QC Status Model

Add lightweight QC status to analysis runs/tables:

```python
QCStatus = Literal["pass", "warning", "fail", "not_checked"]
```

Recommended record:

```python
@dataclass
class QCRecord:
    source: str
    status: QCStatus
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    reviewed_by_user: bool = False
    notes: str | None = None
```

QC status should be stored with:

- segmentation result
- measurement table
- analysis run
- sample summary

## Manual Correction Workflow

Use napari's native label editing.

Required app behavior:

1. User edits labels layer manually.
2. User clicks refresh measurement.
3. App recomputes measurement table from current labels.
4. Provenance records refresh.
5. QC record notes that manual edit occurred if detectable or user confirms.

Optional later:

- detect labels layer data change event
- mark measurement table stale
- prompt user to refresh

## Tools

### compute_segmentation_qc

```python
compute_segmentation_qc(labels_layer: str, image_layer: str | None = None) -> dict
```

Returns QC metrics and warnings.

### compute_measurement_qc

```python
compute_measurement_qc(table_name: str) -> dict
```

Returns table-level QC metrics and warnings.

### compute_timecourse_qc

```python
compute_timecourse_qc(table_name: str) -> dict
```

Returns time-course QC metrics and warnings.

### create_label_outline

```python
create_label_outline(labels_layer: str, name: str | None = None) -> dict
```

Adds outline/boundary image layer for visual review.

### jump_to_object

```python
jump_to_object(table_name: str, label: int) -> dict
```

Uses centroid columns to move viewer to object if possible.

### mark_qc_status

```python
mark_qc_status(source: str, status: QCStatus, notes: str | None = None) -> dict
```

Allows user to mark a result/sample as pass/warning/fail.

## UI Requirements

### QC Dock

Add a QC/review dock or panel.

Minimum:

- selected result/table
- QC status
- warnings list
- key metrics
- mark pass/warning/fail buttons
- notes field
- refresh QC button

### Object Review Panel

Can be part of table dock or QC dock.

Show for selected object:

- label id
- area/volume
- mean/max/min intensity
- centroid
- source layer
- QC flags if any

### Time-Course Plot Panel

Show:

- selected ROI trace
- mean trace
- optional all traces

This panel should be linked to time-course tables.

### Batch QC Summary

For experiment-level workflows:

- sample name
- group
- object count
- mean intensity
- QC status
- warnings
- failed run indicator

## Report Requirements

Reports should include a QC section.

Single-sample report:

- segmentation QC metrics
- measurement QC warnings
- manual review status
- notes

Experiment report:

- per-sample QC status
- excluded/failed samples
- warnings by sample
- whether results were manually reviewed

Do not hide QC failures. If a sample failed or was excluded, report it clearly.

## Agent Behavior

When segmentation produces warnings:

- summarize the issue
- suggest likely next action
- do not proceed blindly if segmentation is unusable

Examples:

- zero objects -> suggest different target channel, preprocessing, or diameter
- many tiny objects -> suggest denoising or larger diameter
- many border objects -> warn about partial cells
- missing voxel size -> report measurements in pixels only

When user says:

- "괜찮아 보여" -> mark QC pass
- "이건 제외해" -> mark QC fail/excluded
- "다시 측정해" after label edit -> refresh measurement

## Tests

### QC Metrics

- empty labels produce fail/warning
- normal labels produce pass/no severe warning
- tiny-object-heavy labels produce warning
- border-touching labels produce warning
- missing scale produces physical-unit warning

### Measurement QC

- all-zero intensity table warning
- saturated intensity warning
- NaN/null warning
- broad area distribution warning

### Time-Course QC

- flat trace warning
- missing timepoint warning
- normal trace passes

### Visualization Tools

- outline layer created from labels
- jump_to_object uses centroid columns
- invalid label gives useful error

### Manual Review

- mark_qc_status updates QC record
- refresh measurement after label edit updates table
- stale measurement marking if implemented

### Report

- QC section includes warnings
- failed sample appears in report
- manually reviewed status appears

## Acceptance Criteria

- User can visually review labels over target image.
- User can select a measurement row and highlight/jump to the object.
- App computes basic segmentation QC metrics.
- App computes basic measurement QC metrics.
- App can mark QC status pass/warning/fail.
- Time-course tables can be plotted or summarized for review.
- Reports include QC status and warnings.
- Existing fast tests pass.

## Suggested Implementation Order

1. Add QC data model and in-memory registry.
2. Implement segmentation QC metrics.
3. Implement measurement QC metrics.
4. Implement label outline tool.
5. Improve table-to-label selection/jump behavior.
6. Add simple QC dock.
7. Add time-course plot panel.
8. Integrate QC status into reports.
9. Add batch QC summary support.
10. Add tests.

## Open Questions

- What thresholds should define too few/too many objects?
- Should QC thresholds be user-configurable per tissue/experiment?
- Should excluded samples remain in summaries with an exclusion flag or be
  omitted from group statistics?
- Should manual edits be detected automatically or only user-marked?
- Which plotting backend should be used for time-course traces?
