# Phase 7 Spec: Analysis Methods Expansion

## Goal

Expand Imajin's analysis capabilities after the core workflows, execution
system, persistence, and QC layers are stable.

The priority is not to add every possible algorithm. The priority is to add
methods that are common, interpretable, and useful for Drosophila fluorescence
imaging experiments.

Primary emphasis:

- live/time-series reporter analysis, especially GCaMP and CaLexA-like signals

Secondary emphasis:

- cell morphology and tissue summaries
- colocalization extensions
- registration/drift correction
- batch-level statistics
- advanced neural process analysis as a separate track

## Non-Goals

- Do not add methods that cannot be validated or explained.
- Do not make advanced methods part of the default workflow.
- Do not hide assumptions inside black-box analysis.
- Do not implement connectome database integration until a backend is selected.

## Method Categories

1. Time-course reporter analysis
2. Cell morphology and tissue-level summaries
3. Colocalization and channel relationship analysis
4. Registration and drift correction
5. Batch-level statistics
6. Advanced neural process analysis

## 1. Time-Course Reporter Analysis

This is the highest-priority expansion.

Use cases:

- GCaMP live imaging
- CaLexA reporter time courses
- fluorescence response over time in gut, neurons, or fat body
- ROI/cell intensity traces

### Inputs

Required:

- time-course measurement table
- ROI labels or object ids
- target channel

Optional:

- time interval metadata
- stimulus/event timing
- baseline window
- sample/group annotations

### Normalization Methods

Add tools for:

- raw intensity trace
- baseline subtraction
- `F/F0`
- `DeltaF/F0`
- z-score normalization
- min-max normalization for visualization only

Recommended function:

```python
normalize_timecourse(
    table_name: str,
    value_col: str = "mean_intensity",
    method: Literal["raw", "baseline_subtract", "f_over_f0", "delta_f_over_f0", "zscore"] = "delta_f_over_f0",
    baseline: tuple[int, int] | None = None,
    group_cols: list[str] | None = None,
) -> dict
```

Rules:

- `F0` should be computed per ROI/object by default.
- If baseline is not provided, default to early frames only if clearly stated.
- Store normalization parameters in table metadata/provenance.
- Do not overwrite raw measurements.

### Response Metrics

Add tools for:

- peak amplitude
- time to peak
- area under curve (AUC)
- mean response in window
- baseline mean
- response duration above threshold
- onset time
- decay time if practical

Recommended function:

```python
extract_timecourse_features(
    table_name: str,
    value_col: str,
    baseline_window: tuple[int, int] | None = None,
    response_window: tuple[int, int] | None = None,
    threshold: float | None = None,
) -> dict
```

Output:

- one row per ROI/object
- sample/group columns retained
- feature columns

### Smoothing

Add optional trace smoothing:

- rolling mean
- Savitzky-Golay
- Gaussian 1D

Rules:

- smoothing should be optional
- raw trace remains preserved
- smoothing parameters are recorded

### Event / Stimulus Alignment

Support event timing later in this phase.

Data model:

```python
@dataclass
class EventAnnotation:
    name: str
    time_index: int | None = None
    time_s: float | None = None
    notes: str | None = None
```

Use cases:

- stimulus onset
- drug addition
- optogenetic stimulation
- mechanical stimulation

Tools:

- annotate_event
- align_timecourse_to_event

### Time-Course Plots

Generate plot-ready data and optional figures:

- selected ROI trace
- all ROI traces
- mean ± SEM by sample
- mean ± SEM by group
- event-aligned traces

Plot generation should be optional. Result tables should remain the primary
data product.

## 2. Cell Morphology and Tissue-Level Summaries

Use cases:

- gut cell size distribution
- fat body cell area
- cell density
- morphology changes across treatment
- z-stack object volume

### Per-Cell Morphology

Add or expose:

- area
- perimeter
- circularity
- eccentricity
- major/minor axis length
- solidity
- extent
- volume for 3D labels
- surface area if reliable

Recommended:

```python
measure_cell_morphology(
    labels_layer: str,
    image_layer: str | None = None,
    properties: list[str] | None = None,
) -> dict
```

Physical units should be included when voxel size is available.

### Tissue-Level Summaries

Summaries:

- object count
- object density per area/volume
- area fraction
- intensity distribution
- cell size distribution

These should feed into experiment-level reports.

## 3. Colocalization and Channel Relationship Analysis

Existing:

- Manders coefficients
- Pearson correlation

Expand carefully.

### Per-Cell Colocalization

Compute channel relationship per segmented object.

Metrics:

- per-cell Pearson
- per-cell Manders
- mean intensity channel A/B
- ratio A/B

Recommended:

```python
measure_per_cell_colocalization(
    labels_layer: str,
    image_a: str,
    image_b: str,
    threshold_a: float | str = "otsu",
    threshold_b: float | str = "otsu",
) -> dict
```

### Threshold Strategy

Make thresholding explicit:

- zero
- Otsu
- scalar
- percentile

Record threshold method and threshold value.

### Visualization

Optional:

- scatter plot table
- channel intensity ratio table
- colocalization mask layer

## 4. Registration and Drift Correction

Important for live imaging.

### Use Cases

- tissue moves slightly during time-series imaging
- ROI measurement drifts away from cells
- channels need alignment

### Initial Scope

Start simple:

- translation-only frame registration
- reference frame selection
- apply shifts to image stack
- record shifts over time
- before/after QC summary

Possible methods:

- phase cross-correlation from scikit-image
- optical flow later if needed

Tools:

```python
estimate_drift(
    image_layer: str,
    reference_frame: int = 0,
    time_axis: int | str = "t",
) -> dict

apply_drift_correction(
    image_layer: str,
    shifts_table: str,
) -> dict
```

Rules:

- corrected layer is new layer
- original data is not overwritten
- shifts are saved as a table

## 5. Batch-Level Statistics

After Phase 3 experiment summaries exist, add basic statistical helpers.

### Initial Tests

Support:

- t-test
- Mann-Whitney U
- one-way ANOVA
- Kruskal-Wallis

Add later:

- multiple comparisons correction
- effect size
- confidence intervals

Recommended:

```python
compare_groups(
    table_name: str,
    value_col: str,
    group_col: str = "group",
    test: Literal["auto", "ttest", "mannwhitney", "anova", "kruskal"] = "auto",
) -> dict
```

Rules:

- prefer sample-level summaries over pooled cell-level statistics for group
  inference unless user explicitly asks otherwise
- report n samples and n objects separately
- do not overstate significance

## 6. Advanced Neural Process Analysis

Keep separate from default cell-analysis workflow.

Possible methods:

- Sholl analysis
- Strahler order
- branch pruning
- skeleton smoothing
- soma/process separation
- neurite length distribution
- branch angle analysis
- eventual FlyWire/neuPrint/navis integration

Do not implement database comparison until backend and data format are chosen.

## Tool Organization

Avoid overwhelming the default tool surface.

Suggested categories:

- Time Course
- Morphology
- Colocalization
- Registration
- Statistics
- Advanced Neural

The LLM can access all relevant tools, but manual UI should expose methods in
workflow panels.

## Report Integration

Reports should include method-specific sections:

### Time-Course

- normalization method
- baseline window
- response window
- extracted features
- event alignment if used

### Morphology

- measured shape features
- physical unit handling

### Colocalization

- channels
- threshold method
- metric definitions

### Registration

- reference frame
- registration method
- maximum/median shift

### Statistics

- test used
- sample n
- object n
- p-value
- effect size if available

## Tests

### Time-Course

- `DeltaF/F0` per ROI
- baseline window handling
- zero baseline handling
- AUC
- peak amplitude/time
- event alignment

### Morphology

- known 2D shape area/perimeter/circularity
- physical unit conversion
- 3D volume conversion

### Colocalization

- per-cell perfect overlap
- per-cell no overlap
- threshold methods

### Registration

- synthetic shifted movie recovers known shifts
- corrected movie aligns with reference

### Statistics

- two-group comparison
- multi-group comparison
- sample-level vs object-level warning

## Acceptance Criteria

- Time-course tables can be normalized without overwriting raw data.
- GCaMP-like response features can be extracted per ROI.
- Morphology measurements include shape descriptors.
- Per-cell colocalization is available when requested.
- Simple drift correction works on synthetic shifted movies.
- Group comparison tools produce clear, conservative outputs.
- Advanced neural tools remain isolated.
- Fast test suite passes.

## Suggested Implementation Order

1. Time-course normalization.
2. Time-course feature extraction.
3. Time-course plotting data.
4. Morphology measurements.
5. Per-cell colocalization.
6. Translation-only drift correction.
7. Batch statistics.
8. Advanced neural metrics.

## Open Questions

- What baseline default should be used when user does not specify one?
- Should CaLexA be treated as time-course or endpoint reporter by default?
- Which morphology metrics are most useful for gut/fat body cells?
- Should group statistics default to sample means rather than pooled cells?
- Which neural database/backend should be targeted first later?
