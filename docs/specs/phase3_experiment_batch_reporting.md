# Phase 3 Spec: Experiment, Batch, and Reporting Workflow

## Goal

Support the way experiments are actually organized: multiple files and
replicates grouped by user-confirmed biological conditions, with a repeatable
analysis recipe and a final report.

The core user request should become possible:

> "These files are control 1-4 and these are treatment 1-4. Analyze them the
> same way and make a report."

This phase builds on:

- Phase 1: metadata and channel handling
- Phase 2: reliable single-sample cell analysis

## Non-Goals

- Do not implement neural/connectome workflows here.
- Do not infer group/condition/replicate from filenames.
- Do not build a full LIMS.
- Do not require a rigid filename convention.
- Do not require all files to be loaded in napari simultaneously.

## Core Principle

Batch analysis should be **agent-assisted, user-confirmed**, and reproducible.

The app can help the user organize files, but structured experiment metadata
must come from user confirmation.

Metadata facts:

- file path
- image dimensions
- channel metadata
- voxel size
- acquisition settings

User-confirmed meaning:

- sample name
- group
- condition
- genotype
- tissue
- region
- replicate
- notes

## User Workflow

### 1. Register Files

The user may:

- load files one at a time
- drag/drop multiple files
- tell the agent paths or folders
- use files already open in napari

The app should store file records without requiring every file to remain loaded
in memory.

### 2. Build Sample Table

The agent should help build a sample table conversationally.

Example:

User:

```text
J41 + 1234 vF midgut R3 1-4 are control.
J41 + 5678 vF midgut R3 1-4 are treatment.
```

Allowed behavior:

- ask the user to confirm the mapping
- store the confirmed group/sample metadata
- keep original filenames as text

Disallowed behavior:

- silently parse `J41`, `vF`, `midgut`, `R3`, or trailing numbers into fields
- infer control/treatment from filename without user confirmation

### 3. Define Analysis Recipe

The app should capture a reusable recipe:

- target channel query or annotation
- segmentation method and parameters
- preprocessing choice
- measurement properties
- time-course settings if applicable
- colocalization channel pairs if requested

The recipe should be explicit enough to re-run.

### 4. Run Per-Sample Analysis

For each sample/file:

1. Load data.
2. Resolve target channel.
3. Run the recipe.
4. Store result tables.
5. Attach sample/group columns.
6. Record provenance.
7. Release memory if needed before moving to the next file.

### 5. Summarize Across Groups

Aggregate per-cell/per-ROI measurements into sample-level and group-level
tables.

The first version should support simple summaries:

- object count per sample
- mean/median intensity per sample
- mean/median area or volume per sample
- standard deviation
- standard error
- group mean/median
- number of samples per group
- number of objects per group

Statistical tests can come later.

### 6. Generate Report

Report should include:

- experiment metadata
- sample/group table
- acquisition metadata summary
- analysis recipe
- per-sample result summary
- group-level result summary
- methods/provenance
- warnings and exclusions

## Data Models

### ExperimentSession

Recommended model:

```python
@dataclass
class ExperimentSession:
    id: str
    name: str | None = None
    created_at: str | None = None
    files: dict[str, FileRecord] = field(default_factory=dict)
    samples: dict[str, SampleAnnotation] = field(default_factory=dict)
    recipes: dict[str, AnalysisRecipe] = field(default_factory=dict)
    runs: dict[str, AnalysisRun] = field(default_factory=dict)
    notes: str | None = None
```

This does not need to replace all current global state immediately. It can start
as a serializable structure used for batch/report workflows.

### FileRecord

```python
@dataclass
class FileRecord:
    file_id: str
    path: str
    original_name: str
    file_type: str | None = None
    metadata_summary: dict[str, Any] = field(default_factory=dict)
    load_status: Literal["unloaded", "loaded", "failed"] = "unloaded"
    notes: str | None = None
```

Rules:

- `original_name` is stored exactly.
- No hard-coded parsing into condition/tissue/replicate.
- `metadata_summary` stores acquisition facts only.

### SampleAnnotation

```python
@dataclass
class SampleAnnotation:
    sample_id: str
    sample_name: str
    group: str | None = None
    file_ids: list[str] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    notes: str | None = None
    extra: dict[str, str] = field(default_factory=dict)
```

`extra` is for user-confirmed fields such as:

- genotype
- tissue
- sex/stage
- region
- replicate
- condition

The app should not invent these fields from filenames.

### AnalysisRecipe

```python
@dataclass
class AnalysisRecipe:
    recipe_id: str
    name: str
    target_channel: str | None = None
    preprocessing: list[dict[str, Any]] = field(default_factory=list)
    segmentation: dict[str, Any] = field(default_factory=dict)
    measurement: dict[str, Any] = field(default_factory=dict)
    timecourse: dict[str, Any] | None = None
    colocalization: list[tuple[str, str]] = field(default_factory=list)
    notes: str | None = None
```

Examples:

- target channel: `"green"` or user-confirmed target annotation
- segmentation: `{"tool": "cellpose_sam", "do_3D": true, "diameter": null}`
- measurement: `{"properties": ["area", "centroid", "mean_intensity"]}`

### AnalysisRun

```python
@dataclass
class AnalysisRun:
    run_id: str
    sample_id: str
    file_id: str
    recipe_id: str
    status: Literal["pending", "running", "complete", "failed"]
    table_names: list[str] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
```

## Tools

### register_files

Register files without necessarily loading all into napari.

```python
register_files(paths: list[str]) -> dict
```

Return:

- file records
- supported/unsupported status
- metadata summary if cheap to inspect

### annotate_samples

Create/update sample annotations from user-confirmed mappings.

```python
annotate_samples(
    samples: list[dict[str, Any]]
) -> dict
```

Each sample dict can contain:

- sample_name
- group
- files
- notes
- extra

### list_experiment

Return current experiment/session structure:

- files
- samples
- recipes
- runs

### create_analysis_recipe

Store a reusable analysis recipe.

```python
create_analysis_recipe(
    name: str,
    target_channel: str,
    segmentation: dict | None = None,
    measurement: dict | None = None,
    preprocessing: list[dict] | None = None,
    timecourse: dict | None = None,
)
```

### run_recipe_on_samples

Run a stored recipe over selected samples.

```python
run_recipe_on_samples(
    recipe_name: str,
    sample_names: list[str] | None = None,
)
```

Behavior:

- iterate samples one by one
- load file
- run recipe
- store outputs with sample/group columns
- record success/failure per sample

### summarize_experiment

Aggregate result tables.

```python
summarize_experiment(
    measurement: str,
    group_by: str = "group",
    sample_col: str = "sample_name",
)
```

Should return:

- sample-level summary table
- group-level summary table

### generate_experiment_report

Generate report using:

- experiment structure
- recipe
- runs
- summaries
- provenance

## Agent Behavior

### When User Provides File Groups

User:

```text
These four are control and these four are treatment.
```

Agent should:

1. Identify referenced files or ask for clarification if not enough.
2. Show concise mapping.
3. Ask for confirmation if mapping is ambiguous.
4. Call `annotate_samples`.

### When User Requests Batch Analysis

User:

```text
Analyze all samples the same way.
```

Agent should:

1. Check sample annotations exist.
2. Check target channel/recipe exists.
3. If missing, ask one focused question.
4. Create or reuse recipe.
5. Run per-sample analysis.
6. Summarize results.
7. Offer report generation or generate if explicitly requested.

### Avoid Over-Asking

Ask only for required missing information:

- Which files belong to which group?
- Which channel is target?
- Is this a z-stack or time-course workflow if metadata is ambiguous?

Do not ask for optional details before analysis can proceed.

## Tables

Per-object measurement tables should include:

- sample_id
- sample_name
- group
- file_id
- source_file
- source_layer
- label
- measurement columns

Time-course tables should include:

- sample_id
- sample_name
- group
- file_id
- source_file
- source_layer
- label
- time_index
- time_s if available
- intensity columns

Summary tables:

- sample-level table
- group-level table

## Memory and Loading Policy

Batch analysis should not require all files loaded simultaneously.

Recommended:

- register many files cheaply
- load one file/sample at a time for analysis
- close/remove intermediate layers after each sample if needed
- retain result tables and metadata

If user wants visual inspection, keep selected layers loaded.

## Report Requirements

Experiment report sections:

1. Overview
   - experiment name
   - number of groups
   - number of samples
   - number of files
2. Sample table
   - sample name
   - group
   - file
   - user notes/extra fields
3. Acquisition metadata summary
   - per file or collapsed if identical
4. Analysis recipe
   - target channel
   - segmentation settings
   - measurement settings
5. Results
   - sample-level summary
   - group-level summary
6. Methods
   - provenance-backed method paragraph
7. Warnings
   - failed samples
   - missing metadata
   - ambiguous channels resolved by user

## Tests

### Experiment State

- register files creates file records
- sample annotations store user-confirmed group
- filename is not parsed into structured fields
- experiment state can be serialized/deserialized

### Recipe

- create recipe with target channel
- recipe can be listed
- recipe preserves segmentation/measurement settings

### Batch Run

Use tiny synthetic files or fake loader.

- run recipe on two samples
- result tables include sample/group columns
- failed sample does not abort entire batch
- memory cleanup path does not remove result tables

### Summary

- sample-level summary computes mean/median/count
- group-level summary groups by user-confirmed group
- missing group handled gracefully

### Report

- report includes sample table
- report includes recipe
- report includes group summary
- report does not infer filename tokens
- report lists failed samples

## Acceptance Criteria

- User can register multiple files.
- User can annotate files/samples into groups without filename hard-coding.
- User can define one analysis recipe.
- App can apply recipe to multiple samples one by one.
- Result tables contain sample/group/file columns.
- App can generate sample-level and group-level summaries.
- Report includes experiment organization and analysis results.
- Fast test suite passes.

## Suggested Implementation Order

1. Add serializable experiment/session models.
2. Add file registration tools.
3. Expand sample annotation tools.
4. Add analysis recipe model/tools.
5. Add recipe runner for one sample.
6. Extend to multiple samples.
7. Add sample/group columns to result tables.
8. Add summary tools.
9. Add experiment report generator.
10. Add tests.

## Open Questions

- Should experiment state be stored in JSON first, or move directly to SQLite?
- Should batch runs remove napari layers after each sample by default?
- What initial group statistics are enough: mean/median/std/SEM/count?
- Should plots be generated in this phase or deferred?
- How should failed samples be retried?
