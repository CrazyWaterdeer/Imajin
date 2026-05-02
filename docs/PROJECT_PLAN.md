# Imajin Project Plan

## Purpose

Imajin is a napari-based analysis app for general confocal and live fluorescence
imaging. The primary use case is Drosophila tissue imaging, especially gut,
brain, and VNC samples, but the design should remain general enough for other
users and file naming styles.

The app should help users move from raw microscopy files to reproducible
single-sample and experiment-level results:

1. Load microscopy files.
2. Extract acquisition metadata.
3. Identify target and reference channels.
4. Segment cells or ROIs.
5. Measure intensity, size, and morphology.
6. Measure intensity over time for live/time-series data.
7. Organize multiple files into user-confirmed groups.
8. Generate reproducible reports with methods and group summaries.

Neural process tracing is an important advanced workflow for Drosophila
brain/VNC data. It should be available as a focused workflow while staying out
of the default cell-analysis path. Connectome comparison is a later optional
extension.

## Core Product Principles

### Metadata vs Meaning

File metadata provides physical and acquisition facts:

- axes and shape
- voxel size
- channel count
- excitation/emission wavelengths
- objective
- detector/filter/laser settings where available
- time-series and z-stack structure

Biological meaning must come from the user:

- target channel
- counterstain channel
- control/treatment group
- genotype or condition
- tissue region
- replicate identity

The app may suggest interpretations, but it should not silently turn suggestions
into facts.

### No Hard-Coded Filename Parsing

Filenames are user-defined and inconsistent across datasets. The app must not
hard-code parsing rules for genotype, sex, tissue, condition, region, or
replicate.

Allowed:

- store the original filename
- show the filename to the user
- let the agent ask how to interpret it
- store user-confirmed annotations

Disallowed:

- automatic control/treatment inference from filename
- project-specific regex for sample names
- hidden assumptions based on token order

### Simple Channel Roles

Keep the user-facing channel model simple:

- `target`: default channel for segmentation, intensity, size, morphology, and
  time-course measurement
- `counterstain`: reference/localization channel
- `ignore`: excluded from analysis
- `unknown`: not yet assigned

Channel color is separate from role:

- `green`
- `red`
- `uv`
- `ir` / far red

Far red is only a color. It is often phalloidin, TOPRO, or another reference
marker, but it may also be a target in some experiments. Do not infer its role
from color alone.

## Supported Data Shapes

The internal model should support missing axes cleanly.

- `CZYX`: multichannel z-stack confocal
- `TCZYX`: time-series z-stack confocal
- `TCYX`: live imaging or 2D time-lapse
- `CYX`: multichannel 2D image
- `YX`: single-channel 2D image

The canonical conceptual axis order is `T, C, Z, Y, X`, but loaded arrays should
preserve file axes where practical and carry explicit axis metadata.

## Main Workflows

### Single Z-Stack Cell Analysis

1. Load file.
2. Extract metadata and channel information.
3. Resolve or ask for target channel.
4. Segment cells/ROIs in the target channel.
5. Measure per-cell area, centroid, mean/max/min intensity.
6. Optionally compare target channels for colocalization.
7. Generate report entries with metadata and methods.

### Live Imaging / Time-Series Analysis

1. Load movie or time-series file.
2. Extract metadata, especially `T` axis and time interval if available.
3. Resolve or ask for target channel.
4. Create or load ROI labels.
5. Measure ROI intensity over time.
6. Optionally add simple tracking or registration later.
7. Export time-course table and report summary.

Initial priority is correct ROI intensity-over-time measurement. Tracking is
useful, but secondary.

### Experiment-Level Batch Analysis

1. Load or register multiple files.
2. Ask user to assign samples to groups.
3. Store user-confirmed sample annotations.
4. Apply the same analysis recipe to each sample.
5. Add sample/group columns to all result tables.
6. Summarize results by sample and group.
7. Generate a final report with acquisition metadata, methods, per-sample
   results, and group-level statistics.

The agent should help the user build the group table conversationally, but the
stored structured metadata must come from user confirmation.

### Neural Process Module

This is an important secondary workflow, not just a distant plugin idea.
Baseline process reconstruction should live in core because Drosophila brain
and VNC imaging commonly needs it.

1. Enhance or segment neural processes.
2. Build 3D skeletons or traces.
3. Extract morphology metrics.
4. Review, prune, and export traces.
5. Later, compare against external Drosophila connectome resources.

External database connectors can be plugins, but local reconstruction,
measurement, QC, and export should not wait for database/backend choices.

## Architecture Direction

### Current Useful Center

The current design has a good core idea:

- manual dock and LLM chat both call the same tool functions
- tool calls produce provenance
- napari is the visual workspace
- tables are stored centrally and shown in a table dock

This should be preserved.

### Needed Structural Improvements

#### Analysis Session

Introduce an `AnalysisSession` or equivalent state owner that holds:

- viewer reference
- tables
- sample annotations
- channel annotations
- provenance logger
- current file/dataset metadata
- model caches
- dispatch/execution services

This should replace scattered global state over time.

#### Tool Execution Service

Manual and LLM calls should go through one execution path:

- validation
- provenance driver
- worker/main-thread dispatch
- cancellation
- progress reporting
- error handling

The current manual dock runs tools synchronously. That should be changed so
large analyses do not block the UI.

#### Core Analysis Layer

Separate pure analysis functions from napari adapter functions.

Suggested structure:

- `analysis/segmentation.py`
- `analysis/measurement.py`
- `analysis/timelapse.py`
- `analysis/registration.py`
- `analysis/colocalization.py`
- `analysis/neurotrace.py`

Tools should mostly:

1. snapshot layer data
2. call pure analysis code
3. add output layers/tables
4. record results

#### IO Metadata Layer

Create a consistent metadata model across loaders:

- `DatasetMetadata`
- `ChannelMetadata`
- `AcquisitionMetadata`

LSM should be treated as the first high-priority implementation target because
it contains rich Zeiss metadata.

## Data Models

### Channel Metadata

Physical/acquisition facts from files.

Important fields:

- channel index
- display name
- dye name
- inferred color
- excitation wavelength
- emission wavelength/range
- laser power
- detector gain
- pinhole
- detector/filter names

Metadata may include a role suggestion, but not a confirmed biological role.

### Channel Annotation

User-confirmed meaning:

- layer name
- role: `target`, `counterstain`, `ignore`, `unknown`
- marker
- biological target
- notes

Default role should be `unknown`.

### Sample Annotation

User-confirmed experiment metadata:

- sample name
- group
- files
- layers
- notes
- extra free-form key/value fields

Filename-derived guesses should not populate this automatically.

### Analysis Recipe

A reusable recipe should eventually capture:

- target channel query or annotation
- segmentation method and parameters
- measurement properties
- time-course settings
- preprocessing steps
- colocalization pairs, if any

Recipes allow applying the same workflow across control/treatment replicates.

## Reporting Direction

Reports should distinguish:

1. Acquisition metadata
2. User annotations
3. Analysis operations
4. Results and summaries

Reports should include:

- file type
- axes and shape
- voxel size
- z-slices/timepoints
- objective
- channel wavelengths/dyes/filters/detectors
- target/counterstain annotations
- segmentation and measurement methods
- result tables
- sample/group summaries

Reports must not claim unconfirmed biological meaning.

## Prioritized Roadmap

### Phase 1: Stabilize Metadata and Channel Handling

- finalize channel metadata model
- improve LSM extraction
- align OME/CZI metadata outputs
- make `resolve_channel` metadata-first, annotation-second
- separate confirmed roles from suggestions
- update reports to include acquisition metadata

### Phase 2: Strengthen Core Cell Analysis

- use target channel annotations consistently
- add physical-unit measurement columns
- improve z-stack handling and anisotropic voxel awareness
- make preprocessing/segmentation/measurement more robust for target-channel
  workflows

### Phase 3: Time-Series and Live Imaging

- refine ROI intensity-over-time tables
- preserve time interval metadata
- add optional drift correction/registration
- add simple ROI tracking later
- add plots/export for time-course data

### Phase 4: Experiment-Level Workflow

- add project/experiment session model
- add sample/group assignment workflow
- support applying an analysis recipe to multiple files
- add group-level result tables
- generate experiment-level reports

### Phase 5: UI and Execution Unification

- move manual dock to background execution
- add progress/cancel for long jobs
- expose workflow-oriented panels instead of a flat tool list
- improve error messages and recovery

### Phase 6: Neural Process Analysis

- build a proper process reconstruction pipeline
- add pruning, QC, and morphology metrics
- add SWC/CSV export
- keep external Drosophila database backends optional
- add connectome comparison only after backend selection

## Open Questions

- Which segmentation method should be the default for weak reporter channels?
- Should z-stack cell measurement default to 2D per-slice, 3D objects, or both?
- How should time intervals be represented when TIFF metadata is missing?
- What minimum group-level statistics should the first experiment report include?
- Should project/session state be saved as JSON, SQLite, or another format?

## Immediate Next Step

Before adding more analysis features, implement the metadata/channel cleanup:

1. metadata-first channel resolution
2. confirmed vs suggested channel roles
3. acquisition metadata in reports
4. no filename hard-coding

This prevents later batch/report functionality from being built on unstable
assumptions.
