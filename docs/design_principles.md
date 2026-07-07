# Imajin Design Principles

The enduring design and architecture principles behind imajin — what the app treats
as fact vs. interpretation, how it models channels and samples, and the shape of the
codebase. (Extracted from the original project plan; the completed roadmap and
status notes were dropped.) For what the tools *do*, see
[analysis_capabilities.md](analysis_capabilities.md).

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

## Architecture — the useful center

The design has a good core idea, preserved as it evolves:

- manual dock and LLM chat both call the same tool functions
- tool calls produce provenance
- napari is the visual workspace
- tables are stored centrally and shown in a table dock

Pure analysis functions live under `analysis/` and are kept separate from the napari
adapter layer in `tools/`; manual and LLM calls go through one execution path
(validation, provenance, worker/main-thread dispatch, cancellation, error handling).

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

A reusable recipe captures:

- target channel query or annotation
- segmentation method and parameters
- measurement properties
- time-course settings
- preprocessing steps
- colocalization pairs, if any

Recipes allow applying the same workflow across control/treatment replicates.

## Reporting

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
