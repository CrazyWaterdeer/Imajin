# Phase 6B Spec: Neural Process Analysis

## Goal

Build a real neural process analysis workflow for Drosophila brain and VNC
confocal z-stacks.

This should cover local image-to-trace reconstruction, morphology measurement,
manual review, QC, and export. External connectome/database comparison should
remain optional and can be added later through plugins.

## Current State

The existing neural tooling is an early prototype:

- `skeletonize(layer)` converts a binary/Labels layer into a skeleton image.
- `extract_branch_metrics(skeleton_id)` creates a branch table with skan.
- `compute_morphology_descriptors(skeleton_id)` returns simple aggregate
  descriptors.
- `query_connectome(...)` is a stub.
- `classify_neuron_type(...)` is a stub.
- The neural tracer sub-agent correctly tells the user that connectome/NBLAST
  features are deferred.

This is useful as a starting point, but it is not yet a complete neural process
analysis module.

## Product Position

Neural process analysis should be a core advanced workflow, not only a plugin.

Core should provide:

- process enhancement
- process segmentation
- 3D skeletonization
- branch/node metrics
- pruning and review
- Sholl-style analysis
- trace export
- report-ready morphology summaries

Plugins can later provide:

- FlyWire connector
- neuPrint connector
- navis/NBLAST integration
- organism-specific reference datasets
- advanced model-based tracing backends

## Supported Data Assumptions

Initial target:

- Drosophila brain or VNC confocal z-stack
- sparse or moderately sparse reporter labeling
- one target channel selected by the user
- optional counterstain channel for anatomical context
- voxel scale available from metadata when possible

Out of scope for the first implementation:

- dense neuropil identity resolution
- fully automatic neuron identification from connectome
- reliable tracing through strongly overlapping labeled neurons
- synapse-level connectomics

The UI and agent should be honest when the input is too dense or ambiguous for
local image-based reconstruction.

## Workflow

### 1. Select Target Channel

The user selects a channel by:

- layer name
- channel role
- color alias such as green, red, UV, IR, or far red
- marker alias when available

The selected target is the signal used for process reconstruction.

### 2. Enhance Processes

Add a tool that prepares a z-stack for neurite/process segmentation.

Suggested tool:

```python
enhance_neural_processes(
    layer: str,
    method: str = "tubeness",
    sigma: float | tuple[float, ...] | None = None,
    background: str | None = "rolling_ball",
    normalize: bool = True,
) -> dict
```

Supported methods can start small:

- gaussian denoise
- rolling-ball or morphological background subtraction
- vesselness/tubeness filter when available
- percentile normalization

Requirements:

- preserve physical scale
- create a new image layer
- record preprocessing parameters in provenance
- avoid irreversible mutation of the original layer

### 3. Segment Processes

Add a process-specific segmentation step. Cellpose cell segmentation is not a
good default for thin processes.

Suggested tool:

```python
segment_neural_processes(
    layer: str,
    threshold: str = "otsu",
    min_size_um3: float | None = None,
    fill_holes: bool = False,
    keep_largest: bool = False,
) -> dict
```

Requirements:

- operate on 2D or 3D image layers
- respect voxel scale for size filtering
- output a binary/Labels layer
- return foreground fraction and component count
- warn if foreground fraction is too high or too low

Initial threshold modes:

- otsu
- yen
- triangle
- local/adaptive if feasible
- manual scalar threshold

### 4. Skeletonize in 3D

Keep the existing `skeletonize` tool but improve it for neural process work.

Required improvements:

- reject non-binary continuous images unless explicitly thresholded first
- preserve physical spacing
- support pruning short spurs after skeletonization
- return skeleton image layer and structured skeleton object
- compute node/edge tables, not only branch summary
- handle disconnected components explicitly

Suggested output tables:

- `skeleton_nodes`
- `skeleton_edges`
- `skeleton_components`

### 5. Review and Edit

Tracing needs manual review because confocal process signals are often noisy.

Minimum review features:

- show skeleton overlay on target channel
- show branch table
- select branch in table and highlight it in viewer
- mark branch as accepted/rejected
- prune branches below a length threshold
- keep original skeleton and reviewed skeleton separately

Suggested tools:

```python
prune_skeleton(
    skeleton_id: str,
    min_branch_length_um: float,
    remove_isolated: bool = True,
) -> dict
```

```python
set_branch_qc(
    skeleton_id: str,
    branch_ids: list[int],
    status: str,
    reason: str | None = None,
) -> dict
```

### 6. Soma and Region Annotation

Many useful neural metrics require a reference point or region.

Support optional annotations:

- soma point
- soma mask
- brain/VNC region label
- ROI volume

Suggested tools:

```python
set_soma_location(
    skeleton_id: str,
    point_layer: str | None = None,
    mask_layer: str | None = None,
) -> dict
```

```python
assign_neural_region(
    skeleton_id: str,
    region_layer: str,
) -> dict
```

These should be optional. Basic branch metrics should work without soma
annotation.

## Metrics

### Core Morphology Metrics

At minimum:

- total process length
- number of branches
- number of endpoints
- number of junctions
- number of connected components
- mean/median branch length
- branch length distribution
- tortuosity
- bounding box dimensions in physical units
- skeleton volume occupancy

### Branch Order Metrics

Add after basic branch tables are stable:

- terminal branch count
- internal branch count
- Strahler order
- path distance from soma if soma is annotated
- euclidean distance from soma if soma is annotated

### Sholl Analysis

Add a Sholl-style tool.

Suggested API:

```python
compute_sholl_analysis(
    skeleton_id: str,
    center: str = "soma",
    radius_step_um: float = 5.0,
    max_radius_um: float | None = None,
) -> dict
```

Outputs:

- intersections by radius
- peak intersection count
- radius at peak
- area under Sholl curve
- table name

### Intensity Along Processes

For reporter expression, skeleton geometry alone may not be enough.

Add later:

- mean intensity along skeleton
- intensity per branch
- intensity profile by path distance from soma
- intensity in process mask vs background

## Data Model

Introduce explicit neural analysis records instead of only keeping skan objects
in a private registry.

Suggested models:

```python
@dataclass
class NeuralTraceRecord:
    trace_id: str
    source_layer: str
    mask_layer: str | None
    skeleton_layer: str
    spacing: tuple[float, ...]
    units: tuple[str, ...] | None
    status: str
    parameters: dict[str, Any]
```

```python
@dataclass
class NeuralTraceQC:
    trace_id: str
    accepted: bool | None
    rejected_branch_ids: list[int]
    notes: str | None
```

Project persistence should eventually save these records and linked tables.

## Export

Add export support before database integration.

Required exports:

- branch metrics CSV
- node/edge CSV
- skeleton image TIFF
- SWC when a graph representation is available

Suggested tool:

```python
export_neural_trace(
    skeleton_id: str,
    output_path: str,
    format: str = "swc",
) -> dict
```

SWC export should document limitations if soma/root assignment is missing.

## Agent Behavior

The agent should distinguish these requests:

- "measure cells" -> core cell analysis workflow
- "trace processes" -> neural process workflow
- "compare to connectome" -> explain that local trace/export is available but
  DB comparison requires a backend/plugin
- "identify neuron type" -> do not overclaim; return local morphology summary
  and state that reference matching is not implemented unless a backend exists

The neural specialist prompt should be updated after the tools exist.

## UI Requirements

Add a neural process workflow panel or grouped tool section.

Minimum controls:

- target channel selector
- enhance button/parameters
- segment button/parameters
- skeletonize button/parameters
- prune length threshold
- metrics button
- export button

Visualization:

- original signal
- enhanced signal
- process mask
- skeleton overlay
- branch/node table
- QC status

## Tests

### Unit Tests

- process enhancement preserves shape and scale
- segmentation produces binary/Labels layer
- segmentation reports foreground fraction and component count
- skeletonization preserves spacing
- branch table contains physical length columns
- pruning removes short branches
- Sholl output has expected radius/intersection columns
- export writes expected file type

### Integration Tests

- synthetic 3D branching process -> enhance -> segment -> skeletonize -> metrics
- disconnected components are reported
- too-dense mask produces warning/QC flag
- reviewed/pruned trace keeps provenance
- report includes neural morphology summary when trace data exists

### Regression Tests

- existing `test_tools_trace.py` behavior remains valid or is migrated
- connectome/classification stubs continue to disclose `not_implemented`
  until real backends exist

## Acceptance Criteria

- A user can reconstruct a simple 3D neural process from a z-stack target
  channel without using a connectome backend.
- The app produces branch/node/morphology tables in physical units.
- The user can visually inspect and prune the skeleton.
- The app can export trace data for external analysis.
- Reports can include a neural morphology summary.
- The agent does not claim neuron identity unless a reference backend is
  actually configured.

## Suggested Implementation Order

1. Add `enhance_neural_processes`.
2. Add `segment_neural_processes`.
3. Upgrade `skeletonize` outputs and validation.
4. Add branch/node/component tables.
5. Add pruning.
6. Add core morphology summary.
7. Add Sholl analysis.
8. Add trace export.
9. Add UI grouping/review support.
10. Add report section.
11. Only then add connectome/reference comparison plugins.

## Open Questions

- Should the first implementation assume sparse single-neuron labeling?
- Is soma annotation usually available or should it be optional by default?
- Which export format matters most first: SWC, CSV graph tables, or both?
- Should process segmentation be threshold-based first, or should a learned
  model be added early?
