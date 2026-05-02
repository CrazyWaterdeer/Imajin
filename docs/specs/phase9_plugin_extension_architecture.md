# Phase 9 Spec: Plugin and Extension Architecture

## Goal

Keep Imajin's core app focused while allowing advanced, optional, or
organism-specific functionality to be added cleanly.

The core app should remain a general confocal/live fluorescence image analysis
tool. Specialized functionality should be packaged as plugins when it would
otherwise make the core app too complex.

## Why Plugins Matter

Imajin will likely need support for:

- connectome backends
- alternative segmentation models
- specialized live imaging analysis
- custom report sections
- new file metadata extractors
- external database connectors

Putting all of this directly in core would make the app hard to maintain and
hard to use.

## Core vs Plugin Boundary

### Core App Should Own

- file loading for common microscopy formats
- normalized metadata model
- channel annotations
- cell segmentation/measurement basics
- time-course ROI measurement basics
- experiment/session state
- provenance
- report generation framework
- execution service
- baseline neural process reconstruction and morphology export
- core UI shell
- plugin discovery/loading

### Plugins Should Own

- specialized algorithms
- optional heavy dependencies
- organism-specific databases
- advanced workflow templates
- custom report sections
- custom UI panels

## Example Plugins

### imajin-drosophila-connectome

Purpose:

- Drosophila connectome/database integration.

Possible features:

- FlyWire connector
- neuPrint connector
- navis integration
- NBLAST/reference morphology comparison

Baseline neural process reconstruction should exist in core. External database
connectors and reference-matching backends can remain plugins until the backend
choice is clear.

### imajin-live-gcamp

Purpose:

- specialized live reporter analysis.

Possible features:

- event alignment templates
- DeltaF/F0 presets
- peak detection presets
- stimulus annotation UI
- trace plot templates

Some basic time-course analysis belongs in core, but complex experimental
templates can be plugin-based.

### imajin-stardist

Purpose:

- alternative segmentation backend.

Possible features:

- StarDist model loader
- segmentation tool registration
- model-specific parameters

### imajin-ilastik

Purpose:

- classifier/project-based segmentation integration.

Possible features:

- ilastik project loading
- pixel classifier inference
- object probability maps

## Plugin Capabilities

Plugins may contribute:

1. tools
2. analysis recipes
3. metadata extractors
4. report sections
5. UI panels
6. settings pages
7. file readers
8. external connectors
9. QC checks

Each contribution type should have a stable interface.

## Plugin Metadata

Each plugin should declare metadata.

Example:

```json
{
  "name": "imajin-drosophila-connectome",
  "version": "0.1.0",
  "display_name": "Drosophila Connectome Tools",
  "description": "FlyWire, neuPrint, and morphology comparison helpers for Drosophila.",
  "imajin_min_version": "0.1.0",
  "capabilities": ["tools", "ui_panels", "report_sections"],
  "optional_dependencies": ["navis", "neuprint-python"],
  "entry_point": "imajin_drosophila_connectome.plugin:register"
}
```

## Entry Point Discovery

Use Python entry points for installed plugins.

Suggested group:

```toml
[project.entry-points."imajin.plugins"]
drosophila_connectome = "imajin_drosophila_connectome.plugin:register"
```

Core app discovers and registers plugins at startup.

Requirements:

- plugin load failures should not crash the app
- failures should be visible in doctor/settings
- plugin version compatibility should be checked

## Plugin Registration API

Suggested API:

```python
def register(plugin_context: PluginContext) -> None:
    plugin_context.register_tool(...)
    plugin_context.register_recipe(...)
    plugin_context.register_panel(...)
    plugin_context.register_report_section(...)
```

### PluginContext

```python
@dataclass
class PluginContext:
    imajin_version: str
    register_tool: Callable
    register_recipe: Callable
    register_metadata_extractor: Callable
    register_report_section: Callable
    register_panel: Callable
    register_settings_page: Callable
```

The exact API can evolve, but plugins should not modify global registries
directly unless through supported registration functions.

## Tool Extension

Plugins should be able to register tools using the same tool registry mechanism
as core.

Requirements:

- plugin tools appear with plugin namespace or metadata
- manual/LLM visibility flags work
- worker/main-thread execution policy works
- provenance records plugin name and version

Suggested tool metadata additions:

- plugin name
- plugin version
- category
- optional dependency status

## Recipe Extension

Plugins can contribute reusable workflow recipes.

Example:

```python
register_recipe(
    name="GCaMP event-aligned response",
    category="timecourse",
    steps=[...],
)
```

Recipes should be visible but not automatically executed without user intent.

## Metadata Extractor Extension

Plugins can add metadata extractors for:

- special file formats
- vendor-specific fields

Rules:

- extractors should add facts, not biological interpretations
- filename parsing should not be a default source of biological meaning
- user-confirmed annotations should override inferred values

## Report Section Extension

Plugins can add report sections.

Example:

- neural morphology summary
- connectome candidate matches
- GCaMP response summary

Report sections should declare:

- required data
- section title
- renderer function

## UI Panel Extension

Plugins can add dock widgets or panels.

Rules:

- plugin UI should not block core app startup if it fails
- plugin panels should be optional
- heavy imports should be lazy

## Settings Extension

Plugins may need settings:

- API endpoints
- database credentials
- model paths
- default parameters

Rules:

- secrets must use app secret/config mechanism
- project files must not store secrets
- plugin settings should be namespaced

## External Connector Policy

External DB/API connectors should be plugins unless broadly useful.

Examples:

- FlyWire
- neuPrint
- CAVE/MICrONS
- Allen data APIs

Connector requirements:

- clear authentication handling
- timeout/error handling
- no secrets in provenance
- reproducible query metadata

## Dependency Policy

Plugins may have optional heavy dependencies.

Core should not require:

- navis
- neuprint-python
- fafbseg
- ilastik
- stardist
- specialized deep learning models

Doctor should report plugin dependency status.

## Safety and Stability

Plugin load failure should:

- be caught
- be reported
- not crash core app
- not prevent unrelated plugins from loading

Plugin tools should:

- validate inputs
- record provenance
- respect cancellation where possible
- avoid mutating unrelated state

## Version Compatibility

Core should check:

- plugin declared minimum imajin version
- plugin API version

If incompatible:

- disable plugin
- show warning
- do not crash

## Provenance

Plugin tool calls should record:

- tool name
- plugin name
- plugin version
- inputs/outputs
- duration
- driver

This matters for report reproducibility.

## UI/UX

Plugin manager UI should eventually show:

- installed plugins
- enabled/disabled status
- version
- capabilities
- dependency status
- errors

Initial version can be a doctor/report output only.

## Tests

### Discovery

- discovers fake plugin via entry point or test registry
- incompatible plugin is skipped
- plugin load error is reported, not raised

### Tool Registration

- plugin tool appears in registry
- plugin tool can be called
- provenance includes plugin metadata if implemented

### UI Registration

- plugin panel registration succeeds
- failed panel registration does not crash app

### Report Extension

- plugin report section appears when data available
- missing data skips section cleanly

### Optional Dependencies

- missing optional dependency disables plugin feature with clear warning

## Acceptance Criteria

- Core app can discover and load plugins.
- Plugin load failure does not crash app.
- Plugins can register tools through supported API.
- Plugin tools work with execution service and provenance.
- Plugin metadata is visible in doctor or plugin listing.
- Connectome functionality has a clear plugin path.

## Suggested Implementation Order

1. Define plugin metadata schema.
2. Define plugin context/registration API.
3. Add plugin discovery via entry points.
4. Add plugin status registry.
5. Allow plugin tool registration.
6. Add doctor output for plugins.
7. Add report section extension point.
8. Add UI panel extension point.
9. Add tests with fake plugins.
10. Move connectome/database integrations toward plugin boundary later.

## Open Questions

- Should plugins be enabled by default after installation?
- Should users be able to disable plugins from the UI?
- Should plugin settings live in global config, project config, or both?
- How strict should API version compatibility be?
