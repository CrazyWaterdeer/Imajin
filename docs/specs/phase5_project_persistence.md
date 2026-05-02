# Phase 5 Spec: Project Persistence and Reproducibility

## Goal

Make Imajin analyses persistent, reloadable, and reproducible.

By this phase, the app will manage:

- file records
- channel metadata and annotations
- sample/group annotations
- analysis recipes
- analysis runs
- result tables
- provenance logs
- reports
- job history

This state must survive app restarts and be shareable enough for review or
re-analysis.

## Non-Goals

- Do not build a full database-backed LIMS.
- Do not copy large raw microscopy files by default.
- Do not store API keys or secrets in project files.
- Do not solve long-term archival formats beyond practical local projects.
- Do not require cloud sync.

## Recommended Storage Strategy

Use a project folder instead of a single binary file.

Initial recommended layout:

```text
my_experiment.imajin/
  project.json
  files.json
  samples.json
  channels.json
  recipes.json
  runs.json
  jobs.json
  tables/
    sample_001_measurements.parquet
    sample_002_measurements.parquet
    group_summary.parquet
  provenance/
    session_abcd1234.jsonl
  reports/
    report.html
    report.md
  logs/
    warnings.jsonl
```

Rationale:

- JSON is easy to inspect and version.
- Parquet is efficient for tables and preserves dtypes better than CSV.
- JSONL provenance is already compatible with the current app direction.
- A folder is easier to debug than a monolithic file.

SQLite can be considered later if project state becomes too complex.

## Core Principles

### Raw Images Are References

Do not copy raw image files into the project by default.

Store:

- original path
- optional relative path
- file size
- modified time
- optional checksum if requested

The project should warn if a file is missing or appears changed.

### Project Files Exclude Secrets

Do not store:

- API keys
- local provider credentials
- user secrets
- private tokens

Provider settings should remain in app config, not project state.

### Version Every Schema

Every project should have a schema version:

```json
{
  "schema_version": 1,
  "imajin_version": "0.1.0"
}
```

Future migrations should be explicit.

### Reproducibility Over Convenience

Store enough to understand and rerun an analysis:

- file references
- metadata summaries
- user annotations
- recipes
- tool parameters
- result table paths
- provenance records

Do not rely only on napari layer state.

## Data Files

### project.json

Top-level project metadata.

Example:

```json
{
  "schema_version": 1,
  "project_id": "proj_20260502_abcd",
  "name": "midgut_gcamp_treatment",
  "created_at": "2026-05-02T10:00:00Z",
  "updated_at": "2026-05-02T11:00:00Z",
  "imajin_version": "0.1.0",
  "notes": ""
}
```

### files.json

Stores registered raw file records.

Fields:

- file_id
- original_path
- relative_path
- original_name
- file_type
- size_bytes
- modified_time
- checksum optional
- metadata_summary
- status
- notes

Rules:

- Do not infer group/sample metadata from filename.
- Keep original filename exactly.
- Use metadata summary for acquisition facts only.

### samples.json

Stores user-confirmed sample annotations.

Fields:

- sample_id
- sample_name
- group
- file_ids
- layer_names if relevant
- notes
- extra user-confirmed fields

`extra` can contain:

- genotype
- tissue
- sex/stage
- region
- replicate
- condition

These values must be user-confirmed.

### channels.json

Stores channel annotations and normalized metadata references.

Separate:

1. physical/acquisition metadata from file
2. user-confirmed annotations

Recommended structure:

```json
{
  "metadata": {
    "file_001:0": {
      "index": 0,
      "color": "green",
      "excitation_wavelength_nm": 488.0,
      "dye_name": "Alexa Fluor 488"
    }
  },
  "annotations": {
    "file_001:0": {
      "role": "target",
      "marker": "GCaMP",
      "biological_target": "gut cells"
    }
  }
}
```

### recipes.json

Stores reusable analysis recipes.

Fields:

- recipe_id
- name
- target_channel
- preprocessing steps
- segmentation settings
- measurement settings
- time-course settings
- colocalization pairs
- notes

Recipes should be executable without relying on LLM memory.

### runs.json

Stores each analysis run.

Fields:

- run_id
- sample_id
- file_id
- recipe_id
- status
- started_at
- finished_at
- table paths
- generated layer names
- summary
- warnings
- error
- provenance session id

### jobs.json

Stores optional job history.

Minimum:

- job id
- title
- source
- status
- start/end time
- linked run id
- error

Long-term job history can be pruned, but analysis run records should remain.

### tables/

Store result tables.

Preferred format:

- Parquet

Fallback:

- CSV if Parquet unavailable

Every table should include enough columns for traceability:

- project_id
- sample_id
- sample_name
- group
- file_id
- source_file
- source_layer
- recipe_id
- run_id

### provenance/

Copy or link session JSONL logs.

Each run should know which provenance file(s) describe it.

### reports/

Generated reports:

- HTML
- Markdown
- optional exported figures later

## Save / Load Behavior

### Save Project

Save should:

1. write JSON metadata files atomically
2. write result tables
3. copy provenance logs
4. update `updated_at`
5. avoid writing secrets

Atomic write recommendation:

- write to temporary file
- fsync if practical
- rename into place

### Load Project

Load should:

1. validate schema version
2. load project metadata
3. load files/samples/channels/recipes/runs
4. check raw file references
5. load result table registry
6. report missing or moved files
7. not automatically load every raw image into napari

### Autosave

Autosave should trigger after:

- sample annotation changes
- channel annotation changes
- recipe creation/update
- analysis run completion
- report generation

Autosave should not trigger during every small UI state change.

## Path Handling

Store both:

- absolute path
- path relative to project folder if possible

On load:

1. try absolute path
2. try relative path
3. if missing, mark file missing and ask user to relink

This helps if a project folder is moved with its raw data folder.

## Missing File Handling

If a raw image file is missing:

- keep sample and run records
- keep existing result tables
- mark file status as missing
- allow user to relink
- do not delete results automatically

If file size or modification time changed:

- warn user
- do not invalidate results automatically
- offer re-run option

## Project Commands / Tools

### create_project

```python
create_project(path: str, name: str | None = None) -> dict
```

Creates project folder and initial JSON files.

### save_project

```python
save_project() -> dict
```

Writes current project state to disk.

### load_project

```python
load_project(path: str) -> dict
```

Loads project metadata and result table registry.

### register_project_files

Can extend Phase 3 `register_files`.

Should associate file records with current project.

### relink_file

```python
relink_file(file_id: str, new_path: str) -> dict
```

Updates missing/moved file reference.

### export_project_summary

```python
export_project_summary(path: str) -> dict
```

Writes human-readable project summary without raw data.

## UI Requirements

Add project controls:

- New Project
- Open Project
- Save Project
- Save Project As
- Project status indicator

Add project summary panel:

- project name
- project path
- number of files
- number of samples
- number of recipes
- number of runs
- missing file warnings

Add relink workflow:

- show missing file
- choose replacement path
- validate size/name if possible

## Report/Reproducibility Requirements

A report should cite:

- project id/name
- recipe id/name
- run ids
- provenance session ids
- raw file names
- file metadata summaries

The report should be reproducible from project state plus raw files.

## Schema Migration

Add a migration module from the start, even if only version 1 exists.

Suggested:

```python
CURRENT_SCHEMA_VERSION = 1

def migrate_project(data: dict) -> dict:
    ...
```

If schema is newer than app supports:

- fail clearly
- do not corrupt project

If schema is older:

- migrate in memory
- ask or backup before writing migrated version

## Tests

### Project Creation

- create project folder
- initial JSON files exist
- schema version present

### Save/Load

- save sample annotations
- save channel annotations
- save recipe
- save run record
- reload and compare

### Tables

- save DataFrame to Parquet
- load table registry
- preserve sample/group columns

### Paths

- absolute path stored
- relative path stored when possible
- missing file marked
- relink updates file record

### Secrets

- ensure API keys are not written to project files

### Migration

- version 1 project loads
- unsupported future version raises clear error

## Acceptance Criteria

- User can create a project folder.
- User can save and reload sample/channel annotations.
- User can save and reload recipes and run records.
- Result tables survive app restart.
- Missing raw files are detected without deleting results.
- Project files contain no API keys.
- Reports can reference project/run/provenance identifiers.
- Fast test suite passes.

## Suggested Implementation Order

1. Define project data models.
2. Implement project folder creation.
3. Implement JSON save/load helpers with atomic writes.
4. Implement table persistence.
5. Connect current sample/channel/recipe/run state.
6. Add project tools.
7. Add minimal project UI/menu actions.
8. Add missing-file detection and relink.
9. Add schema version/migration helper.
10. Add tests.

## Open Questions

- Should project folders use `.imajin/` suffix or plain directory?
- Should result tables default to Parquet even if users prefer CSV exports?
- Should raw files optionally be copied into the project?
- Should autosave be enabled by default?
- Should job history be persisted permanently or only run history?
