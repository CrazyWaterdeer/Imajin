# Unified Analysis Bundle

**Status**: Draft
**Date**: 2026-05-14
**Type**: Design
**Supersedes (partial)**: `2026-05-09-result-bundle-organization-design.md` (extends the bundle layout to cover all output writers and removes flat fallback folders)

## Problem

A typical analysis currently produces files in three places at once:

- A timestamp-prefixed bundle folder (`bundles/<timestamp>_<name>/`) holding `metadata.json`, `labels/`, `qc/`, `tables/combined.csv`.
- Flat category folders next to it (`figures/`, `stats/`, `segmentation_qc/`) where every tool call accumulates files. Different analyses pile up in the same folder with numeric collision suffixes (`stats_compare__measurements__mean_intensity_27.csv`).
- An append-only `manifest.jsonl` at the results root and, when files anchor next to source data, a parallel `<anchor>/.imajin/manifest.jsonl`.

The flat folders defeat the bundle abstraction. After a few sessions it is impossible to tell which figures and which stats CSVs belong to which analysis without opening each file and reading its embedded metadata. The segmentation-QC pathway compounds the problem: the QC PNG is written first to a flat `segmentation_qc/` folder, then copied into `bundle/qc/<sample>.png`, and the original is deleted in a best-effort cleanup that occasionally fails.

Inside the bundle itself there is also redundancy. Stats CSVs are split per `(value_col, level)` even though their schemas are identical. `tables/<name>.spec.json` duplicates what `metadata.json` could carry. Two write paths to the same `samples[].outputs` field exist (one filled by `populate_sample_outputs`, one by `save_result_bundle`).

## Goals

- One analysis = one timestamp-prefixed folder containing every artifact from that analysis (labels, tables, QC, figures, stats, metadata).
- No flat category folders at the results root. No file is written outside an active bundle.
- Each tool call routes its output through the bundle abstraction; no tool has its own fallback.
- Stats CSVs are consolidated by source table; `value_col` and `level` become columns, not filenames.
- `metadata.json` is the single index of bundle contents. No `manifest.jsonl`, no per-table `.spec.json`.

## Non-goals

- Migrating existing bundles or flat folders to the new layout. Old outputs are left in place; the change applies only to new outputs.
- Changing `auto3d_runs/<timestamp>_<name>/` outputs produced by external scripts not part of the in-app tool registry.
- Changing tools that take an explicit user-specified path (`export_table`, `export_neural_trace`, video screenshots). Those continue to honor the caller's path.
- Reorganizing the in-memory `state` tables and QC records. This design is about disk layout.

## Decisions

| Decision | Choice | Notes |
|---|---|---|
| Scope of "one analysis" | Agent task | Speaker opens a bundle at task start; tool calls within the task share it. |
| Standalone tool call (no active bundle) | Auto ad-hoc bundle | First write lazily creates a process-wide `<timestamp>_adhoc` bundle; subsequent ad-hoc writes reuse it. |
| Bundle location | Anchor when known, else `~/Documents/Imajin/results/` | `bundles/` middle directory is removed. |
| Bundle inner layout | Keep current subdirectory layout (`figures/`, `stats/`, `qc/`, `labels/`, `tables/`, `metadata.json`) | Already in use by the workflow path; extended to cover all writers. |
| Existing files | Untouched | No migration script. |
| `tables/combined.csv` vs `tables/<name>.csv` | Keep both | Current code paths already write each. |
| `tables/<name>.spec.json` | Removed; spec folded into `metadata.json` | Reduce file count. |
| `metadata.json` `samples[].outputs` (filesystem mirror) | Reduced | Drop the field; consumers walk the directory. |
| `metadata.json` environment block | Keep | Reproducibility. |
| `metadata.json` `samples[].summary.qc_warnings` | Removed | QC PNG and the in-memory QC record already carry this. |
| Root and anchor `manifest.jsonl` | Removed; output index folded into bundle `metadata.json` | Single source of truth per bundle. |
| `labels/cells/` vs `labels/domain/` | Separate folders, current layout | Two-tier analysis pairs them; separation reads better. |
| Stats CSV layout | Long format per source table | `describe__<table>.csv` covers all `(value_col, level)`; `compare__<table>.csv` covers all `value_col`. |

## New Bundle Layout

```
<root>/<timestamp>_<name>/
├── metadata.json
├── tables/
│   ├── combined.csv                # concat of all registered measurement tables (workflow path)
│   └── <table>.csv                 # individual measurement table
├── labels/
│   ├── cells/<sample>.tif
│   └── domain/<sample>.tif         # two_tier only
├── qc/
│   └── <sample>.png                # segmentation QC overlay (sole location)
├── figures/
│   └── <stem>.png|.svg             # every figure tool output
└── stats/
    ├── describe__<table>.csv
    ├── compare__<table>.csv
    └── timecourse_features__<table>.csv
```

`<root>` is the bundle's parent:

- Anchor present (input files registered) → `<anchor>/<timestamp>_<name>/`
- No anchor → `<user_results_root>/<timestamp>_<name>/` (no `bundles/` wrapper)

`<name>` is the slugified label the Speaker passed to `start_analysis`. Default `"analysis"`. Ad-hoc bundles use `"adhoc"`.

### `metadata.json` Shape (schema_version 3)

```jsonc
{
  "schema_version": 3,
  "recipe_params": { ... },
  "run_context": {
    "kind": "single" | "batch" | "adhoc",
    "tier": "single_tier" | "two_tier" | null,
    "name": "<bundle name>",
    "status": "in_progress" | "complete" | "failed",
    "created_at": "<KST ISO timestamp>",
    "finalized_at": "<KST ISO timestamp>" | null,
    "samples": [
      {
        "sample_name": "...",
        "group": "...",
        "file_id": "...",
        "source_file": "...",
        "source_layer": "...",
        "status": "complete" | "failed",
        "error": null,
        "summary": {
          // qc_warnings field removed; everything else from build_sample_summary kept
        }
      }
    ],
    "n_samples": <int>,
    "n_complete": <int>,
    "n_failed": <int>,
    "folder_set": [...],
    "channel_roles": {...},
    "scope_filters": [...]
  },
  "environment": {
    "python_version": "3.12.3",
    "imajin_version": "0.1.0",
    "deps": { "cellpose": "...", "scikit-image": "...", ... },
    "git_commit": "..."
  },
  "table_specs": {
    "<table_name>": { ... TableEntry.spec ... }
  },
  "outputs": [
    {
      "kind": "labels_cells" | "labels_domain" | "qc_png" | "table_csv" | "figure" | "stats_describe" | "stats_compare" | "stats_timecourse_features" | "tables_combined",
      "path": "labels/cells/foo.tif",
      "created_at": "<ISO>",
      "metadata": { ... }
    }
  ]
}
```

Notes:
- `samples[].outputs` (filesystem mirror) is removed. Consumers needing per-sample artifact lookup compute it from `outputs` (filter by sample slug) or walk the directory.
- `run_context.tables` (the `{"combined": "tables/combined.csv"}` shorthand emitted today by `finalize_bundle_metadata`) is also removed; the same information lives in the `outputs` index with `kind="tables_combined"`.
- `table_specs` replaces the per-file `tables/<name>.spec.json`.
- `outputs` replaces the root and anchor `manifest.jsonl`. It is append-only within the bundle and final-written when the bundle finalizes (or on every write — see "Open implementation choices").

### Stats CSV Schemas

Each `describe_table` or `compare_groups` call writes (or merges) into one of two files per source table. Multiple value columns and levels coexist in the same file.

#### `stats/describe__<table>.csv`

| Column | Notes |
|---|---|
| `value_col` | e.g., `mean_intensity` |
| `level` | `object` or `sample` |
| `sample_aggregation` | `mean` or `median`; empty for object-level rows |
| `group` | group value (may be missing if no group_col was used) |
| `n`, `mean`, `median`, `std`, `sem`, `min`, `p5`, `q1`, `q3`, `p95`, `max`, `iqr`, `cv`, `outlier_iqr_count` | unchanged |

Rows: one per `(value_col, level, group)`. Re-running `describe_table` for the same key overwrites the prior rows; running it for a new `(value_col, level)` appends.

#### `stats/compare__<table>.csv`

| Column | Notes |
|---|---|
| `value_col` |  |
| `test`, `requested_test`, `data_level`, `group_col`, `group_a`, `group_b`, `n_a`, `n_b`, `object_n_a`, `object_n_b`, `mean_a`, `mean_b`, `median_a`, `median_b`, `mean_difference_b_minus_a`, `median_difference_b_minus_a`, `mean_difference_ci95_low`, `mean_difference_ci95_high`, `statistic`, `p_value`, `cohens_d`, `hedges_g`, `cliffs_delta` | unchanged from current `stats_compare__*` schema |
| `n_groups`, `groups`, `analysis_n_total`, `object_n_total`, `eta_squared`, `epsilon_squared` | unchanged; present for multi-group tests |
| `warnings` | unchanged |

Rows: one per `(value_col, test, group_pair_or_multigroup)`.

#### `stats/timecourse_features__<table>.csv`

One file per source table. `value_col` becomes a column. Feature columns (`peak_amplitude`, `time_to_peak`, `auc`, `duration_above_threshold`, ...) unchanged.

## Components and Code Changes

### New / extended

- `imajin/result_bundles.py`
  - `start_analysis(name: str, *, kind="single", tier=None, metadata=None) -> Path` — public entry: resolves root, creates `<timestamp>_<name>/`, seeds `metadata.json` with `status="in_progress"`, sets `_active_bundle`.
  - `ensure_active_bundle(*, kind_hint="adhoc") -> Path` — returns `current_bundle()` if set; otherwise creates a process-wide ad-hoc bundle (`<timestamp>_adhoc`), sets `_active_bundle`, registers an `atexit` finalizer.
  - `finalize_analysis(*, status="complete", samples=None, extra=None)` — wrapper over `finalize_bundle_metadata` that writes `status`, `finalized_at`, drops `qc_warnings`, and removes the `samples[].outputs` field.
  - `bundle_output_path(category: str, filename: str) -> Path` — `ensure_active_bundle()` then `<bundle>/<category>/<filename>`. Parent dirs created on first write.
  - `register_output(kind: str, relative_path: str, metadata: dict | None)` — appends to the bundle's `outputs` index in `metadata.json` (held in-memory; flushed on each write or at finalize).
  - `register_table_spec(table_name: str, spec: dict)` — adds to in-memory `table_specs` map, written at finalize.
  - `register_stats_rows(kind: "describe" | "compare" | "timecourse_features", table: str, rows: list[dict])` — merges rows into the in-memory buffer keyed by `(kind, table)`. On each call, the merged CSV is rewritten so users see partial results without waiting for finalize.

### Modified

- `imajin/results.py`
  - Remove `unique_result_path`, `unique_result_dir`, `results_dir`, `_RESULT_CATEGORY_DIRS`, `record_result`, `_manifest_root`.
  - Keep `slugify_result_name`, `user_results_root`, `results_root`, `_collect_env_info`, `_kst_now`.
  - `create_result_bundle` now writes directly under `results_root()` (no `bundles/` middle directory) unless a `root=` is passed.
- `imajin/tools/figures.py`
  - `_figure_path` → `bundle_output_path("figures", filename)`. No anchor or flat fallback.
  - `_save_figure` calls `register_output("figure", rel_path, metadata)` instead of `record_result`.
- `imajin/tools/stats.py`
  - Replace `_write_stats_csv` and `_stats_csv_path` with `register_stats_rows(...)` calls. Three call sites: `describe_table` (rows for object level and optional sample level), `compare_groups` (rows for each pairwise/multi-group test), `extract_timecourse_features` (rows for each ROI).
  - Drop the `stats_object__`, `stats_sample__`, `stats_compare__` filename stems entirely.
  - `ensure_default_statistics` continues to drive automatic stats over the active bundle.
- `imajin/tools/_segmentation_outputs.py`
  - Remove `_default_qc_png_path`'s anchor-side branch and `unique_result_path` fallback. The default path is `bundle_output_path("qc", f"{sample_slug}.png")`.
  - Remove the cleanup hook that deletes the anchor-side original (no original any more).
- `imajin/tools/_workflow_outputs.py`
  - `_write_analysis_bundle_outputs` keeps the same shape but its `populate_sample_outputs` no longer copies; the QC PNG was already written into the bundle by the segmentation tool.
  - `_remove_copied_standalone_qc` is deleted along with its call site.
- `imajin/tools/results.py`
  - `save_result_bundle` (the explicit tool) keeps its public surface but writes via `bundle_output_path` and `register_output`.
  - `_resolve_output_path` is simplified: if `path` is provided, use it; otherwise route through `bundle_output_path(category, filename)`.
  - Drop the `tables/<name>.spec.json` file write; instead call `register_table_spec(table_name, spec)`.

### Removed

- `imajin/results.py`: `unique_result_path`, `unique_result_dir`, `results_dir`, `_RESULT_CATEGORY_DIRS`, `record_result`, `_manifest_root`.
- Anchor-side `segmentation_qc/` directory writes.
- Root-level `manifest.jsonl` writes (file is no longer created or appended).

## Lifecycle

### Agent task path (primary)

1. Speaker invokes `start_analysis(name="<task label>")` at the beginning of a task it expects to produce outputs.
2. The bundle directory is created and `_active_bundle` is set via `with_active_bundle(...)` for the duration of the task.
3. Tools that produce output call `bundle_output_path(...)` to obtain a destination and `register_output(...)` after writing.
4. When the task finishes, Speaker calls `finalize_analysis(status="complete" | "failed", samples=[...])`. `metadata.json` is rewritten with the final status, sample summaries, table specs, and the consolidated `outputs` index.

### Ad-hoc path (fallback)

1. A tool fires while `current_bundle()` is `None`.
2. The writer calls `ensure_active_bundle()`. On the first call in the process, an ad-hoc bundle is created at `<root>/<timestamp>_adhoc/` with `kind="adhoc"`.
3. The ad-hoc bundle's path is held in a process-global slot (`_process_adhoc_bundle`) and reused for every subsequent ad-hoc write in the same process.
4. At process exit (atexit) or main-window close, the ad-hoc bundle is finalized with `status="complete"`.

## Open implementation choices

These are not blocking for the design; resolve during planning:

1. **`outputs` index flush cadence.** Either write `metadata.json` on every `register_output` (durable but I/O-heavy) or buffer in memory and flush at finalize (one write at end, lossy if the process crashes mid-task). Plan to flush on each output but coalesce same-tick writes via a small debounce.
2. **Stats CSV merge cadence.** Every `register_stats_rows` call rewrites the destination CSV from the buffer. Simpler than append. For typical analyses (<1MB stats files) this is negligible.
3. **Process-global ad-hoc slot vs ContextVar.** A ContextVar is per-task-context; ad-hoc reuse across the GUI's worker threads benefits from a true process-global. Plan to use a module-level `Path | None` guarded by a lock.

## Migration

No data migration. The behavior change is forward-only.

- The existing `bundles/` directory and its contents are left untouched.
- The existing flat `figures/`, `stats/`, `segmentation_qc/` directories are left untouched. New tool calls do not write to them; users can delete them manually when convenient.
- The existing `manifest.jsonl` is left as a historical artifact.

Code paths and tests that depend on the old layout need updating:

- `tests/test_tools_figures.py`, `tests/test_tools_stats.py`, `tests/test_tools_results.py`, `tests/test_tools_segment.py`, `tests/test_results_bundle.py` and any test that asserts a flat `figures/`/`stats/`/`segmentation_qc/` path.
- Anything that imports `unique_result_path`, `unique_result_dir`, `results_dir`, or `record_result` must be replaced with bundle calls.

## Testing

Functional coverage:

- `start_analysis → bundle_output_path → finalize_analysis` round-trip: bundle created with correct path, `metadata.json` shows `status="complete"` and the expected `outputs` index.
- Ad-hoc auto-bundle: a standalone `bar_plot_groups` call from a fresh process creates `<timestamp>_adhoc/figures/...`, registers the output, and writes the metadata at atexit.
- Process-wide ad-hoc reuse: two standalone tool calls in the same process land in the same ad-hoc bundle.
- Stats consolidation: two `describe_table` calls for different `value_col`s produce a single `stats/describe__<table>.csv` with one row per `(value_col, level, group)`.
- Stats overwrite: re-running `describe_table` for the same `(value_col, level)` updates rows in place.
- Segment QC: a `segment_cells_*` call with QC PNG produces exactly one PNG at `qc/<sample>.png`; no anchor-side `segmentation_qc/` directory is created.
- Two-tier sample with both `labels/cells/<sample>.tif` and `labels/domain/<sample>.tif`; `outputs` index lists both.
- `metadata.json` schema_version 3, no `qc_warnings`, no `samples[].outputs`, `table_specs` populated.
- No `manifest.jsonl` is created anywhere under `<root>` or `<anchor>/.imajin/`.

Regression coverage:

- Existing recipe-based workflow tests continue to produce identical `combined.csv` and label TIFFs.
- `save_result_bundle` tool returns the same dict shape (bundle path, outputs grouped by kind).
- Backwards-compat reader for older `metadata.json` (schema_version 2 and flat schema): the metadata-reading helpers continue to load old bundles for the table dock and report tool. New writes are schema_version 3 only.

## Out of scope

- Schema-v2 → schema-v3 in-place migration of existing bundles.
- A new "bundle inspector" UI.
- Changing the auto3d workflow's output convention.
- Changing tools that already take an explicit user path.
