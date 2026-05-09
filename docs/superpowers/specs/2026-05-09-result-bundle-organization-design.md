# Result Bundle Organization

**Status**: Draft
**Date**: 2026-05-09
**Type**: Design

## Problem

Analysis outputs are currently scattered across multiple bundle directories or, in some paths, not bundled at all:

1. **Batch (`run_recipe_on_samples`) does not bundle.** A 14-sample batch produces 14 independent bundle folders, one per sample's `analyze_target_cells` call. There is no top-level folder grouping the batch, so navigating "this experiment's results" means hunting through unrelated bundles by timestamp.
2. **Two-tier `analyze_target_cells` does not bundle at all.** The two-tier branch (`domain_strategy is not None`) deliberately skips `save_result_bundle`. Domain-mask TIFFs and the long-format combined table only live in napari/agent state; nothing reaches disk unless the user explicitly invokes `save_result_bundle` afterwards.
3. **Per-sample table-attachment skips two-tier.** `run_recipe_on_samples` calls `attach_sample_columns_to_table` only on the cells-only `table_name`. Two-tier's `tier_table_name` never receives `sample_name`/`group`/`file_id` columns, so a downstream "combine all samples' tier tables" operation cannot produce a sample-aware long-format table.

These gaps block the next phase (statistics, graphs) from having a predictable input location.

## Proposed Solution

A single, uniform bundle layout used by all analysis paths — batch and standalone, single-tier and two-tier. One bundle = one analysis run, named `<KST_timestamp>_<recipe_or_target>`. Type-first internal layout so QC and label files can be browsed in bulk.

The bundle is the contract for downstream stats/graphs. Stats reads `tables/combined.csv` and `metadata.json`; graphs writes into `figures/`; stats writes into `stats/`. No code outside the bundle layer needs to know how the layout works.

## Key Decisions and Rationale

- **Folder name = `<timestamp>_<recipe_name>` for batch, `<timestamp>_<target>__single|__two_tier` for standalone.** Recipe name is the natural batch identifier (matches user's mental model of "this experiment"). Standalone bundles encode tier in the suffix so a glance reveals what kind of analysis ran.
- **KST (UTC+9) timestamp.** User-facing folder names should reflect local working time; ISO `created_at` retains explicit `+09:00` offset for unambiguous machine parsing.
- **Type-first internal layout (`labels/`, `tables/`, `qc/`, `stats/`, `figures/`) with tier subdirs (`labels/cells/`, `labels/domain/`).** Browsing all 14 QC PNGs at once is a common workflow; sample-first nesting fights it. Tier subdirs let the user inspect "all 14 domain masks" or "all 14 cells masks" independently. A 14-sample two-tier batch produces 28 TIFFs; flat naming with `__domain` / `__cells` suffixes loses scannability.
- **Single combined long-format `tables/combined.csv` only.** Per-sample CSVs duplicate information already addressable by filtering on `sample_name`. The `tier` column (`cells` / `domain`) extends naturally for two-tier without splitting into multiple files.
- **All paths use the new layout (Q7-A).** Single-tier standalone migrates from `labels/<name>.tif` to `labels/cells/<name>.tif`. The cost is one path string in `save_result_bundle`; the benefit is a single reader for stats/graphs with no kind-based branching.
- **Two-tier and batch get bundles (Q3-B); single-tier behaviour parity preserved.** Standalone single-tier still auto-creates a bundle (existing UX); the layout migrates to the unified one.
- **Batch creates one parent bundle; per-sample `analyze_target_cells` writes into it directly.** No nested bundles. The runner passes the parent path via an internal `_bundle_path` keyword to suppress per-sample auto-bundling.
- **`metadata.json` carries recipe snapshot, sample index with per-sample summaries, environment info, and counts.** Reproducibility (recipe + env) and downstream introspection (per-sample status / object counts / qc warnings) live in one machine-parseable file. Stats/graphs read from here, not by walking the filesystem.
- **`save_result_bundle` tool stays.** Useful for ad-hoc "bundle these layers" requests outside the analysis pipeline. Internals migrate to the new layout.
- **Existing on-disk bundles untouched.** No code currently reads bundle layout externally (`read_bundle_metadata` is the only reader and is internal). Old bundles remain on disk; only new bundles use the new layout.

## Architecture

### Bundle root location (unchanged)

- Project open: `<project_root>/reports/bundles/`
- No project: `~/Documents/Imajin/results/bundles/` (Windows mapped automatically per existing `_windows_documents_dir`)

### Bundle layout

```
20260509_180023_CaLexA_J20_1234_rectum/
├── metadata.json
├── labels/
│   ├── cells/
│   │   ├── J20_1234_..._rectum_1.tif
│   │   └── ... (×N)
│   └── domain/                          # only populated when tier="two_tier"
│       └── ...
├── tables/
│   └── combined.csv                     # long-format; sample_name/group/file_id/source_file/source_layer/tier columns
├── qc/
│   ├── J20_1234_..._rectum_1.png
│   └── ... (×N)
├── stats/                               # empty placeholder (next phase)
└── figures/                             # empty placeholder (next phase)
```

`labels/domain/` is created (mkdir) for all bundles for layout uniformity but stays empty for single-tier.

### `metadata.json` schema

```jsonc
{
  "kind": "batch" | "single",
  "tier": "single_tier" | "two_tier",
  "name": "CaLexA_J20_1234_rectum",
  "status": "complete" | "cancelled",        // top-level status
  "created_at": "2026-05-09T18:00:23+09:00",
  "imajin_version": "0.1.0",
  "python_version": "3.12.3",
  "deps": {"cellpose": "...", "scikit-image": "...", "tifffile": "..."},
  "git_commit": "818cca0",
  "recipe": { /* full snapshot of recipe at run time */ },
  "samples": [
    {
      "sample_name": "...",
      "group": "saline",
      "file_id": "...",
      "source_file": "/mnt/c/.../J20 + 1234 vF saline injected rectum 1.lsm",
      "source_layer": "Ch2-T1_avg_z",
      "status": "complete" | "failed" | "skipped",
      "error": null,
      "outputs": {
        "labels_cells": "labels/cells/<slug>.tif",
        "labels_domain": "labels/domain/<slug>.tif" | null,
        "qc_png": "qc/<slug>.png" | null
      },
      "summary": {
        "n_cells": 234,
        "n_domain_components": 3,
        "domain_area_um2": 12345.6,
        "qc_warnings": []
      }
    }
  ],
  "tables": {"combined": "tables/combined.csv"},
  "n_samples": 14,
  "n_complete": 12,
  "n_failed": 2
}
```

### Code touchpoints

`imajin/results.py`
- `create_result_bundle(name, kind, metadata)`: emit new layout (`labels/cells`, `labels/domain`, `tables`, `qc`, `stats`, `figures`); KST timestamp.
- New helpers:
  - `_kst_now() -> datetime` — single source of timezone.
  - `_collect_env_info() -> dict` — `imajin_version` (via `importlib.metadata`), `python_version`, key dep versions, `git_commit` (best-effort, graceful on failure).

`imajin/tools/results.py`
- `save_result_bundle`: write to `labels/cells/<slug>.tif` (no longer `labels/<slug>.tif`); rest of API unchanged.

`imajin/tools/workflows.py`
- `analyze_target_cells`:
  - When invoked outside a batch (no parent bundle in context), creates its own bundle for both single-tier and two-tier branches (single-tier preserves existing UX, two-tier gains it).
  - When invoked from inside `run_recipe_on_samples`, writes outputs into the parent bundle instead of creating its own. The hand-off mechanism (e.g. `contextvars.ContextVar` set by the runner, or an internal helper called separately from the public tool entry) is an implementation detail; the design constraint is that the parent path is **not** part of the public `@tool` schema, so the LLM cannot accidentally pass it.
  - Two-tier branch writes both `cells` and `domain` label TIFFs into the bundle's `labels/cells/` and `labels/domain/`; QC PNG into `qc/`.
- `run_recipe_on_samples`:
  - At entry, create parent bundle (`kind="batch"`, `tier=` derived from `recipe.domain` presence, `metadata.recipe` = full snapshot).
  - Run each sample with the parent bundle as the active target (per the hand-off mechanism above).
  - For two-tier, attach sample columns to `tier_table_name` (currently only `table_name` is attached — fix concurrent with this work).
  - On loop end (success, failure, or cancellation), write `tables/combined.csv` (concat of per-run measurement tables) and finalize `metadata.json` (`status`, `samples`, `n_*`).
  - Wrap loop in `try / finally` so cancellation still produces a usable bundle with `status="cancelled"` and skipped samples marked.

`imajin/agent/prompts.py`
- One-line addition documenting that `run_recipe_on_samples` now returns `bundle_path` and that all per-sample outputs live inside it.

No new dependencies. `git` and `importlib.metadata` are stdlib / pre-installed.

## Edge Cases

- **Failed sample**: `samples[i].status = "failed"`, error message captured, no label/QC files written for that sample, `n_failed` incremented, no row in `combined.csv`.
- **Cancellation mid-batch**: `try / finally` guarantees `metadata.json` and `combined.csv` are written for whatever ran. Top-level `status="cancelled"`; remaining samples marked `status="skipped"`.
- **Slug collision**: Defensive check — if `labels/cells/<slug>.tif` already exists in the same bundle, raise `ValueError` rather than silently overwrite. Practical risk is near-zero for unique sample_name inputs.
- **Empty batch (`sample_names=[]`)**: Unchanged early-return path; no bundle is created.
- **Single-tier standalone**: `labels/domain/` is created but stays empty; `samples` array length 1; `tier="single_tier"`.

## Testing Strategy

New tests in `tests/test_phase3_experiment.py` (reuses existing fixtures and stub patterns). Plus minimal unit coverage in a small `tests/test_results_bundle.py` for layout primitives.

Unit:
1. `create_result_bundle` produces all subdirs (`labels/cells`, `labels/domain`, `tables`, `qc`, `stats`, `figures`) with KST timestamp in folder name and ISO `created_at` carrying `+09:00`.
2. `_collect_env_info` includes `imajin_version`, `python_version`, falls back gracefully when `git` is unavailable.

Integration (run_recipe_on_samples):
3. Single-tier batch (2 samples) → bundle has `labels/cells/<slug>.tif × 2`, empty `labels/domain/`, `combined.csv` with 2-sample-aware rows, `metadata.kind="batch"`, `metadata.tier="single_tier"`.
4. Two-tier batch (2 samples) → both `labels/cells/` and `labels/domain/` populated, `combined.csv` has `tier` column with both values, `metadata.tier="two_tier"`.
5. Failed sample → `samples[i].status="failed"` in metadata, no files written for that sample, `combined.csv` excludes its rows, `n_failed=1`.
6. Cancellation → `CancelledError` raised by stubbed `analyze_target_cells` mid-loop; bundle still has `metadata.json` with top-level `status="cancelled"`, completed samples logged, remaining marked `skipped`.
7. Two-tier `tier_table_name` carries `sample_name` columns (regression test for the attachment-bug fix).

Standalone:
8. Two-tier standalone `analyze_target_cells` → bundle named `<ts>_<target>__two_tier`, `kind="single"`, samples length 1.
9. Single-tier standalone → bundle named `<ts>_<target>__single`, `labels/cells/<target>.tif` (new path), `kind="single"`.

Regression:
10. Existing single-tier batch tests (`test_run_recipe_on_samples_single_sample_attaches_columns`, `_multi_sample_one_fails`, `_auto_loads_sample_local_target_and_cleans_layers`, etc.) updated to assert against the new layout. These tests currently rely on per-call bundles; they shift to asserting the parent batch bundle structure.
