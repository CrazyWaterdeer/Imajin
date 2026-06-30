# Advance-to-Next-File (memory cleanup) + current-file pointer — Design

Status: design (revised after one Codex review; ready to plan)
Date: 2026-06-30

## Problem

Files are 1-2 GB, so the user works **one at a time**: load -> (optional ROI) -> analyse -> next. But
loading the next file never unloads the finished one, so memory accumulates. The user wants the screen
cleared when advancing. Capability exists (`unload_file_layers`, batch `serial_cleanup`); the
interactive path never triggers it. Also, the batch-progress ledger has no "current file" pointer.

## Approach (corrected after review)

Codex's core point: auto-unloading inside `load_file` based on a weak "analysed" signal risks **data
loss, over-removal, and surprise**. So:

- **A dedicated tool `advance_to_file`** ("I'm done with the current file; unload it and load the
  next"), NOT a side effect of `load_file`. `load_file` stays idempotent; the **batch runner is
  unaffected** (it keeps calling `load_file`).
- **Narrow blast radius:** unload only the **currently-loaded file(s) being left**, and only when they
  are **analysed** (a `complete` AnalysisRun) — never arbitrary history, never an unanalysed file
  unless `force_unload=True` (protects an in-progress ROI / unsaved work).
- **Guarded ownership:** a layer belongs to file F iff `source_path == F` **or** it is a
  `metadata.source_layer` descendant of such a layer — **excluding any layer whose own `source_path`
  is a different canonical path** (prevents the name-chain crossing into another file's tree).

## Design

### 1. `advance_to_file` tool (`tools/files.py`)

```python
@tool(phase="1", llm=True)
def advance_to_file(path: str, force_unload: bool = False) -> dict[str, Any]: ...
```
1. `new_canon = canonical(path)`.
2. `loaded = {source_path of every loaded image layer}`; `leaving = loaded - {new_canon}`.
3. `complete = {r.file_id for r in list_runs() if status=="complete"}` (canonicalised; include both
   path-keyed interactive runs and registered `file_id` runs by resolving `FileRecord.file_id ->
   canonical(path)`), so a registered/batch-analysed file is recognised too (Codex #2).
4. For each `f in leaving`: if `f in complete or force_unload` -> unload `_file_layer_tree(f)`; else
   keep it and add a warning `"<f> is loaded but not analysed; not unloaded (force_unload=True to
   discard)"`.
5. `load_file(new_canon)`; return `{loaded: ..., unloaded_files: [...], unloaded_layers: [...],
   kept_unanalysed: [...], warnings: [...]}` (transparency — Codex #3/#4).

`_file_layer_tree(canonical_path) -> set[str]` (in `tools/layers.py` or `files.py`):
- seed = layers with `_layer_source_path(layer) == canonical_path`;
- transitively add layers whose `metadata.source_layer` is a name already in the set, **but skip any
  layer whose own `_layer_source_path` is a non-None canonical path != `canonical_path`** (guard against
  crossing trees after a name reuse/rename — Codex #1).

### 2. Catch the boundary mask (`tools/boundary.py`)

`boundary_mask_from_shapes` currently records `reference_layer` but not `source_path`. Propagate the
reference image's `source_path` into the boundary layer's metadata so its (image-sized) mask is part of
the reference file's tree (Codex #6 under-removal). One line.

### 3. Current-file pointer (`agent/context.py`)

In `batch_progress_data` / `summarize_batch_progress`, derive `current` = loaded **image** files
(grouped by `source_path`, multi-channel collapses to one) whose path is **not** in the complete set.
- 0 -> `current: None`; 1 -> `current: <label>`; >1 -> list + count (ambiguous; Codex #7).
- **Headless-safe:** wrap viewer access in try/except; no viewer -> `current: None`, never raise
  (Codex #8). `get_batch_progress` keeps working with no viewer.

### 4. Prompt note (`agent/prompts.py`)

"To move to the next file in a sequential batch, call `advance_to_file(next_path)` — it unloads the
finished (analysed) current file to free memory and loads the next. Use plain `load_file` only for a
one-off load; `advance_to_file` refuses to discard an unanalysed file unless `force_unload=True`."

## Decisions / risks (from review)

- **Ownership (#1/#6):** guarded source_layer closure + source_path; still may miss an exotic
  derivative that propagates neither (`reference_layer`-only, `source_layers` plural). Residual is small
  (those layers are rare/small); a fully-robust fix (propagate a uniform `root_source_path` on every
  derived layer) is noted as a follow-up, not done here.
- **Done trigger (#2):** only currently-loaded files being left, recognised via complete runs keyed by
  path OR registered file_id->path. A file analysed only on a projection/timepoint (layer-keyed run, no
  source_path) won't be auto-recognised -> it is simply kept (safe default), not wrongly swept.
- **Unsaved work (#3):** unanalysed loaded files are never swept without `force_unload`; post-analysis
  manual label edits that were not re-saved are a documented limitation (analysis already saved labels
  at analysis time; the user can `save_labels` / re-analyse).
- **Surprise (#4):** the side effect lives in `advance_to_file`, not `load_file`; result reports exactly
  what was unloaded/kept.
- **Batch (#5):** unchanged — batch uses `load_file` (inert) + its own `serial_cleanup`.

## Files

| File | Change |
| --- | --- |
| `src/imajin/tools/files.py` | `advance_to_file` tool + `_file_layer_tree` (guarded) |
| `src/imajin/tools/boundary.py` | boundary layer metadata propagates the reference image `source_path` |
| `src/imajin/agent/context.py` | derive `current` (grouped by source_path, headless-safe) |
| `src/imajin/agent/prompts.py` | one line: use `advance_to_file` to advance |
| `tests/test_tools_files.py` (+ a small new file) | the hostile cases below |

## Acceptance / hostile tests (Codex #9)

- File A loaded (image + labels + a MIP + a boundary mask), A has a complete run; B not loaded.
  `advance_to_file(B)` -> **all four** of A's layers gone, B loaded.
- A layer with a **different** `source_path` that happens to chain off A is **not** swept (guard).
- A loaded-but-**unanalysed** file C is **kept** (warning) on `advance_to_file(B)`; with
  `force_unload=True` it is unloaded.
- Multi-channel A (two image layers, same source_path) -> both gone; `current` groups them as one file.
- `batch` with `keep_layers=True` / `serial_cleanup` is unaffected (still uses `load_file`).
- `summarize_batch_progress` / `get_batch_progress` with **no viewer** -> `current: None`, no error.
- `current` = the loaded image whose file has no complete run.

## Non-goals

- Per-file ROI persistence / resume-mid-file (separate next increment).
- A uniform `root_source_path` on all derived layers (follow-up if residual under-removal bites).
- Auto-unload inside `load_file`; memory-threshold triggers; cross-session state.
