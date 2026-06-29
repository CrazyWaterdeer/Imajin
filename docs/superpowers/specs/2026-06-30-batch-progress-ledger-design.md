# Batch-Progress Ledger + Re-run Guard for the In-App Agent — Design

Status: design (revised after one Codex review; ready to plan)
Date: 2026-06-30

## Problem

The chat agent loses multi-file progress: it analyses file 1, then a few turns later re-asks a path it
already has and **re-analyses file 1**, because:

- `runner._compact_messages` summarises older turns into a lossy block (keeps "recent user intents" +
  "tool activity counts", **not** which files are done).
- The only per-turn context (`runner._runtime_system_prompt` -> `context.summarize_viewer_state`) is
  layers / tables / samples / channels — not analysed-vs-pending.
- `put_run` (`AnalysisRun`: pending/running/complete/failed) is written **only by the batch runner**;
  the interactive `analyze_target_cells` records nothing.

## Goal & approach (corrected after review)

Codex's key point: a passive prompt injection makes state **visible** but does not **enforce** it — the
model can still ignore it. So this is two things, not one:

1. **Enforcement (the actual fix):** `analyze_target_cells` records its outcome and **refuses to
   recompute an already-analysed (file, method) unless `rerun=True`**. This makes "re-analyse file 1"
   impossible by default, regardless of what the model remembers.
2. **Visibility (the mitigation):** a compact **Batch progress** ledger injected every turn (rebuilt
   from durable session state, so compaction can't erase it) + a `get_batch_progress` tool for detail.

Reuse the existing `AnalysisRun` registry as the single source of truth (Codex #7 — do not add a
divergent cache). Deterministic batch orchestration (removing the LLM from the per-file loop) stays a
separate, later step.

## Design

### 1. Record completion in the interactive path (`tools/workflows.py`)

On `analyze_target_cells` success **and** failure, write an `AnalysisRun` via `put_run`:
- `file_id` = **canonical** source path of the target layer (`_canonical_path_text(_layer_source_path(
  snapshot))`, both in `tools/files.py`); fall back to the target layer name when there is no
  `source_path` (Codex #2).
- `sample_id` = the file label (stem); `recipe_id` = `f"interactive:{method}"` where method is
  `two_tier` / `single` / the segmentation method, so different methods on one file coexist and don't
  suppress each other (Codex #3, #8).
- `status` = `complete` / `failed`; `table_names` = the result table; `summary` = `{n_objects,
  bundle_path, method}` (Codex #6 provenance, #11 status).

### 2. Re-run guard (`tools/workflows.py`)

Add `rerun: bool = False` to `analyze_target_cells`. Before doing work, look up a **complete**
`AnalysisRun` for this `(file_id, recipe_id)`; if one exists and `rerun` is False, **return early**:
```
{ok: True, already_analysed: True, labels_layer/table from the prior run,
 message: "<file> was already analysed (table <name>); pass rerun=True to recompute."}
```
The guard defers to the user: an explicit user "re-run / recompute / changed parameters" sets
`rerun=True` (the agent passes it). The guard only blocks *silent duplicate* work (Codex #4). A `failed`
prior run does **not** block (it should be retried).

### 3. Ledger summary (`agent/context.py`)

`summarize_batch_progress(max_labels=8, max_chars=600) -> str | None`, from `list_runs()` (done/failed)
+ `iter_file_records()` (universe), keyed by canonical path:
```
Batch progress: analysed 2, failed 0, registered-pending 9.
  analysed: mF rectum 1 [..._two_tier], mF rectum 2 [...]
  pending: mF rectum 3, mF rectum 4 (+7 more)
  Re-analyse only when the user asks to rerun or changes parameters.
```
- If files were registered, show `registered-pending`. If **not** registered, show
  `pending: unknown (no file registry — call register_files to track the batch)` instead of implying
  completeness (Codex #5).
- Counts first, then a few short labels; **hard char cap** (Codex #9). Return `None` when there are no
  runs and no registered files (inject nothing for single-shot use).

### 4. `get_batch_progress` tool (`tools/experiment.py`)

A read-only tool returning the full structured progress + `next_pending` (first registered file with no
complete run), so the agent can fetch detail on demand without the per-turn injection carrying it all
(Codex #10).

### 5. Inject every turn (`agent/runner.py`)

`_runtime_system_prompt` appends the ledger after `summarize_viewer_state` (only when non-`None`),
rebuilt fresh each turn (compaction-proof).

### 6. Prompt guidance (`agent/prompts.py`)

One rule: "A **Batch progress** section lists files already analysed (with their result table) and
files still pending. Do **not** re-analyse a file shown as analysed **unless the user explicitly asks
to rerun/recompute or changes parameters** (then pass `rerun=True`); do not re-ask for a path already
registered/loaded; when continuing a batch, pick the next *pending* file. For a multi-file batch,
`register_files` first so pending is tracked."

## Decisions / risks (carried + from review)

- **Enforcement vs intent (Codex #1, #4):** the guard blocks only silent duplicates; explicit user
  rerun always wins via `rerun=True`. Parameter changes count as a rerun (the agent passes `rerun=True`
  when the user changes params).
- **Key canonicalisation (#2):** canonical absolute path; layer-name fallback only when no source_path.
  Not hashing files (out of scope) — a path move mid-session is rare and `rerun=True` recovers.
- **Multi-channel/method (#3, #8):** key includes method; analysing a different channel/method is not
  suppressed.
- **Stale results (#6):** the run stores table/bundle; if the table was deleted the agent can still
  rerun. Deep validity (params drifted) is handled by the user asking to rerun.
- **Universe unknown (#5):** explicit "unknown — register_files" wording; never imply batch complete.
- **Token cost (#9):** compact, char-capped; full detail behind `get_batch_progress`.
- **Session boundary (#12):** progress is per-session (`AnalysisRun` in the session); a brand-new batch
  in the same session inherits prior runs — acceptable; `reset_runs()` clears it.
- **No result change:** segmentation/measurement outputs are unchanged; this adds a record, a guard
  (opt-out via rerun), a tool, and a prompt section.

## Files

| File | Change |
| --- | --- |
| `src/imajin/tools/workflows.py` | record an `AnalysisRun` on success/failure; `rerun` guard |
| `src/imajin/agent/context.py` | `summarize_batch_progress` |
| `src/imajin/tools/experiment.py` | `get_batch_progress` read tool (+ `next_pending`) |
| `src/imajin/agent/runner.py` | inject the ledger in `_runtime_system_prompt` |
| `src/imajin/agent/prompts.py` | one rerun-aware rule |
| `tests/...` | guard blocks silent dup + allows `rerun`; record on success/failure; ledger string (analysed/pending/unknown/none); injection present/absent; tool output |

## Acceptance

- `analyze_target_cells` twice on the same layer (same method): the 2nd returns `already_analysed=True`
  and does **no** segmentation; with `rerun=True` it recomputes. A `failed` first run does not block.
- `list_runs()` has the interactive run with canonical `file_id` + `recipe_id=interactive:<method>`.
- `summarize_batch_progress`: analysed/failed/pending counts; "unknown" when unregistered; `None` when
  empty; within the char cap.
- `_runtime_system_prompt` includes the ledger iff progress exists.
- `get_batch_progress` returns structured progress + next_pending.
- No regression in agent/runner/workflow tests.

## Non-goals

- Deterministic batch orchestration / removing the LLM from the per-file loop (later step).
- Cross-session persistence; file-hash identity; a UI progress panel; changing analysis outputs.
