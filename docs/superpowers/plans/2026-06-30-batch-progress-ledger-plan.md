# Batch-Progress Ledger + Re-run Guard — Implementation Plan

Follows `docs/superpowers/specs/2026-06-30-batch-progress-ledger-design.md` (revised after 1 Codex
review). This plan: pre-Codex-review. Branch `feat/batch-progress-ledger` off master, commit-by-commit
with `.venv/bin/python -m pytest <files> -q` gates.

Reuses the existing `AnalysisRun` registry (`session.put_run` / `list_runs`); no new store.
Key facts: `_layer_source_path(snapshot)` (`tools/files.py:15`) already returns the **canonical** source
path (it calls `_canonical_path_text`); `batch_runner.py:580` calls `analyze_target_cells` (so the batch
path must pass `rerun=True` to skip the interactive guard).

## Commit 1 — re-run guard + record `AnalysisRun` (enforcement)

`tools/workflows.py` `analyze_target_cells`:
- Add `rerun: bool = False` param (and document it).
- After the target layer + `method` are resolved, derive the key:
  `file_key = _layer_source_path(snapshot) or seg_input_layer`; `recipe_key = f"interactive:{mode}"`
  where `mode` is `"two_tier"` when `domain_strategy` else `"single"` (matches the bundle suffix).
- **Guard:** scan `list_runs()` for an entry with `file_id == file_key`, `recipe_id == recipe_key`,
  `status == "complete"`. If found and not `rerun`, return early without segmenting:
  ```
  {ok: True, already_analysed: True, labels_layer: <prior>, cells_layer: <prior>,
   table_name: <prior>, message: "<label> was already analysed (table <name>); pass rerun=True
   to recompute.", warnings: [...]}
  ```
  A `failed` prior run does NOT block.
- **Record:** at the success return(s), `put_run(sample_id=<file stem>, file_id=file_key,
  recipe_id=recipe_key, status="complete", table_names=[table], layer_names=[cells/domain],
  summary={"n_objects":..., "bundle_path":..., "method":mode})`. At the `empty_mask` failure return,
  record `status="failed"`. (Other early returns — resolve error / review-skip — are setup issues, not
  analysed files; no record.)
- Use a single private helper `_record_interactive_run(file_key, recipe_key, status, result)` to keep
  the return sites small.

`tools/batch_runner.py:580`: pass `rerun=True` to `analyze_target_cells(...)` (batch manages its own
dedup + `put_run`; the interactive guard must not block it).

`tests/test_phase2_workflow.py`:
- analysing the same layer twice (same method) -> 2nd result has `already_analysed is True` and **no new
  cells layer / no recompute** (assert the labels layer count did not increase, or spy the segment
  step); `rerun=True` recomputes.
- after success, `list_runs()` has one entry with `file_id == _layer_source_path(snapshot)` and
  `recipe_id == "interactive:two_tier"`, `status == "complete"`.
- a failure (e.g. empty target) records `status == "failed"` and does **not** block a later real run.

**Gate:** `pytest tests/test_phase2_workflow.py -q`.

## Commit 2 — ledger summary + `get_batch_progress` tool

`agent/context.py` `summarize_batch_progress(max_labels=8, max_chars=600) -> str | None`:
- `runs = list_runs()`; group by `file_id`; `analysed` = files with a `complete` run, `failed` = files
  whose latest run failed and none complete.
- `universe = [canonical(path) for rec in iter_file_records()]`. `pending = universe - analysed`.
- Build a compact string: counts first, then up to `max_labels` short labels (file stem +
  `[table]`), with `(+N more)`; **hard-truncate to `max_chars`**.
- If `universe` is empty, show `pending: unknown (call register_files to track the batch)` instead of a
  pending list. Return `None` when `runs` and `universe` are both empty.

`tools/experiment.py` `get_batch_progress()` (`@tool`, llm=True): structured
`{analysed: [...], failed: [...], pending: [...], universe_known: bool, next_pending: str | None}` from
the same sources, for on-demand detail.

`tests/test_phase2_workflow.py` (or a small `tests/test_batch_progress.py`):
- with no runs/files -> `summarize_batch_progress()` is `None`; with one complete run -> contains
  "analysed 1"; with registered files -> shows pending; unregistered -> "unknown".
- `get_batch_progress` returns the analysed/pending split + `next_pending`.
- char cap respected for long labels.

**Gate:** `pytest tests/test_phase2_workflow.py tests/test_batch_progress.py -q`.

## Commit 3 — inject every turn + prompt rule

`agent/runner.py` `_runtime_system_prompt`: after the `summarize_viewer_state` block, also
`call_on_main(summarize_batch_progress)`; if non-empty append
`"\n\nBatch progress:\n{ledger}"`. Keep the existing early-return when there is no viewer context, but
still include the ledger if it is non-empty (the ledger does not need a viewer).

`agent/prompts.py`: add the rerun-aware rule from the spec (do not re-analyse an analysed file unless
the user asks to rerun / changes params -> `rerun=True`; do not re-ask a known path; pick the next
pending file; `register_files` first for batches).

`tests/test_runner.py` (or extend): a `Runner` whose session has a complete `AnalysisRun` ->
`_runtime_system_prompt()` contains "Batch progress"; with no runs/files -> it does not. (Use the
existing runner test fixture / monkeypatch `summarize_batch_progress`.)

**Gate:** `pytest tests/test_runner.py tests/test_phase2_workflow.py tests/test_batch_progress.py -q`,
then full `pytest -q`.

## Verification before done

1. Full suite green; report counts.
2. Manual: construct a session, run `analyze_target_cells` once, confirm `summarize_batch_progress()`
   returns an "analysed 1" line and a 2nd call returns `already_analysed`.

## Risks (carried)

- Guard must defer to explicit user rerun (`rerun=True`); batch passes it. A `failed` run never blocks.
- Key is the canonical source path; a file with no `source_path` falls back to the layer name.
- Injection is rebuilt each turn from the session (compaction-proof); char-capped to bound token cost.

## Changelog — plan -> rev.1 (accepted Codex plan-review findings)

- **P0 guard placement:** snapshot the **original target layer** and compute the normalized method
  *before* `_run_preprocess_step`; place the guard there (a duplicate call must not even run preprocess
  or the domain precompute). Key from the original target's `source_path`, not the preprocessed layer.
- **P0 method in key:** `recipe_key = f"interactive:{method}:{mode}"` (`method` = normalized
  segmentation method, `mode` = `two_tier|single`), so a different method on the same file is not
  blocked.
- **P0 batch double-record:** add a private `batch_managed: bool = False` to `analyze_target_cells`;
  when True it skips **both** the guard and the interactive `put_run` (batch_runner owns dedup +
  recording). `batch_runner.py:580` passes `batch_managed=True` (instead of just `rerun=True`).
- **P0 ledger keyspace:** `summarize_batch_progress` / `get_batch_progress` normalise every run's
  `file_id` through `FileRecord.file_id -> canonical(path)` (and registered universe the same way), so a
  run stored under `ctrl_1` and a universe entry under a path resolve to one key; otherwise completed
  batch files show as pending.
- **P1 early-return contract:** at record time store the **full success result dict** in
  `summary["result"]`; the guard early-return returns `{**stored_result, "already_analysed": True,
  "message": ...}` so `result_bundle_path` / `tier_table_name` / `segmentation_threshold_scope` /
  `cells_layer` etc. are preserved for callers.
- **P1 latest-complete:** select the **newest** complete run for the key (iterate runs reversed).
- **P1 guard test rigor:** monkeypatch `_run_segmentation_step`, `_run_preprocess_step`, and
  `_precompute_domain_layer` to fail/count on a 2nd call and assert none ran, plus no new run/table.
- **P1 runner injection:** compute viewer-context and ledger in **separate** try/excepts and append each
  when non-empty — the ledger appears even when `summarize_viewer_state` returns `""`/raises (add that
  test).
- **P2 two-tier table:** the stored full result already carries `primary_table_name`/`tier_table_name`,
  so the ledger/message reference the combined table, not just the cells table.
