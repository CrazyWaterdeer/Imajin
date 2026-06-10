# Session State Extraction — Refactor Plan

> **For agentic workers:** implement this plan **one commit at a time**, running the
> test suite (`uv run pytest -q -m "not slow and not integration"`) after every
> commit. Each commit below is designed to leave the codebase green. Steps use
> checkbox (`- [ ]`) syntax for tracking. Do not batch commits unless explicitly
> asked.

**Date:** 2026-06-10
**Status:** Planned (not started)
**Owner decisions baked in:** target = de-globalize + relocate `state`; migration
depth = **keep free-function API as thin shims, migrate internals only** (no
caller-side churn, no signature changes); deliverable = this doc.

**Goal:** Turn `src/imajin/agent/state.py` from a 1101-line half-migrated
god-module — an `AnalysisSession` dataclass shadowed by ~11 hand-synced
module-level alias globals — into a single foundation module
`src/imajin/session.py` whose public free-function API is unchanged but whose
internals read and write exactly one place: `current_session()`. Then relocate it
out from under `agent/` so the `agent → tools → agent.state` import cycle is
broken and the function-local "lazy import" workarounds can be removed.

**Non-goal:** No signature changes. `get_table`, `put_table`, `put_file`,
`resolve_target_channel`, `current_session`, `set_current_session`,
`bulk_state_update`, etc. keep their names and signatures. The one public-surface
change is the import path: `imajin.agent.state` → `imajin.session` (verified zero
out-of-repo importers; the old path survives as a shim through Phase 2 and is
removed in C17). Caller bodies are otherwise untouched.

---

## Problem Statement

`agent/state.py` already defines `AnalysisSession`, but it is used as a single
process-global singleton (`_CURRENT_SESSION = AnalysisSession()`) that is shadowed
by a parallel set of module-level alias globals pointing into the singleton's
dicts:

```
_VIEWER  _TABLES  _QC_RECORDS  _FILES  _RECIPES  _RUNS  _RUN_COUNTER
_SAMPLES  _CHANNELS  _TABLE_LISTENERS
```

plus three un-aliased notification globals (`_STATE_CHANGE_DEPTH`,
`_PENDING_STATE_REASONS`, `_PENDING_TABLES_CHANGED`). The module's own docstring
admits the situation: *"The module-level variables below remain as compatibility
aliases while the codebase is migrated away from scattered globals."*

The ~60 free functions read these aliases (e.g. `put_file` mutates `_FILES`, not
`current_session().files`). The only thing keeping the two views consistent is
`set_current_session`, which manually rebinds all ten aliases on every session
swap. This is fragile: any new field or any function that forgets the alias
contract silently desynchronizes from the session. It is the classic signature of
an incomplete "globals → object" migration that stalled at the halfway point.

Two structural consequences fall out of this:

1. **Layering inversion.** `state.py` lives under `agent/`, yet it is the
   foundational dependency of the whole tool layer — 21 of 32 files in `tools/`
   import `agent.state`, plus 4 in `ui/`. Meanwhile `agent/` (runner, execution,
   qt_tool_runner, specialists) imports `tools`. The result is a dependency cycle
   `agent → tools → agent.state` that is currently "managed" by pushing many
   `state` imports inside function bodies (lazy imports): experiment.py has 9,
   batch_runner.py 5, segment.py 3, report.py 3, napari_ops.py 3, and more.
   Function-local imports are a smell that the module sits at the wrong layer.

2. **No direct test coverage.** There is no `test_session.py` / `test_state.py`.
   The session is exercised only indirectly through ~21 tool test files. The one
   safety net is the `autouse` fixture in `tests/conftest.py`, which resets the
   per-family registries between tests. Refactoring the most central state module
   in the codebase with no characterization test is the main risk to manage.

## Solution

Two ordered phases plus a cleanup phase, all behind a stable public API:

- **Phase 0 — Net.** Add `tests/test_session.py` characterization tests that pin
  the *observable* behavior of the session module (CRUD round-trips per family,
  id-collision suffixing, bulk-update coalescing, session-swap isolation,
  snapshot/restore round-trip, channel/layer resolution). Must pass on the
  current code unchanged.

- **Phase 1 — De-globalize in place.** One record family per commit: rewrite that
  family's free functions to read/write `current_session().<field>`, drop the
  field from `set_current_session`'s rebind block, and delete the alias global. By
  the end `set_current_session` is a one-liner and the only module-level mutable
  state is the single `_CURRENT_SESSION` plus immutable lookup constants. File
  stays at `agent/state.py` throughout — the risky logic change happens with
  import paths frozen.

- **Phase 2 — Relocate.** `git mv` the now-clean module to `imajin/session.py`,
  leave `agent/state.py` as a `from imajin.session import *` re-export shim so all
  103 existing import sites stay green, migrate import sites directory-by-directory
  (`tools/`, `ui/`, `agent/`, the two root modules `anchor.py` /
  `result_bundles.py` with their coupled test patch targets, `tests/`), then delete
  the shim.

- **Phase 3 — Un-lazy.** With `session` a top-level module that imports nothing
  from `imajin`, the **state** cycle (`agent → tools → agent.state`) is gone.
  Convert the function-local `from imajin.session import …` workarounds back to
  top-level imports, one module per commit, verifying import-time cleanliness after
  each. (The unrelated `tools → agent.execution` / `tools → agent.qt_dispatch`
  coupling stays; it is not part of this refactor.)

The free-function API (`get_table`, `put_file`, `resolve_target_channel`, …) is
preserved exactly — they become thin delegations over `current_session()`. This
is the smallest, safest migration depth and was the explicit owner choice.

---

## Current-State Facts (evidence gathered 2026-06-10)

| Fact | Value |
|---|---|
| `agent/state.py` size | 1101 lines |
| Alias globals to remove | 10 (`_VIEWER`, `_TABLES`, `_QC_RECORDS`, `_FILES`, `_RECIPES`, `_RUNS`, `_RUN_COUNTER`, `_SAMPLES`, `_CHANNELS`, `_TABLE_LISTENERS`) |
| Un-aliased notification globals | 3 (`_STATE_CHANGE_DEPTH`, `_PENDING_STATE_REASONS`, `_PENDING_TABLES_CHANGED`) |
| Immutable lookup constants (keep as module-level) | `_CHANNEL_COLOR_ALIASES`, `_CHANNEL_ROLE_ALIASES` |
| External imports of the private aliases | **none** (safe to delete) |
| `imajin.*` imports inside `state.py` | **none** (foundation-clean → safe top-level placement) |
| Total `agent.state` import sites (src + tests) | 103 (tools 21 files, ui 4, agent 2, **root 2 — `anchor.py`, `result_bundles.py`**, tests 21, conftest 1, …) |
| Public API used by other modules | `current_session`, `set_current_session`, `reset_session`, `bulk_state_update` (12 references across execution.py, provenance.py, batch_runner.py, report.py, specialists.py, test_execution_service.py, test_tool_registry.py) |
| Function-local `state` imports to un-lazy (Phase 3) | experiment 9, batch_runner 5, segment 3, report 3, napari_ops 3, workflows 2, stats 2, layers 2, channels 2, view 1, _workflow_outputs 1 |
| Dedicated test for state today | none |
| conftest safety net | `autouse` fixture resets samples/channels/files/recipes/runs/qc per test (note: it does **not** reset tables — preserve that) |
| Tests that patch a moved fn | `test_anchor.py` 3× + `test_tools_results.py` 2× patch `imajin.agent.state.list_files`; both consumers import it function-locally → patch targets must move with `anchor.py` (C15) |

---

## Commits

### Phase 0 — Characterization net

- [x] **C0. Add `tests/test_session.py` (passes on unchanged code).**
  Pin observable behavior only — never assert that a particular module global
  exists. Cover, at minimum:
  - file family: `put_file` returns a slug id; a second file with the same
    original name gets the `_2` suffix; `get_file` round-trips; `update_file_status`
    mutates status/notes; `list_files`/`iter_file_records` reflect inserts;
    `reset_files` clears.
  - one round-trip test each for recipes, runs (incl. run-counter increment), qc
    records, samples, channel annotations, tables.
  - tables + listeners: `on_tables_changed(cb)` registers once (idempotent on the
    same callable); a `put_table`/`update_table` fires `cb`; `reset_tables` clears.
  - `bulk_state_update`: a block that mutates tables N times fires the
    tables-changed listener **exactly once**, and only after the block exits.
  - session isolation: after `set_current_session(AnalysisSession())`, tables/files
    from the old session are invisible, and mutating the new session does not touch
    the old object's dicts. `reset_session()` yields an empty session.
  - `snapshot_session_state()` / `restore_session_state()` **full round-trip for
    all six families** (files, samples, channels, recipes, runs, qc): populate
    each, snapshot, swap to a fresh session, restore, snapshot again, assert equal.
    This is the net for the direct `_FILES[...]` / `_RUN_COUNTER[0]` writes that
    live *inside* `_restore_session_state_impl` (state.py:879, 949) — which C1/C3
    must migrate but the samples/channels-only round-trip would not catch.
  - **run-counter continuation:** restore runs containing `run_0007`, then assert
    the next `put_run()` returns `run_0008` (pins the restore-path `_RUN_COUNTER`
    update at state.py:949 that C3 deletes the alias for).
  - channel helpers: `canonical_channel_color`, `canonical_channel_role`
    (incl. the `ValueError` on an unknown role), and `resolve_target_channel` /
    `resolve_layer_name` against a fake viewer with stub layers (reuse the layer
    stubs already used in `tests/test_tools_channels.py` / `test_tools_view.py`).

  Imports target `imajin.agent.state` for now; Phase 2 retargets them.

### Phase 1 — De-globalize internals (file stays at `agent/state.py`)

Each commit follows the same recipe: (a) rewrite **every** function that reads or
writes the alias — including the restore/snapshot/resolve helpers, not just the
obvious CRUD set — to use `current_session().<field>`; (b) remove that field's
`global`/rebind line from `set_current_session`; (c) delete the alias global,
**gated on `rg "\b<ALIAS>\b" src/imajin/agent/state.py` showing no references
outside the removed definition line** — not on a hand-listed function set (the
per-commit "Touches" lists below were verified against the source, but the `rg`
check is the authority). Run `test_session.py` plus the family's existing tool
tests after each.

- [x] **C1. files** → `current_session().files`. Touches `put_file`, `get_file`,
  `iter_file_records`, `list_files`, `update_file_status`, `reset_files`, **and the
  direct `_FILES[...]` write in `_restore_session_state_impl` (state.py:879).**
  Delete `_FILES`. (tests: `test_session.py`, `test_phase2_workflow.py`, plus the
  C0 six-family restore round-trip)
- [x] **C2. recipes** → `current_session().recipes`. Touches `put_recipe`,
  `get_recipe`, `list_recipes`, `reset_recipes`. Delete `_RECIPES`.
  (tests: `test_recipe_import.py`)
- [x] **C3. runs** → `current_session().runs` and `.run_counter`. Touches
  `put_run`, `get_run`, `list_runs`, `reset_runs`, **and the run-counter
  continuation `_RUN_COUNTER[0] = max(...)` in `_restore_session_state_impl`
  (state.py:949).** Delete `_RUNS`, `_RUN_COUNTER`. (guarded by the C0
  run-counter-continuation test)
- [x] **C4. qc records** → `current_session().qc_records`. Touches `put_qc_record`,
  `get_qc_record`, `list_qc_records`, `reset_qc_records`. Delete `_QC_RECORDS`.
  (tests: `test_tools_qc.py`, `test_qc_dock.py`)
- [x] **C5. samples** → `current_session().samples`. Touches `put_sample`,
  `list_samples`, `get_sample`, `reset_samples`, and the sample half of
  `snapshot/restore_session_state`. Delete `_SAMPLES`. **Note:
  `attach_sample_columns_to_table` is named for samples but actually reads/writes
  `_TABLES` (state.py:720–734), not `_SAMPLES`. Migrate its body to
  `current_session().tables` — do it here (safe: `_TABLES` is not deleted until
  C7) or defer it to C7, but never point it at `.samples`.**
  (tests: `test_phase3_experiment.py`, `test_tools_experiment.py`)
- [x] **C6. channel annotations** → `current_session().channels`. Touches
  `put_channel_annotation`, `list_channel_annotations`, `reset_channel_annotations`,
  the channel half of snapshot/restore, **and the five `_CHANNELS` reads in the
  resolver layer that the CRUD list misses: `resolve_layer_name` (state.py:620),
  `_confirmed_target_layers` (982), `_image_layer_names` (998), and
  `resolve_target_channel` (1041, 1064).** Delete `_CHANNELS`. Leave
  `_CHANNEL_COLOR_ALIASES` / `_CHANNEL_ROLE_ALIASES` as module constants.
  (tests: `test_tools_channels.py`, plus `resolve_target_channel` coverage from
  the C0 channel-resolution tests)
- [x] **C7. tables + listeners** → `current_session().tables` and
  `.table_listeners`. Touches `get_table`, `get_table_entry`, `iter_table_entries`,
  `put_table`, `set_table`, `update_table`, `list_tables`, `reset_tables`,
  `on_tables_changed`, `_emit_tables_changed`, **the `_TABLES.items()` read in
  `snapshot_session_state` (state.py:831), and `attach_sample_columns_to_table`
  (state.py:720–734) unless it was already migrated in C5.** Delete `_TABLES`,
  `_TABLE_LISTENERS`. (tests: `test_table_dock.py`, `test_tools_measure.py`)
- [x] **C8. viewer** → `current_session().viewer` only. Simplify `set_viewer` to
  set `current_session().viewer`; `get_viewer` already reads it. Delete `_VIEWER`.
  (tests: `test_tools_view.py`)
- [x] **C9. notification machinery onto the session.** Add
  `state_change_depth`/`pending_tables_changed` (and, if kept, pending reasons) as
  `AnalysisSession` fields; rewrite `_state_changed`, `_tables_changed`,
  `bulk_state_update` to use them. Behavior is identical for single-session usage
  (the only real usage) and a session swap now resets the counters coherently.
  Delete the three module globals. Guarded by the C0 bulk-update test.
- [x] **C10. collapse `set_current_session` + refresh docstring.**
  `set_current_session` becomes `global _CURRENT_SESSION; _CURRENT_SESSION =
  session`. Update the `AnalysisSession` docstring to state that free functions
  delegate to `current_session()` and that there are no compat aliases. Confirm
  `grep -nE "^_[A-Z_]+ =" src/imajin/agent/state.py` shows only `_CURRENT_SESSION`
  and the two channel-alias constants.

### Phase 2 — Relocate out of `agent/`

- [x] **C11. move + shim.** `git mv src/imajin/agent/state.py
  src/imajin/session.py`. Recreate `src/imajin/agent/state.py` as a one-line
  re-export shim: `from imajin.session import *  # noqa: F401,F403` (default star
  re-exports every public name; external code imports only public names, verified).
  Whole suite stays green with zero call-site edits.
- [x] **C12. migrate `tools/` imports** `imajin.agent.state` → `imajin.session`
  (21 files; keep each import lazy/top-level exactly as it is — only the path
  changes).
- [x] **C13. migrate `ui/` imports** (4 files).
- [x] **C14. migrate `agent/` imports** (the actual sites are `context.py` and
  `review_checkpoint.py`; execution/provenance don't import state). Shim
  `agent/state.py` left untouched.
- [x] **C15. migrate root modules + coupled test patch targets.** The two
  root-level `src/imajin/` consumers that Phase 2's directory buckets miss:
  `anchor.py` (function-local `from imajin.agent.state import list_files` at
  line 36 — also rewrite the `imajin.agent.state.list_files` reference in its
  docstring at line 32 so the C17 grep gate is clean) and `result_bundles.py`
  (`from imajin.agent.state import get_table` at line 15). **In the same commit**,
  retarget the only mock sites that patch a moved function:
  `imajin.agent.state.list_files` → `imajin.session.list_files` in
  `tests/test_anchor.py` (3×) and `tests/test_tools_results.py` (2×). They must
  move *with* `anchor.py` — both consumers import `list_files` function-locally, so
  a patch on the old path silently misses once the source reads from
  `imajin.session`. (`get_table` is not patched anywhere, so `result_bundles.py`
  needs no coupled test edit.)
- [x] **C16. migrate `tests/` imports**, including `conftest.py` last within this
  commit (it is `autouse`, so a mistake fails everything — change it deliberately
  and run the full suite). The `list_files` patch-target strings in
  `test_anchor.py` / `test_tools_results.py` were already handled in C15 — only
  plain `import imajin.agent.state` / `from imajin.agent.state import …` lines
  remain here.
- [x] **C17. delete the shim.** Confirm `grep -rn "agent.state" src tests` is empty
  (the `anchor.py` docstring reference was rewritten in C15), remove
  `src/imajin/agent/state.py`. Green.

### Phase 3 — Remove circular-import workarounds

`session` now imports nothing from `imajin`, so the **state** edge is gone and
`agent → tools → session` is a DAG. (The residual `tools → agent.execution` /
`tools → agent.qt_dispatch` top-level imports in `batch_runner.py`, `workflows.py`,
`results.py` — paired with `execution.py`'s lazy `from imajin.tools…` imports — are
a *separate* agent/tool-boundary coupling, out of scope here; do **not** un-lazy
`execution.py`'s tool imports.) Convert the function-local **session** imports back
to module-top imports, one source module per commit, and after each run
`python -c "import imajin.tools.<mod>"` plus the suite to confirm no cycle reappeared.

- [x] **C18. experiment.py** (9 lazy imports → top-level).
- [x] **C19. batch_runner.py** (5).
- [ ] **C20. segment.py / report.py / napari_ops.py** (3 each) — one commit each
  if any still trips a cycle; otherwise batch the clearly-safe ones.
- [ ] **C21. remaining** (workflows, stats, layers, channels, view,
  _workflow_outputs). Finish with a repo-wide check that no `from imajin.session`
  remains inside a function body except where a genuine cycle still requires it
  (document any such exception inline).

---

## Decision Document

- **Keep the free-function API; migrate internals only.** No caller changes, no
  signatures touched. The functions become thin delegations over
  `current_session()`. (Owner choice: smallest, safest depth.)
- **`AnalysisSession` becomes the single source of truth.** All per-session
  mutable state lives on the dataclass instance returned by `current_session()`.
  The only surviving module-level mutable name is `_CURRENT_SESSION`. Immutable
  lookup tables (`_CHANNEL_COLOR_ALIASES`, `_CHANNEL_ROLE_ALIASES`) stay as module
  constants — they are not session state.
- **Notification counters move onto the session** (C9) so a session swap resets
  them; behavior is unchanged under the single-session usage that actually occurs.
- **New module location: `src/imajin/session.py`** (top-level, foundation layer).
  Chosen over keeping it under `agent/` because it is depended on by `tools/`,
  `ui/`, and `agent/` alike; placing it at the root removes the dependency cycle.
  Public import path becomes `imajin.session`. A re-export shim at
  `imajin/agent/state.py` keeps the old path alive during migration and is deleted
  in C17 (verified zero out-of-repo importers, so no permanent deprecation layer is
  kept; if external importers later surface, drop C17 and keep the shim with a
  `DeprecationWarning` instead).
- **De-globalize before relocate.** The risky logic change (Phase 1) happens with
  import paths frozen; the relocation (Phase 2) is then a mechanical, behavior-free
  move. This keeps each phase single-concern.
- **Module stays a single file, not a package.** Splitting `session.py` into
  `records/`, `channels`, `layers` submodules is deferred. If pursued later, making
  it `imajin/session/__init__.py` preserves the `imajin.session` import path with
  no further caller churn.

## Testing Decisions

- **What a good test asserts here:** observable behavior through the public
  surface — "put X then get X returns X", "reset clears", "a fresh session is
  isolated from the old one", "a bulk block fires the listener once". A good test
  must **not** assert that a specific module global exists or that internals read a
  particular variable; those are exactly the implementation details this refactor
  deletes. The C0 net is written to survive Phase 1 unchanged.
- **Module under test:** the session module (`imajin.agent.state` → `imajin.session`).
  `tests/test_session.py` is new; it is the first commit and the gate for every
  subsequent one.
- **Prior art / fixtures to reuse:** the `autouse` reset fixture in
  `tests/conftest.py`; layer/viewer stubs in `tests/test_tools_view.py` and
  `tests/test_tools_channels.py`; session-swap and annotation patterns already
  present in `tests/test_phase2_workflow.py` and `tests/test_phase3_experiment.py`.
- **Preserve the conftest quirk:** the autouse fixture resets the per-family
  registries but deliberately does **not** reset tables. Do not "tidy" this during
  the refactor; a test relies on table state surviving where samples/files do not.
- **Per-commit gate:** `uv run pytest -q -m "not slow and not integration"` after
  every commit. Heavy `slow`/`integration` paths are unaffected by this refactor.

## Out of Scope

- Splitting the large tool modules (`segment.py` 1322, `stats.py` 989,
  `batch_runner.py` 957, `report.py` 734). Separate effort; the existing
  `_segmentation_outputs.py` / `_trace_*.py` / `_workflow_*.py` helper-module
  pattern is the template when it happens.
- Splitting the relocated `session.py` into a sub-package.
- Unifying the manual-dock and LLM execution paths / moving the manual dock to
  background execution (PROJECT_PLAN Phase 5). Behavioral, not a pure refactor.
- Any change to `analysis/` — it is already napari-free and well-separated; leave
  it.
- Any change to the `@tool` registry, provenance, or result-bundle code beyond the
  import-path move.
- Public API/signature changes of any session free function.

## Further Notes

- **Why this is the keystone:** with `session` de-globalized and relocated, the
  two follow-on refactors get materially easier — un-lazying imports (Phase 3) is
  only safe once the cycle is broken, and any later split of the big tool modules
  is simpler when those modules import a clean top-level `session` rather than
  reaching into `agent.state` from inside functions.
- **Rollback granularity:** every commit is independently revertible and green. If
  Phase 2 or 3 surfaces an unexpected cycle, stop at the last green commit; the
  Phase 1 de-globalization stands on its own as a complete, shippable improvement.
- **Squash guidance:** Phase 1's C1–C8 are intentionally one-family-per-commit for
  reviewability and bisectability. They may be squashed into a single
  "de-globalize session state" commit before merge if the reviewer prefers, but
  keep them separate during development so a regression bisects to one family.
