# trace.py Tool-Module Split — Refactor Plan

> **For agentic workers:** implement one commit at a time, running
> `IMAJIN_RESULTS_DIR=$(mktemp -d) uv run pytest -q -m "not slow and not integration"`
> after every commit (the env var avoids a dead `/mnt/e` WSL mount that otherwise
> fails ~24 unrelated tests). Each commit leaves the suite green. Don't batch.

**Date:** 2026-07-04
**Status:** Planned (not started).
**Owner decisions baked in (reuse the proven segment split, issue #3):**
- Same template: **C0 equivalence + registry-golden guard → (minimal) scaffolding
  extraction → family package split**, behind unchanged tool names / signatures /
  json-schema and the `imajin.tools.trace.<name>` public + private surface.
- No behavior change; byte-equivalent tool outputs, guarded by C0.

**Goal:** Turn `src/imajin/tools/trace.py` (959 lines, **15 `@tool` functions**, all
`subagent="neural_tracer"`) into a `trace/` package split by family. Like
`segment`, it is already a thin `@tool` wrapper layer over extracted helper modules
(`_trace_export`, `_trace_image`, `_trace_store`, `_trace_tables`); the split is
mostly relocation, not dedup.

**Non-goal:** No tool name / signature / schema / output / flag change. No change to
`analysis/morphology_*` or the `_trace_*` helper modules. No change to the skeleton
registry semantics.

---

## Problem Statement

`tools/trace.py` is the neural-morphology tool surface: 15 specialist
(`subagent="neural_tracer"`) tools spanning distinct families — image enhancement,
skeletonization + skeleton edits, morphometry (Sholl / descriptors), SWC/CSV export,
connectome lookup, and neuron-type classification / reference library — all in one
959-line file. The numeric cores already live in `analysis/morphology_*` and the
`tools/_trace_*` helper modules; `trace.py` is the wrapper glue. Unlike `segment`
there is **almost no repeated boilerplate** (each tool is distinct), so the win is
navigability from a by-family package, not a big dedup.

### Constraints

1. **The `@tool` registry keys on `func.__name__`; the signature is the schema.**
   All 15 names + signatures + json-schemas are frozen, and the modules must be
   imported at package init so registration fires (`tools/__init__.py` does
   `from imajin.tools import trace`). **All 15 carry `subagent="neural_tracer"`**
   (so `manual=False`); `worker` varies per tool — the registry-golden pins these.
2. **`imajin.tools.trace.<name>` is load-bearing** for the 15 tools **and** three
   re-exported names read via the module: **`trace.reset_skeletons`** (18 test uses),
   **`trace._entry`** (`test_tools_morphology.py`), and **`trace.list_trace_records`**
   — imported by **`report.py`** (`from imajin.tools.trace import list_trace_records`
   at lines 492, 746; Codex found this — it is a *source* dependency, not just a
   test). All three come from `_trace_store`. **No test monkeypatches a `trace`
   private** (verified) — so, unlike segment's `_get_cellpose_model`, re-export is
   sufficient and **no patch-site relocation is needed**.
5. **Cross-family dependency (Codex):** `_skeleton_feature_vector` (a classify
   helper) calls `compute_morphology_descriptors` (a morphometry tool). So
   `classify.py` must import it **from the `morphometry` submodule directly**
   (`from imajin.tools.trace.morphometry import compute_morphology_descriptors`),
   not via the package (whose `__init__` is mid-import); morphometry (C5) is created
   before classify (C8).
3. **`_materialize`** (`trace.py:58`) is the only shared local helper — a 2-line
   `return materialize_array(arr)` used by the enhance / skeleton / set-soma /
   assign-region tools. The four classification helpers
   (`_reference_library_path`, `_load_reference_or_none`, `_add_persistence_features`,
   `_skeleton_feature_vector`) are used **only** by the classify family and move with it.
4. **Test env:** set `IMAJIN_RESULTS_DIR` (see the header note).

## Solution

- **Phase 0 — Net.** `tests/test_trace_equivalence.py`: registry/signature/json-schema
  golden for all 15 tools (incl. `subagent`/`worker`/`manual` flags); the
  public + private surface (`reset_skeletons`, `_entry`, 15 tools); and output
  equivalence on a fixed pipeline — skeletonize a seeded mask, then run
  extract_branch_metrics / compute_morphology_descriptors / compute_sholl_analysis /
  export_neural_trace / query_connectome (no-backend) / classify_neuron_type +
  find_similar_neurons (no-reference), pinning result dicts (volatile paths dropped)
  and stored-table shapes. Deterministic (connectome + classify degrade to typed
  no-op statuses without the optional extras). Gate for every later commit.

- **Phase 1 — Minimal scaffolding.** Move `_materialize` into `_trace_image.py`
  (an image helper); callers import it there. (One tiny commit; trace has no larger
  shared skeleton.)

- **Phase 2 — Split into a `trace/` package.** `git mv tools/trace.py
  tools/trace/__init__.py`, then move one family per commit into a submodule,
  `__init__` importing each submodule (registration) and re-exporting the 15 tools
  plus the `reset_skeletons` / `_entry` (and other read) aliases. Surface intact
  after every commit.

---

## Current-State Facts (evidence gathered 2026-07-04)

| Fact | Value |
|---|---|
| `tools/trace.py` size | 959 lines |
| `@tool` functions | 15, **all `subagent="neural_tracer"`** (so `manual=False`); `worker=True` on the compute-heavy ones |
| Families | enhance (2), skeleton (6), morphometry (2), export (1), connectome (1), classify (3 + 4 private helpers) |
| Already-extracted helper modules | `_trace_export`, `_trace_image`, `_trace_store`, `_trace_tables` |
| Shared local helper | `_materialize` (→ `_trace_image`); classify helpers move with classify |
| Surface read via `trace.<name>` (beyond 15 tools) | `reset_skeletons` (18×, tests), `_entry` (2×, tests), **`list_trace_records` (report.py source import)** — all from `_trace_store` |
| Monkeypatched trace privates | **none** (re-export suffices; no patch relocation) |
| Cross-family edge | classify's `_skeleton_feature_vector` → `compute_morphology_descriptors` (classify.py imports from the morphometry submodule) |
| Public path to preserve | `imajin.tools.trace.<tool>` (registry + `tools/__init__.py` + tests) |
| Tests | `test_tools_trace.py` 217, `test_tools_morphology.py` 574, `test_subagent_neural_tracer.py` 108 |
| Skeleton registry | `_SKELETON_REGISTRY` in `_trace_store`; `reset_skeletons` clears it (conftest autouse also calls it) |

---

## Commits

### Phase 0
- [ ] **C0. `tests/test_trace_equivalence.py`.** Registry/signature/schema golden for
  all 15 tools + flags; surface (`reset_skeletons`, `_entry`, 15 tools via `hasattr`
  + `get_tool(name).func is`); and a fixed-pipeline output-equivalence set (pin
  result dicts minus volatile paths + stored-table row counts / skeleton QC). Uses
  the `viewer` fixture + a seeded Y-shaped mask (reuse `test_tools_trace.py`
  patterns); `reset_skeletons()` per case. Passes on unchanged code.

### Phase 1
- [ ] **C1. Move `_materialize` to `_trace_image.py`.** Rewrite trace's callers to
  import it from there; drop the local def. (No surface change — not read via `trace.`.)

### Phase 2 (behavior-free relocation; surface intact after each commit)
- [ ] **C2. Module → package.** `git mv tools/trace.py tools/trace/__init__.py`;
  verify `imajin.tools.trace.<name>` + `reset_skeletons` + `_entry` resolve and the
  registry is unchanged. Pure move.
- [ ] **C3. `trace/enhance.py`** — enhance_neural_processes, segment_neural_processes.
- [ ] **C4. `trace/skeleton.py`** — skeletonize, extract_branch_metrics,
  prune_skeleton, set_branch_qc, set_soma_location, assign_neural_region.
- [ ] **C5. `trace/morphometry.py`** — compute_sholl_analysis,
  compute_morphology_descriptors.
- [ ] **C6. `trace/export.py`** — export_neural_trace.
- [ ] **C7. `trace/connectome.py`** — query_connectome.
- [ ] **C8. `trace/classify.py`** — classify_neuron_type, add_reference_neuron,
  find_similar_neurons + the four classification helpers. `classify.py` imports
  `compute_morphology_descriptors` from `imajin.tools.trace.morphometry` (submodule-
  direct, per Constraint 5).
- [ ] **C9. Freeze `__init__`.** Keep only submodule imports (registration +
  re-export of the 15 tools) and the `reset_skeletons` / `_entry` /
  **`list_trace_records`** aliases from `_trace_store`; add `__all__`; confirm no
  stale `tools.trace` paths and the C0 golden is green.

---

## Decision Document

- **Same proven template as segment (issue #3), lighter Phase 1.** trace has no
  repeated wrapper skeleton, so scaffolding extraction is just relocating
  `_materialize`; the value is the by-family package.
- **Family grouping:** enhance / skeleton / morphometry / export / connectome /
  classify — by concern and shared private helpers, not line count.
- **Preserve the surface via re-export; no patch relocation.** The 15 tools plus
  `reset_skeletons` and `_entry` are re-exported on the `trace` package. Because no
  test monkeypatches a trace private (they only *read* `trace._entry` /
  `trace.reset_skeletons`), re-export is patch-through-equivalent here — the segment
  `_get_cellpose_model` relocation problem does not recur. (If a future test patches
  a trace-called helper, move that patch to the submodule that calls it — see the
  segment plan and the tool-module-split note.)
- **Registry-golden pins `subagent="neural_tracer"` + `worker`/`manual`/`llm` per
  tool** — these govern which agent sees each tool; a silent flip would be a
  behavior change a plain output test misses.
- **`analysis/morphology_*` and `_trace_*` helper modules are untouched.**

## Testing Decisions

- **What C0 asserts:** observable output (result dicts + stored-table shapes +
  skeleton QC) for a fixed skeletonize→analyze pipeline, plus the registry entry,
  signature, json-schema, and the import surface for all 15 tools — never that a
  helper lives at a path (that moves).
- **Determinism:** connectome and classification degrade to typed statuses
  (`backend_unavailable` / `no_reference`) without the optional `connectome` extra /
  a reference library, so C0 needs neither; skeletonization on a seeded mask is
  deterministic.
- **Prior art:** the `viewer` fixture, the Y-mask skeletonize pattern, and the
  `reset_skeletons()` reset in `tests/test_tools_trace.py`; the `trace._entry` /
  `_write_swc` usage in `tests/test_tools_morphology.py`; the subagent tool-set
  assertions in `tests/test_subagent_neural_tracer.py` /
  `tests/test_tools_morphology.py` (they check `tools_for_anthropic("neural_tracer")`
  — the split must keep all 15 in that set).
- **Per-commit gate:** `IMAJIN_RESULTS_DIR=$(mktemp -d) uv run pytest -q -m "not slow
  and not integration"` after each commit; full suite at each phase boundary.

## Out of Scope

- Any tool name/signature/schema/flag/output change; any change to
  `analysis/morphology_*` or the `_trace_*` helper modules or the skeleton registry.
- Splitting the other large tool modules (`stats.py`, `batch_runner.py`,
  `report.py`) — separate efforts using this template.

## Further Notes

- **Rollback granularity:** every commit is independently revertible and green;
  Phase 1 alone (the `_materialize` move) is trivial and the package split (Phase 2)
  can stop at any family boundary.
- Built via the spec/plan → Codex-review workflow (one round); tool-module-split
  monkeypatch caveat and the `IMAJIN_RESULTS_DIR` test-env note are in project memory.
