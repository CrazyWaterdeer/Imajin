# segment.py Tool-Module Split — Refactor Plan

> **For agentic workers:** implement this plan **one commit at a time**, running
> `uv run pytest -q -m "not slow and not integration"` after every commit. Each
> commit below is designed to leave the codebase green. Steps use checkbox
> (`- [ ]`) syntax for tracking. Do not batch commits unless explicitly asked.

**Date:** 2026-07-04
**Status:** Planned (not started) — revised after one Codex review round (2026-07-04).
**Owner decisions baked in:**
- Shape = **both**: extract the repeated tool-wrapper scaffolding first (dedup,
  import path frozen), **then** split the slimmed tools into a `segment/` package
  by family.
- Dedup = **in scope**: collapse the ~6×-duplicated boilerplate into shared
  helpers (not a mechanical cut/paste).
- Safety net = **equivalence guard**: a characterization/golden test (C0) that pins
  current tool outputs — result dict, layer metadata, **label array**, and the
  **registry entry + json-schema** — before touching anything; run before/after.
- Private surface = **keep re-exporting, but patch where used.** Readable aliases
  stay on the package (`imajin.tools.segment.<name>`) so read-only callers and the
  workflow need no change. **Amended by the Codex review:** three of these aliases
  are *monkeypatch targets*, and a re-exported value is not patch-through once the
  tool body moves to a submodule, so those three patch sites are relocated to the
  helper's real module (≈6 lines across three test files). See the "Monkeypatch
  compatibility" constraint below.

**Goal:** Turn `src/imajin/tools/segment.py` (1600 lines, 8 `@tool` functions)
from a single god-module — where six segmentation tools each repeat the same
load→guard→boundary→min_size→segment→emit-labels→save-QC-PNG→build-payload
skeleton — into (1) a small set of shared scaffolding helpers and (2) a
`segment/` package split by tool family, behind an **unchanged public and private
import surface** and **byte-equivalent tool outputs**.

**Non-goal:** No behavior change. No tool name or signature change. No algorithm
change (the numeric cores already live in `analysis/`). No change to which
warnings each tool emits, or the order in which each tool validates its inputs.

---

## Problem Statement

`tools/segment.py` is the largest file in the project (1600 lines). It is already
a *thin wrapper layer*: every heavy algorithm was previously extracted into
`analysis/` (`target_pipeline.py` — its docstring reads *"Extracted from
`tools.segment.segment_target_objects`"* — plus `segmentation.py`,
`segmentation_auto3d.py`, `domain_segmentation.py`, `roi_quality.py`). What remains
is mostly **glue copy-pasted across the eight tools**:

- **Load + axis/dim guard** (6 tools): `snapshot_layer` → `materialize_array` →
  `_layer_axes_for_seg` → reject time-series (`"T" in axes`) → reject non-2D/3D.
  Five allow 2D|3D; `segment_3d_cells_auto` requires 3D ZYX. ~15 lines each.
- **Boundary-mask load + resolve (+ broadcast warning)** (4 tools): snapshot the
  boundary layer, `materialize_array`, `_resolve_boundary_mask(raw, shape)`. Three
  (`segment_3d_cells_auto`, `segment_target_objects`, `auto_segment_target`) then
  append the *"2D ROI broadcast across all N Z planes"* warning;
  `segment_expression_domain` **does not** (it builds a 2D outline and clips
  ROI-locally instead).
- **Effective `min_size` from physical units** (2 tools): identical `xy_area` +
  `_min_size_from_physical(...)` + `max(16, min(512, round(xy_area*0.00005)))`.
- **QC-PNG save block** (6 tools): a ~25-line `try/except`. ~150 lines of
  near-verbatim duplication.
- **Secondary-outline projection for the QC overlay** (3 tools): project the
  boundary mask to a 2D `int32` outline — but **each tool sources the mask
  differently** (target re-snapshots the boundary layer; auto-target reuses the
  already-resolved bool; domain uses the raw boundary array).

A change to any cross-cutting behavior must be edited in up to six places, and the
eight distinct tools are hard to navigate in one 1600-line file. This is the exact
split the **session-state-extraction plan deferred**, naming the template: *"the
existing `_segmentation_outputs.py` / `_trace_*.py` / `_workflow_*.py` helper-module
pattern."*

### Constraints that shape the refactor

1. **The `@tool` registry keys on `func.__name__`; the signature is the input
   schema.** `registry.tool()` registers each function under its name and builds a
   pydantic input model from `get_type_hints(func)` + the signature, cached on the
   `ToolEntry` at decoration time. So **every tool name and signature is frozen**,
   and the modules holding the tools must be imported at package init so
   registration fires (`tools/__init__.py` does `from imajin.tools import segment`).
   The json-schema is the LLM-facing contract and must not drift.

2. **`imajin.tools.segment.<name>` is load-bearing for public tools AND private
   helpers.**
   - `_workflow_steps.py` calls `_segment.segment_target_objects`,
     `_segment.segment_intensity_regions`, `_segment.segment_3d_cells_auto`,
     `_segment.cellpose_sam`, `_segment._voxel_spacing`, and
     `from imajin.tools.segment import segment_expression_domain`.
   - `tests/` reach six private helpers via the module (read and/or patch):
     `_threshold_noise_floor`, `_write_segmentation_qc_png`, `_prepare_corrected`,
     `_boundary_bbox_slices`, `_voxel_spacing`, **and `_get_cellpose_model`**
     (omitted from the first draft — it is patched by four test sites).

3. **Monkeypatch compatibility (the load-bearing subtlety).** `monkeypatch.setattr`
   rebinds a *name in a module namespace*; it only affects call sites that look the
   name up **as a global in that same module** at call time. Three helpers are
   patched by tests and the patch must keep intercepting the real call after the
   split:
   - `_get_cellpose_model` — patched at `tests/test_tools_segment.py:66`,
     `tests/test_review_mode_workflow.py:43`, `tests/test_phase2_workflow.py:49`,
     `tests/test_phase3_experiment.py:481`.
   - `_prepare_corrected` — patched at `tests/test_tools_segment.py:876` (read at 870).
   - `_boundary_bbox_slices` — patched at `tests/test_tools_segment.py:892` (read at 911).

   Re-exporting these on `segment/__init__` makes them *readable* but **not
   patch-through**: once `cellpose_sam` lives in `segment/cellpose.py`, its
   `_get_cellpose_model(...)` call resolves `cellpose.py`'s global, which a patch on
   `segment.__init__` does not touch. The fix is "patch where it is used": give each
   patched helper **one canonical home**, call it there, and point the patch sites
   at that home (see C2, C11).

4. **No formatter config** (`pyproject` has no ruff/black/line-length). Match the
   surrounding hand-formatting by eye (~88-col wraps, trailing-comma multilines).

## Solution

Three ordered phases, all behind a stable import surface and byte-equivalent
outputs, mirroring the session-extraction pattern (*change-in-place first, relocate
second*):

- **Phase 0 — Net.** A characterization/golden guard (C0) that pins, for each tool,
  the **registry entry + `inspect.signature` + json-schema**, and for the six
  segmentation tools + `correct_roi`, the **result dict + written layer metadata +
  label array** on small deterministic synthetic inputs (Cellpose stubbed). Plus a
  monkeypatch-interception check and a `review_target_roi` error-path check. Must
  pass on the current code unchanged; gate for every later commit.

- **Phase 1 — Extract shared scaffolding (dedup), import path frozen.** Add a
  sibling `tools/_segmentation_io.py` (parallel to `_segmentation_outputs.py`) and
  move the copy-pasted blocks into it, **one block per commit**, rewriting the tools
  to call the helpers. `segment.py` stays one file the whole phase, so the risky
  dedup lands with import paths and the registry untouched. Two behavior-preserving
  subtleties are called out per commit: domain's `threshold_strategy` check stays
  *before* the shared guard, and the boundary broadcast-warning stays *opt-in* (not
  emitted by domain). The file shrinks ~1600 → ~950 lines.

- **Phase 2 — Split into a `segment/` package (behavior-free relocation).**
  `git mv` the slimmed file to `segment/__init__.py`, then move one tool family per
  commit into a submodule, `__init__` importing every submodule (registration) and
  re-exporting all eight tools plus the readable private aliases. The full import
  surface **and monkeypatch-through behavior** must hold after *every* commit, not
  just the last — each commit runs C0 and stays green.

The eight tools keep their names, signatures, schemas, and outputs throughout; C0
proves it commit by commit.

---

## Current-State Facts (evidence gathered 2026-07-04, verified against source)

| Fact | Value |
|---|---|
| `tools/segment.py` size | 1600 lines |
| `@tool` functions | 8: `cellpose_sam`, `segment_3d_cells_auto`, `segment_intensity_regions`, `segment_target_objects`, `auto_segment_target`, `segment_expression_domain`, `correct_roi`, `review_target_roi` |
| Tools sharing the load→guard→QC-PNG skeleton | 6 (all but `correct_roi`, which delegates to `segment_target_objects`, and `review_target_roi`, which opens a dock) |
| QC-PNG save block duplication | ~25 lines × 6 tools |
| Boundary load+resolve | 4 tools; broadcast-warning appended by **3** (`segment_3d_cells_auto`, `segment_target_objects`, `auto_segment_target`) — **not** `segment_expression_domain` |
| Secondary-outline sourcing | differs per tool: target **re-snapshots**; auto-target reuses resolved bool; domain uses raw boundary array |
| Domain input-validation order | `threshold_strategy != "noise_floor"` raises **before** the layer is loaded |
| Algorithms already in `analysis/` | `target_pipeline`, `segmentation`, `segmentation_auto3d`, `domain_segmentation`, `roi_quality` |
| Existing helper-module precedent | `tools/_segmentation_outputs.py` (196 lines) |
| Public path to preserve | `imajin.tools.segment.<tool>` (registry + `tools/__init__.py` + `_workflow_steps.py`) |
| Private surface to preserve — **readable** (6) | `_threshold_noise_floor`, `_write_segmentation_qc_png`, `_voxel_spacing`, `_prepare_corrected`, `_boundary_bbox_slices`, `_get_cellpose_model` |
| Private surface — **patch-through** (3, relocate patch sites) | `_get_cellpose_model` → canonical in `_segmentation_io.py`; `_prepare_corrected`, `_boundary_bbox_slices` → canonical in `segment/target.py` |
| Monkeypatch sites to update | `_get_cellpose_model` ×4 (`test_tools_segment.py:66`, `test_review_mode_workflow.py:43`, `test_phase2_workflow.py:49`, `test_phase3_experiment.py:481`); `_prepare_corrected` ×1 + read (`test_tools_segment.py:876`, 870); `_boundary_bbox_slices` ×1 + read (`test_tools_segment.py:892`, 911) |
| Existing test coverage | `tests/test_tools_segment.py` = 1049 lines |
| Formatter config | none — match surrounding style |
| napari manifest / UI blast radius | `napari.yaml` registers only the reader; UI manual-dock + LLM runner are registry-driven — no hardcoded `tools.segment` path (safe under the package split) |

---

## Commits

### Phase 0 — Characterization / equivalence + registry-golden net

- [ ] **C0. Add `tests/test_segment_equivalence.py` (passes on unchanged code).**
  Four layers of assertion, observable-behavior only (never "helper X exists at
  path Y"):
  1. **Registry/schema golden — all 8 tools.** `get_tool(name)` exists;
     `ToolEntry` flags (`phase`, `manual`, `llm`, `worker`, `vision_hint`) match
     pinned values; `inspect.signature(imajin.tools.segment.<tool>)` matches a
     pinned string; `entry.input_model.model_json_schema()` matches a pinned dict.
     Invoke at least one tool through `call_tool(name, **kwargs)` (exercises the
     registry/validation path end-to-end, not just a direct call).
  2. **Output golden — 6 segmentation tools + `correct_roi`.** On small seeded
     synthetic inputs (Cellpose stubbed with a deterministic fake model), assert the
     **full result dict**, the **written layer `.metadata`**, AND the **label-layer
     array** (`viewer.layers[res["labels_layer"]].data`, exact equality or a stable
     hash) equal pinned values, after dropping only the volatile absolute
     `qc_png_path` string. For the two boundary-intricate tools
     (`segment_target_objects`, `segment_expression_domain`) cover no-boundary and
     2D-ROI-on-3D-stack; for `segment_target_objects` also assert crop == no-crop
     label arrays.
  3. **QC-PNG written (don't just drop the path).** For ≥1 representative tool,
     force `save_qc_png=True` into a tmp bundle and assert the labels layer's
     `metadata["qc_png_path"]` is set and lands under the bundle `qc/` dir.
  4. **Monkeypatch-interception + `review_target_roi`.** Assert that patching the
     canonical helper location still stubs the call (a fake `_get_cellpose_model`
     patched at its canonical home makes `cellpose_sam` return the fake mask), and
     add a `review_target_roi` error-path check (no viewer / missing layer returns
     the `ok: False` payload). This is the gate for every commit below.

### Phase 1 — Extract shared scaffolding (file stays at `tools/segment.py`)

Each commit adds one helper to a new `tools/_segmentation_io.py`, rewrites the
callers, runs C0 + the existing segment tests, and stays green. No signature,
schema, warning-set, or output change.

- [ ] **C1. `load_and_guard`.** Add
  `load_and_guard(image_layer, *, tool_name, dims) -> (snapshot, data, axes)`
  wrapping `snapshot_layer` → `materialize_array` → `_layer_axes_for_seg` →
  time-series rejection → dim rejection, where `dims` selects the `"2d_or_3d"` vs.
  `"3d_only"` (`segment_3d_cells_auto`) message. Move `_layer_axes_for_seg` here too.
  Rewrite the six tools' opening block. Keep the exact error strings (tool name
  interpolated). **`segment_expression_domain` keeps its `threshold_strategy`
  validation *before* the `load_and_guard` call** — the guard must not become the
  first statement there, or an invalid-strategy + bad-layer call would change which
  error is raised.
- [ ] **C2. Relocate the Cellpose model accessor (patch-through fix).** Move
  `_CACHED_MODELS` + `_get_cellpose_model` to `_segmentation_io.py`. `cellpose_sam`
  and `segment_3d_cells_auto` call it **module-qualified**
  (`_seg_io._get_cellpose_model(...)`) so there is a single patch location covering
  both. Update the four `_get_cellpose_model` monkeypatch sites to
  `imajin.tools._segmentation_io._get_cellpose_model`. Keep a readable
  `_get_cellpose_model` alias on `segment` for symmetry. (Split from the load guard
  per the review: different blast radius, and it is a patch target.)
- [ ] **C3. `resolve_boundary` + opt-in `boundary_broadcast_warning`.** Add
  `resolve_boundary(boundary_mask, raw_shape) -> (bool_mask | None, raw | None)`
  (snapshot + `materialize_array` + `_resolve_boundary_mask`) and a *separate*
  `boundary_broadcast_warning(bool_mask, raw) -> str | None`. Convert
  `segment_3d_cells_auto` and `segment_target_objects` first, then
  `auto_segment_target` — all three call `resolve_boundary` **and** append the
  warning exactly as today. `segment_expression_domain` calls **only**
  `resolve_boundary` (no warning) and keeps its ROI-local noise-floor + outline
  logic — its `domain_warnings` set is unchanged.
- [ ] **C4. `effective_target_min_size`.** Extract the
  `_min_size_from_physical(...) or max(16, min(512, round(xy_area*0.00005)))`
  fallback shared by `segment_target_objects` and `auto_segment_target`. Leave
  `segment_intensity_regions`' different `... or int(min_size)` inline.
- [ ] **C5. `finalize_qc_png` (two explicit layer args).** Add
  `finalize_qc_png(image, masks, labels_layer_obj, source_layer_obj, *, method, save_qc_png, qc_png_path, secondary_outline_mask=None) -> (path, error, skipped)`
  wrapping the ~25-line `try/except`. It needs **both** the new labels layer object
  (for `.name` and the `metadata["qc_png_path"]` write) and the source layer object
  (for `_default_qc_png_path(labels_name, source_obj)` and
  `source_layer=source_obj.name`) — a single `source` param is ambiguous. Convert
  the two outline-free tools (`cellpose_sam`, `segment_intensity_regions`) first,
  then the four that pass a `secondary_outline_mask`. **Largest single dedup.**
- [ ] **C6. `project_boundary_outline_2d` (projection only).** Extract just the
  `(m if m.ndim == 2 else np.any(m, axis=0)).astype(np.int32)` projection. Each
  caller keeps its **current input source**: `segment_target_objects` keeps its
  boundary-layer re-snapshot, `auto_segment_target` feeds the resolved bool, and
  `segment_expression_domain` feeds the raw boundary array. Do **not** unify the
  sourcing (that would change target's snapshot timing in a live viewer). End of
  Phase 1: `segment.py` ≈ 950 lines, C0 green throughout.

### Phase 2 — Split into a `segment/` package (behavior-free)

**Invariant for every commit here:** the full public surface (8 tools), the
readable private aliases, and monkeypatch-through for the 3 patched helpers must all
hold, verified by C0 after each commit.

- [ ] **C7. Module → package.** `git mv src/imajin/tools/segment.py
  src/imajin/tools/segment/__init__.py`. `imajin.tools.segment` now resolves to the
  package; attribute access and the registry are unchanged. Because C2 already moved
  `_get_cellpose_model` into `_segmentation_io`, its patch target is stable
  regardless of this move. Run the whole suite. Pure move, zero edits elsewhere.
- [ ] **C8. `cellpose.py`.** Move `cellpose_sam` to `segment/cellpose.py` (imports
  helpers + calls `_seg_io._get_cellpose_model` module-qualified). `__init__` adds
  `from .cellpose import cellpose_sam`.
- [ ] **C9. `auto3d.py`.** Move `segment_3d_cells_auto` (+ `_candidate_summary`,
  `_json_ready`, its only users). `__init__` re-exports.
- [ ] **C10. `intensity.py`.** Move `segment_intensity_regions`. `__init__`
  re-exports.
- [ ] **C11. `target.py` (+ target-only patch sites).** Move
  `segment_target_objects`, `auto_segment_target`, `correct_roi` together
  (`correct_roi`/`auto_segment_target` call `segment_target_objects` in-module, so
  co-locating avoids a cross-module call). Keep `_prepare_corrected` and
  `_boundary_bbox_slices` as `target.py` module globals (imported `as _alias`). **In
  the same commit**, update their monkeypatch/read sites
  (`tests/test_tools_segment.py:870,876,892,911`) to `segment.target._prepare_corrected`
  / `segment.target._boundary_bbox_slices` so C0 stays green at this commit.
  `__init__` re-exports the three tools (and keeps readable aliases).
- [ ] **C12. `domain.py`.** Move `segment_expression_domain`. `__init__` re-exports.
- [ ] **C13. `review.py`.** Move `review_target_roi`. `__init__` now holds only
  submodule imports + re-exports.
- [ ] **C14. Final surface verification.** Add `__all__`; assert the C0
  registry/schema/patch golden is green; `grep -rn "tools.segment" src tests` shows
  no broken paths; confirm the readable aliases (`_threshold_noise_floor`,
  `_write_segmentation_qc_png`, `_voxel_spacing`, `_get_cellpose_model`,
  `_prepare_corrected`, `_boundary_bbox_slices`) all resolve on the package.

---

## Decision Document

- **Both, sequenced: dedup in place, then relocate.** Phase 1 runs with the file
  frozen at `tools/segment.py` so the behavior-changing edit lands with import paths
  and the registry untouched; Phase 2 is a mechanical, behavior-free relocation.
  Single-concern phases, each independently shippable.
- **Preserve every tool name, signature, and json-schema.** They are the registry
  keys and the LLM/manual tool contract; C0 golden-pins all three.
- **Re-export readable aliases; patch where used.** The package `__init__`
  re-exports all eight tools and the six private aliases for read access
  (`_workflow_steps.py` and read-only tests need no change). Because monkeypatch
  rebinds a *namespace name*, the three patched helpers get one canonical call home
  — `_get_cellpose_model` in `_segmentation_io.py` (shared by cellpose + auto3d),
  `_prepare_corrected`/`_boundary_bbox_slices` in `segment/target.py` — and their
  ~6 patch/read sites move there. This amends the initial "zero test churn" reading:
  churn is limited to those patch-target strings; no tool-call sites in tests change.
  (Rejected alternative: routing submodule helper calls back through the package
  object to keep literal zero churn — it creates an import cycle and a reach-into-
  the-package smell.)
- **`_get_cellpose_model` is part of the compatibility surface.** It was missing
  from the first draft's five names; the real preserved private surface is six.
- **Shared scaffolding lives in `tools/_segmentation_io.py`**, parallel to
  `_segmentation_outputs.py` (which stays the low-level PNG writer). It holds
  input-prep (`load_and_guard`, `resolve_boundary` + `boundary_broadcast_warning`,
  `effective_target_min_size`, `project_boundary_outline_2d`, the Cellpose model
  accessor) and the `finalize_qc_png` orchestrator that wraps `_save_qc_png`.
- **`finalize_qc_png` takes both the labels layer and the source layer objects.**
  The path helper needs `_default_qc_png_path(labels_name, source_obj)` and the PNG
  writer needs `source_layer=source_obj.name`; a single `source` param is ambiguous.
- **Secondary-outline sourcing is preserved per tool.** Only the 2D projection is
  shared; target keeps its re-snapshot, so no live-viewer timing/mutation behavior
  changes.
- **Domain's warning-set and validation order are preserved.** The broadcast
  warning stays opt-in (domain never emits it) and `threshold_strategy` is still
  validated before the layer loads.
- **Family grouping:** `cellpose` / `auto3d` / `intensity` / `target`
  (`segment_target_objects` + `auto_segment_target` + `correct_roi`) / `domain` /
  `review` — grouped by in-module call edges, not line count.
- **Do NOT unify the per-tool metadata/return dicts.** Their keys are genuinely
  tool-specific; a shared builder would be over-abstraction and would obscure each
  tool's contract. Only the mechanical cross-cutting blocks are deduped.
- **`analysis/` is untouched.**

## Testing Decisions

- **What a good test asserts here:** observable output through the public tool
  surface — the returned dict, the written layer metadata, **and the label array**
  for a fixed input — plus the **registry entry, signature, and json-schema** for
  each tool, plus that monkeypatching the canonical helper location still
  intercepts the call. A good test must **not** assert that a particular helper
  module/global exists at a path; those are what Phase 1/2 move. C0 is written to
  survive both phases unchanged.
- **Why dict+metadata alone is insufficient:** a crop/scatter or boundary bug can
  leave the summary dict equal while the label geometry changes — so C0 compares the
  label arrays directly, and pins crop == no-crop for `segment_target_objects`.
- **Why schema-golden matters:** the json-schema is the LLM-facing contract; a
  silent signature drift (reordered/renamed kwarg) would pass a behavior test but
  break tool-calling. C0 pins `model_json_schema()` for all 8.
- **Determinism:** the five non-Cellpose tools are deterministic on seeded input;
  `cellpose_sam` and the `include_cellpose_sam` branch run against a stubbed model
  so C0 never downloads weights or touches a GPU.
- **Prior art / fixtures to reuse:** the `viewer` and `synthetic_blob_image`
  fixtures and the `_FakeModel` / `monkeypatch` Cellpose-stub in
  `tests/test_tools_segment.py`; its boundary/crop-equivalence tests
  (`test_target_objects_crop_matches_no_crop`,
  `test_segment_target_objects_boundary_mask_keeps_only_inside`); the headless
  `analysis.target_pipeline.segment_target_array` path; the registry assertions in
  `tests/test_tool_registry.py` and `tests/test_manual_dock.py`.
- **Per-commit gate:** `uv run pytest -q -m "not slow and not integration"` after
  every commit. `slow`/`integration` (real-model) paths are unaffected.

## Out of Scope

- Any change to a tool's name, signature, parameters, defaults, schema, emitted
  warnings, validation order, or output values (byte-equivalent outputs are a hard
  requirement, guarded by C0).
- Any change to `analysis/` (algorithms already extracted and separated).
- Any change to the `@tool` registry, provenance, or the manual/LLM execution paths.
- Unifying the per-tool metadata/return dicts, or unifying the secondary-outline
  *sourcing* (deliberately rejected above as over-abstraction / behavior change).
- Splitting the other large tool modules (`stats.py` 1016, `trace.py` 959,
  `batch_runner.py` 959, `report.py` 806) — separate efforts using this template.
- Repointing the *read-only* private-helper access to `analysis/` (declined by the
  owner in favor of readable re-exports; only the 3 patch targets relocate).

## Further Notes

- **Rollback granularity:** every commit is independently revertible and green.
  Phase 1 stands alone as a shippable dedup even if Phase 2 is never done; if Phase 2
  surfaces an unexpected import/registration problem, stop at the last green commit.
- **Why the model accessor moves first (C2):** relocating `_get_cellpose_model` into
  `_segmentation_io` in Phase 1 decouples its patch target from the later package
  split, so C8 (moving `cellpose_sam`) needs no test edit — the surface stays green
  by construction.
- **Squash guidance:** Phase 1's C1–C6 and Phase 2's C8–C14 are one-concern-per-
  commit for bisectability. They may be squashed into two commits ("dedup segment
  scaffolding", "split segment into package") before merge, but keep them separate
  during development so a regression bisects to one block.
- **Provenance:** this plan was revised after one Codex review round per the
  project's spec/plan → Codex-review workflow. The review's ten findings (monkeypatch
  patch-through, weak C0, missing label-array check, domain warning/validation-order
  preservation, `finalize_qc_png` arg ambiguity, outline sourcing, and commit
  sequencing) are all folded in above.
</content>
