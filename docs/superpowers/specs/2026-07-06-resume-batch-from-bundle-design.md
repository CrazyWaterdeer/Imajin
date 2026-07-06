# Resume a Batch Analysis from an Existing Result Bundle — Design

Status: design (revised after one Codex review; ready to plan)
Date: 2026-07-06

## Problem

Sessions are ephemeral (README). The in-app progress ledger (`AnalysisRun` via
`put_run`/`list_runs`, see [[agent-batch-state-management]]) lives only in the current
session. A user who analyses half a folder, closes the app, and returns later has **no
in-app record** of what is done or which parameters were used — the only durable record is
the **result bundle on disk**.

The agent cannot resume from it today: `import_recipe_from_bundle` recovers `recipe_params`
but explicitly **not file scope**; there is no read for "which files does this bundle already
cover?"; and `run_recipe_on_samples` always makes a **new** bundle.

User intent: open a folder, say "analyse the remaining files", and have the agent read the
existing bundle, diff done-vs-pending, and continue the rest **with the same settings**,
appending to the **same** bundle.

## Goal

Given a folder with a prior bundle, the agent can (1) recover the exact recipe [req 5],
(2) determine analysed-vs-pending by reading the bundle [req 1], (3) analyse the pending
files with those params, appending to the same bundle [req 4]. Primary flow is the manual
one-at-a-time stepping (1–2 GB files) reusing the rerun-guard + ledger; a secondary
`resume_from_bundle` on `run_recipe_on_samples` covers small-file batches.

## What the review changed (Codex, gpt-5.5 xhigh — 1 pass)

The naive "match canonical absolute paths, infer done-ness from CSV rows, mutate the old
bundle, trust the prompt" design was too brittle. The revised design rests on four things:

1. A **durable per-sample index** in the bundle (not CSV-row inference). [Codex #3, #8, #13]
2. **Anchor-relative file identity** (not absolute paths). [Codex #1] — decisive for this
   user, who just moved the same data between WSL (`/mnt/…`) and Windows (`D:\…`); absolute
   paths would never match across that.
3. A **read-only plan phase** separate from the **commit phase**, with correctness enforced
   in tools, not the prompt. [Codex #4, #5, #6, #11, #12]
4. **Deterministic slugs + exact guard-key seeding**, validated *before* 1–2 GB of work.
   [Codex #2, #7]

## Design

### 0. Durable per-sample index in `metadata.json` (foundation) — `result_bundles.py`

Written at analysis time by **both** the interactive (`analyze_target_cells`) and batch
paths, via one helper. Each entry:
```
{ key,                 # source_file path RELATIVE to the bundle anchor (primary identity)
  source_file_abs,     # canonical absolute (secondary / display)
  size, mtime,         # cheap conflict/staleness signal (not a hash)
  sample_slug,
  method, mode,        # so the resume seed reproduces interactive:{method}:{mode} exactly
  status,              # complete | failed
  table, outputs }     # what was written
```
This is authoritative done-tracking: a sample is **complete** because it has a consistent
index entry — independent of how many measurement rows it produced (a zero-object sample is
still complete). [Codex #3] Legacy bundles without the index fall back to best-effort
(CSV `source_file` ∪ `outputs` slugs), clearly flagged as `legacy_inferred`. [Codex #13]

### 1. File identity — anchor-relative key (`analysis/resume.py` helper)

Primary key = source_file relative to the bundle's stored input anchor; `source_file_abs`
canonical absolute as secondary; `size`+`mtime` to flag moved/changed files. Matching
`register_files`/ledger keys go through the same helper. [Codex #1, alt #3]

### 2. Deterministic sample slug (`result_bundles.py`)

Slug derives from the stable key (`slugify(rel_stem) + "_" + short_hash(key)`), so
re-appending the same file is idempotent (same slug) and different files never collide.
Collision is checked **before** analysis, not after. [Codex #7] The existing
`write_label_layer` "collision suspected" guard remains as a backstop.

### 3. Read-only plan — `plan_resume(directory | bundle_path)` (`tools/bundle_resume.py`)

Non-mutating. Finds candidate bundle(s) for the folder (match on stored anchor/scope, not
just proximity), checks schema/version compatibility, and returns a **resumable plan**:
```
{ bundle, compatible: bool, reason?,
  recipe: {...},
  analysed:  [key…],         # complete in the bundle
  pending:   [key…],         # on disk, not in the bundle
  missing:   [key…],         # in the bundle, not on disk now
  out_of_scope: [key…] }     # on disk, outside the bundle's original scope
```
If **>1** candidate bundle covers the folder, return them all with per-bundle diffs and
**require an explicit choice** (no "newest wins"). [Codex #4, #5, #6, #12, alt #4]

### 4. Commit / open — `open_result_bundle(bundle_path)` (`tools/bundle_resume.py`)

Enforced in the tool (not the prompt) [Codex #11]:
- Verify compatibility; if incompatible, **create a linked continuation bundle**
  (`continues: <prior>`) instead of mutating the old one. [Codex #10, alt #2]
- `promote_to_process_bundle(bundle)` so `analyze_target_cells` appends here.
- `import_recipe_from_bundle` → register the recipe; **make it the active recipe** and
  **disable per-file ROI auto-correction** by default (strict reproducibility; opt back in
  explicitly). [Codex #9]
- Seed the ledger: for each `analysed` entry, `put_run(status="complete",
  file_id=<canonical>, recipe_id=f"interactive:{method}:{mode}")` using the entry's own
  method/mode so it **exactly matches** the guard key `analyze_target_cells` computes
  (`workflows.py:149`), or the guard won't skip. [Codex #2]
- Return the plan's diff + `{recipe_name, analysed: N, bundle}`.

### 5. Per-sample append = commit-last (`result_bundles.py`)

For each newly-analysed sample: write outputs (labels/qc/table rows) first, then append the
**samples-index entry last** — that entry is the commit point. A crash mid-sample leaves no
index entry, so the sample reads as pending and is re-analysed (idempotent). Full atomic
table merge is a non-goal; commit-last gives the key safety property. [Codex #4]

### 6. One resume service — `analysis/resume.py`

Discovery, identity, compatibility, diff, and slug policy live in one module. Both the
manual `open_result_bundle` and the automated `run_recipe_on_samples(resume_from_bundle=…)`
call it — no duplicated policy. [Codex #14]

### 7. Automated batch — `run_recipe_on_samples(..., resume_from_bundle=None)`

When set: resolve via the service, `promote_to_process_bundle(existing)` instead of
`_create_parent_bundle`, skip `analysed` samples, append the rest.

### 8. Prompt (`agent/prompts.py`) — advisory only

Pipeline "resume batch" (triggers "이어서 분석", "남은 파일", "continue the rest", "resume"):
`plan_resume(dir)` → if 1 compatible bundle, `open_result_bundle` → `register_files` →
loop pending with `advance_to_file` + `analyze_target_cells` → summary. If >1 bundle or
incompatibility, surface the plan and ask. Correctness is enforced by the tools; the prompt
just sequences them. [Codex #11]

## Decisions / risks

- **Identity (Codex #1):** anchor-relative primary; abs secondary; size/mtime flags moves.
  No content hash (out of scope). Cross-platform (WSL↔Windows) works because rel-to-anchor
  is mount-agnostic.
- **Done = consistent index entry (Codex #3, #8):** not CSV rows; zero-object samples count
  as complete. Legacy bundles = best-effort, flagged.
- **Guard-key seeding (Codex #2):** seed uses each entry's method/mode →
  `interactive:{method}:{mode}` identical to `workflows.py:149`; validated by a test that
  a seeded file then returns `already_analysed=True`.
- **Slugs (Codex #7):** deterministic from stable key, checked pre-analysis.
- **Append safety (Codex #4):** commit-last per sample; incompatible schema → continuation
  bundle, never mutate-and-corrupt (Codex #10).
- **Reproducibility (Codex #9):** resume locks the recipe AND disables per-file
  `correct_roi` auto-correction unless the user re-enables it.
- **Enforcement not prompt (Codex #11):** tools promote the bundle, set the recipe, and
  seed the guard; the prompt is advisory.
- **4-way diff (Codex #12):** analysed / pending / missing-from-disk / out-of-scope, all
  reported.
- **No result change:** segmentation/measurement outputs unchanged; this adds a metadata
  index, identity/slug helpers, read+commit tools, a ledger seed, a batch param, prompt.

## Files

| File | Change |
| --- | --- |
| `result_bundles.py` | durable per-sample `samples` index (§0); deterministic slug (§2); commit-last append (§5) |
| `analysis/resume.py` (new) | identity, discovery, compatibility, diff, slug — the one service (§1, §6) |
| `tools/bundle_resume.py` (new) | `plan_resume` (§3), `open_result_bundle` (§4); `read_result_bundle` read |
| `tools/batch_runner.py` / `workflows.py` | `resume_from_bundle` via the service (§7); auto-correction lock on resume |
| `agent/prompts.py` | advisory "resume batch" pipeline (§8) |
| `tests/…` | index write+read (incl. zero-object + legacy); anchor-relative match (moved file); slug determinism/collision; plan 4-way diff + incompat→continuation; open seeds guard→skips done, new file appends same bundle; batch resume; prompt |

## Acceptance

- A bundle written after §0 carries a per-sample `samples` index with anchor-relative keys.
- `plan_resume(dir)` returns recipe + the 4-way diff; a zero-object completed sample is
  `analysed`, not `pending`; a bundle from a different mount still matches by rel-key.
- `>1` bundle → plan lists both, requires an explicit choice.
- `open_result_bundle`: recipe active, auto-correction off, ledger seeded →
  `analyze_target_cells` on a done file returns `already_analysed=True`; a new file appends
  to the **same** bundle (index grows, no new dir); incompatible schema → continuation bundle.
- `run_recipe_on_samples(resume_from_bundle=B)` skips done, appends new, via the shared service.
- No regression in bundle/recipe/batch/agent tests.

## Non-goals

- Bundle schema *removal*/rewrite (index is additive); content-hash identity; cross-machine
  path portability beyond anchor-relative; removing the LLM from the per-file loop
  (deterministic orchestration — later); a UI resume panel.
