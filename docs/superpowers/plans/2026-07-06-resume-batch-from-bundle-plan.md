# Resume a Batch Analysis from an Existing Result Bundle — Plan

Spec: `docs/superpowers/specs/2026-07-06-resume-batch-from-bundle-design.md`
(revised after one Codex review). Date: 2026-07-06.

Phased into small, independently-testable commits. **P0 is a prerequisite** (durable
per-sample index + identity/slug) and changes no behaviour. P2 delivers the user's primary
value (manual one-at-a-time resume). Each phase runs the suite with `IMAJIN_RESULTS_DIR`
set (see [[imajin-test-env-results-dir]]) and merges via [[merge-then-push-workflow]].

## P0 — Foundation: durable sample index + identity/slug (no behaviour change)

- **C0.1** `analysis/resume.py`: `bundle_anchor(bundle)`, `rel_key(source_path, anchor)`
  (anchor-relative, mount-agnostic), `file_signature(path) -> {size, mtime}`,
  `sample_slug_for(key)` (`slugify(rel_stem)+"_"+short_hash(key)`, deterministic). Pure,
  unit-tested — incl. WSL `/mnt/d/x` vs Windows `D:\x` producing the **same** rel_key under
  a shared anchor. [Codex #1, #7]
- **C0.2** `result_bundles.py`: on every sample write (interactive + batch), append a
  `samples` index entry to `metadata.json` **last** (commit point): `{key, source_file_abs,
  size, mtime, sample_slug, method, mode, status, table, outputs}`. Add
  `read_sample_index(bundle)` that returns the index, or a `legacy_inferred` best-effort
  set (CSV `source_file` ∪ `outputs` slugs) when absent. Tests: index round-trips; a
  **zero-object** complete sample is present; legacy bundle infers + flags. [Codex #3, #8, #13]

## P1 — Read-only plan (the "agent reads the bundle" core)

- **C1.1** `analysis/resume.py`: `find_bundles_for(directory)` (match on stored anchor/scope,
  not proximity); `compatibility(bundle)` (schema_version/imajin_version); `diff(bundle,
  disk_files) -> {analysed, pending, missing, out_of_scope}` on rel_keys. Tests: moved file
  still matches; >1 candidate returned; incompatible flagged; 4-way diff correct. [Codex #5, #6, #10, #12]
- **C1.2** `tools/bundle_resume.py`: `plan_resume(directory | bundle_path)` (read-only) and
  `read_result_bundle(bundle_path)` tools returning recipe + diff + compatibility; when >1
  compatible bundle, return all and require an explicit choice (no auto-pick). Tests + registry-golden. [Codex #4, alt #4]

## P2 — Commit / open (manual one-at-a-time resume works end-to-end)

- **C2.1** `tools/bundle_resume.py`: `open_result_bundle(bundle_path)` —
  compatibility check (incompatible → linked **continuation bundle**, `continues:<prior>`);
  `promote_to_process_bundle`; `import_recipe_from_bundle` + set active recipe + **disable
  per-file ROI auto-correction**; seed the ledger via `put_run(status="complete",
  file_id=<canonical>, recipe_id="interactive:{method}:{mode}")` from each entry's own
  method/mode. Tests: after open, `analyze_target_cells` on a done file →
  `already_analysed=True`; a new file appends to the **same** bundle (index grows, no new
  dir); slug deterministic; incompatible → continuation. [Codex #2, #9, #10, #11]

## P3 — Automated batch resume (secondary path)

- **C3.1** `tools/batch_runner.py` + `workflows.py`:
  `run_recipe_on_samples(resume_from_bundle=None)` resolves via the service, promotes the
  existing bundle, skips `analysed`, appends the rest; auto-correction locked on resume.
  Tests: skips done, appends new, one shared service (no duplicated policy). [Codex #14]

## P4 — Prompt pipeline (advisory)

- **C4.1** `agent/prompts.py`: "resume batch" pipeline — `plan_resume(dir)` → (1 compatible)
  `open_result_bundle` → `register_files` → pending loop (`advance_to_file` +
  `analyze_target_cells`) → summary; (>1 or incompatible) surface plan and ask. Correctness
  is enforced by the tools; prompt only sequences. Test: `build_system_prompt` contains the
  pipeline; no format break. [Codex #11]

## Sequencing & risk

- Land **P0** first (additive metadata, zero behaviour change) — every later phase depends
  on the index. Then **P1** (pure read, safe). **P2** is the first user-visible resume and
  the riskiest (writes/guard) — carries the most tests. **P3/P4** are additive.
- Ship each phase as its own merge; the feature is usable after P2 (manual resume) even if
  P3/P4 slip.

## Open items to confirm during P1/P2 (from review, low-stakes)

- Continuation-bundle vs in-place append as the **default** (spec favours in-place when
  compatible, continuation only on incompatibility) — revisit if append-atomicity bites.
- Whether `out_of_scope` disk files should be offered for opt-in inclusion or just reported.
