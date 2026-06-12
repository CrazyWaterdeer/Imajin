# ROI scorer v2.1 — implementation plan (rev.1, post-Codex)

Date: 2026-06-12
Status: draft (follows spec `2026-06-12-roi-scorer-v2-design.md`; revised after one Codex
review of this plan — changelog at bottom)

Builds the evidence-based, context-aware ROI confidence from the v2.1 spec. v1's structural
`score_roi_quality` stays as **Layer 1**; everything new lives in a new leaf module
`analysis/roi_quality.py` (imports `score_roi_quality` from `segmentation_auto3d`; imported
by `target_pipeline` + tools; auto3d never imports it → no cycle). v1 fields are preserved
and characterized before anything is redefined.

Commit order (revised per Codex): **A0 → A → B → F0 → C → D → E1 → E2/E3 → F1.** Calibration
(F0) lands *before* the layers that need calibrated constants (C/D); v1 back-compat (A0)
lands first.

## A0 — characterize v1 before redefining anything (back-compat guard)

- Enumerate current `roi_score` / `roi_confidence` producers (`segment_target_objects`,
  `auto_segment_target`, `segment_3d_cells_auto`) and consumers (runner vision gate,
  tests, layer metadata, saved result dicts).
- Add **characterization tests** pinning v1 field presence/values + the gate's
  `roi_confidence ∈ {low,medium}` trigger, so the v2.1 redefinition can't silently regress
  callers that assume `roi_confidence == confidence_from_score(roi_score)`.
- Decide field strategy: `roi_confidence` becomes v2.1 but **superset-compatible**
  (still high/medium/low, gate still works); `roi_score` (v1 structural) **kept**;
  `distribution_flag` / `confidence_drivers` / `correction_materiality` are **additive**.

## A — Layer 0 routing (hard prerequisite, nothing left "open")

- **A1** `object_class(meta) -> "blob"|"domain"|"neuron"|"unclassified"` from
  `object_unit`/`segmentation_method` (`target_objects`→blob, `expression_domain`→domain,
  trace outputs→neuron; unknown→unclassified).
- **A2** `effective_object_count(labels, spacing) -> (n_eff, distributed: bool)` with a
  **fixed, concrete spatial-spread rule** (e.g. ≥K non-border objects whose centroids span
  ≥ a set fraction of the image bbox) — not left open, since A gates everything.
- **A3** `route(object_class, n_eff, distributed) -> set` — blob+enough → structural +
  distribution + vision; domain/sparse/unclassified → structural + vision; **neuron →
  structural + vision (NO distribution, and no morphology-consistency layer in v2.1 —
  deferred, see non-goals).** Tests per branch.

## B — size extraction (reuse physical units)

- **B1** `object_sizes_physical(labels, spacing) -> (sizes_um, border_mask, n_eff_sized)` —
  regionprops area (2D µm²) or volume (3D µm³), anisotropy-corrected; **exclude
  border-touching objects** and return the *post-exclusion* usable count so C/A see the
  same effective N. Never mix 2D/3D. Tests: 2D area, 3D anisotropic volume, border exclusion.

## F0 — calibration harness + constants module (before C/D)

- **F0** `tests/test_roi_quality.py` synthetic-perturbation generator (split / merge /
  erode / dilate / background-flood of good multi-object masks) + a calibration-table
  scaffold, and a **constants module** in `roi_quality` holding the tunable cutoffs
  (`min_effective_n`, log-size thresholds, multimodality choice). C/D import these
  constants; F1 later asserts sens/spec floors against them. (Codex #1/#10: calibration
  must precede and feed back, not trail.)

## C — Layer 2 distribution anomaly flag (weak, medium-only)

- **C1** `distribution_flag(sizes_um, *, n_eff) -> {flag, reason, metric, abstained}` on
  **log-size**, robust stats, using the F0 constants. Takes the **effective N** (post-border,
  post-spread) — not `len(sizes)` (Codex #8). Secondary mode near 2× →
  `possible_undersegmentation`; heavy small tail → `possible_oversegmentation`; **abstain**
  below `min_effective_n`. **Never a score delta, never `low`, never `high`.** Tests:
  bimodal→flag, tight→no flag, small-tail→flag, low-N→abstain, **broad lognormal→no false
  flag**, and an explicit **"flag never yields low/high"** test (Codex gap).

## D — confidence mapping v2.1 + correction materiality

- **D0 (prereq)** preserve raw + corrected per-object metrics through the auto-correct loop
  (the loop already keeps `history[0]`; add the per-object size vectors) so D2 has both
  snapshots before mutation (Codex #6).
- **D1** `roi_confidence_v2(structural_score, structural_metrics, *, route, n_eff,
  object_class, dist_flag, correction_gap) -> (confidence, drivers)` — **signature now takes
  the Layer-0 route + n_eff** (Codex #2). `high` only on strong structural evidence AND no
  dist flag AND enough effective N; domains/sparse/arbor **capped at medium**; `low` only on
  gross structural failure; else medium. `drivers` = which layer/metric decided it.
- **D2** `correction_materiality(raw_qc, corrected_qc) -> gap_flag`. Tests: high-only-on-
  structural, caps for domain/sparse/neuron, **no distribution scoring for those classes**,
  drivers legibility/stability.

## E1 — wire into the pipeline + tools (split, back-compat tested)

- **E1a** `target_pipeline` computes v2.1 (route → sizes → dist_flag → confidence_v2) and
  returns it on `TargetSegmentation`.
- **E1b** `segment_target_objects` / `auto_segment_target` surface `roi_confidence` (v2.1),
  `distribution_flag`, `confidence_drivers`, `correction_materiality` in result + layer
  metadata; keep `roi_score`. **Back-compat tests** for old fields + serialized metadata
  (Codex #5). **Ships together with E2** (below) so the richer `medium` can't spam overlays.
- **E1c** tool-surface tests; expression domains/neurons verified to never get distribution.

## E2 — vision gate escalation budget (lands with E1b, not after)

- Cap repeated overlay attachment for the same sample/recipe unless metrics worsen, so v2.1's
  more-frequent `medium` doesn't spam the agent (Codex #7). Until this lands, E1b stays
  behind a flag.

## E3 — prompt

- `distribution_flag` is a *possible-segmentation-issue* signal ("worth a look"), **not** a
  phenotype verdict and **not** a correction trigger; distinguish it from structural `low`.

## F1 — end-to-end validation + regression floors

- Run full v2.1 over the F0 perturbation set; assert the **separate** signals —
  `roi_confidence`, `distribution_flag`, `vision_escalated`, `drivers` — **not** a conflated
  label (Codex #9). Compute sensitivity/specificity; **regression tests fail if sens/spec
  drops below floors** in the constants module (Codex #10). Negative test: broad biological
  lognormal variation does not trigger correction via the loop or prompt.

## Non-goals / risks

- **Non-goals:** morphology-consistency layer for neurons (deferred — v2.1 routes neurons to
  structural+vision only); absolute-size priors as default (opt-in only); changing auto3d
  ranking or v1 `score_roi_quality`/`confidence_from_score`.
- **Risks:** Layer-0 mis-routing poisons everything → `unclassified` stays conservative
  (no distribution). Distribution layer fails safe (abstain), never asserts. Redefining
  `roi_confidence` → A0 characterization + back-compat tests are the guard.

## Changelog — plan rev.1 (accepted Codex plan-review findings)

Accepted: #1 split F into F0 (pre-C/D calibration) + F1; #2 D1 takes route + n_eff; #3 drop
the unbacked neuron-morphology branch → structural+vision (deferred); #4 split E1 into
E1a/E1b/E1c; #5 add A0 v1 characterization + back-compat tests; #6 D0 preserves raw/corrected
per-object metrics; #7 E2 escalation budget ships with E1b (flag otherwise); #8 C1 takes
effective N; #9 validation asserts separate signals; #10 constants module + sens/spec
regression floors so F feeds back. Plus the gap-tests (flag-never-low/high,
no-distribution-for-domain/neuron, serialized back-compat, drivers legibility,
lognormal-no-correction) and fixing A2's spatial-spread rule rather than leaving it open.
