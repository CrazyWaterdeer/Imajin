# ROI quality scorer v2.1 — design spec

Date: 2026-06-12
Status: design (revised after one Codex review; ready to plan)

Follows the shipped ROI-judgment ladder (`2026-06-10`/`2026-06-12`). v1's
`score_roi_quality` + `confidence_from_score` work, but the confidence criterion is too
coarse. **v2.1** revises v2 after a single Codex (gpt-5.5) review; the changelog at the
bottom records which findings were accepted.

## Why v2 — what v1 actually judges, and its flaws

v1 scores (100 minus penalties) on **intensity / coverage / topology only**: coverage
(`mask_fraction`), signal containment (`top_bright_outside_fraction`), contrast
(`separation_snr`), and crude shape-tail checks (`tiny_object_fraction`,
`largest_to_median_ratio`).

Flaws: (1) **context-blind** — same function for cells, nuclei, lipid droplets, puncta, a
domain, or a traced neuron; (2) **"absence of penalty = high"** — confidence from *not
tripping* a penalty, not positive evidence, so it stays silent on errors it can't see;
(3) **no distribution model** — only extreme-tail checks.

## Principles (v2.1)

1. **Internal consistency over fixed per-type priors.** The Drosophila compartment survey
   proved fixed priors fragile: 2–3 orders of magnitude range, same type varies hugely by
   tissue, and **size change is often the experimental result** (lipid droplets under
   diet/genotype). Penalizing size *deviation* would corrupt the measured signal.
2. **Consistency is necessary, not sufficient (Codex #4).** A uniformly-wrong mask (every
   pair merged, every nucleus eroded) yields a *tight, unimodal* distribution. So internal
   consistency **cannot be positive evidence of correctness** — only the *absence of
   heterogeneity*. It may therefore only **withhold/lower** confidence, never grant it.
3. **Evidence-based `high` comes from structure, not distribution.** `high` requires
   positive **structural** evidence (good inside/outside separation + signal containment),
   optionally reinforced by an opt-in absolute-size check or by vision — never from
   distribution coherence alone.
4. **Distribution = weak anomaly flag, biology-safe.** Multimodality / high spread can be
   real biology (cell cycle, polyploidy, regional variation, the phenotype itself), so the
   distribution layer **never penalizes a score and never yields `low`** — at most it
   routes to `medium` ("look at this"). Never "mistake biology for error" (Codex #1).
5. **Escalate on weak evidence** (hard/dense/low-contrast, too few objects) → vision/user.

## Layer 0 — routing is a hard prerequisite (Codex #6)

The whole scorer depends on knowing the object class; mis-routing produces misleading QC,
so this is a **required precondition**, not an open item. Route on:
- **object class** — blob-like ROIs (cells/nuclei/droplets/puncta) vs single **domain** vs
  **neuron/arbor**. Source, in priority order: explicit `object_unit`/segmentation-method
  metadata already on the labels layer → workflow/tool that produced it → agent/user
  statement. If unknown → treat as "unclassified" and **do not** apply the distribution
  layer; fall back to structural + vision.
- **effective object count** — not just N labels: N from one small crop are not N
  independent examples (Codex gap). Require enough *and* spatially distributed objects
  before the distribution layer is even eligible.

Routing outcome:
- blob-like + enough effective objects → L1 + L2 + (L3 if weak).
- domain (1 object) / sparse / unclassified → L1 + L3 only.
- neuron/arbor → morphology metrics (Sholl, branch stats — already built) + L3; **no
  size-distribution.**

## Layer 1 — structural (keep v1, the source of positive evidence)

Coverage / containment / contrast / topology. Cheap, universal; catches gross failures
(flooded background, missed signal, no contrast) → these give `low`. Strong separation +
containment is the **only** thing that earns `high`.

## Layer 2 — distributional anomaly flag (new, weak, medium-only)

Computed from per-object **size** (defined below), **only** when Layer 0 admits it:
- operate on **log-size** (sizes are skewed/lognormal — raw CV is wrong; Codex #3); use
  robust stats (median/MAD on log).
- **multimodality** (e.g. a secondary mode near 2×) → *possible* under-segmentation;
  **small-size tail** → *possible* over-segmentation/fragments.
- **statistical honesty (Codex #2):** at low effective N these tests are low-power and
  bandwidth-sensitive. The layer **abstains** below a reliability threshold (report
  "insufficient N") rather than asserting coherence. A *missed* second peak must never be
  read as positive evidence.
- output is a **flag with the offending metric**, routing to `medium` and surfacing the
  overlay — never a score delta, never `low`, never `high`.

## Layer 3 — visual

When L1 is not strongly positive, or L2 raised a flag, or N is insufficient, or raw≠
corrected (below) → attach the QC overlay (Phase A gate, already wired) / open
`review_target_roi`. **Escalation budget (Codex gap):** cap how often `medium` forces a
look so common valid phenotypes don't trigger constant review (e.g. once per
sample/recipe unless metrics worsen).

## Confidence mapping (v2.1)

- **high** — strong L1 structural evidence (separation ≫ noise, signal contained) AND L2
  raised no flag (or was N/A) AND enough effective objects; OR an opt-in absolute-size
  check is satisfied. Domains/sparse/arbors **cannot be `high`** on L1 alone → `medium`
  (closes the v2 escape hatch, Codex #5).
- **low** — gross L1 failure only (zero objects, region-level merge, over-wide, near-empty).
- **medium → show image** — L1 ok-but-not-strong, OR an L2 anomaly flag, OR insufficient
  effective N, OR a material raw-vs-corrected gap, OR a hard sample.

## Size definition (Codex #7 — make it exact)

Reuse the app's existing physical-unit measurement (`min_size_from_physical`, voxel
spacing) — don't reinvent:
- **3D labels** → volume in µm³ (anisotropic voxel-corrected); **2D** → area in µm²; never
  mix the two in one distribution.
- regionprops label area/volume (not raw connected-component pixels), physical units.
- **border objects** flagged and excluded from the distribution (truncation inflates the
  small-size tail and fakes multimodality — Codex gap).
- projection vs 3D is explicit; an opt-in absolute expectation in µm³ must not silently
  apply to a 2D area / projection (units provenance — Codex gap).

## Raw vs corrected masks (Codex #10)

The shipped auto-correct loop changes masks before scoring, so distinguish:
- **raw-mask QC** (as first segmented), **corrected-mask QC** (after the loop), and a
  **"correction changed measurement materially"** flag (e.g. per-object size/count shifted
  beyond a tolerance). Score the as-measured mask, but surface when correction moved the
  result — otherwise L2 may validate a correction artifact.

## Validation design (Codex #9 — replace "real-data tuning")

- **labeled masks** stratified by compartment / tissue / imaging protocol.
- **synthetic perturbations** of good masks: split, merge, erode, dilate, background-flood
  — we already have a synthetic-perturbation harness (`tests/test_target_pipeline.py`) to
  extend.
- report **sensitivity/specificity** of each confidence outcome (`show-image` / `low` /
  `high`) against the perturbations — not threshold anecdotes. Tune log-size cutoffs and
  the effective-N reliability threshold against this, not by eye.

## Reporting / legibility (Codex gap)

Every confidence verdict must expose **which layer drove it** + the metric/flag, so users
can tell a *real phenotype* from a *segmentation warning*. Carry this in the result dict
and the QC record.

## Reuse (cheap to build on)

- Per-object physical sizes already from `measure` / regionprops + `min_size_from_physical`.
- Phase A vision gate already wired — Layer 3 plugs in.
- Neuron morphology metrics (Sholl, branch) already built — Layer 0 arbor route.
- `score_roi_quality` / `confidence_from_score` are the integration points.
- Synthetic-perturbation tests already exist — extend for validation.
- `morphology_reference` CSV pattern fits the opt-in absolute-size library.

## Open decisions (for the plan)

- Exact log-size reliability threshold (min effective N) and multimodality test (dip vs
  KDE-mode vs 2-component) — **fixed empirically via the validation harness**, not guessed.
- Effective-object-count definition (count + spatial spread).
- Whether L2's flag and L1's confidence are one field or two surfaced signals (lean: two —
  `roi_confidence` + a separate `distribution_flag`, for legibility).

## Changelog — v2 → v2.1 (accepted Codex findings)

Accepted: #1 biology≠error (→ L2 never penalizes/`low`), #2 weak stats at low N (→ abstain
+ reliability threshold), #3 use log-size/robust stats, #4 consistency≠correctness (→ L2
can't grant `high`), #5 close the `high` escape hatch, #6 Layer 0 elevated to prerequisite,
#7 exact size definition, #8 resolution guard tied to PSF/voxel (kept opt-in), #9 validation
design via synthetic perturbations + sens/spec, #10 raw vs corrected, plus gaps (border
objects, effective N, layer-attribution reporting, escalation budget, absolute-size units
provenance). Noted as already-supported (specify, don't build): physical-unit/anisotropy
sizing and the synthetic-perturbation harness.
