# Stage C — registration → NBLAST connectome lookup (implementation plan, rev.1)

Date: 2026-06-15
Status: draft (follows spec `2026-06-15-connectome-registration-design.md` [3 Codex rounds];
this plan had 1 Codex round — changelog at bottom)

**Thin wrapper** around `navis` / `navis-flybrains` / `neuprint-python` (do NOT reimplement
NBLAST/transforms/fetch). Value = napari-native workflow + QC + review + export. New code in
`analysis/connectome_registration.py` + `tools/connectome.py`; reuse the connectome scaffold,
execution service, and v2.1 evidence idioms.

## Sequencing principle

Validate the **real product path** (incl. the QC safety gate) on synthetic data BEFORE the
dashboard. Reorder per Codex: foundations + minimal QC land in M1 so C-gate (go/no-go) tests
the actual path, not a synthetic NBLAST toy. Brain → hemibrain only.

## Milestone 0 — foundations (no/low network)

- **C0a navis-flybrains dependency.** Add `navis-flybrains` to the `connectome` extra (it is
  NOT there today — only navis + neuprint-python); env/version compatibility check. (Codex #5)
- **C0b CMTK bundle: layout + authoring + validator.** Name the exact accepted CMTK layout
  (affine + nonlinear, direction, `streamxform`/`reformatx` semantics); a `register_import`
  CLI/wizard that ingests a CMTK output dir → validates → runs a **point round-trip smoke
  test** → writes checksums → emits the app bundle. Not synthetic-affine only. (Codex #4)
- **C0c frame validation + preflight.** `validate_trace_frame(trace, stack_meta)` (affine/
  voxel/axis); run-preflight (assets/versions/neuPrint-token/disk/format/trace↔stack present).

## Milestone 1 — prove the real core path (go/no-go gate)

- **C1 bridge + warp (split, thin navis-flybrains wrapper).** (a) apply the bundle transform
  to SWC; (b) flybrains import + template resolution + **bridge-asset download/cache/checksum
  + offline behavior**; (c) `warp_trace_to(JRC2018F→JRCFIB2018F)` integration. Synthetic-
  transform unit tests; live assets behind a marker. (Codex #6)
- **C2 hemibrain dotprops + NBLAST wrapper.** Small dataset-versioned hemibrain set; build
  dotprops (8 nm→µm); `nblast_query()` with an **explicit fly score matrix** (current wrapper
  passes none). Cache **raw skeletons + resampled points + metadata**, not only version-
  sensitive pickled dotprops. (Codex #10)
- **C3 minimal registration-QC gate (moved up — the safety core).** Hard sanity checks
  (bbox/containment in template & connectome volume, scale, handedness, Jacobian extremes,
  plausible neuropil); pass/warn/fail; **NBLAST quarantined unless QC passes.** Present in M1
  so C4 exercises it. (Codex #1)
- **C4 synthetic validation = GO/NO-GO (anti-circular, pre-registered).** Hemibrain neurons →
  warp/degrade/**crop into LM-like queries** → run the **full real path (incl. C3 QC)** →
  recovery under fragment regimes. Anti-circularity (Codex #3): **bodyId self-exclusion,
  held-out body/type splits, injected registration/local-warp noise, shaft-only fragments,
  out-of-connectome containment.** Adversarial controls (wrong-region/shifted/scrambled/stale
  bridge) with **expected** outcomes. **Thresholds pre-registered before running** (Codex #2):
  fragment classes, top-k floors, bootstrap CIs, stop criteria. GATE: real-path recovery must
  beat the persistence baseline meaningfully AND clear the floors → else stop + report.

## Milestone 2 — the product (only if M1 passes)

- **C5 cache-at-scale + prefilter + execution + batch model.** Full dataset-versioned
  hemibrain dotprops cache (ROI-partitioned, refresh job); coarse ROI/bbox prefilter; run via
  the execution service (progress + **cooperative cancel only — no resumability today; persist
  artifacts explicitly, don't claim resume**, Codex #7); typed network-failure states;
  version-drift cache invalidation; **batch-ready data model now** (one transform → many
  traces, shared cache, per-query failure isolation, aggregate export — Codex #8).
- **C6 candidate-evidence result + priors.** Top-N + score gap + precomputed null/background
  percentile + uniqueness + mirror (homologs scored separately) + declared coverage + type/
  dataset-version/annotation provenance; user **priors as evidence columns**.
- **C7 napari workflow + review + export.** Overlay query+candidates; review state (accept/
  reject/inconclusive + notes) → session; export with **mandatory citations** (hemibrain
  CC-BY, JRC2018, navis/NBLAST) + provenance.
- **C8 economics + ownership.** Measure adoption economics (setup/run time, useful top-k rate,
  traces-to-amortize-one-registration); a prospective **blinded expert-review** protocol on
  real data (Codex #9); name a maintainer + refresh cadence + minimal smoke-test corpus.

## Preliminary go/no-go result (2026-06-15) — GO

`scripts/bench_nblast_typeid.py` ran NBLAST on the SAME 72 hemibrain neurons / 9 types and
fragmentation regimes as the persistence benchmark (apples-to-apples). NBLAST is far more
fragmentation-robust: ceiling top-1 0.639 / top-5 0.944; fragment ~50% 0.472 / **0.806**;
fragment ~30% 0.569 / **0.875** — vs persistence collapsing to 0.208/0.472 and 0.139/0.417
(≈chance). So the partial-trace failure that killed registration-free matching is handled by
NBLAST → building the registration pipeline (which unlocks NBLAST) is justified.

Caveats: (a) the 30%>50% non-monotonicity is single-random-subtree noise (n=72) — the
*conclusion* is robust, the exact curve is not; (b) this is **within-hemibrain (EM↔EM)** —
it proves NBLAST's fragment-robustness only. **Registration error + LM→EM cross-modality is
NOT yet tested** — that is the proper C4 (inject registration/local-warp noise + held-out
splits). This is therefore a *preliminary* GO; C4 remains the real gate before M2.

## Non-goals
Reimplementing NBLAST/transforms/fetch; MANC/VNC, FlyWire/FANC, in-app CMTK/ANTs automation,
batch UI; any match as a definitive identification.

## Risks
M1 may fail the gate → honest stop. GPL-v3 (navis) vs Imajin license → resolve before ship.
flybrains assets + neuPrint network external → offline/failure paths required. Execution
service has no resumability → don't over-promise.

## Changelog — plan rev.1 (accepted Codex plan-review)
#1 minimal QC gate moved into M1 (C3) so C4 tests the real path; #2 pre-registered go/no-go
thresholds; #3 anti-circular synthetic design (self-exclusion, held-out splits, injected
registration noise, shaft-only, out-of-volume); #4 CMTK layout + authoring tool early (C0b);
#5 add navis-flybrains dep (C0a); #6 split bridge/warp (C1); #7 split C5 + scope resumability
honestly; #8 batch-ready data model in C5; #9 economics + expert-review (C8); #10 explicit fly
score matrix + cache raw+metadata (C2).
