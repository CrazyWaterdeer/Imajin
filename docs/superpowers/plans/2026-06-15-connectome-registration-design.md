# Stage C — confocal→template registration → NBLAST connectome lookup (design spec)

Date: 2026-06-15
Status: design (Codex iteration — round 1 folded in; rounds 2–3 pending, each prompted for NEW issues)

The persistence benchmark (2026-06-14) showed registration-free matching collapses on
partial confocal traces. The valid path registers a trace into a Drosophila template and
runs NBLAST against connectome neurons in the same space. Outcome framing (round 1): this
produces **QC-gated candidate matches requiring anatomical review**, not definitive
identifications.

## Goal

Given a confocal trace (SWC) from an nc82-stained sample, return ranked, QC-gated candidate
connectome matches with uncertainty + provenance — by: validate the trace's coordinate
frame, register the sample into a template, bridge to the connectome space, NBLAST the
warped trace against fetched connectome dotprops, gate on registration quality. Region-aware
(brain / VNC are separate template+connectome families).

## Grounded facts (prior investigation + live checks)

- `navis-flybrains`: templates `JRC2018F` (brain), `JRCVNC2018F` (VNC), `JRCFIB2018F`
  (="hemibrain"); bridging incl. `FANC↔JRCVNC2018F`; `navis.xform_brain` auto-bridges.
- neuPrint (token verified): `hemibrain:v1.2.1`, `manc:v1.2.3`; `fetch_skeleton` → SWC-shaped
  node tables in **8 nm voxels**. FlyWire/FAFB + FANC are CAVE-based (separate token).
- nc82 (anti-Brp) is the standard fly registration reference; JRC2018 built with ANTs,
  individuals aligned with CMTK. NBLAST tolerates partial arbors **only when the query has
  discriminative morphology** (round 1 #5) — tracts/shafts/fragments give high but
  non-specific scores.

## Coordinate-frame contract (round 1 #1/#2 — load-bearing)

- The SWC trace MUST be in the **exact** coordinate frame of the nc82 stack used for
  registration (same crop, axis order, voxel size, no derived/resampled stack). Mandatory
  pre-flight: validate the trace's affine/voxel/axis metadata against the registered stack;
  refuse if absent or inconsistent (silent mismatch invalidates the warp).
- **Transform direction is explicit** per accepted format (moving→fixed vs fixed→moving; a
  point warp may need the inverse). A point round-trip smoke test confirms direction before
  use.

## External-transform contract (round 1 #3 — strict, not "any CMTK/ANTs file")

MVP consumes an externally produced transform but only via a strict bundle: source-stack
metadata (voxel size, axes, dims), target template **name+version**, transform **direction**
+ units, transform provenance (engine+params), and a **point-transform smoke test** that
must pass. One format first — **CMTK** (matches flybrains tooling). In-app CMTK/ANTs
automation is deferred.

## Pipeline (3 steps; cost very asymmetric)

1. **Sample→template registration (the crux, external for MVP).** nc82 → `JRC2018F` /
   `JRCVNC2018F`. App consumes the validated transform bundle.
2. **Template→connectome bridging.** `navis.xform_brain` (JRC2018F→JRCFIB2018F=hemibrain;
   JRCVNC2018F→MANC; →FAFB/FANC later). Each bridge adds deformation error (#4) → candidate
   retrieval, not identity.
3. **Warp trace + NBLAST.** Apply 1–2 to SWC coords → dotprops (navis, fly score matrix) →
   NBLAST vs fetched region-appropriate connectome dotprops. **Mirror** the query too and
   score bilateral homologs separately (#10).

## Registration trust gate (round 1 #11–#15 — local, not global)

NBLAST is **quarantined** unless registration QC passes; "a transform was provided" is not
evidence of quality (#15). QC is **local to the traced arbor**, not whole-brain overlap (#11):
- hard sanity checks: warped-trace bounding box inside the template; expected neuropil
  containment; gross scale; handedness; transform **Jacobian** extremes; does the warped
  trace land in plausible neuropil (#14).
- local nc82↔template overlap **near the arbor**; a positive-control known-neuron round-trip
  (mechanics only — necessary not sufficient, #12).
- output **pass / warn / fail** with visible metrics; thresholds are **not universal** —
  calibrate per region/modality/objective/voxel/prep; until then surface raw metrics (#13).

## Containment & routing nuance (round 1 #8)

`region` → (template, connectome) is necessary but coarse: hemibrain is a **partial** brain,
MANC is adult VNC+neck; the traced neuron may fall **outside** the connectome volume or be
only partly represented. Check warped-trace containment in the connectome volume and warn on
out-of-volume / low-overlap.

## Result presentation (round 1 #16–#19 — uncertainty, not a lone hit)

Return, per query: top-N candidates with `nblast_score`, **score gap** to runner-up, a
**null/background percentile** (is this score better than random?), uniqueness flag,
**mirror status**, query **coverage** (axon/dendrite/fragment/uncertain-polarity — declared),
connectome **type + dataset version + annotation provenance** (#17), and (where feasible)
neuropil-compartment overlap. Frame as "candidates requiring review"; never a single
authoritative `bodyId`.

## Reproducibility metadata (round 1 #20)

Cache + every result record store: connectome dataset version, neuPrint server, navis +
navis-flybrains versions, transform IDs/checksums, NBLAST score matrix, dotprops params,
query preprocessing (resample/units), and the QC verdict + metrics.

## Operational reality & asset supply chain (round 2 #1–#4)

- **Registration runbook + owner.** External-transform MVP is unusable without a documented
  path: which CMTK/ANTs install (container), the parameter set for nc82→JRC2018F/JRCVNC2018F,
  hardware/runtime budget, logs, and who runs it. Ship a runbook, not just a contract.
- **CMTK-in-practice.** Name the exact accepted layout (affine chain + nonlinear warp,
  direction, `reformatx`/`streamxform` semantics, template conventions) — "CMTK" alone is
  ambiguous.
- **Bundle authoring tool.** A CLI/wizard that ingests a CMTK output dir, extracts metadata,
  runs the point smoke test, writes checksums, and emits the app-consumable bundle. Manual
  JSON assembly will fail in practice.
- **flybrains bridge supply chain.** `navis.xform_brain` needs downloaded bridging
  registrations: define source, cache location, checksums, license/redistribution, **offline
  behavior**, and handling when navis-flybrains updates/renames templates.

## Scale & performance plan (round 2 #5–#7)

- **Precomputed, dataset-versioned connectome dotprops caches** (hemibrain/MANC), with body
  filters + ROI partitions + refresh jobs — not on-demand fetch+dotprops per query.
- **NBLAST at scale:** coarse prefilter (ROI / bounding-box / neuropil containment) before
  expensive scoring; chunked, memory-bounded, parallel, with progress + cancel.
- **Null/background percentile** needs **precomputed** distributions stratified by region,
  query length, arbor coverage, and fragment/polarity class.

## Validation plan (round 2 #8–#10) — reuses the persistence-benchmark pattern

- **Synthetic (no curated LM→EM pairs needed):** take connectome neurons, warp/degrade/crop/
  downsample them into LM-like queries, run the FULL pipeline, measure source/type recovery
  under controlled fragment regimes (extends `scripts/bench_persistence_typeid.py`).
- **Adversarial negative controls:** wrong-region transform, spatially shifted trace,
  out-of-volume trace, low-information shaft, scrambled dotprops, stale bridge version —
  each must fail or lose rank predictably; regression floors like the v2.1 F1 harness.
- **Prospective expert review:** blinded anatomical review of top-N on real confocal data;
  report inter-reviewer agreement + top-k enrichment / plausible-type recovery — never
  "matched bodyId" as truth.

## Execution & batch architecture (round 2 #11–#12)

- Registration/NBLAST run via the existing **execution service** (worker thread): queued
  jobs, artifact IDs, logs, progress, cancellation, resumability, results attached to the
  session — never on the UI thread.
- **Batch-ready data model now (even though batch UI is deferred):** one sample transform
  applies to many traces → model sample-level transform reuse, many-query execution, shared
  caches, per-query failure isolation, aggregate export.

## Failure, versioning & preflight (round 2 #13–#15)

- **Network/auth failure semantics:** retries, resumable fetches, rate-limit handling,
  corrupt-cache detection; distinguish **"no candidate"** from **"lookup failed"** (typed
  states, never silent empty).
- **Version-drift cache invalidation:** before reusing any cache/result, check runtime
  navis / navis-flybrains / template / bridge / connectome versions against stored metadata;
  mixed-version → warn or refuse, never silently score.
- **Run preflight (data prerequisites):** nc82 image availability + supported format,
  trace↔stack association, neuPrint token scope, local cache disk budget, and required
  template/bridge assets present — checked before a run starts.

## Auth / secrets
- neuPrint token (have it): `~/.config/neuprint/token`, env-injected, never logged.
- CAVE token (FlyWire/FANC, deferred): `~/.cloudvolume/secrets/cave-secret.json`.

## Strategy, product framing & licensing (round 3)

- **Thin wrapper, not a reimplementation (#1).** Do NOT reimplement NBLAST, fly transforms,
  or neuPrint/CAVE access — `navis` / `navis-flybrains` / `neuprint-python` own those. The
  app's defensible value is the **napari-native workflow**: trace↔stack association, visual
  registration QC, candidate overlay, accept/reject/inconclusive review state, and
  methods-ready report export. If that workflow isn't the value, users are better served by
  navis/natverse/fafbseg notebooks.
- **The product is a candidate-evidence dashboard (#8/#9), not an "NBLAST lookup".** Accept
  user priors as **evidence columns** (side, neuropil, driver line, soma location,
  neurotransmitter, polarity, expected type family) alongside the NBLAST score; let the user
  review, annotate, save bodyIds/types/provenance into the session, and export
  citations+figures.
- **Narrowed MVP (#2/#10):** ONE region + ONE connectome + ONE upstream stack —
  **externally-registered brain trace → hemibrain → candidate table + overlay + exportable
  review report.** Defer MANC, FlyWire/FANC, batch, null-calibration polish, generalized
  asset handling.
- **Go/no-go economics (#3/#4):** worth it for many-traces-per-sample / unfamiliar classes /
  low connectome fluency / reproducible triage; an expert may beat it for one high-quality
  trace with known context. Define adoption thresholds up front (median setup+run time,
  top-k expert-useful rate, traces needed to amortize one registration, how often output
  changes the biological conclusion) — abandon if unmet, like the persistence benchmark's
  decision rule.
- **Licensing/data-use matrix (#5/#6, partly verified):** hemibrain data is **CC-BY** →
  derived dotprops caches are redistributable *with attribution + citation* (surface
  citations in every export). `navis`/`navis-flybrains` are **GPL-v3** → an Imajin
  distribution-license compatibility decision (favours the thin-wrapper architecture);
  confirm before shipping. JRC2018 templates: cite; don't bundle — let navis-flybrains
  download. FlyWire/FANC terms (likely more restrictive, e.g. NC) are a **mandatory check at
  that deferred phase**, especially for redistributing authenticated-data-derived caches.
- **Ownership/sustainability (#7):** name a maintainer, a refresh cadence for the
  navis/flybrains/neuPrint/template/napari support matrix, a deprecation policy, and a
  **minimal smoke-test corpus** (a couple of known traces → expected candidate enrichment)
  run on upgrades — else it becomes frozen researchware at the first env break.

## Scope / staging
- **MVP (one path, brutally narrow):** validate frame → consume external CMTK transform
  bundle → bridge → registration-QC gate → NBLAST vs **hemibrain only** → candidate-evidence
  table + napari overlay + review state + methods/citation export.
- **Deferred:** MANC (VNC), FlyWire/FANC (CAVE), in-app CMTK/ANTs automation, batch,
  null-calibration polish.

## Honest risks
- Registration is the crux; garbage-in→garbage-out; nc82 aligns neuropil *texture*, not
  individual neuron geometry (#7) → local deviation even after good global alignment.
- LM→EM cross-modality + multi-bridge deformation → candidate ranking, not identity.
- Sex/age/genotype/prep/expansion bias alignment; templates aren't universal ground truth
  (#9). Mirror flips create plausible wrong matches (#10).

## Open decisions (to refine with Codex rounds 2–3)
- The exact CMTK transform-bundle schema + how the point smoke test is defined.
- Where local-QC + Jacobian thresholds come from (calibration data?).
- NBLAST null/background model for the percentile.
