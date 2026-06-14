# Connectome live lookup (registration-free) — design spec

Date: 2026-06-14
Status: design (pre-Codex-review)

Completes the neural-tracing → connectome thread. Tier-1 tracing (skeletonize,
morphometrics, persistence, local labelled-library matching) already ships. This wires the
**live neuPrint fetch + registration-free matching** so a confocal trace can be compared
to real connectome neurons with **just the token** — NBLAST-grade comparison stays deferred
behind template registration (Stage C).

## Grounded facts (verified live, 2026-06-14)

- Token authenticates (neuPrint 1.7.10). Datasets present: `hemibrain:v1.2.1` (brain),
  `manc:v1.2.3` (VNC), `male-cns:v1.0` (whole CNS), `optic-lobe`, etc.
- `fetch_neurons(NeuronCriteria(...))` → DataFrame with `bodyId, instance, type, pre, post,
  upstream, downstream, size, status, cropped, …` (type labels + connectivity).
- `fetch_skeleton(bodyId, format='pandas')` → `rowId, x, y, z, radius, link` = **SWC
  structure** → feeds `persistence_features_from_swc` directly.
- **Units:** hemibrain/MANC skeleton coords are in **8 nm voxels**. Persistence features are
  path-length-based (scale-sensitive), so fetched skeletons MUST be scaled to µm (×0.008)
  to be comparable to the user's micron traces.

## Scope

**In (token only, no registration):** live region-aware fetch + registration-free
persistence matching of a query trace against a *scoped* candidate set, returning ranked
neuron-type suggestions with neuPrint metadata.

**Deferred:** NBLAST-grade comparison (needs Stage C: nc82→template registration);
FlyWire/FANC (CAVE auth, separate token); morphometric (non-persistence) enrichment of
fetched neurons.

## Design

### Region-aware backend
- `query_connectome(skeleton_id, db="neuprint", region=..., dataset=..., ...)`. Region →
  default dataset: `brain`→`hemibrain:v1.2.1`, `vnc`→`manc:v1.2.3`; explicit `dataset`
  overrides (e.g. `male-cns:v1.0`). Same `neuprint.Client`, dataset string differs.
- Token: read from env `NEUPRINT_APPLICATION_CREDENTIALS` (the user keeps it in
  `~/.config/neuprint/token`, chmod 600); never logged.

### Candidate scoping (REQUIRED — no match-all)
Matching one trace against ~25k+ neurons (a skeleton fetch + persistence vector each) is
infeasible per query. So a candidate filter is **mandatory**:
- by **type** pattern (e.g. `type="PFN.*"`), and/or
- by **ROI** (e.g. neurons innervating `FB`), and/or
- by **status='Traced', cropped=False**.
The tool fetches only the scoped candidates' skeletons. If the filter is missing or too
broad (candidate count over a cap, e.g. > ~300), it returns a typed `needs_scoping` status
asking the user to narrow — rather than fetching thousands.

### Fetch + cache
- `fetch_neurons` (criteria) → candidate bodyIds + metadata; `fetch_skeleton` per candidate.
- **Cache** skeletons + computed persistence vectors to disk keyed by `(dataset, bodyId)`
  (skeletons are large — ExR1 was 22k nodes), so repeat matches don't re-fetch.

### Registration-free matching (the core)
- Convert each fetched skeleton → SWC (×0.008 → µm) → `persistence_features_from_swc`.
- Compute the query trace's persistence vector (already available via Tier-1).
- Rank candidates by persistence-vector similarity (k-NN, same metric as the local
  matcher); return top-k `{bodyId, type, instance, similarity, pre, post, roi}`.
- Reuse `morphology_match` / `morphology_persistence`; do not reinvent the matcher.

### query_connectome modes
- `mode="match"` (default, token-only) → real ranked matches (registration-free).
- `mode="nblast"` → stays `needs_registration` (honest) until Stage C exists.

### Graceful degradation (typed statuses, no exceptions)
`backend_unavailable` (no extra) → `needs_token` (no token) → `needs_scoping` (filter too
broad/absent) → `ok` (matches). Mouse dbs (`microns`/`allen`) stay `off_domain`.

## Honest caveats
- **Cross-modality, registration-free → approximate.** Persistence compares branching
  topology/scale, not spatial position; confocal traces (partial, lower-res) vs EM
  (complete) differ. Good for "roughly which type", not spatial-overlap identification.
  The result must say so; NBLAST (registered) remains the gold standard.
- **Scale sensitivity:** correct µm conversion of fetched skeletons is load-bearing; a
  wrong factor silently degrades matches. Pin it in a test.
- **Partial/cropped query traces** bias persistence; surface `cropped`/coverage caveats.

## Reuse
- `connectome_neuprint.py` token/status scaffold; `morphology_persistence` (registration-free
  vectors); `morphology_match` (k-NN); trace store for the query vector; the typed-status
  degradation idiom.

## Benchmark verdict (2026-06-14) — pivot to registration

`scripts/bench_persistence_typeid.py` measured persistence-vector type recovery on 72
hemibrain neurons across 9 types (token-only, EM-only). Findings:

- **Ceiling is real:** complete neurons, same modality/scale → top-1 0.528, top-5 0.750
  (chance 0.099; shuffled-label control 0.069). Robust to ±20% scale (top-1 ~0.46) and
  lowres+jitter (top-5 0.68).
- **Fragmentation collapses it:** ~50% subtree → top-1 0.208 / top-5 0.472; ~30% subtree →
  top-1 0.139 / top-5 0.417 ≈ chance. Confocal traces are typically *partial*, so the
  registration-free matcher is unreliable for the real use case — shipping it would mislead
  (confirms Codex #4/#5 empirically).

**Decision:** do NOT ship registration-free persistence matching as a type-ID/suggestion
tool. The ceiling result is the argument *for* registration — registration brings complete
traces into a comparable space where the signal (and NBLAST, stronger than persistence)
actually works. Pivot to **Stage C (nc82 → template registration → NBLAST)**, or, only if
the user routinely has complete-cell traces, a narrow "morphology-similar EM skeletons (not
identification)" browser hard-gated to reject fragments.

## Open decisions (for the plan)
- Candidate cap before `needs_scoping` (propose ~300).
- Cache location/format (propose `<results_root>/connectome_cache/<dataset>/<bodyId>.json`).
- Similarity metric + whether to blend connectivity (pre/post) into ranking (lean: pure
  morphology first; connectivity later).
- Whether to expose a `build_connectome_reference(type/ROI)` prefetch tool for fast repeat
  matching, or fetch-on-query only (lean: fetch-on-query + cache for v1).
