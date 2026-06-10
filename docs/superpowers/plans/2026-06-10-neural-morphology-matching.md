# Neural Morphology Matching — Implementation Plan

> **For agentic workers:** implement this plan **one commit at a time**, running
> `uv run pytest -q -m "not slow and not integration"` after every commit. Each
> commit is designed to leave the codebase green. Steps use checkbox (`- [ ]`)
> syntax for tracking. Do not batch commits unless explicitly asked.

**Date:** 2026-06-10
**Status:** Planned (not started)
**Revision:** v2 — Tier 1 changed from NBLAST to **registration-free morphometric
features** after a code-grounded self-review (see *Review Findings* at the end).
v1's "local NBLAST" Tier 1 had a hidden co-registration prerequisite and pulled a
heavy optional dependency; both are now pushed to Tier 2 where they belong.

**Owner decisions baked in:** layered backend (not "wire up all four DBs"). **Tier
1 = local morphometric similarity/classification** using the morphology descriptors
the app *already* computes + `scikit-learn` (already a dependency) — **no new
dependency, no spatial registration, offline.** **Tier 2** = NBLAST (`navis`,
optional extra) + template registration + Drosophila connectome DBs (neuPrint →
FlyWire). **Out of scope** = MICrONS / Allen (mouse, off-domain). Deliverable of
*this* document: the plan.

**Goal:** Turn the two `not_implemented` stubs in `tools/trace.py`
(`classify_neuron_type` at line ~760, `query_connectome` at line ~736) into a real,
offline morphometric-matching capability, behind an abstraction that lets NBLAST and
external connectome databases plug in later without changing the tool surface.

**Non-goal:** No new heavy dependency in Tier 1. No spatial registration in Tier 1.
No external network calls in Tier 1. No change to the shipped tracing pipeline
(enhance → segment → skeletonize → metrics → prune/QC → export), which is complete
and tested.

---

## Problem Statement

`tools/trace.py` exposes 13 `@tool`s for the `neural_tracer` specialist; 11 are real
and tested. Two are deliberate stubs returning `{"status": "not_implemented"}`:

- `query_connectome(skeleton_id, db, k)` — nearest reference neurons from an external
  connectome DB (`db ∈ {flywire, neuprint, microns, allen}`).
- `classify_neuron_type(skeleton_id, reference)` — predict a neuron type.

**Why not NBLAST for the first deliverable.** NBLAST (the standard morphology
similarity used by neuPrint/FlyWire) compares neurons by nearest-neighbour point
distances + tangent dot-products **in a shared coordinate frame** — it is *not*
rotation/translation invariant. Comparing an arbitrary confocal trace against an
arbitrary reference library is only valid after both are registered into a common
template brain (`navis-flybrains`). That registration is exactly the work this plan
defers to Tier 2, so an "NBLAST Tier 1" would secretly depend on Tier 2 (or produce
unreliable scores). It also pulls a heavy optional dependency (`navis`) for a
capability that can't yet be trusted.

**What the codebase already gives us, for free.** `compute_morphology_descriptors`
(`trace.py:642`) already emits ~12 scalar shape features (total length, branch /
endpoint / junction counts, terminal vs internal branches, components, bbox,
occupancy), and `_branch_summary` exposes `branch_length` + `euclidean_distance` (→
tortuosity) and `branch_type_code`. `compute_sholl_analysis` adds a radial profile
(peak, peak radius, AUC). `scikit-learn` is already a dependency. A **morphometric
feature-vector** comparison over these is registration-free, rotation/translation
invariant, needs **no new heavy dependency**, and is a standard neuron-classification
approach (L-Measure-style). That is the right Tier 1; NBLAST is the right Tier 2.

**Two honest constraints (unchanged from v1).**

- **Species lock-in.** neuPrint / FlyWire are *Drosophila*; MICrONS / Allen are
  *mouse*. App domain is Drosophila confocal (`PROJECT_PLAN.md:22`) → MICrONS/Allen
  excluded; the current `db` enum lists them only as placeholders.
- **Connectome lookup needs template registration** (Tier 2 prerequisite).

## Solution

A matcher abstraction plus three tiers; only Tier 1 is committed in detail.

- **Tier 1 — morphometric features (this plan).** A shared `extract_feature_vector`
  over the existing descriptor output, split into **scale-invariant** features
  (counts, ratios, tortuosity, occupancy — always usable) and **absolute** features
  (lengths, bbox — used only when the skeleton is in physical µm). A **reference
  library** = a labelled CSV of feature vectors, built offline from the user's own
  labelled traces (`add_reference_neuron`). Classification / nearest-neighbour search
  via `scikit-learn` (`StandardScaler` + k-NN). Both stubs become real
  (`classify_neuron_type`, plus a new `find_similar_neurons`), and degrade gracefully
  to a typed `status` when no reference library is configured.

- **Tier 2 — NBLAST + connectome (sketched).** Add `navis` as an optional extra
  (`uv sync --extra connectome`), template registration (`navis-flybrains`), and a
  neuPrint reference source behind the same matcher interface; `query_connectome`
  becomes real. Then FlyWire (CAVE auth). Outlined, not committed.

- **Tier 3 — out of scope.** MICrONS / Allen (mouse): rejected with an "off-domain"
  message, not faked.

Tool surface stays stable: `classify_neuron_type` / `query_connectome` keep their
signatures; additions are `find_similar_neurons` and `add_reference_neuron`.

---

## Current-State Facts (evidence gathered 2026-06-10)

| Fact | Value |
|---|---|
| Stub tools | `query_connectome` (`trace.py:~736`), `classify_neuron_type` (`trace.py:~760`); both `subagent="neural_tracer"`, `phase="6B"` |
| **Current stub ignores `skeleton_id`** | `classify_neuron_type` returns a canned dict without touching `_SKELETON_REGISTRY` (`trace.py:767`) — a real impl that calls `_entry(id)` must guard the no-reference path *before* the lookup (see N4 / finding H3) |
| Descriptor source for features | `compute_morphology_descriptors` (`trace.py:642`) → total_length, length_unit, mean/median_branch_length, n_branches, n_endpoints, n_junctions, n_components, n_terminal/internal_branches, bbox_scaled, skeleton_volume_occupancy |
| Scale-invariant extras | `_branch_summary` → `branch_length`,`euclidean_distance` (tortuosity), `branch_type_code` (`_trace_tables.py:38`); Sholl → peak/peak_radius/AUC (`trace.py:561`) |
| Physical-units detector | `_scale_is_physical(spacing) = any(\|v-1\|>1e-9)` (`_trace_tables.py:33`); pixel-scale ⇒ `units=None` on the record (`_trace_store.py:110`) |
| QC-record keys (collision risk) | `compute_morphology_descriptors` writes `put_qc_record(skeleton_id, kind="neural_morphology")` (`trace.py:670`); segmentation writes by **layer name** (`trace.py:254`). A classifier QC write keyed by `skeleton_id` would **overwrite** the morphology record (finding H2) |
| Classifier dep | `scikit-learn>=1.5` already declared (`pyproject.toml`) and installed ✓ — **Tier 1 needs no new dependency** |
| NBLAST deps (Tier 2 only) | navis / navis-flybrains / neuprint all MISSING; install is `uv sync --extra connectome` (repo uses `uv`, not pip) |
| `@tool` registry | new `@tool(subagent="neural_tracer")` defaults `llm=True`,`manual=False` ⇒ auto-visible via `tools_for_anthropic("neural_tracer")` (`registry.py:128,148`) |
| Stub tests to update | `test_tools_trace.py:79-87` assert `status=="not_implemented"` (call `classify_neuron_type("any_id")` with a **non-existent** id) |
| Specialist prompt | `agent/specialists/neural_tracer.py:27` ("stubbed for now") — update in N6 |
| `query_connectome` db validation | accepts `{flywire,neuprint,microns,allen}`, else `ValueError` (`trace.py:748`); no test exercises microns/allen ⇒ rejecting them in N6 is safe |
| Offline mode exists | `docs/specs/phase8_distribution_onboarding.md:43` ("Offline / No-LLM User") — Tier 1 keeps that intact (no new dep, no network) |

---

## Commits

### Phase 0 — Net

- [x] **N0. Characterization tests + reference fixtures.** Add
  `tests/test_tools_morphology.py`. Pin the *current* stub contract
  (`classify_neuron_type` / `query_connectome` return a dict with a `status` key,
  today `"not_implemented"`) so the change is observable. Add a fixture that
  skeletonizes 2-3 toy masks (straight line, Y-branch, bushy multi-branch; reuse the
  mask builders in `test_tools_trace.py`), runs `compute_morphology_descriptors`, and
  writes a labelled reference CSV in `tmp_path`. Uses only existing deps. Must pass
  on current code unchanged.

### Phase 1 — Tier 1: morphometric matching (no new dependency)

- [x] **N1. Feature extractor + unit guard.** `analysis/morphology_features.py`:
  `extract_feature_vector(descriptors, branch_df=None) -> {features: dict,
  units_physical: bool}`. **Scale-invariant core** (always present): n_branches,
  endpoint/junction counts, terminal_fraction, internal_fraction,
  endpoints_per_junction, mean/median branch-length ratio, mean tortuosity
  (`branch_length/euclidean_distance`), volume_occupancy. **Absolute** (gated on
  `units_physical`): total_length_um, mean_branch_length_um, bbox dims. Pure function,
  no new deps. Tests: invariant features identical for a pixel-scale vs µm-scale
  version of the same shape (this is the M1 guard); absolute features present only
  when physical.

- [ ] **N2. Reference library I/O.** `analysis/morphology_reference.py`:
  `load_reference_library(path)`, `append_reference(path, feature_vector, label,
  name)`. CSV schema = one row per neuron: `name, label, <feature columns>,
  units_physical`. Pure pandas. Tests (use N0 fixtures): round-trips, raises a clear
  error on empty/missing library, requires a `label` column, and refuses to mix
  physical and non-physical rows without falling back to the invariant subset.

- [ ] **N3. Matcher core.** `tools/_trace_classify.py`:
  `match_against_library(query_fv, library, *, k) -> {ranked: [{name,label,distance}],
  predicted, confidence, status}`. `StandardScaler` fit on the library + k-NN in
  feature space; when query and library disagree on `units_physical`, restrict to the
  invariant subset and flag it. `confidence` = a bounded function of the
  nearest/2nd-nearest distance ratio. Pure `scikit-learn` (already present). Tests:
  Y-branch query ranks a branched label above a linear one; single-row library still
  classifies; empty library ⇒ `status="no_reference"`.

- [ ] **N4. Make `classify_neuron_type` real.** Rewrite the stub. **Ordering (fixes
  H3):** resolve the reference library *first*; if missing/empty return
  `{status:"no_reference", ...}` **without** calling `_entry(skeleton_id)` — so the
  existing-style call with a bogus id still returns a graceful status, not a KeyError.
  Only with a library present do we look up the skeleton, run
  `compute_morphology_descriptors` → `extract_feature_vector` → `match_against_library`
  and return `{skeleton_id, predicted_type, confidence, runner_up, status:"ok"}`.
  Keep the signature (`reference="default"` ⇒ the default library path, else a path).
  **Fix H2:** write the classification QC record under a distinct key
  `f"{skeleton_id}::classification"`, never plain `skeleton_id` (which belongs to the
  morphology descriptors). Update `test_tools_trace.py:85` to assert `"no_reference"`
  when unconfigured, plus a real classification test against an N0 library.

- [ ] **N5. New tools: `add_reference_neuron` + `find_similar_neurons`.** Both
  `@tool(subagent="neural_tracer", phase="6B")`. `add_reference_neuron(skeleton_id,
  label, library_path)` appends the current skeleton's feature vector — closing the
  loop so a user builds a labelled library from their own traces, fully offline.
  `find_similar_neurons(skeleton_id, reference, k=10)` returns top-k nearest by
  feature distance (the local answer to "nearest by morphology"; distinct from
  `query_connectome`, which stays external/Tier 2). Same no-reference / no-skeleton
  ordering as N4. Tests: build → append → classify round-trip; ranking sanity.

- [ ] **N6. Specialist prompt + keep `query_connectome` honest.** In
  `neural_tracer.py`: rewrite the "stubbed for now" paragraph — morphometric
  classification and nearest-neighbour search are **available locally/offline**;
  NBLAST + connectome DB lookups are Tier 2 (need `navis` + template registration).
  Reject `query_connectome` `db ∈ {microns, allen}` with an explicit mouse/off-domain
  message; `{neuprint, flywire}` keep returning `"not_implemented"` with a "Tier 2 —
  needs backend + token + registration" note. Tests: `tools_for_anthropic
  ("neural_tracer")` now includes the two new tools; prompt no longer claims
  classification is stubbed.

- [ ] **N7. Docs.** `PROJECT_PLAN.md` Phase 6: mark local morphometric matching done;
  describe the reference-library workflow; record the Tier 2/3 roadmap + species
  caveat + that Tier 2 adds the `connectome` extra (`uv sync --extra connectome`).
  Short README note. No code.

### Phase 2 — Tier 2: NBLAST + Drosophila connectome (roadmap, NOT committed here)

- Add `[project.optional-dependencies] connectome = ["navis", "navis-flybrains",
  "neuprint-python"]`; verify it installs under the `numpy>=1.26,<3` pin on
  Python 3.11-3.12 **before** relying on it (v1 assumed this; do not).
- `navis` NBLAST adapter isolated in one module behind the matcher interface; SWC
  bridge via the existing `_write_swc`. Force single-threaded NBLAST in tests.
- Template registration (`navis-flybrains`): confocal trace → standard brain space —
  the real prerequisite for valid cross-dataset NBLAST.
- neuPrint reference source (token-configured) feeding the same matcher; then FlyWire
  (CAVE auth). `query_connectome(db="neuprint")` becomes real here.
- Every backend lazy + optional + behind the Tier-1 graceful-degradation contract.

### Phase 3 — Out of scope

- MICrONS / Allen (mouse) — rejected with an "off-domain" message.
- NBLAST in Tier 1; learned classifiers; persisting reference libraries in
  session/project state; a GUI for reference management.

---

## Decision Document

- **Tier 1 = morphometric features, not NBLAST.** Registration-free, rotation/
  translation invariant, reuses descriptors we already compute + `scikit-learn`
  (already a dep) ⇒ **zero new dependency, offline, both stubs become real.** NBLAST
  is spatial-overlap-based and needs co-registration, so it belongs in Tier 2 next to
  the `navis-flybrains` registration that makes it valid.
- **Scale handling is explicit (M1).** Features are split into scale-invariant
  (always used) and absolute (gated on physical µm). Mixing pixel- and µm-scale
  neurons falls back to the invariant subset with a flag — never a silent invalid
  comparison.
- **Graceful degradation over exceptions.** No reference library ⇒
  `status:"no_reference"`, decided *before* any skeleton lookup, so a bogus id never
  crashes the tool (H3). Mirrors the current stub's tolerance.
- **Distinct QC key (H2).** Classification writes `f"{id}::classification"`, leaving
  the `neural_morphology` QC record (keyed by plain `id`) intact.
- **Stable tool surface.** Signatures of `classify_neuron_type` / `query_connectome`
  unchanged; add `find_similar_neurons` + `add_reference_neuron`. `query_connectome`
  stays the external-DB tool (Tier 2).
- **Species honesty.** MICrONS/Allen rejected as off-domain rather than faked.

## Testing Decisions

- **Everything in Tier 1 is CI-testable with current deps** (no navis, no network).
  This is the main payoff of the reframe: no `skipif`-gated paths, no heavy optional
  install to exercise the real behaviour.
- **Observable assertions only:** tool return contracts (`status`, `predicted_type`,
  `ranked` ordering), not classifier internals.
- **The M1 invariance test is load-bearing:** the same shape at pixel vs µm scale must
  yield identical scale-invariant features (guards the unit-mismatch bug class).
- **Reuse fixtures:** `viewer`, the mask builders in `test_tools_trace.py`, conftest's
  skeleton reset; reference CSVs generated in `tmp_path` (N0).
- **Per-commit gate:** `uv run pytest -q -m "not slow and not integration"`.

## Out of Scope

- External network calls (all of Tier 1 is local/offline).
- NBLAST / `navis` / template registration / connectome fetching (Tier 2 roadmap).
- MICrONS / Allen (mouse).
- Changes to the shipped tracing pipeline.
- Persisting reference libraries in session/project state.

## Review Findings (v1 self-review that produced this v2)

A code-grounded review of the v1 draft surfaced the following; v2 resolves them.

- **H1 (High) — NBLAST Tier 1 rested on a deferred prerequisite.** NBLAST needs
  co-registration; v1 put registration in Tier 2. → v2 makes Tier 1 registration-free
  (features) and moves NBLAST to Tier 2. Also eliminates v1's M2/M3/M4 from Tier 1
  (no `uv`/`pip` extra, no unverified `navis` version claim, no `make_dotprops` on
  tiny/2D skeletons).
- **H2 (High) — QC key collision.** `compute_morphology_descriptors` already writes
  `put_qc_record(skeleton_id,…)` (`trace.py:670`); a classifier writing the same key
  overwrites it. → N4 uses `f"{id}::classification"`.
- **H3 (High) — stub ignores `skeleton_id`, but a real impl can't.** Current stub
  returns canned data; the existing test calls `classify_neuron_type("any_id")`
  (`test_tools_trace.py:86`). A real `_entry(id)` lookup would `KeyError`. → N4 checks
  reference availability *before* the lookup and the test moves to a real id for the
  success path.
- **M1 (Med) — pixel-vs-µm validity.** `_scale_is_physical` ⇒ `units=None` for
  pixel-scale data; absolute-length features are then incomparable. → N1 splits
  invariant vs absolute features and gates the absolute ones.
- **M2/M3/M4 (Med, NBLAST-only) — uv-not-pip install, unverified navis/3.12/numpy
  compat, dotprops on small/2D skeletons.** → deferred to Tier 2, with explicit
  "verify before relying" notes.
- **L (Low) — confirmed safe:** rejecting `microns/allen` breaks no test; adding
  tools doesn't break the fake-provider specialist test
  (`test_subagent_neural_tracer.py` asserts named calls, not a tool count).
