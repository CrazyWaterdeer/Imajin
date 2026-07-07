# Plan: Spots, spatial relationships, object colocalization, filament tracer

Implementation plan for the approved scope from `imaris_gap_analysis.md`:
Spots detection → spatial relationships → object-based colocalization, plus a
connectivity-aware filament tracer with diameter analysis.

Status: IMPLEMENTED (2026-07-08). Revised after one Codex review (valid points
folded in — see "Codex triage" below), then Phases 0–5 built, tested, and merged
to master. Deferred, as planned: deconvolution, dendritic-spine detection,
`min_path` AutoPath.

Delivered (12 new @tools, full suite 909 passed):
- Phase 0 — `analysis/coords.py` coordinate contract; fixed `set_soma_location`.
- Phase 1 — `detect_spots`, `compute_spots_qc` (`tools/spots.py`).
- Phase 2 — `assign_objects_to_parents`, `measure_distance_to_reference`,
  `nearest_neighbor_distances` (`tools/spatial.py`).
- Phase 3 — `costes_threshold`, `costes_significance`, `object_colocalization`
  (extended `tools/coloc.py`).
- Phase 4 — `propose_filament_bridges`, `build_rooted_tree` (`tools/trace/tracer.py`).
- Phase 5 — `measure_filament_diameter`, `compute_tree_topology`
  (`tools/trace/filament_analysis.py`); SWC export now writes measured radii.

## Principles this plan must respect

- **One function, two drivers**: every capability is a `@tool(...)` in
  `tools/*.py` that both the chat and a magicgui manual-dock form call — identical
  results, identical provenance. No logic in the UI layer.
- **Manual-dock layer pickers are name-driven** (verified in
  `ui/manual_dock.py::_layer_param_names`): a parameter gets a layer dropdown only
  if it is named `layer`, ends with `_layer`, or is in
  `{image_a, image_b, mask, boundary_mask}`. **All new layer arguments must end in
  `_layer`** (`objects_layer`, `parents_layer`, `reference_layer`, `channel_layer`)
  or the dock stays a free-text box.
- **Metadata vs meaning / physical units**: spatial outputs in µm from voxel
  spacing; never parse filenames.
- **House patterns**: read layers on the main thread via
  `call_on_main(snapshot_layer, name)`, `_materialize(...)` to numpy, emit session
  tables through `session.put_table` / `_trace_tables._put_table`, `worker=True`
  for heavy calls, QC surfaced in the QC dock, results captured in bundles/recipes.
- **Tables feed the existing pipeline**: any new measurement is a DataFrame that
  `describe_table` / `compare_groups` / `plot_*` / `generate_report` already accept,
  carrying sample/parent ids so aggregation and pseudoreplication warnings hold.

## Phase 0 — Object & coordinate contract (foundation; do first)

Codex's top finding: without one explicit coordinate convention, Phases 1–3 will
bake in incompatible assumptions and distances can be **double-scaled** (a Point
stored in µm that napari then multiplies by layer `scale` again). Establish this
before any detection tool.

New helper `analysis/coords.py` (pure, unit-tested, no Qt):

- **Canonical rule**: geometry lives in napari **data/index coordinates** on the
  layer; physical µm is derived on demand via `scale` (and `translate` for world
  position). Points layers store **data coordinates** so napari renders them
  correctly; tables carry **both** raw index columns and `_um` columns (mirror
  `measure.py::_add_physical_columns`). Nothing is pre-multiplied into µm in the
  layer itself.
- Helpers: `data_to_world` / `world_to_data` (scale + translate), per-object
  centroid and voxel-coordinate extraction, axis/`C`/`T` handling reusing
  `analysis/arrays.layer_axes_from_metadata`.
- **Fix the existing inconsistency** Codex flagged: `set_soma_location` takes
  point-layer coords as-is while mask-derived soma is scaled. Route both through
  `coords.py` so soma, skeleton nodes, spots, and distances share one convention.
- Tests: round-trip data↔world under non-unit `scale` + non-zero `translate`;
  anisotropic 3D; a soma-from-points vs soma-from-mask equality test that pins the
  fixed convention.

## Phase 1 — Spot / puncta detection

New module `tools/spots.py`.

- `detect_spots(channel_layer, min_diameter_um, max_diameter_um, mode="2d_projection"
  |"3d", threshold="auto", subtract_background=True, boundary_mask=None,
  exclude_border=True, overlap=0.5)`
  - `skimage.feature.blob_log` (default) / `blob_dog`. **Anisotropic sigma**: derive
    per-axis sigma from diameter ÷ voxel spacing; in 3D allow **separate lateral vs
    axial diameter** (confocal axial PSF is worse than lateral — a spherical
    physical diameter is not the observed blob), or a PSF-derived axial default.
  - **`mode`** is explicit: `2d_projection` (detect on a projection — different QC)
    vs true `3d`. Do not silently mix.
  - **Background first**: optional rolling-ball / white-tophat pre-step (reuse
    `preprocess`) so LoG isn't fooled by uneven background.
  - **`threshold="auto"` is concretely defined**, not vague: threshold relative to
    a robust noise estimate (e.g. `k · MAD` of the background), documented and
    logged in provenance; expose the multiplier.
  - `exclude_border` drops edge-truncated spots; `overlap` controls deblending of
    touching puncta (blob_log's overlap merge).
  - `boundary_mask=` restricts detection (reuse `segment_target_objects` convention).
  - Outputs: (a) a napari **Points** layer in **data coords** (canonical), (b) a
    session table with index coords + `_um` columns, estimated lateral/axial
    diameter, per-channel intensity sampled at spot centers (small window, not
    regionprops — regionprops is Labels-only), local SNR / quality; (c) optional
    dilated-seed **Labels** only for downstream label-only tools.
- QC `compute_spots_qc`: count, density (per µm²/µm³ in mask), fraction below
  quality; on synthetic data, **precision/recall** vs planted ground truth.
- Manual dock form + provenance + recipe step.
- Tests: synthetic puncta (borrow `calcium_synth` style) with known counts/
  positions → recovered count + subpixel error; anisotropic-spacing; dense/touching
  puncta stress; boundary-mask; empty image; `2d_projection` vs `3d` parity where
  expected.

## Phase 2 — Object-to-object spatial relationships

New module `tools/spatial.py`. Depends on Phase 0.

- `assign_objects_to_parents(objects_layer, parents_layer)` — objects = Points or
  Labels; parents = Labels. Points → containing parent by index lookup; Labels →
  max-overlap **with explicit edge policies**: background/unassigned, boundary/tie,
  objects spanning multiple parents, parent holes, cropped-border objects,
  time-varying labels. Emit per-object rows (`parent_id`, `overlap_fraction`,
  `assignment_ambiguous`) **and** a per-parent summary (`n_objects`, density). This
  is "spots per cell" — with ambiguity made visible, not silently resolved.
- `measure_distance_to_reference(objects_layer, reference_layer, signed=False)` —
  `distance_transform_edt(sampling=spacing)` of the reference. **Points**: sample
  EDT at point coords. **Labels objects**: per-object **min boundary distance**
  (min EDT over each object's voxels), preserving per-object identity — *not*
  centroid-of-object over a label union (Codex #7). Optional signed (negative
  inside). → `distance_um`.
- `nearest_neighbor_distances(objects_layer, other_layer=None, k=1)` —
  `scipy.spatial.cKDTree`; within-set or between two sets → `nn_distance_um`
  (+ k-NN mean). Clustering/dispersion readout.
- All emit tables → `compare_groups` / `plot_group_distribution` / `plot_scatter`.
- Tests: geometric fixtures with known answers (grid of points in known parents;
  object at known distance from a plane; two rings with known NN); µm correctness
  under anisotropic scale; ambiguous-assignment fixture asserts the flag fires.

## Phase 3 — Object-based colocalization + Costes

Extend `tools/coloc.py` (keep existing Manders/Pearson). Depends on Phase 2.

- `costes_threshold(image_a, image_b, mask=None)` — regression-based automatic
  threshold. Document sensitivity to background/bleedthrough/saturation and the
  masked-pixel selection; require a specimen `mask` for meaningful results.
- `costes_significance(image_a, image_b, mask=None, n=200, block="auto")` —
  block-scramble one channel and recompute Pearson r. **Block size tied to
  PSF/autocorrelation** (`"auto"` estimates it), not arbitrary; `n` default raised
  from 100 → 200 and documented as the floor. Report observed r, null summary,
  percentile/p — **framed as exploratory**, with caveats, matching Imajin's honest-
  stats stance.
- `object_colocalization(objects_a_layer, objects_b_layer, within_layer)` — overlap
  fraction and NN-distance vs a **null model constrained by biology**: resample
  within the specimen `within_layer` mask, respecting object count and size, not the
  whole FOV (else it just reports density / segmentation bias, Codex #10). Reuses
  Phase 2 machinery.
- Tests: analytic Costes cases (correlated / anticorrelated / independent); overlap
  cases (fully overlapping vs disjoint); a **channel-shift negative control** that
  must read as non-colocalized.

## Phase 4 — Filament tracer (split into 3 stages; honest MVP)

Builds on `enhance_neural_processes` (Frangi/Sato) and the **existing**
`_trace_tables.store_graph_tables` (skan Skeleton → node/edge/component tables with
connected components already implemented). Framed honestly: this is a
skeleton-graph tracer, **not** an Imaris-parity ML tracer. Documented failure
modes: crossings, touching/fused neurites, loops, low-SNR fragments, 2D
projections. Not parallel-safe — the three stages are serial.

- **4a `extract_filament_graph(layer, ...)`** — skeletonize (existing) → skan graph
  → node/edge/component tables (existing). No bridging yet; just a clean,
  identity-preserving graph with QC (component count, endpoint count).
- **4b `propose_filament_bridges(trace_id, max_gap_um, ...)`** — candidate endpoint
  joins via cKDTree, but **evidence-gated, not distance-only** (Codex #3): require
  tangent/direction continuity, vesselness/intensity support sampled along the
  candidate segment, gap length relative to local radius, and no crossing through
  other labels. Every candidate written to a **bridge QC table** (accepted/
  rejected + reason) for review before it mutates topology.
- **4c `build_rooted_tree(trace_id, soma=None)`** — turn the (bridged) undirected
  skeleton graph into a valid neuron tree (Codex #4): break cycles (MST/BFS from
  soma), snap/choose soma, deterministic parent ordering, explicit disconnected-
  component/multi-root policy, and provenance for every bridged/dropped edge. Only
  now are SWC parent pointers valid. Feeds existing `extract_branch_metrics`,
  `compute_sholl_analysis`, `classify_neuron_type`, `export_neural_trace`.
- Tests: synthetic Y/branched arbor with a planted gap → single rooted tree,
  correct branch count + parent ordering; X-crossing fixture → assert bridges are
  *proposed and gated*, and document that binary-skeleton pixel-sharing at true
  crossings is a known limitation (cannot be asserted away).

## Phase 5 — Filament analysis (diameter + topology; spines deferred)

Depends on Phase 4c. Scoped honestly per Codex #11.

- `measure_filament_diameter(trace_id)` — EDT of the segmented mask sampled along
  skeleton nodes → local radius → per-branch diameter profile (mean/min/max).
  **Exclude junction neighborhoods** (radius inflates there) and state the
  segmentation-dependence caveat. Serves dendrite and vessel width.
- Tree-topology metrics into `compute_morphology_descriptors`: branch order,
  Strahler number, path length to soma.
- Extend `export_neural_trace` SWC/CSV with radius columns.
- **Dendritic-spine detection is deferred** to a separate, explicitly experimental
  effort after topology + diameter are validated — spine necks are often below
  confocal resolution and "reuse the blob detector" is insufficient.
- Tests: cylinder of known radius → diameter within tolerance (junction-excluded);
  Strahler on a known tree.

## Validation (beyond synthetic)

Synthetic fixtures gate CI (deterministic). Add a documented **real-data
acceptance protocol** the user runs before trusting a phase in production:

- Spots: a densely-punctate confocal stack + manual counts; sub-resolution beads
  for PSF/scale.
- Coloc: a channel-shift **negative control** (must read non-colocalized).
- Tracer: comparison against a manual trace on a real neuron.

## Sequencing & recommendation

- **Phase 0 first** (coordinate/object contract) — everything depends on it.
- **Thread A**: `1 → 2 → 3` (spots → spatial → object coloc), each unlocks the next.
- **Thread B**: `4a → 4b → 4c → 5`, strictly serial (not parallel-safe).

Recommend **Phase 0 → Phase 1** as the immediate start (Phase 1 also unlocks 2, 3).
Thread B can begin after Phase 0. Ship each stage behind its own tools + tests.

## Codex triage (what was accepted / declined)

Accepted and folded in: coordinate contract as Phase 0 (#1); honest tracer framing +
3-way split + evidence-gated bridging + valid-tree construction (#2, #3, #4,
sequencing B); spot background/PSF-anisotropy/2D-vs-3D/threshold-definition/border/
deblend/precision-recall (#5, #6); object-boundary min-distance + identity + edge
policies (#7, #8); Costes `n`↑/PSF block size/mask-constrained null/exploratory
framing (#9, #10); diameter junction caveat + **spine deferral** (#11); real-data
validation protocol (#12); manual-dock `*_layer` naming; all four open decisions
(skan not sknw; defer `min_path`; deconvolution stays a separate later effort;
Points canonical + optional Labels).

Declined/limited: EDT-based diameter is kept (not dropped) with an explicit
junction-exclusion + segmentation-dependence caveat rather than deferred — it is
useful and cheap when honestly scoped.

## Open decisions — resolved

1. **Graph lib**: use `skan` (already a core dep; `store_graph_tables` uses it). No
   `sknw`. Declare `networkx` directly if imported (currently transitive via skan +
   scikit-image).
2. **`min_path` AutoPath**: **deferred** — experimental/manual-seed only, after 4a–4c
   are solid.
3. **Deconvolution**: **separate later effort**, provenance-tracked preprocessing —
   not in this plan's scope.
4. **Spot output**: **Points canonical** (data coords + µm table columns) + optional
   dilated-seed Labels for label-only downstream tools.
