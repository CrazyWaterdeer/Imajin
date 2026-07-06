# Channel-as-Mask: Inside/Outside Signal Analysis — Implementation Plan

Status: plan (revised after one Codex review; ready to implement)
Date: 2026-07-07
Spec: `docs/superpowers/specs/2026-07-07-channel-mask-inside-outside-design.md`

Branch `feat/mask-logic-inside-outside` off `master`. Commit-by-commit; each commit runs its
own tests **and** the affected suites as a gate. Merge `--no-ff` to master + push when green
(per merge-then-push workflow).

Delivers: `mask_logic` (boolean set-ops primitive), `partition_inside_outside` (opinionated
inside/outside wrapper), and a `measure_intensity` `region`-column enhancement — so a
segmented channel scopes another channel's analysis and inside-vs-outside is a short, correctly
paired recipe over existing tools.

## Cross-cutting design rules (apply to both new tools)

- **Reference input = the highest-`ndim` input** among those supplied (3D beats 2D; ties →
  `a_layer` / `region_layer`). All other inputs are aligned to the reference's shape via
  `resolve_boundary_mask` (which broadcasts a 2D mask across the reference's Z). Output layer
  `scale` and shape come from the **reference**. This fixes the 2D-region + 3D-`within` case:
  a region from a 2D MIP broadcasts into the 3D specimen and the output carries the 3D scale
  (Codex-plan #1/#10).
- **`_align(target_shape, arr, *, broadcast_2d_to_3d) -> (bool_mask, broadcast_z)`**: wraps
  `resolve_boundary_mask`; returns whether a 2D→3D broadcast happened; raises `ValueError` when
  a broadcast is needed but `broadcast_2d_to_3d=False`, or on any incompatible shape (Codex-plan
  #6). Verified safe: boolean ops on the read-only broadcast view allocate fresh arrays.
- **No `translate` mirroring, no Image-vs-Labels detection.** `snapshot_layer` carries only
  `name/data/scale/metadata` and `add_labels_from_worker` has no `translate` param
  (Codex-plan #2), so we follow the `coloc`/`measure` precedent: propagate **scale only**. The
  gray-level "you passed a raw channel" heuristic is **dropped for v1** — it can't tell a
  300-cell Labels layer from an Image without the layer kind, and would false-positive
  (Codex-plan #2 + scope). The masks-in contract (`foreground = data > 0`) is documented instead.
- **Worker/main-thread + headless:** `worker=True`; every viewer touch via
  `call_on_main(snapshot_layer, …)` / `call_on_main(add_labels_from_worker, …)`, exactly like
  `coloc.py`/`measure.py`. No `import napari` at module scope.

## Commit order

### Commit 1 — `erode_binary_um` helper (physical-radius erosion)

- `src/imajin/analysis/segmentation.py`: add `erode_binary_um(binary, *, spacing, radius_um)`
  mirroring `dilate_binary_um` (`segmentation.py:253`) exactly — same per-axis pixel-radius
  structure — using `scipy.ndimage.binary_erosion`. `radius_um <= 0` → return `binary` unchanged.
- Tests (`tests/test_segmentation_helpers.py`, new): erode shrinks a filled square by ~radius/px;
  anisotropic spacing → per-axis radius; `0`/negative → noop; erode of a thin line → empty;
  `dilate∘erode` on a solid blob ≈ identity (closing).

**Gate:** `pytest tests/test_segmentation_helpers.py tests/test_tools_segment.py -q`.

### Commit 2 — `measure_intensity` emits a `region` column from `label_names` metadata

- `src/imajin/tools/measure.py`:
  - `_add_region_column(df, label_names)`: build `mapping` by coercing each key with a
    **guarded** `int(k)` (skip keys that don't coerce — no raise, Codex-plan #9); if `"label"`
    in `df.columns`, `df.insert(<after label>, "region", df["label"].map(mapping))`.
    `df["label"].map` yields `NaN` for unmapped labels and **never drops rows**. No-op when
    `label_names` is falsy / not a dict, or `"label"` absent.
  - `measure_intensity`: after `df = _add_physical_columns(...)`, read
    `label_names = (labels.metadata or {}).get("label_names")` (the `snapshot_layer` result
    carries `.metadata`) and apply.
  - `refresh_measurement`: same one-liner so a refreshed table keeps `region` (Codex-plan #8).
- Tests (`tests/test_tools_measure.py`): `{1,2}` labels layer with
  `metadata={"label_names": {1:"inside",2:"outside"}}` + red image → table has `region` per row;
  **no metadata → no `region`** (back-compat); **string keys** `{"1":…}` (JSON round-trip) map;
  **partial mapping** (`{1:"inside"}`) → label 2 row present with NA region (not dropped);
  **refresh** keeps `region`.

**Gate:** `pytest tests/test_tools_measure.py -q`.

### Commit 3 — `mask_logic` tool + pure core + registration

- `src/imajin/tools/masks.py` (new):
  - `_foreground(arr) -> np.ndarray[bool]`: `np.asarray(arr) > 0`.
  - `_align(target_shape, arr, *, broadcast_2d_to_3d) -> (bool, broadcast_z)` (as above).
  - `_reference_index(shapes) -> int`: index of the max-`ndim` input (ties → first).
  - `_combine_masks(op, a, b, within) -> np.ndarray[bool]` (headless): `not`=`~a`, `and`=`a&b`,
    `or`=`a|b`, `subtract`=`a&~b`; then `& within` when given; raises `ValueError` if a binary op
    is missing `b`. The unit-tested heart.
  - `@tool(phase="7", worker=True) def mask_logic(op, a_layer, b_layer=None, within_layer=None,
    broadcast_2d_to_3d=True, name=None)`:
    1. `op in {not,and,or,subtract}` else `{ok:False,error}`. `not` with a `b_layer` → keep going,
       warn "b_layer ignored for op=not".
    2. snapshot each supplied layer; axes-guard each via `layer_axes_from_metadata` → reject unless
       `YX`/`ZYX` (segment-tool message: extract a timepoint/slice first).
    3. reference = max-ndim input; `target_shape`/`scale` from it. `_align` the others (collect
       `broadcast_z = any`). incompatible/broadcast-disallowed → `{ok:False,error}`.
    4. scale/translate agreement across inputs → `scale_mismatch` bool + warning (metadata-only;
       not fatal — a hand-built `scale=1` mask vs µm image is legitimate).
    5. `mask = _combine_masks(op, a, b, within)`.
    6. `layer = call_on_main(add_labels_from_worker, mask.astype(int32),
       name=name or f"{a_layer}_{op}", scale=ref.scale, metadata={source_layer, source_path, op,
       axes, broadcast_z, scale_mismatch, mask_voxels, mask_fraction})`.
    7. return `{ok, op, mask_layer: layer.name (actual, post-dedupe), voxels, fraction, empty,
       broadcast_z, scale_mismatch, axes, warnings}`; empty → `ok:True, empty:True` + warning.
- `src/imajin/tools/__init__.py`: `from imajin.tools import masks  # noqa: F401, E402`.
- Tests (`tests/test_tools_masks.py`, new):
  - **registration:** `imajin.tools.get_tool("mask_logic")` returns an entry (assert here, **not**
    in `test_tool_registry.py` — its autouse `reset_registry` clears `_REGISTRY`, Codex-plan #7).
  - **pure `_combine_masks`:** four ops' truth tables; `within` clips each; `subtract(a,b)` ==
    `_combine_masks("and", a, _combine_masks("not", b), None)` cross-check; missing `b` raises;
    all-zero result returned.
  - **`_align`:** 2D→3D broadcast identical per plane, `broadcast_z=True`; `broadcast_2d_to_3d
    =False` raises; incompatible shape raises.
  - **tool (`viewer`):** `subtract(specimen,green)` region == `not(green, within=specimen)`;
    **2D `a` + 3D `b`** → 3D output with the 3D reference scale (guards #1/#10); output feeds
    `measure_intensity` (single cross-tool feed — dropped the manders+segment dual, scope);
    T-axis / 4D layer → `ok:False`; scale-mismatch → warning **and** `scale_mismatch` in metadata;
    `not` with a `b_layer` → warned; missing `b` for `and` → `ok:False`; empty result →
    `ok:True, empty:True`.

**Gate:** `pytest tests/test_tools_masks.py -q`.

### Commit 4 — `partition_inside_outside` + guard band + headline recipe

- `src/imajin/tools/masks.py`:
  - `_partition(region_aligned, within_aligned, *, broadcast_z, spacing, buffer_um) ->
    (labels int32, stats)` (headless):
    - `bounded = region_aligned & within_aligned`.
    - **guard band** (`buffer_um > 0` and `spacing` present): morphology on `bounded`, at the
      **right dimensionality** — when `broadcast_z` (region constant across Z), operate on the 2D
      YX plane (`bounded[0]`) with `erode_binary_um`/`dilate_binary_um` then re-broadcast across Z
      (never erode along Z, Codex-plan #4); when natively 3D, full-3D morphology. Then
      `inside = erode(bounded)`, `outside = within_aligned & ~dilate(region_aligned_bounded)`.
      When `buffer_um <= 0` or `spacing` missing: `inside = bounded`,
      `outside = within_aligned & ~region_aligned`; missing-spacing-with-buffer → warning +
      buffer skipped.
    - `labels = 1*inside + 2*(outside & ~inside)` (disjoint by construction; `& ~inside` defensive).
    - stats: `inside_voxels`, `outside_voxels`,
      `region_clipped_fraction = (region_aligned & ~within_aligned).sum() / max(region_aligned.sum(),1)`
      (raw region vs within — independent of morphology, Codex-plan #3).
  - `@tool(phase="7", worker=True) def partition_inside_outside(region_layer, within_layer=None,
    boundary_buffer_um=0.0, allow_full_frame_outside=False, broadcast_2d_to_3d=True, name=None)`
    (**`within_layer` is Optional** so the schema/dock allow the full-frame opt-in, Codex-plan #5):
    1. `within_layer` falsy and not `allow_full_frame_outside` → `{ok:False,error}` ("outside needs
       a specimen bound; pass within_layer or allow_full_frame_outside=True").
    2. snapshot + axes-guard region (+within). reference = max-ndim of {region, within};
       `target_shape`/`scale`/`spacing`(via `voxel_spacing`) from the reference; `_align` region and
       within to it (`broadcast_z = any`).
    3. full-frame opt-in: `within_aligned = ones(target_shape, bool)` + loud background warning.
    4. `labels, stats = _partition(...)`.
    5. `comparable = inside_voxels>0 and outside_voxels>0`; not comparable → warn "inside/outside
       comparison impossible" (Codex-plan #7-severity). all-zero labels → `{ok:False,error}`.
       `region_clipped_fraction > 0.2` → strong warning (misregistration / wrong layer).
    6. `layer = call_on_main(add_labels_from_worker, labels, name=name or
       f"{region_layer}_partition", scale=ref.scale, metadata={source_layer:region_layer,
       source_path, label_names:{1:"inside",2:"outside"}, within_used, boundary_buffer_um,
       region_clipped_fraction, comparable, broadcast_z})`.
    7. return `{ok, partition_layer: layer.name, inside_voxels, outside_voxels, within_used,
       region_clipped_fraction, boundary_buffer_um, comparable, broadcast_z, label_names, warnings}`.
- Tests (`tests/test_tools_masks.py`):
  - **pure `_partition`:** disjoint inside/outside; guard band removes an edge annulus from both;
    **broadcast guard band keeps top/bottom Z planes** (the #4 regression — 2D-broadcast region +
    buffer, assert every Z plane identical and non-empty); `region_clipped_fraction` correct when
    region spills past within; empty inside / empty outside → `comparable=False`.
  - **tool + recipe (`viewer`), 2D and the 3D-within case:** green blob + red inside&outside within
    a specimen ROI → `segment_intensity_regions("green")` → `partition_inside_outside(region,
    specimen)` → `measure_intensity(partition,["red"])`: two rows with `region∈{inside,outside}`,
    inside red mean > outside, `log2(inside/outside) > 0`, physical columns present, inside∪outside
    ⊆ specimen. Plus a **2D region + 3D within + 3D red** end-to-end (output scale is the 3D one).
  - `within_layer` omitted → error; `allow_full_frame_outside=True` → warned full-frame outside.
  - `boundary_buffer_um>0` shrinks inside / pushes outside off the edge; **negative buffer → off**;
    **buffer with missing spacing → warn + skipped**.
  - `broadcast_2d_to_3d=False` on a 2D region vs 3D within → error (no silent full-frame).
  - empty-inside and empty-outside partitions still measure (1 row) with the warning.
  - duplicate output name → returned `partition_layer` is the actual (deduped) layer name.

**Gate:** `pytest tests/test_tools_masks.py -q` then the **full suite** `pytest -q`.

### Commit 5 — docs pointer (small)

- Append a short "Channel-as-mask (inside vs outside)" recipe to `README.md`: the 4-step recipe +
  the **paired-stats caveat** (per-sample `log2(inside/outside)`, replicate-level test; **not**
  `compare_groups(group_col="region")`). No code.

**Gate:** none (docs).

## Verification (beyond unit tests, the `verify` step)

Run the headline recipe end-to-end in a headless session on a synthetic 2-channel image; print the
inside/outside table + `log2(inside/outside)` and assert the sign and the `region` column — this
exercises segment→partition→measure exactly as a user would.

## Risks / mitigations

- **read-only broadcast view + `&`/`~`:** boolean ops allocate fresh arrays (verified); a test
  asserts a 2D→3D combine succeeds and is per-plane identical.
- **3D erosion erasing broadcast Z planes:** guard-band morphology runs in 2D-then-broadcast when
  `broadcast_z`; explicit top/bottom-plane test.
- **2D-region + 3D-within scale bug:** reference = max-ndim input, output scale from reference;
  dedicated end-to-end test asserts the 3D scale and physical columns.
- **`label_names` malformed / partial:** guarded `int(k)` + `.map`→NA, row-preserving; tests cover.
- **scale mismatch:** warning + metadata (not fatal); test asserts both.

## Out of scope (per spec)

Auto specimen detection; per-object inside/outside classification; a paired-test mode in
`compare_groups` (follow-up); Z-varying set ops; thresholding inside the logic tools; gray-level
"raw channel" heuristic and `translate` propagation (deferred — no snapshot API, unreliable).
