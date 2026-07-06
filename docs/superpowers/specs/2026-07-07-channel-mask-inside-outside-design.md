# Channel-as-Mask: Inside/Outside Signal Analysis — Design

Status: design (revised after one Codex review; ready to plan)
Date: 2026-07-07

> **Refinements during plan review (see the plan doc).** Three details below were sharpened
> when the implementation plan was reviewed against the real repo API: (a) the **reference
> layer is the highest-`ndim` input** (not always `a_layer`) so a 2D region + 3D `within`
> outputs at the 3D shape/scale; (b) **`translate` is not propagated** — `snapshot_layer` /
> `add_labels_from_worker` carry only scale (we follow the coloc/measure precedent); (c) the
> **gray-level "raw channel" warning is dropped for v1** — without the layer kind it can't be
> told from a many-label segmentation and would false-positive. The masks-in contract
> (`foreground = data > 0`) is documented instead.

## Problem

A user wants to **use one channel as a mask to scope analysis of another channel**, and
compare the two sides. The motivating case: red and green channels — measure the **red
signal inside the green domain** vs. the **red signal outside the green domain**, then
compare.

The "inside" half already works end-to-end today:

- Turn green into a mask: `segment_intensity_regions("green")` (`segment/intensity.py:30`)
  or `segment_expression_domain("green")` (`segment/domain.py:45`) → a Labels layer.
- Measure red inside that mask: `measure_intensity(labels_layer="green_regions",
  image_layers=["red"])` (`measure.py:243`) → per-region regionprops (mean/max/min/area).
- Or restrict a coloc/correlation metric to the mask:
  `manders_coefficients("red","green", mask="green_regions")` (`coloc.py:38`),
  `pearson_correlation(..., mask=...)` (`coloc.py:88`).

Every mask parameter accepts **any** Labels/Image layer by name: `resolve_boundary`
(`_segmentation_io.py:93`) → `resolve_boundary_mask` (`segmentation.py:185`) materialises
the layer's array as `> 0`, with automatic 2D→3D Z-broadcast; coloc uses
`snapshot_layer(mask).data > 0`. So a segmented channel plugs straight in.

**The gap is the "outside" half.** Across all 101 registered tools there is **no**
mask invert / complement / set-difference operation (verified: `iter_tools()` grouped by
`invert|complement|logic|subtract|difference|xor` → empty). Every mechanism restricts to
*inside* a mask. There is no supported way to produce "outside green," and therefore no
way to compare the two sides.

A second, quieter gap: a **naive** complement (`~green`) is the entire image background,
so a red-intensity mean "outside green" computed against the whole frame is dominated by
black voxels and is scientifically meaningless. "Outside" almost always means *outside the
domain but still inside the specimen/tissue* — the complement must be **bounded** by a
region the user supplies.

## Goal

Two small, composable tools that (a) supply the missing boolean-mask primitive and (b)
make the specific inside-vs-outside comparison a short recipe over the **existing**,
tested measurement tools — with the statistics done correctly (paired, per biological
replicate), not as a naive two-group test.

```
green ─segment→ green_mask ┐
                           ├─ partition_inside_outside(region=green_mask, within=specimen)
specimen (drawn/segmented) ┘        → partition Labels {1=inside, 2=outside∖region}
                                    → measure_intensity(partition, ["red"])   # 2 rows: inside, outside
                                    → per-sample contrast log2(inside/outside)
                                    → across replicates: paired/one-sample test of the contrast
```

`mask_logic` is the general primitive; `partition_inside_outside` is the *opinionated,
safe-by-default* workflow wrapper (bounded outside required; guard band optional).

## Verified facts (current repo)

- `resolve_boundary_mask(raw, target_shape)` (`segmentation.py:185`) returns a bool mask
  aligned to `target_shape`: exact-shape match → writable bool; 2D `(Y,X)` vs 3D `(Z,Y,X)`
  → **read-only broadcast view** across Z; any other combo → `ValueError`. Reusable as the
  shape-alignment primitive; boolean ops (`a & b`) allocate fresh arrays, so the read-only
  view is safe to combine.
- `measure_intensity` runs `regionprops_table` per label; a `{0,1,2}` label map yields
  **exactly two rows** (labels 1 and 2; 0 is background), each with
  `mean/max/min_intensity_<red>`, `area`, `centroid`, and physical columns
  (`area_um2`/`volume_um3`) from the layer `scale`. Confirmed by the existing pattern in
  `tests/test_tools_measure.py:33-50` (label map {1,2} + image `ch_red` → `mean_intensity_ch_red[1|2]`).
- `compare_groups` (`stats.py:424`) supports `test ∈ {ttest, welch, mannwhitney, anova,
  kruskal}` — **no paired / signed-rank test.** So inside/outside from one image must NOT be
  fed to it as two independent groups (see Statistics below).
- ManualDock renders a param as a **layer dropdown** iff it is `layer`, `*_layer`, or one of
  `{image_a, image_b, mask, boundary_mask}` (`ui/manual_dock.py:19-27`). Naming every new
  layer param `*_layer` → **no manual_dock change needed**.
- `layer_axes_from_metadata(md, ndim)` (`analysis/arrays.py`) is how `boundary.py` and the
  segment tools derive/guard axes; reused here to reject T/4D/nonstandard-axis inputs.
- `_dilate_binary_um(mask, spacing, radius_um)` (`analysis/segmentation.py`) exists for the
  optional guard band; skimage `binary_erosion` supplies the erode side.

## New tool 1 — `mask_logic` (the missing primitive)

```python
@tool(phase="7", worker=True)
def mask_logic(
    op: str,                          # not | and | or | subtract
    a_layer: str,                     # primary mask (Labels or binary Image; >0 = foreground)
    b_layer: str | None = None,       # second mask; required for and/or/subtract
    within_layer: str | None = None,  # optional bound: result is intersected with this
    broadcast_2d_to_3d: bool = True,  # allow a 2D mask to apply across a 3D stack's Z
    name: str | None = None,
) -> dict[str, Any]: ...
```

Set rules (element-wise, `foreground = layer > 0`), stated directly:

| `op` | result (before `within`) | needs `b_layer` |
| --- | --- | --- |
| `not` | `~a` | no |
| `and` | `a & b` | yes |
| `or` | `a \| b` | yes |
| `subtract` | `a & ~b` | yes |

- `within_layer` given → final result `& within`. This is what makes `not` usable:
  `mask_logic("not","green", within_layer="specimen")` = `specimen & ~green` (bounded
  outside). `subtract("specimen","green")` is the same region, spelled more explicitly.
- **`xor` dropped from v1** (rare in bioimage; add later if needed).
- **Shape / axes:** every input must be a 2D `YX` or 3D `ZYX` **spatial** layer
  (`layer_axes_from_metadata`); a `T`/4D/channel-first/nonstandard-axis layer → `ok=False`
  with the segment-tool-style message ("extract a timepoint/slice first"). Target shape =
  `a_layer`; `b`/`within` are aligned via `resolve_boundary_mask`. A 2D→3D broadcast is
  allowed only when `broadcast_2d_to_3d=True` (default; matches `boundary_mask_from_shapes`);
  set it False to force exact-shape matches. Incompatible shapes → `ok=False`.
- **Scale/translate:** if inputs disagree on `scale` (or `translate`), the combination may be
  physically meaningless (same array shape ≠ same physical grid). Emit a prominent warning and
  record `scale_mismatch` in metadata — **not** a hard error, because a hand-built mask
  legitimately carrying `scale=(1,…)` against a µm-scaled image is a real, recoverable case.
- **Input sanity:** if `a`/`b` is an **Image** layer with many distinct nonzero gray levels
  (heuristic: non-integer dtype, or > 256 unique nonzero values), warn that it is being
  binarised at `>0` and the user likely wants to threshold/segment first. Inputs are intended
  to be binary/label masks.
- **Output:** an `int32` **Labels** layer, `1`/`0`, same shape + `scale` + `translate` as
  `a_layer`. Feeds `measure_intensity`, `manders_coefficients(mask=…)`,
  `segment_*(boundary_mask=…)` unchanged. `source_layer`/`source_path` chain to `a_layer` so
  `advance_to_file` unloads it.
- **Empty policy:** all-zero result → `ok=True`, `empty=True`, warning. An empty set is a
  legitimate set-algebra answer (e.g. red fully inside green). Downstream tools decide severity.
- **Returns:** `{ok, op, mask_layer, voxels, fraction, empty, broadcast_z, scale_mismatch,
  axes, warnings}`; rejected input → `{ok: False, error}`.

## New tool 2 — `partition_inside_outside` (opinionated workflow wrapper)

```python
@tool(phase="7", worker=True)
def partition_inside_outside(
    region_layer: str,                    # the "inside" domain (e.g. segmented green)
    within_layer: str,                    # REQUIRED specimen/tissue bound for a valid "outside"
    boundary_buffer_um: float = 0.0,      # exclude an ambiguous guard band at the region edge
    allow_full_frame_outside: bool = False,  # opt in to within-less, background-dominated outside
    broadcast_2d_to_3d: bool = True,
    name: str | None = None,
) -> dict[str, Any]: ...
```

Produces **one** `int32` Labels layer, ready for a single `measure_intensity` call. Set rules
(disjoint by construction — no label-precedence needed):

- label **1 = inside** = `region & within`.
- label **2 = outside** = `within & ~region`.
- If `within_layer` is omitted, the call **fails** unless `allow_full_frame_outside=True`,
  in which case `outside = ~region` over the full frame and a **loud warning** is attached
  ("outside includes all background; intensity means will be near-zero/misleading"). This is
  never the default path (Codex #2).
- **Guard band (`boundary_buffer_um` > 0):** erode `inside` and dilate `region` by the radius,
  so `outside = within & ~region_dilated` and `inside = erode(region) & within`. The
  ambiguous PSF/bleed-through/chromatic-shift boundary band belongs to neither region (Codex
  bioimage pitfall). Reuses `_dilate_binary_um` + skimage `binary_erosion`; needs `scale`.
- **Clipping report (Codex #8):** if `region` extends beyond `within`,
  `region_clipped_fraction = (region ∖ within).sum() / region.sum()`. A large fraction
  (> 0.2) raises a strong warning — likely misregistration, wrong layer, or a bad specimen mask.
- **Comparability (Codex #7):** `comparable = inside_voxels > 0 and outside_voxels > 0`. If
  either side is empty, warn that the inside/outside comparison is impossible (not merely that
  one region is absent). Hard-error only if the whole map is 0.
- **Label semantics for measurement:** the layer's metadata carries
  `label_names = {1: "inside", 2: "outside"}`. `measure_intensity` is taught to read this and
  emit a `region` column automatically (see below), so the recipe needs no fragile post-step.
- **Returns:** `{ok, partition_layer, inside_voxels, outside_voxels, within_used,
  region_clipped_fraction, boundary_buffer_um, comparable, broadcast_z, label_names, warnings}`.

**This measures red inside the green *domain*, not per green object** (Codex #6). All green
sub-regions collapse into one "inside." Per-object (per-cell) inside/outside classification is
a separate future feature (non-goal below).

## Small enhancement — `measure_intensity` emits a `region` column

`measure_intensity` (and its worker path) reads optional `label_names: dict[int,str]` from the
labels layer metadata; when present, it adds a `region` column mapping each row's `label` →
name. General (any categorical labels layer benefits), ~5 lines, no signature change, existing
callers unaffected (absent metadata → no new column). Preferred over a caller-side post-step
(Codex open-Q #2).

## Statistics — the comparison must be paired (Codex #1)

Inside and outside come from the **same image**; they are paired, not independent samples.
Feeding them to `compare_groups(group_col="region")` as two groups is **statistically invalid**
(and `compare_groups` has no paired test anyway). Correct workflow, documented in the tool
result and report:

1. **Per image:** `measure_intensity` → inside & outside red means. Form the paired **contrast**
   `log2(mean_inside / mean_outside)` (or `inside − outside`). One number per sample.
2. **Across biological replicates:** collect the per-sample contrast; test it against 0 with a
   **one-sample / paired** test (e.g. Wilcoxon signed-rank or one-sample t on the log-ratio).
   The biological sample — not pixels, voxels, labels, or the two partition rows — is the
   replicate.

v1 produces the correct paired per-sample quantities (inside, outside, ratio) and **explicitly
warns against** the naive two-group test; it does not add a new statistical test. A paired-test
mode for `compare_groups` (signed-rank / one-sample) is a clean, separate follow-up (noted for
the plan, not built here).

## Scientific caveats (surfaced in tool warnings + report text)

- **Background:** interpret inside/outside means only after background correction; even bounded
  "outside" can include empty tissue, holes, autofluorescence, uneven illumination, saturation.
- **No circularity:** segment green with settings fixed across the batch; do **not** tune the
  green threshold using the red signal — that makes the comparison circular.
- **Boundary ambiguity:** PSF blur / bleed-through / chromatic shift blur the domain edge; use
  `boundary_buffer_um` to exclude a guard band for a cleaner contrast.
- **Physical units under Z-broadcast:** a 2D ROI broadcast across Z yields a *cylinder*, so
  `volume_um3` "outside/inside" is an extrusion, not the true 3D specimen volume — flagged by
  `broadcast_z` + warning.

## Non-goals (v1)

- **Auto specimen/tissue detection.** `within` is user-supplied (drawn ROI via
  `boundary_mask_from_shapes`, or a segmented counterstain/tissue channel).
- **Per-cell inside/outside classification + per-cell stats.** v1 is domain-level.
- **Paired statistical test in `compare_groups`.** Documented; separate follow-up.
- **Z-varying set ops.** A 2D mask broadcasts across Z (toggle + warning); genuinely
  per-plane logic is out of scope.
- **Thresholding inside the logic tools.** Masks in, mask out; channel→mask stays the segment
  tools' job.

## Files touched

| File | Change |
| --- | --- |
| `src/imajin/tools/masks.py` (new) | `mask_logic`, `partition_inside_outside`, headless-testable cores `_combine_masks(op,a,b,within)` / `_partition(region, within, buffer, spacing)` on plain arrays; axes/scale guards |
| `src/imajin/tools/measure.py` | read optional `label_names` from labels metadata → `region` column (worker path too) |
| `src/imajin/tools/__init__.py` | `from imajin.tools import masks  # noqa` |
| `tests/test_tools_masks.py` (new) | pure-core unit tests + tool tests via `viewer` fixture, incl. the full green→partition→`measure_intensity` recipe and the paired-contrast sanity |
| `tests/test_tools_measure.py` | add: `label_names` metadata → `region` column present/absent |

No `manual_dock.py` change (all new params end in `_layer`). No change to coloc/segment — the
outputs feed their existing mask/labels inputs.

## Test plan

- **Pure core (`_combine_masks`, plain numpy):**
  - `not/and/or/subtract` truth tables on hand-built 2D masks.
  - `within` clips every op (result ⊆ within); `not`+`within` = `within & ~a`;
    `subtract(a,b)` == `and(a, not(b))` on random masks (algebraic cross-check).
  - 2D `b`/`within` vs 3D `a` broadcasts across Z (identical per plane) when allowed; disallowed
    (`broadcast_2d_to_3d=False`) or otherwise-incompatible → `ValueError`.
  - all-zero result returned (not raised), `empty=True`.
- **Pure core (`_partition`):** inside = `region & within`; outside = `within & ~region`
  (disjoint); guard band removes an annulus from both; `region_clipped_fraction` correct when
  region spills past within; empty inside / empty outside → `comparable=False`; full-frame
  outside only when explicitly allowed.
- **`measure_intensity` `region` column:** labels layer with `label_names={1:"inside",
  2:"outside"}` → table has a `region` column with the right per-row names; without the
  metadata → no `region` column (back-compat).
- **Tool (`viewer` fixture):**
  - `mask_logic("subtract","specimen","green")` region == `mask_logic("not","green",
    within_layer="specimen")`; output is a Labels layer with right `scale`/`translate`; feeds
    `manders_coefficients(mask=…)` and `segment_target_objects(boundary_mask=…)`.
  - **Headline recipe (paired):** synthetic red/green — green blob with red both inside and
    outside it, all within a specimen ROI. `segment_intensity_regions` →
    `partition_inside_outside` → `measure_intensity(["red"])` gives two rows (`region` in/out);
    assert inside red mean > outside red mean, physical columns present, inside∪outside ⊆
    specimen, and `log2(inside/outside) > 0`.
  - `within_layer` omitted → error unless `allow_full_frame_outside=True` (then warned).
  - `boundary_buffer_um>0` shrinks inside and pushes outside away from the edge (guard band
    belongs to neither).
  - 2D partition over a 3D stack broadcasts across Z; `broadcast_z` True + warned.
  - scale-mismatched inputs → warning + `scale_mismatch` recorded.

## Changelog — design → revised (accepted Codex findings)

- **#1 (stats):** rewrote the statistics section — paired per-sample contrast + replicate-level
  test; explicit warning against the naive `compare_groups(group_col="region")` two-group test;
  noted `compare_groups` has no paired mode (future follow-up).
- **#2 (bounded outside):** `within_layer` is **required**; full-frame outside is an explicit
  `allow_full_frame_outside=True` opt-in.
- **#3 (Z-broadcast):** kept default broadcast (repo convention — sibling
  `boundary_mask_from_shapes`) but added a `broadcast_2d_to_3d` toggle and a `broadcast_z`
  warning naming the cylinder/volume consequence. *Rejected* forcing opt-in (would diverge from
  the established MIP-draw workflow).
- **#4/#5 (physical alignment / axes):** axes guard (2D YX / 3D ZYX only; reject T/4D/nonstd);
  scale/translate-mismatch warning + metadata.
- **#6 (domain not per-object):** documented explicitly; per-object deferred.
- **#7 (empty severity):** partition `comparable` flag + "comparison impossible" warning;
  `mask_logic` empty stays `ok=True`/`empty=True`.
- **#8 (clipping visibility):** `region_clipped_fraction` with a strong warning past 0.2.
- **#9 (set rule):** stated the disjoint rule directly; dropped "precedence" language.
- **#10 (>0 leaky):** input-sanity warning when an Image layer with many gray levels is
  binarised; documented masks-in contract.
- **Bioimage buffer zone:** added optional `boundary_buffer_um` guard band.
- **Open-Q2 (region column):** `measure_intensity` reads `label_names` metadata (general, small)
  instead of a fragile caller post-step.
- **Open-Q3 (xor):** dropped from v1.
