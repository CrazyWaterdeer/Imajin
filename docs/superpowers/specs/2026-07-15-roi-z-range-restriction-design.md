# Z-axis restriction for hand-drawn ROI analysis — Design

- **Date:** 2026-07-15
- **Status:** Approved (brainstorming) → Codex-reviewed (gpt-5.6-sol) + revised → pending user spec review
- **Author:** Jin (with Claude Code)
- **External review:** one read-only Codex (gpt-5.6-sol) pass folded in — the
  "zero downstream change / restricts all analysis" claim was over-broad and is now
  scoped per tool; the 2D-drawing-reference vs 3D-target-volume problem is resolved
  via the MIP's `source_layer` metadata (+ an explicit `target_layer`); interval
  intersection + index convention pinned; single-slice / `min_z_planes` and the
  `measure` (no `boundary_mask`) claims corrected; API dtype/keys/manual-dock
  decisions pinned; a pure `normalize_z_range` + shared Labels writer added.

## Summary

Today a hand-drawn ROI restricts analysis in **XY only**: the user draws a shape on
a 2D max-projection (MIP), `boundary_mask_from_shapes` rasterises it against that 2D
reference into a 2D `(Y, X)` mask, and the mask is **broadcast across every Z plane**
when segmentation resolves it. There is no way to also restrict along Z.

This adds a **contiguous Z-slice range** to that mechanism. The user names the slice
range in chat (e.g. "slices 5–15"); the drawn XY ROI then applies only within that Z
sub-range, and a sibling tool covers the no-ROI case. The mechanism is: **produce a
3D `(Z, Y, X)` boundary mask that is the rasterised XY region (or the full frame) on
`z0..z1` and zero elsewhere.** `resolve_boundary_mask` already accepts an exact-shape
3D mask, so this mask reaches the analysis tools through the existing plumbing.

**What the Z restriction actually guarantees (scoped per tool — this is the load-
bearing correction from review):**

- **`segment_target_objects` / `auto_segment_target`:** the boundary mask **gates
  detection** (it is applied during threshold→label), so excluded Z planes produce
  no labels. This is a *true* Z restriction of the segmentation result.
- **`detect_spots(mode="3d")`:** detection runs on the full volume; detected spot
  **centers are filtered** by the 3D mask, so only spots whose localized `(z, y, x)`
  lies in the domain survive. A good approximation; detection neighbourhoods still
  span all Z.
- **`detect_spots(mode="2d_projection")`:** detection runs on `raw.max(axis=0)` over
  **all** Z, then centers are filtered. Out-of-range signal can therefore still shape
  the projection. This mode gives only weak Z restriction; the agent should prefer
  `mode="3d"` for a Z-restricted spot analysis (documented, see Component 4).
- **Measurement:** `measure_intensity` has **no** `boundary_mask` parameter. The Z
  restriction reaches measurement only **transitively** — you measure the Labels that
  segmentation already restricted. Restricting *pre-existing* labels to a Z range is
  out of scope (would need a separate label∩range op).

## Goals

- Restrict a hand-drawn-ROI **segmentation** to a contiguous Z-slice range named in
  chat by slice number — no new drawing surface.
- Same mechanism restricts segmentation to a Z sub-range **without** a drawn ROI.
- Reuse the existing 3D-mask plumbing: no change to how `resolve_boundary_mask`,
  `segment_*`, or `measure_*` consume a boundary mask.
- Fully backward compatible: `z_range=None` behaves exactly as today.

## Non-goals (YAGNI)

- **No new drawing surface** (no orthogonal-view / per-slice drawing). Z is spoken,
  by slice number.
- **No µm / relative-position parsing in the tool** — the canonical parameter is
  slice indices; any depth/"top third" phrasing is mapped to indices by the agent.
- **No non-contiguous Z selection** — a single `[z0, z1]`.
- **No new detection-time cropping** for spots and **no domain-aware QC rework** in
  this iteration (see Risks — both are documented caveats / future work, not silent
  behaviour changes).
- **No `boundary_mask` on `measure_*`** and no label∩range op for existing labels.

## Decisions (from brainstorming + review)

| Question | Decision |
|---|---|
| What the Z restriction attaches to | **Both** — drawn XY ROI + Z range (primary), *and* a Z-only restriction with no ROI |
| How the Z range is expressed | **Slice numbers**; canonical parameter `z_range=(z0, z1)` |
| API shape | **Approach A**: `z_range` on `boundary_mask_from_shapes` **+** sibling `boundary_mask_from_z_range` |
| Index convention | **0-based, inclusive, napari Z-slider indices** — the agent maps human phrasing ("first/top 10" → `(0, 9)`; 1-based "slice 5" disambiguated against the slider) |
| Drawing vs target reference | `reference_layer` = drawing surface (2D MIP or 3D stack); the target Z count comes from an explicit `target_layer`, else the MIP's `source_layer` metadata, else error |
| Interval semantics | `used = [max(0, z0), min(Z-1, z1)]`; **error** if `z0 > z1` (inverted) or `used` is empty (whole request out of range) — never silently clamp a fully-out-of-range request onto one edge slice |
| Mask dtype | int32 for both tools (Labels layers are int; matches `broadcast_yx_to_ref`) |

## Architecture

`resolve_boundary_mask` returns an exact-shape 3D mask unchanged and only *broadcasts*
a raw 2D `(Y, X)` mask against a 3D target. So the feature is entirely about
*producing* a 3D `(Z, Y, X)` mask filled on `z0..z1`. Verified consumer behaviour:

- `segment_target_objects` computes its speed-crop footprint as
  `np.any(_boundary_raw > 0, axis=0)` (correct XY footprint of a partial-Z mask) and
  `boundary_bbox_slices` gives the **Z axis a full slice** — so the crop is XY-only and
  the partial-Z structure (zero planes) is preserved and suppresses detection there. ✔
- `auto_segment_target` passes the full 3D mask straight through. ✔
- `detect_spots` filters centers by a per-voxel 3D mask lookup (`bmask[tuple(vox.T)]`),
  so a center in an excluded plane is dropped — but see the Summary caveat about *where
  detection runs*.
- `project_boundary_outline_2d` uses `np.any(mask, axis=0)` for the **QC overlay only**
  → it shows the XY footprint and does not visualise the Z restriction (cosmetic).
- `masks.make_boundary_mask`: an exact-shape 3D mask returns a **writable** bool array;
  boolean ops downstream allocate fresh arrays, so nothing mutates the mask in place.

```
draw ROI on MIP (2D) ──► boundary_mask_from_shapes(shapes, reference=MIP,
                                                    z_range=(z0,z1))
                              │  YX rasterised on the MIP; Z count from the MIP's
                              │  source_layer (or explicit target_layer)
                              ▼
   normalize_z_range → fill_yx_to_z_range → 3D (Z,Y,X) int32 Labels layer
                              ▼
   resolve_boundary_mask (exact-shape, no broadcast)
                              ▼
   segment_target_objects / auto_segment_target  → labels only in XY∩[z0,z1]
                              ▼
   measure_intensity(labels)  → measured over the restricted labels (transitive)
```

## Components

### Component 1 — `normalize_z_range` (pure, shared)

`normalize_z_range(z_range, z_count) -> (z0, z1, note | None)` in `boundary.py`.

- `used = (max(0, z0), min(z_count - 1, z1))`.
- Error if `z0 > z1` ("z_range start > end").
- Error if `used[0] > used[1]` — the request lies entirely outside `[0, z_count-1]`
  (do **not** collapse it onto slice 0 or `Z-1`).
- If `used != (z0, z1)`, return a note ("requested Z 3–99 → used 3–40 of 41").
- 0-based inclusive throughout.

### Component 2 — `fill_yx_to_z_range` (pure, viewer-free)

`fill_yx_to_z_range(mask2d, z_count, z_range) -> mask3d int32`, beside
`broadcast_yx_to_ref`. Allocates `zeros((z_count, Y, X), int32)`, writes `mask2d` into
`out[z0:z1+1]` (after `normalize_z_range`), returns int32. Unit-testable with plain
arrays. The Z-only tool passes an all-ones `mask2d`.

### Component 3 — extend `boundary_mask_from_shapes`

New params: `z_range: tuple[int, int] | None = None`, `target_layer: str | None = None`.

- `z_range=None` → **unchanged** (rasterise, `broadcast_yx_to_ref`).
- `z_range` set:
  1. Rasterise the (2D or 3D) shape's YX against `reference_layer` exactly as today,
     yielding `mask2d` and the YX dims.
  2. Resolve `z_count` (and thus the Z axis) of the **target volume**:
     - `target_layer` if given (must be a 3D ZYX layer whose YX matches the reference);
     - else, if `reference_layer` is itself 3D ZYX, use it;
     - else, if `reference_layer` metadata has `source_layer` (a MIP), use that stack;
     - else error: "z_range needs the 3D stack; draw on its MIP (I resolve the source)
       or pass target_layer".
  3. Validate: target is 3D ZYX; target YX == reference YX (else error). Then
     `fill_yx_to_z_range(mask2d, z_count, z_range)`.
- Because the produced mask is a real 3D array, the "2D ROI broadcast across all Z"
  note is **not** emitted; a Z-restriction note ("ROI restricted to Z z0–z1 of Z") is
  added to `warnings` instead, plus any clamp note from `normalize_z_range`.

### Component 4 — sibling `boundary_mask_from_z_range`

`boundary_mask_from_z_range(reference_layer, z_range, name=None)` — the no-ROI case.
`reference_layer` must be a 3D ZYX stack (or a MIP that resolves to one via
`source_layer`); builds a full-frame `(Z, Y, X)` mask, all-True on `z0..z1`. Registered
`@tool(llm=True, manual=False)` (agent/verbal-driven; sidesteps the magicgui-tuple
question). Shares Component 1/2 and a common Labels-layer writer with
`boundary_mask_from_shapes`.

### Component 5 — agent prompt guidance

Extend the existing hand-drawn-ROI block in `agent/prompts.py`:

- When the user names a Z slice range ("slices 5–15", "the top 10 slices"), pass
  `z_range=(z0, z1)` to `boundary_mask_from_shapes` (draw on the MIP; the tool resolves
  the source stack). With no drawn ROI, use `boundary_mask_from_z_range`.
- Indices are **0-based** napari-slider indices; map "first/top N" → `(0, N-1)`; when a
  user says a 1-based ordinal, align it to the slider.
- For a **Z-restricted spot** analysis, prefer `detect_spots(mode="3d")` (the mask
  filters spot centers); note that `2d_projection` still projects over all Z.
- A **single-slice** range (`z0 == z1`) is a 2D question — run 2D segmentation
  (`segment_target_objects` in 2D), not `segment_3d_cells_auto` (whose
  `min_z_planes ≥ 2` default drops single-plane objects).

## Data flow

```
User: "이 영역 그리고 5~15 슬라이스만 세그멘테이션"   (ROI + Z range)
  └─ max_projection(stack)               # MIP carries source_layer=stack
  └─ user draws ROI on the MIP
  └─ boundary_mask_from_shapes(shapes, reference=MIP, z_range=(5, 15))
        → resolves stack via MIP.source_layer → 3D mask on Z[5..15]
  └─ segment_target_objects(image=stack, boundary_mask=…) → measure
        → labels (and measurements) restricted to XY-ROI ∩ Z[5..15]

User: "5~15 슬라이스만 세그멘테이션"   (no ROI)
  └─ boundary_mask_from_z_range(stack, z_range=(5, 15)) → 3D mask → segment → measure
```

## Edge cases & error handling

- **2D reference, no resolvable 3D target** → error (see Component 3 resolution order).
- **Interval** per Component 1: clamp both endpoints to `[0, Z-1]`; error on inverted or
  empty intersection (never silently onto an edge slice).
- **Single-slice `z0 == z1`** → valid mask; agent routes to 2D analysis (Component 5).
- **Reference YX ≠ target YX** → error (the drawing surface and stack must share XY).
- **> 3D input (CZYX / TZYX)** → out of scope; require `extract_timepoint` / a single
  channel first, matching the existing `dims="2d_or_3d"` / `ZYX` guards.
- **`z_range=None`** everywhere preserves today's behaviour exactly.

## Testing

Deterministic unit tests (helpers are viewer-free):

- `normalize_z_range`: clamps `(3, 99)`→`(3, Z-1)`+note; errors on `(10, 2)` and on a
  fully-out-of-range `(Z, Z+5)`; inclusive (`z1` kept).
- `fill_yx_to_z_range`: mask is zero on every slice outside `[z0, z1]`, equals the 2D
  input inside; dtype int32; single-slice fills exactly one plane.
- `boundary_mask_from_shapes(z_range=…)`: source resolved from MIP `source_layer`;
  explicit `target_layer` honoured; YX-mismatch errors; out-of-range Z slices sum to 0;
  in-range slices match the `z_range=None` rasterisation. Regression: `z_range=None`
  still broadcasts (existing tests pass).
- `boundary_mask_from_z_range`: full-frame True on `[z0, z1]`, False elsewhere; shape ==
  target; non-3D reference errors.
- Downstream: a 3D partial-Z mask passes `resolve_boundary_mask` unchanged and
  `segment_target_objects` yields labels only within `[z0, z1]` (assert no labels on an
  excluded plane).
- Writability (contract): the produced mask is writable (`flags.writeable`), so a
  consumer that ever writes will not hit the broadcast-view read-only trap.

## Risks & edge cases

- **Scoped guarantee, honestly documented.** Only segmentation labeling is a *true*
  Z gate. Spots (esp. `2d_projection`) and auto-selection QC see the full stack; the
  agent guidance steers spot users to `mode="3d"`, and QC/auto behaviour is a
  documented caveat below — not a silent change.
- **QC / auto-selection over the full stack.** `target_object_qc` and the
  auto-correct / auto-3D rankers compute `mask_fraction` and inside-vs-outside signal
  over the whole stack, so excluded Z planes count as "outside." For manual
  `segment_target_objects` this only affects **reported** QC numbers (labels are
  correct); for the **auto** paths it could shift the selected parameters/candidate.
  Making QC domain-aware (crop metrics/projection to `[z0, z1]`) is noted as future
  work; the QC outline likewise shows the XY footprint only.
- **Off-by-one / numbering.** 0-based inclusive is stated in both tool docstrings and
  the prompt; a test pins inclusivity and the agent disambiguates 1-based phrasing.
- **Silent empty result.** Inverted or fully-out-of-range requests error rather than
  producing an all-zero mask that segments nothing.
- **Mutability/warnings drift (doc-level).** `broadcast_yx_to_ref` already returns a
  writable int32 array; the read-only view arises only from `resolve_boundary_mask`
  broadcasting a raw 2D mask. The new masks are writable exact-shape arrays.

## Out of scope / future

- Orthogonal-view / per-slice ROI drawing.
- µm-depth or relative-position as first-class tool inputs.
- Non-contiguous Z selection.
- Detection-time Z cropping for spots; domain-aware QC/auto metrics.
- A `boundary_mask` (or label∩range) restriction applied to *pre-existing* labels.
