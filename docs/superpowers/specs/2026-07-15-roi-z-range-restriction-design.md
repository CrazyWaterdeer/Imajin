# Z-axis restriction for hand-drawn ROI analysis — Design

- **Date:** 2026-07-15
- **Status:** Approved (brainstorming) → pending Codex review + user spec review
- **Author:** Jin (with Claude Code)

## Summary

Today a hand-drawn ROI restricts analysis in **XY only**: the user draws a
polygon/rectangle/ellipse on a 2D (max-)projection, `boundary_mask_from_shapes`
rasterises it, and the resulting YX mask is **broadcast across every Z plane** of a
3D stack. There is no way to also restrict the analysis along Z.

This adds a **Z-slice range** to that mechanism. The user names a slice range in
chat (e.g. "slices 5–15", "the top 10 slices"); the drawn XY ROI then applies only
within that Z sub-range. A sibling tool covers the no-ROI case ("just analyse slices
5–15 of the whole frame"). Nothing downstream changes: the segmentation, spots, and
measurement tools already accept a full 3D `(Z, Y, X)` boundary mask, so a mask that
is filled only on the chosen Z slices flows through the existing pipeline unchanged.

## Goals

- A user can restrict a hand-drawn-ROI analysis to a contiguous Z-slice range by
  saying the slice numbers in chat — no new drawing surface required.
- The same mechanism restricts analysis to a Z sub-range **without** a drawn XY ROI.
- Zero change to the analysis tools (segment / spots / measure): the feature is
  entirely about *producing* the right boundary mask.
- Fully backward compatible: an existing `boundary_mask_from_shapes` call with no
  `z_range` behaves exactly as today (2D → broadcast across all Z).

## Non-goals (YAGNI)

- **No new drawing surface.** We do not add orthogonal-view (XZ/YZ) ROI drawing or
  per-slice hand drawing. Z is specified verbally, by slice number.
- **No µm / relative-position parsing in the tool.** The user chose slice-number
  expression; the canonical tool parameter is slice indices. (If the user ever
  phrases it as depth or "top third", the agent can translate to slice indices from
  metadata it already sees, but that is agent behaviour, not tool surface.)
- **No non-contiguous Z selection.** The range is a single contiguous `[z0, z1]`.
  Arbitrary slice sets are out of scope.
- **No change to the 2D-broadcast default**, the analysis tools, or the manual dock's
  existing ROI flow.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| What does the Z restriction attach to | **Both** — the drawn XY ROI + a Z range (primary), *and* a Z-only restriction with no ROI |
| How is the Z range expressed | **Slice numbers** (index range); the tool's canonical parameter is `(z0, z1)` |
| API shape | **Approach A**: extend `boundary_mask_from_shapes` with `z_range`, **plus a separate sibling tool** for the no-ROI case |
| Index convention | **0-based, inclusive** — matches napari's Z slider; the agent maps the user's phrasing to slider indices |

## Architecture

The insight is that `resolve_boundary_mask` (in `analysis/segmentation.py`) already
accepts an **exact-shape 3D mask** and only *broadcasts* when handed a 2D `(Y, X)`
mask against a 3D target. So the whole feature is: build a 3D `(Z, Y, X)` mask that
is the rasterised XY region (or the full frame) on `z0..z1` and zero elsewhere, and
hand it to the unchanged downstream. No consumer of `boundary_mask` changes.

```
boundary_mask_from_shapes(shapes, ref, z_range=(z0,z1))  ─┐
boundary_mask_from_z_range(ref, z_range=(z0,z1))          ─┤→ 3D (Z,Y,X) mask, filled
                                                           │   only on z0..z1
   (both call the viewer-free helper `fill_yx_to_z_range`) │
                                                           ▼
   existing Labels boundary layer → resolve_boundary_mask (exact-shape path,
   no broadcast) → segment_target_objects / spots / measure  (UNCHANGED)
```

## Components

### Component 1 — extend `boundary_mask_from_shapes`

File: `src/imajin/tools/boundary.py`. Add a parameter
`z_range: tuple[int, int] | None = None`.

- `z_range=None` → **unchanged** current behaviour: rasterise YX, then
  `broadcast_yx_to_ref` broadcasts across all Z (2D→3D read-only view for a 3D ref).
- `z_range=(z0, z1)` → rasterise YX, then `fill_yx_to_z_range` produces a 3D `(Z, Y,
  X)` int32 array with the YX region on slices `z0..z1` inclusive and zero elsewhere.
- The produced Labels layer is a genuine 3D array, so the existing
  `boundary_broadcast_warning` (which fires only when `boundary_raw.ndim == 2`) does
  **not** fire; instead the tool returns a note: `"ROI restricted to Z slices z0–z1
  of {Z}"`.

### Component 2 — new sibling tool `boundary_mask_from_z_range`

File: `src/imajin/tools/boundary.py`. Signature
`boundary_mask_from_z_range(reference: str, z_range: tuple[int, int], name: str | None
= None) -> dict[str, Any]`.

- Builds a full-frame `(Z, Y, X)` mask that is all-True on `z0..z1`, all-False
  elsewhere, and adds it as a Labels boundary layer, same as the shapes tool.
- Naming parallels `boundary_mask_from_shapes` so the two read as a pair. Registered
  the same way (`@tool`, assistant + manual as the existing boundary tool is).

### Component 3 — viewer-free helper `fill_yx_to_z_range`

File: `src/imajin/tools/boundary.py`, beside `broadcast_yx_to_ref`, matching the
existing "viewer-free core + thin tool wrapper" pattern so it is unit-testable with
plain arrays.

```
fill_yx_to_z_range(mask2d, ref_shape, z_range) -> (mask3d int32, note_or_None)
```

- Requires `len(ref_shape) == 3`. Validates/normalises `z_range` (see edge cases),
  writes `mask2d` into `out[z0:z1+1]`, returns the int32 array and an optional
  clamp/summary note. The Z-only tool passes an all-ones `mask2d` (full frame).

### Component 4 — agent prompt guidance

File: `src/imajin/agent/prompts.py`. Extend the existing hand-drawn-ROI guidance (the
`max_projection` → `boundary_mask_from_shapes` block) with one line: when the user
names a Z slice range ("slices 5–15", "the top 10 slices"), pass `z_range=(z0, z1)`
to `boundary_mask_from_shapes`; when there is no drawn ROI, use
`boundary_mask_from_z_range`. Indices are 0-based to match the napari Z slider. Per
the project's agent-guidance-over-hardcoding principle, the natural-language → slice
mapping lives here, not in the tool.

## Data flow

```
User: "이 영역 그리고 5~15 슬라이스만 분석"  (draw ROI + Z range)
  └─ (if a stack) max_projection so the ROI is drawn on the flat 2D projection
  └─ user draws the ROI on a Shapes layer
  └─ boundary_mask_from_shapes(shapes, reference, z_range=(5, 15)) → 3D mask
  └─ segment_target_objects(boundary_mask=…) → measure …           # unchanged
     result: objects inside (XY ROI ∩ Z[5..15]) only

User: "5~15 슬라이스만 분석"  (no ROI)
  └─ boundary_mask_from_z_range(reference, z_range=(5, 15)) → 3D mask
  └─ segment_target_objects(boundary_mask=…) → measure …
```

Behaviour is inherited from the existing `boundary_mask` semantics: detection is
suppressed where the mask is zero, so slices outside `[z0, z1]` are simply excluded.

## Edge cases & error handling

- **2D reference (no Z axis).** `z_range` on a 2D `(Y, X)` image is meaningless →
  error: "z_range needs a 3D (Z, Y, X) image; this layer is 2D".
- **Out-of-bounds range.** Clamp to `[0, Z-1]` and return a note (e.g. requested
  3–99 → used 3–40). If the clamped range is empty or `z0 > z1`, error with a clear
  message.
- **Reference dimensionality > 3 (T/C axes).** Out of scope — the existing broadcast
  path only handles 2D/3D; `z_range` likewise supports only a 3D `(Z, Y, X)`
  reference (the per-channel segmentation input). Anything else → error.
- **Interaction with the broadcast warning.** With `z_range` set, the mask is a real
  3D array, so the "2D ROI broadcast across all Z" note is not emitted; the
  Z-restriction note replaces it.
- **`z_range=None`** everywhere preserves today's behaviour exactly.

## Testing

Deterministic unit tests (no viewer needed for the helper):

- `fill_yx_to_z_range`: for a 3D ref, the returned mask is zero on every slice
  outside `[z0, z1]` and equals the 2D input on every slice inside; dtype int32.
- `boundary_mask_from_shapes(..., z_range=(z0,z1))`: mask sum over out-of-range Z is
  0; in-range slices match the `z_range=None` YX rasterisation of the same shapes.
- Regression: `boundary_mask_from_shapes(..., z_range=None)` still broadcasts across
  all Z (existing tests continue to pass).
- `boundary_mask_from_z_range`: full-frame True on `[z0, z1]`, False elsewhere; shape
  equals the reference; a 2D reference errors.
- Validation: 2D reference + `z_range` → error; out-of-bounds → clamp + note; `z0 >
  z1` / empty → error.
- Downstream acceptance: a 3D partial-Z mask passes `resolve_boundary_mask`'s
  exact-shape branch unchanged (already covered, plus an explicit assertion that
  `segment` / `measure` accept such a mask).

## Risks & edge cases

- **Off-by-one / index convention.** 0-based inclusive is the one convention to get
  right; the tool docstring and the agent guidance both state it, and the agent
  aligns the user's phrasing to the napari slider. A test pins inclusivity
  (`z1` slice is filled).
- **Silent empty result.** A clamped-to-empty or inverted range must error, not
  quietly produce an all-zero mask that segments nothing.
- **Manual-dock ergonomics.** `z_range` as a pair may render awkwardly in a magicgui
  form; if so, the manual surface can expose two integer fields (`z_start`, `z_end`)
  while the tool keeps the tuple — decided at plan time.

## Out of scope / future

- Orthogonal-view or per-slice ROI drawing.
- µm-depth or relative-position ("top third") as first-class tool inputs.
- Non-contiguous Z selection.
