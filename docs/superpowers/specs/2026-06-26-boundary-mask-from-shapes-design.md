# Boundary Mask from Hand-Drawn Shapes — Design

Status: design (revised after one Codex review; ready to plan)
Date: 2026-06-26

## Problem

A user wants to **draw a shape by hand and have segmentation find ROIs only inside it.**
The backend already supports this: `segment_target_objects` / `auto_segment_target` /
`segment_3d_cells_auto` take a `boundary_mask: str` layer, and `target_pipeline.threshold_and_label`
correctly scopes the threshold to the mask, restricts the binary with `& boundary_mask`, and
intersects the final labels (`intersect_labels_with_mask`). Tested in `tests/test_tools_segment.py:726`.

The gap is the **input bridge**. `boundary_mask` requires a Labels/Image layer whose pixel array
matches the target image shape (`segment.py:709-717`). A hand-drawn napari **Shapes** layer
(polygon/rectangle/ellipse) is a list of *vertex arrays*, not a pixel mask, so passing it errors out
(ragged-array build / shape mismatch). The polygon rasteriser `_rasterize_polygon` (`review_dock.py:555`)
is wired only to ROI add/remove-regions correction **and rasterises every shape as a polygon — so a
napari ellipse (its 4 bbox corners) becomes a rectangle.** No `@tool` turns a drawing into a boundary
mask, and there is no 2D→3D broadcast for Z-stacks.

## Goal

One small, single-responsibility tool that converts a hand-drawn Shapes layer into a boundary
**Labels** layer matching a reference image, which then feeds the existing, tested `boundary_mask`
path unchanged.

`draw Shapes → boundary_mask_from_shapes(...) → segment_*(..., boundary_mask=<that layer>)`

## Verified facts (headless napari 0.7.0 probe)

- `Shapes.data` is a `list` of `(N, D)` vertex arrays; `Shapes.shape_type` is a `list` of lowercase
  strings (`polygon`/`rectangle`/`ellipse`/`line`/`path`).
- **`Shapes.to_labels(labels_shape=(Y, X))` and `to_masks(mask_shape=(Y, X))` work headlessly and
  fill a rotated, non-square ellipse correctly** — its 4 bbox corners are background, its centre is
  foreground (area 281 px vs bbox 360 px). This is napari's *own* rendering geometry, so it matches
  what the user drew. We use it instead of hand-rolled skimage ellipse/rotation math.
- `Image` and `Shapes` layers both expose `data_to_world()` / `world_to_data()`; an Image with
  `scale=(0.3,0.3), translate=(5,2)` round-trips data(10,10)→world(8,5)→data(10,10). We convert
  shape vertices through world space so a Shapes layer with a different transform than the image does
  not produce a silently offset/scaled mask.

## Non-goals (v1)

- Z-limited boundaries (a shape applying to only some Z planes). v1 broadcasts one YX region across
  the whole stack — "draw once, constrain the stack" — made **explicit** in metadata + a warning.
- Invert/exclude; subtraction/holes from nested shapes (multiple shapes are unioned only — documented).
- `line`/`path` (open curves, no interior) — skipped with a warning; never inferred-closed.
- A Labels/Image ndarray input ("I painted a mask"): out of scope for *this* tool; a painted Labels
  layer that already matches the stack dims can be passed straight to `boundary_mask`.
- Auto-running segmentation. The tool only produces the mask layer; the user/agent calls segment next.
- Axis-permuting/affine reference transforms — assume YX / ZYX aligned references (same assumption the
  segment tools already make); scale+translate are handled, a permuting affine is not (documented).

## New tool

```python
@tool(phase="2", worker=False, vision_hint=False)   # main-thread GUI tool, like review_target_roi
def boundary_mask_from_shapes(
    shapes_layer: str,        # the napari Shapes layer the user drew on
    reference_layer: str,     # image/labels layer whose shape+scale the mask must match
    name: str | None = None,  # output layer name; default f"{reference_layer}_boundary"
) -> dict[str, Any]: ...
```

- Both params end in `_layer`, so the ManualDock already renders them as **layer dropdowns**.
- Output: an `int32` Labels layer, `1` inside the union of drawn area-shapes, `0` outside, same shape,
  `scale`, and `translate` as `reference_layer`.
- Returns: `{ok, boundary_layer, n_shapes, n_used, skipped_shape_types, mask_voxels, mask_fraction,
  axes, broadcast_z, warnings}`; on failure `{ok: False, error, ...}`.

## Algorithm (all on the main thread)

1. Resolve `shapes_layer`; require it to be a napari `Shapes` instance (else `ok=False`, clear error).
   Read `data` (list of vertex arrays) and normalise `shape_type` to a lowercase `str` per shape.
2. Resolve `reference_layer`; read `data.shape`, `scale`, `translate`, transforms, and axes
   (`layer_axes_from_metadata`). Reject time-series/4D (`"T" in axes`, same message style as the
   segment tools — `extract_timepoint` first). Accept 2D `YX` and 3D `ZYX`.
3. Validate dimensionality: every shape's vertex `D` must equal the reference `ndim` (2 or 3). A
   mismatch (e.g. a 2D Shapes layer over a 3D image) is a `ok=False` fail-loud, not a guess.
4. Partition shapes by normalised type: **area** = {`polygon`, `rectangle`, `ellipse`}; **skip** =
   {`line`, `path`} (counted into `skipped_shape_types`, warned). **Unknown type → `ok=False`**
   (do not silently skip). Drop vertices containing NaN/Inf; drop shapes left with `< 3` vertices.
5. Coordinate convert each area shape: `world = shapes_layer.data_to_world(V)`;
   `ref = reference_layer.world_to_data(world)`; take `yx = ref[:, -2:]`. (Row-wise if the transform
   is not vectorised.) This yields vertices in the reference's **YX index space**.
6. Rasterise via napari: build a temporary 2D `napari.layers.Shapes(yx_polys, shape_type=area_types)`
   and call `.to_labels(labels_shape=(Y, X))`; `mask2d = result > 0` (union of all area shapes).
   napari clips out-of-image geometry, so partially/fully out-of-bounds shapes need no special-casing.
7. If reference is 3D ZYX, `mask = np.broadcast_to(mask2d[None], (Z, Y, X))` materialised to int32 and
   set `broadcast_z=True`; else `mask = mask2d.astype(int32)`.
8. **Empty policy:** if `mask` has no foreground voxel → `ok=False` with an error naming the cause
   (no area shapes / all shapes off-image / degenerate). Never emit a usable-looking all-zero layer.
9. `add_labels_from_worker(mask, name=name or f"{reference_layer}_boundary", scale=ref.scale,
   metadata={...})`; mirror `translate` onto the new layer so the boundary overlays the image. The
   warning list includes the Z-broadcast note when `broadcast_z`.

## Coordinate handling (Codex #9)

Vertices are converted **shapes-data → world → reference-data** using each layer's own transform, so a
boundary is correct even when the Shapes layer's transform differs from the image's `scale`/`translate`
(the common case — a freshly-added Shapes layer defaults to identity while a µm-scaled image does not).
Rasterisation then happens in the reference's integer index space, so there is no scale/offset drift.
A reference whose affine *permutes* axes is out of scope (documented non-goal) — refs are YX/ZYX.

## Secondary fix — Gap C (ManualDock dropdown for `boundary_mask`)

`_layer_param_names` (`ui/manual_dock.py:19-27`) recognises `layer`, `*_layer`, and
`{image_a, image_b, mask}`, so the `boundary_mask` param renders as a free-text box. Fix with an
**explicit allowlist** — add `"boundary_mask"` to that set — **not** a broad `endswith("_mask")` rule
(which could capture a future non-layer `*_mask` param). Audit confirms `boundary_mask` is today the
only `*_mask` *tool* param (`secondary_outline_mask` lives only in a private non-`@tool` helper).

## Files touched

| File | Change |
| --- | --- |
| `src/imajin/tools/boundary.py` (new) | `boundary_mask_from_shapes` tool + headless-testable `_rasterize_shapes_yx(yx_polys, area_types, (Y,X))` (builds a temp napari Shapes, `to_labels`) + YX→ZYX broadcast helper |
| `src/imajin/tools/__init__.py` | `from imajin.tools import boundary  # noqa` (registration) |
| `src/imajin/ui/manual_dock.py` | add `"boundary_mask"` to the layer-param allowlist |
| `tests/test_boundary_mask.py` (new) | pure rasteriser + broadcast unit tests using real `napari.layers.Shapes` data |
| `tests/test_tools_boundary.py` (new) | tool test via the `viewer` fixture (incl. scaled/translated image) → mask layer → `segment_target_objects(boundary_mask=...)` keeps only inside |

`review_dock._rasterize_polygon` is **left untouched** (its add/remove-regions path is unaffected and
already tested); no shared refactor in v1 to keep blast radius small.

## Test plan

- **Pure (`_rasterize_shapes_yx`, real napari Shapes data):**
  - rotated non-square **ellipse fills an ellipse, not its bbox** — assert the 4 bbox corners are 0 and
    the centre is 1 (the exact property the probe verified).
  - rectangle (4 corners) fills the quad; polygon fills its interior.
  - `line`/`path` excluded (skipped + warned), unknown type → error.
  - union of two disjoint shapes; degenerate (NaN/Inf, <3 verts) dropped without raising.
  - small shape at the image border — assert no off-by-one spill outside `(Y,X)`.
  - YX→ZYX broadcast yields the identical mask on every Z plane.
- **Tool (`viewer` fixture):**
  - Image with bright blobs inside and outside a drawn polygon → `boundary_mask_from_shapes` →
    `segment_target_objects(boundary_mask=...)`; assert inside-blob kept, outside-blob dropped, and
    `threshold_scope == "boundary_mask"` (mirrors the existing boundary test).
  - **Scale/translate correctness:** image with `scale=(0.3,0.3)`, `translate=(5,2)` and a Shapes layer
    at default transform; assert the mask lands on the intended pixels (guards Codex #9), not offset.
  - **3D:** ZYX reference, polygon on one slice → mask non-empty on every Z; segmentation constrained
    on all planes; `broadcast_z` True and warned.
  - Empty Shapes layer / only a `line` → `ok=False`.
  - Real `viewer.add_shapes(..., shape_type="ellipse")` and `"rectangle"` round-trips (Codex #15) — the
    tool rasterises the actual `layer.data`, catching napari's true vertex/`shape_type` representation.

## Residual risks

- **napari `to_labels` per-version geometry**: verified on 0.7.0 (the installed version, pinned via
  `uv.lock`); the real-Shapes tests re-verify on whatever version CI runs, so a future bump can't
  silently change fills.
- **Coordinate edge cases**: only scale+translate are handled; an axis-permuting affine reference is a
  documented non-goal and shares the segment tools' existing YX/ZYX assumption.
- **Scope creep** into z-aware / exclude / hole boundaries — explicitly deferred.

## Changelog — design → revised (accepted Codex findings)

- **#3/#10 (ellipse + raster convention):** replaced hand-rolled skimage polygon/ellipse math with
  napari-native `Shapes.to_labels` (probe-verified rotated-ellipse correctness). Removed the fragile
  4-corner rotation recovery entirely.
- **#9 (coords):** added shapes-data→world→reference-data conversion; no longer assume identical
  transforms / raw index vertices.
- **#5 (threading):** tool is now `worker=False`, fully main-thread (matches `review_target_roi`).
- **#1/#2 (axes/slice):** validate shape ndim == reference ndim, fail loud; Z-broadcast made explicit.
- **#6/#7 (shape_type):** normalise to lowercase; area vs skip partition; unknown → fail loud; never
  infer a closed path.
- **#8 (ndarray fallback):** dropped from v1 scope.
- **#12 (empty):** unified — any all-zero result is `ok=False`.
- **#14 (Gap C):** explicit `boundary_mask` allowlist instead of a broad `*_mask` suffix.
- **#11/#15 (tests):** degenerate-shape defenses + real napari rectangle/ellipse round-trips added.
- **#13 (union/holes):** documented v1 limitation (no subtraction/holes).
