# Boundary Mask from Hand-Drawn Shapes — Implementation Plan

Follows the spec `docs/superpowers/specs/2026-06-26-boundary-mask-from-shapes-design.md`.
This plan: rev.1 (revised after 1 Codex review; changelog at bottom).

Branch: `feat/boundary-mask-from-shapes` (already created off `master`). Commit-by-commit, each with
its test gate. Run `.venv/bin/python -m pytest <files> -q` per commit (never bare `pytest`).

## Facts the implementation relies on (probe-verified, napari 0.7.0)

- `napari.layers.Shapes(data, shape_type=[...])` and `napari.layers.Image(...)` construct **standalone
  (no viewer/qapp)**; `.data` is a `list` of `(N,D)` arrays, `.shape_type` a `list` of lowercase strs.
- `Shapes(...).to_labels(labels_shape=(Y,X))` fills a **rotated** ellipse as an ellipse (bbox corners
  background, centre foreground) — napari's own geometry. Standalone layers expose
  `data_to_world()` / `world_to_data()`.
- Default `viewer` fixture is a **`_FakeViewer`** under offscreen (`conftest.py:5,78`): it has
  `add_image/add_labels` but **no `add_shapes`, no real `Shapes`, no transforms**. `_FakeLayerList`
  is a real `list` subclass with name-keyed lookup — so a **real standalone napari layer can be
  `viewer.layers.append(...)`-ed** and found by `get_layer(name)`. (Same gap `test_review_dock.py`
  handles with its own `_make_real_napari_viewer`.)
- `call_on_main` (`qt_dispatch.py:16`) runs inline when no dispatcher (tests/scripts).

## napari imports stay lazy (Codex P3)

`boundary.py` must **not** import `napari` at module top — only inside functions — so
`import imajin.tools` and `python -c "import imajin.tools; get_tool(...)"` stay headless-safe.
Module-top imports: `from __future__ import annotations`, `numpy as np`, `typing.Any`,
`from imajin.tools.registry import tool`, `from imajin.agent.qt_dispatch import call_on_main`,
`from imajin.session import get_layer, get_viewer`,
`from imajin.analysis.arrays import layer_axes_from_metadata, materialize_array`.

---

## Commit 1 — pure helpers (no viewer, no live layers)

`src/imajin/tools/boundary.py` helper section:

```python
_AREA = {"polygon", "rectangle", "ellipse"}
_SKIP = {"line", "path"}

def _clean_shape(v) -> np.ndarray | None:
    """Whole-shape validation: drop a shape with any NaN/Inf or < 3 finite verts.
    (Do NOT delete individual rows — that corrupts rectangle/ellipse control points.)"""
    a = np.asarray(v, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or not np.isfinite(a).all() or a.shape[0] < 3:
        return None
    return a

def rasterize_shapes_yx(yx_polys, area_types, target_yx) -> np.ndarray:
    """Union bool mask (target_yx) via napari's own rasteriser. Filters (poly,type) TOGETHER."""
    from napari.layers import Shapes  # lazy
    Y, X = target_yx
    pairs = [(p, t) for p, t in zip(yx_polys, area_types) if p is not None and len(p) >= 3]
    if not pairs:
        return np.zeros((Y, X), dtype=bool)
    polys, types = zip(*pairs)
    lab = Shapes(list(polys), shape_type=list(types)).to_labels(labels_shape=(Y, X))
    return np.asarray(lab) > 0

def broadcast_yx_to_ref(mask2d, ref_shape) -> tuple[np.ndarray, bool]:
    if len(ref_shape) == 3:
        return np.broadcast_to(mask2d[None], (ref_shape[0], *mask2d.shape)).astype(np.int32), True
    return mask2d.astype(np.int32), False
```

`tests/test_boundary_mask.py` (no viewer; build real `napari.layers.Shapes` and feed `.data`/`.shape_type`):

- **rotated ellipse, rotation-sensitive (Codex #3, P2-test):** 30°, non-square ellipse. Assert (a) 4
  bbox corners `False`, centre `True`, area `< bbox`; AND (b) a point a short step along the **rotated
  major axis** is `True` while the same step along the **axis-aligned** major axis is `False` — so an
  unrotated ellipse would fail.
- rectangle (4 corners) → quad filled; polygon → interior filled.
- **`area_types` pairing bug:** pass `[good_poly, NaN_poly, good_poly]` with types
  `["polygon","ellipse","rectangle"]` through `rasterize_shapes_yx`; the NaN one drops and the
  remaining masks still match their own types (no shift), union has both good shapes.
- `line`/`path` excluded upstream (tested at tool level); border shape → `mask.shape == (Y,X)`, no spill.
- `broadcast_yx_to_ref`: 2D→`(False, int32)`; 3D `(4,Y,X)`→ every plane equals `mask2d`, `True`.

**Gate:** `.venv/bin/python -m pytest tests/test_boundary_mask.py -q` green.

---

## Commit 2 — core + tool + registration

`src/imajin/tools/boundary.py`:

`_apply_transform(tf, coords)` — call `tf(coords)`; on TypeError/ValueError fall back to row-wise
`np.array([tf(c) for c in coords])`. Used for **both** `data_to_world` and `world_to_data` (Codex).

`_build_boundary_mask_from_layers(s, ref, name) -> dict` (pure of the viewer; takes layer objects):

1. `isinstance(s, napari.layers.Shapes)` (lazy import) else `{ok: False, error}`.
2. Validate `ref`: `data = ref.data`; reject if no `.ndim`/`.shape` or multiscale list/tuple
   (`{ok: False}`). `ndim = data.ndim`; `axes = layer_axes_from_metadata(ref.metadata, ndim)`.
   **Reject `axes not in {"YX","ZYX"}`** (covers `T`, 4D, and permuted axes — Codex) with the
   extract_timepoint-style message.
3. Per shape: `t = str(shape_type[i]).lower()`. Unknown (∉ `_AREA ∪ _SKIP`) → `{ok: False}`. `_SKIP`
   → counted in `skipped_shape_types`. `_AREA` → `_clean_shape`; require vertex `D == ndim`
   (a 2D Shapes over a 3D ref is rejected with a "draw on the 3D image / extract a slice" message —
   the real hand-drawn 3D layer carries the Z coord, so `D == ndim`).
4. Coord-convert each kept shape: `world = _apply_transform(s.data_to_world, V)`;
   `refc = _apply_transform(ref.world_to_data, world)`; `yx = np.asarray(refc)[:, -2:]`.
5. `mask2d = rasterize_shapes_yx(yx_list, types, (Y, X))`;
   `mask, broadcast_z = broadcast_yx_to_ref(mask2d, data.shape)`.
6. `mask.sum() == 0` → `{ok: False, error: "<no area shapes | off-image | degenerate>",
   skipped_shape_types, n_shapes}` (Codex #12 — never a silent empty boundary).
7. Build `info = {ok: True, mask, n_shapes:int, n_used:int, skipped_shape_types,
   mask_voxels:int(mask.sum()), mask_fraction:float(mask.sum()/mask.size), axes,
   broadcast_z:bool, warnings:[...]}` — all numpy scalars cast to plain int/float (Codex). Warn if
   `mask_fraction > 0.98`; append the Z-broadcast note when `broadcast_z`.

`@tool(... llm=True, worker=False)` `boundary_mask_from_shapes(shapes_layer, reference_layer, name=None)`:
wraps everything in one `call_on_main` closure:
```python
def _run():
    s = get_layer(shapes_layer); ref = get_layer(reference_layer)
    info = _build_boundary_mask_from_layers(s, ref, name)
    if not info["ok"]:
        return info
    mask = info.pop("mask")
    out = name or f"{reference_layer}_boundary"
    layer = get_viewer().add_labels(mask, name=out, scale=tuple(ref.scale),
                                    metadata={"source_shapes_layer": shapes_layer,
                                              "reference_layer": reference_layer, **{k:info[k] for k in
                                              ("axes","broadcast_z","mask_voxels","mask_fraction",
                                               "skipped_shape_types")}})
    tr = getattr(ref, "translate", None)
    if tr is not None:
        try: layer.translate = tr
        except Exception: pass
    info["boundary_layer"] = layer.name   # actual name (dup-rename safe — Codex)
    return info
return call_on_main(_run)
```

`src/imajin/tools/__init__.py`: add `from imajin.tools import boundary  # noqa: F401, E402`.

`tests/test_tools_boundary.py` — construct **real** standalone napari layers, inject into the default
fake `viewer` via `viewer.layers.append(...)`:

- **keeps-only-inside + wiring (pin seg knobs — Codex):** real `Image` with a bright blob inside and
  one outside a polygon; append `Image`+`Shapes`; `boundary_mask_from_shapes("shp","img")`; then
  `segment.segment_target_objects("img", boundary_mask=res["boundary_layer"], background_radius=0,
  smoothing_sigma=0, min_size=20, save_qc_png=False)`; assert inside `>0`, outside `==0`,
  `threshold_scope == "boundary_mask"`, `res["ok"]`.
- **scale/translate exactness (Codex #9):** `Image(img, scale=(0.3,0.3), translate=(5,2))` AND a
  `Shapes` layer at a **different** (default) transform; assert the mask's foreground bounding box /
  centroid equals the intended image-index region (exact, not "blob is nonzero"); assert the output
  layer's `translate == ref.translate`, and `res["boundary_layer"] == layer.name`.
- **ellipse + rectangle via real Shapes:** `Shapes([...], shape_type=["ellipse"])` injected → tool ok,
  ellipse boundary excludes bbox corners.
- **3D (real 3D Shapes):** `Image(zyx)`, `Shapes([(z,y,x)...], ["polygon"])` with constant `z`; assert
  `mask.shape == ref.shape`, every Z plane equals the 2D mask, a non-`z` plane is non-empty,
  `res["broadcast_z"] is True`.
- **empty / line-only / non-Shapes input:** empty Shapes, a `line`-only Shapes, and an Image passed as
  `shapes_layer` → `res["ok"] is False` (distinct error messages).

**Gate:** `.venv/bin/python -m pytest tests/test_boundary_mask.py tests/test_tools_boundary.py -q` green.

---

## Commit 3 — Gap C: ManualDock dropdown + docs

`src/imajin/ui/manual_dock.py` `_layer_param_names`: explicit allowlist
`elif p in {"image_a", "image_b", "mask", "boundary_mask"}:` (not a broad `*_mask` suffix — Codex #14).

`tests/test_manual_dock.py`: add a direct unit test —
`_layer_param_names(segment.segment_target_objects.__wrapped__ or .func)` returns a set containing
`"image_layer"` and `"boundary_mask"` and **not** `"min_snr"` (scalar). (The dock calls it on
`entry.func`; assert on the same callable the dock uses.)

`README.md` (segmentation section): one-liner workflow — *draw a Shapes polygon/rectangle/ellipse →
`boundary_mask_from_shapes(shapes_layer, reference_layer)` → `segment_*(..., boundary_mask=<that layer>)`*.

**Gate:** `.venv/bin/python -m pytest tests/test_manual_dock.py tests/test_boundary_mask.py
tests/test_tools_boundary.py tests/test_tools_segment.py -q` green.

## Verification before "done"

1. The combined gate above is green; report pass/fail counts.
2. Headless registry sanity (no napari): `.venv/bin/python -c "import imajin.tools;
   from imajin.tools.registry import get_tool; print(get_tool('boundary_mask_from_shapes').name)"`.

## Residual risks (carried)

- napari `to_labels` geometry is version-dependent — re-checked by the real-Shapes tests on the
  `uv.lock`-pinned version; a bump that changes fills fails them loudly.
- Only scale+translate transforms handled; permuting-affine references are rejected (axes guard), not
  silently mis-rasterised.

## Changelog — design plan → rev.1 (accepted Codex plan-review findings)

- Bugs fixed in-spec: filter `(poly,type)` together; return actual `layer.name` (dup-rename); row-wise
  fallback for **both** transforms; NaN/Inf → drop whole shape; reject `axes ∉ {YX,ZYX}`; validate
  array-like reference; cast numpy scalars; **lazy napari imports**.
- Tests hardened: rotation-sensitive ellipse; exact scale/translate bounds incl. a transformed Shapes
  layer; real 3D Shapes with `mask.shape==ref.shape` + all-planes-equal; mixed valid/invalid through
  `rasterize_shapes_yx`; pinned segmentation knobs; assert translate + returned name; real napari
  layers injected into the fake viewer (no `add_shapes` dependency).
- Clarified (Codex assumption corrected): a 3D hand-drawn layer is `D==3` (Z carried), so `D==ndim`
  holds; the 3D test uses a real 3D Shapes layer and 2D-over-3D is rejected with a helpful message.
