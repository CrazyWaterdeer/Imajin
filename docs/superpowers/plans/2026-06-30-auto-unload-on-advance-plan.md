# Advance-to-Next-File + current-file pointer — Implementation Plan

Follows `docs/superpowers/specs/2026-06-30-auto-unload-on-advance-design.md` (revised after 1 Codex
review). Branch `feat/advance-to-file` off master, commit-by-commit, `.venv/bin/python -m pytest <files>
-q` gates. `load_file` and the batch runner are NOT changed (the side effect lives in a new tool).

Facts: `files.py` already has `get_viewer`, `_layer_source_path`, `_canonical_path_text`,
`_layer_names_for_source_path`, `_remove_layers_by_name`. Derived layers carry `metadata.source_layer`
(parent name) and propagating ones carry `source_path`.

## Commit 1 — `advance_to_file` + guarded layer-tree + boundary source_path

`tools/files.py`:

```python
def _file_layer_tree(canonical_path: str) -> list[str]:
    viewer = get_viewer()
    layers = list(viewer.layers)
    own = {L.name: _layer_source_path(L) for L in layers}
    tree = {n for n, sp in own.items() if sp == canonical_path}
    changed = True
    while changed:
        changed = False
        for L in layers:
            if L.name in tree:
                continue
            sp = own[L.name]
            if sp is not None and sp != canonical_path:
                continue  # belongs to a DIFFERENT file -> never cross trees (Codex #1)
            parent = (getattr(L, "metadata", {}) or {}).get("source_layer")
            if parent in tree:
                tree.add(L.name); changed = True
    return [L.name for L in layers if L.name in tree]  # viewer order

@tool(description="Finish the current file and load the next one: unload the layers of the "
      "currently-loaded ANALYSED file(s) to free memory, then load `path`. Use this to step "
      "through a multi-file batch one file at a time. An unanalysed loaded file is kept unless "
      "force_unload=True. Plain load_file does not unload anything.", phase="1", llm=True)
def advance_to_file(path: str, force_unload: bool = False) -> dict[str, Any]:
    new_canon = _canonical_path_text(str(normalize_user_path(path).resolve()))
    loaded = {sp for sp in (_layer_source_path(L) for L in get_viewer().layers
                            if getattr(L, "kind", "") == "image" or ... ) if sp}
    leaving = loaded - {new_canon}
    complete = _complete_file_paths()   # canonical paths of complete runs (path- or file_id-keyed)
    unloaded_layers, unloaded_files, kept, warnings = [], [], [], []
    for f in sorted(leaving):
        if f in complete or force_unload:
            names = _file_layer_tree(f)
            unloaded_layers += _remove_layers_by_name(names); unloaded_files.append(f)
        else:
            kept.append(f); warnings.append(f"{f} is loaded but not analysed; not unloaded "
                                            "(force_unload=True to discard)")
    load_result = _load_file(new_canon, force_reload=False)
    return {"loaded": new_canon, "load_result": load_result, "unloaded_files": unloaded_files,
            "unloaded_layers": unloaded_layers, "kept_unanalysed": kept, "warnings": warnings}
```

`_complete_file_paths()` helper: `set()` of canonical paths for every `complete` `AnalysisRun`
(`list_runs()`), where `file_id` that is already a path -> canonical; a registered `file_id` ->
`canonical(get_file(file_id).path)` (Codex #2: recognise batch/registered runs too).

Detecting "image" layers: use `getattr(L, "kind", None)` / the napari layer type; for the leaving set,
include any layer with a source_path (so multi-channel + derived count their file). Group by source_path.

`tools/boundary.py`: in `boundary_mask_from_shapes`, add `"source_path": <reference layer's
source_path>` to the output Labels metadata so the (image-sized) boundary mask is part of the reference
file's tree.

`tests/test_tools_files.py` (real napari layers injected into the fake viewer, like test_tools_boundary):
- A: add an Image with `metadata.source_path="/d/A.lsm"`, a labels layer with the same source_path, a
  MIP layer with `metadata.source_layer=<A image name>` (no source_path), and a boundary mask with
  source_path=A. Put a **complete** run for "/d/A.lsm". `advance_to_file("/d/B.lsm")` (B image also
  injected/loadable via a stub) -> all four A layers removed; B present.
- A layer with `source_path="/d/other.lsm"` that chains `source_layer` off A's image is **kept**.
- Unanalysed loaded file C (source_path, no complete run) -> kept + warning; `force_unload=True` removes it.
- multi-channel A (two images same source_path) -> both removed.

(For the load of B without a real file, monkeypatch `files._load_file` to a stub that returns a dict and
adds a B image layer, so the test exercises the unload + the advance contract without disk I/O.)

**Gate:** `pytest tests/test_tools_files.py -q`.

## Commit 2 — current-file pointer + prompt

`agent/context.py` `batch_progress_data`: add `current`:
- try `get_viewer()`; for loaded **image** layers, group `source_path`; `current_files = those not in
  complete set`. `current = None` (0), one label (1), or `{"multiple": [labels]}` (>1). Any
  viewer/exception -> `current = None` (headless-safe, Codex #8). Never raise.
- `summarize_batch_progress`: add a `current: <label>` / `current: (none)` line from that.

`agent/prompts.py`: the `advance_to_file` rule from the spec.

`tests/test_batch_progress.py`: `current` is `None` with no viewer; with a `viewer` fixture, a loaded
image whose file has no complete run shows as `current`; a loaded image whose file IS complete does not.

**Gate:** `pytest tests/test_batch_progress.py tests/test_tools_files.py -q`, then full `pytest -q`.

## Verification before done

1. Full suite green; report counts.
2. Confirm `load_file` and `batch_runner` are byte-unchanged (no auto-unload regression in batch tests).

## Risks (carried)

- Guarded source_layer closure may still miss a derivative propagating neither source_path nor
  source_layer; residual is small; uniform `root_source_path` is a noted follow-up.
- `advance_to_file` only recognises files analysed via a complete run; an unanalysed-but-loaded file is
  always kept unless `force_unload` (safe default).

## Changelog — plan -> rev.1 (accepted Codex plan-review findings)

- **P0 #1 `_complete_file_paths`:** for each `complete` run's `file_id` — `try get_file(fid) ->
  canonical(path)` (registered), `except KeyError`: if it is path-like (`/` or `\`) -> canonicalise,
  else ignore. A layer-name-keyed historical run must never raise.
- **P0 #2 loaded set:** drop kind detection; `loaded = { _layer_source_path(L) for L in
  viewer.layers if _layer_source_path(L) }` (labels share the image source_path; grouping by
  source_path collapses multi-channel).
- **P0 #3 boundary/MIP:** `boundary_mask_from_shapes` adds **`source_layer = reference_layer`** (plus
  `source_path` when the reference has one) so a boundary drawn on a MIP is caught via the
  source_layer chain (MIP -> image), not only the direct source_path. Test the MIP case.
- **P1 #5 metadata:** read it safely — `md = getattr(L, "metadata", None); parent = md.get(
  "source_layer") if isinstance(md, dict) else None`.
- **P1 #6 ordering:** `unloaded_layers` is removal order; tests compare **sets**.
- **P1 #7 current visibility:** `summarize_batch_progress` returns non-`None` when `current` exists even
  if analysed/pending/failed are all empty.
- **Tests #8/#9/#10/#12:** add a registered-`file_id`->path complete-run unload case; a non-path
  `file_id="some_layer"` complete run that must be **ignored** (no crash); the `_load_file` stub takes
  `(path, *, force_reload)`; build the ownership cases with **fake-viewer layers** (`add_image(...,
  metadata={"source_path"/"source_layer": ...})`) so they never skip on missing napari.
- **#14 imports:** `context.py` reads `source_path` from layer metadata **inline** (no
  `import imajin.tools.files`, which would register all tools during agent-context import); use
  `viewer_or_none()`.
- **#16 prompt:** describe `advance_to_file` as **manual one-at-a-time stepping** (for the
  ROI-per-file / too-big-to-batch case), explicitly distinct from `run_recipe_on_samples` (uniform
  batch), so the agent does not confuse the two.
