# Anchor-Folder Bundles & Project Layer Removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop writing analysis results under `~/Documents/Imajin/results/`. Instead, each analysis emits a single self-contained `<timestamp>_<name>/` bundle inside the **anchor folder** (the lone input folder, or the alphabetically-first folder when the analysis spans multiple). The bundle's `metadata.json` is split into a reusable `recipe_params` block and a record-only `run_context` block so the user can re-apply technical settings to a new folder without dragging along sample-specific decisions. The `Imajin project` layer (`project.json` + sibling JSON files + autosave loop + UI menu) is removed; sessions are ephemeral, and continuity comes from pointing the agent at a prior bundle.

**Architecture:**
- **Anchor folder resolution** lives in a new pure function `imajin.results.resolve_anchor_folder()` that consults registered files in session state. Both single-image workflows and `BatchRecipeRunner._create_parent_bundle` route through it.
- **Bundle metadata schema** moves from a flat dict to three top-level blocks: `recipe_params` (portable), `run_context` (this-run only), `environment` (versions/git). Readers tolerate the old flat layout for one release.
- **Project layer removal** strips `_CURRENT_PROJECT` state, `autosave_current_project`, project tools, and the New/Open/Save Project UI menu. `imajin.project` becomes a thin compatibility shim that warns and no-ops while tests are migrated, then is deleted.
- **Recipe import from bundle** is a new tool `import_recipe_from_bundle(bundle_path, name=None)` that reads `recipe_params`, registers it as a session recipe, and returns the recipe name for downstream `run_recipe_on_samples`.

**Tech Stack:** Python 3.12, pytest, pathlib, pandas, tifffile, qtpy/Qt (no UI change forces a Qt test), `contextvars` for active-bundle context.

**Phasing & merge plan:**
- **Phase 1** (Tasks 1–6): Anchor folder routing. Self-contained. Fixes the immediate bug. Project layer untouched (still wins when a project is open). Mergeable.
- **Phase 2** (Tasks 7–11): Metadata schema split + recipe import tool. Builds on Phase 1. Mergeable.
- **Phase 3** (Tasks 12–17): Project layer removal. Deletes project.py, project tools, autosave, UI menu. Mergeable.

Each phase ends in a passing test suite and a green commit. Stop between phases to review before continuing.

---

## File map

### New files
- `src/imajin/anchor.py` — pure anchor-folder resolution from registered files + a loaded layer's file path. Small (<80 lines).
- `tests/test_anchor.py` — unit tests for anchor resolution.
- `src/imajin/tools/recipe_import.py` — `import_recipe_from_bundle` tool.
- `tests/test_recipe_import.py` — covers Phase 2's reuse tool.

### Modified files
- `src/imajin/results.py` — `results_root()` consults anchor before falling back; `create_result_bundle` accepts an optional explicit `root`.
- `src/imajin/result_bundles.py` — `finalize_bundle_metadata` learns to write the three-block schema; `read_bundle_metadata` adds a thin normalizer that exposes both shapes.
- `src/imajin/tools/workflows.py` — `_write_analysis_bundle_outputs` builds `run_context` (channel roles, sample annotations, scope, folder set) and `recipe_params` from the in-flight call.
- `src/imajin/tools/batch_runner.py` — `_create_parent_bundle` builds the same two blocks and passes anchor explicitly.
- `src/imajin/agent/state.py` — Phase 3: remove `_autosave_project`, drop autosave triggers.
- `src/imajin/agent/execution.py` — Phase 3: drop `autosave_current_project` import + call.
- `src/imajin/ui/main.py` — Phase 3: remove New/Open/Save Project menu actions.
- `src/imajin/agent/runner.py` — Phase 3: remove `create_project`/`load_project`/`save_project` from the tool allowlist.
- `src/imajin/agent/prompts.py` — Phase 2/3: mention bundle reuse pattern; drop project mentions where they remain.

### Deleted files (Phase 3)
- `src/imajin/project.py`
- `src/imajin/tools/project.py`
- `tests/test_project_persistence.py`

---

# Phase 1 — Anchor folder routing

Goal of this phase: a single-folder or multi-folder analysis writes its bundle to the anchor folder, while everything else (project layer, autosave, UI) is unchanged.

## Task 1: Pure `resolve_anchor_folder` helper

**Files:**
- Create: `src/imajin/anchor.py`
- Test: `tests/test_anchor.py`

**Design:**
```python
# anchor.py
from pathlib import Path
from typing import Iterable

def resolve_anchor_folder(file_paths: Iterable[str | Path]) -> Path | None:
    """Pick the anchor folder for a set of input file paths.

    Rule: the parent directory of each file is taken; the unique set is sorted
    case-insensitively, and the first entry wins. Returns None if the input is
    empty or every path is unparented.
    """
    parents: set[Path] = set()
    for p in file_paths:
        if not p:
            continue
        parent = Path(p).expanduser().resolve().parent
        parents.add(parent)
    if not parents:
        return None
    return sorted(parents, key=lambda p: str(p).lower())[0]
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_anchor.py
from pathlib import Path

from imajin.anchor import resolve_anchor_folder


def test_returns_none_for_empty_input():
    assert resolve_anchor_folder([]) is None


def test_single_file_returns_its_parent(tmp_path: Path):
    f = tmp_path / "a.lsm"
    f.write_bytes(b"")
    assert resolve_anchor_folder([f]) == tmp_path.resolve()


def test_multi_folder_returns_alphabetically_first(tmp_path: Path):
    a = tmp_path / "2026-05-09"
    b = tmp_path / "2026-05-10"
    a.mkdir()
    b.mkdir()
    (a / "x.lsm").write_bytes(b"")
    (b / "y.lsm").write_bytes(b"")
    anchor = resolve_anchor_folder([b / "y.lsm", a / "x.lsm"])
    assert anchor == a.resolve()


def test_case_insensitive_sort(tmp_path: Path):
    upper = tmp_path / "Zeta"
    lower = tmp_path / "alpha"
    upper.mkdir()
    lower.mkdir()
    (upper / "u.lsm").write_bytes(b"")
    (lower / "l.lsm").write_bytes(b"")
    anchor = resolve_anchor_folder([upper / "u.lsm", lower / "l.lsm"])
    assert anchor == lower.resolve()


def test_ignores_empty_strings():
    assert resolve_anchor_folder(["", None]) is None  # type: ignore[list-item]
```

- [ ] **Step 2: Run to verify failure**

```
uv run --project /home/jin/py314 pytest tests/test_anchor.py -v
```
Expected: ImportError — `imajin.anchor` does not exist yet.

- [ ] **Step 3: Implement the helper**

Create `src/imajin/anchor.py` with the body shown in **Design** above.

- [ ] **Step 4: Run tests; expect green**

```
uv run --project /home/jin/py314 pytest tests/test_anchor.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/anchor.py tests/test_anchor.py
git commit -m "feat(anchor): add resolve_anchor_folder helper"
```

## Task 2: Session-aware anchor resolver

**Files:**
- Modify: `src/imajin/anchor.py` (extend)
- Test: `tests/test_anchor.py` (extend)

**Design:** Wrap the pure helper with a session-aware function that pulls registered file paths from `imajin.agent.state.list_files()` and, as a tiebreaker for non-batch workflows, can accept an explicit `extra_paths` list (e.g. the file path behind the currently-active layer).

```python
def resolve_session_anchor(extra_paths: Iterable[str | Path] | None = None) -> Path | None:
    from imajin.agent.state import list_files

    paths: list[str | Path] = []
    for rec in list_files():
        path = rec.get("path") if isinstance(rec, dict) else getattr(rec, "path", None)
        if path:
            paths.append(path)
    if extra_paths:
        paths.extend(p for p in extra_paths if p)
    return resolve_anchor_folder(paths)
```

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_anchor.py
from unittest.mock import patch

from imajin.anchor import resolve_session_anchor


def test_session_anchor_uses_registered_files(tmp_path: Path):
    a = tmp_path / "alpha"
    a.mkdir()
    file_a = a / "x.lsm"
    file_a.write_bytes(b"")
    with patch("imajin.agent.state.list_files", return_value=[{"path": str(file_a)}]):
        assert resolve_session_anchor() == a.resolve()


def test_session_anchor_merges_extra_paths(tmp_path: Path):
    a = tmp_path / "alpha"
    b = tmp_path / "beta"
    a.mkdir()
    b.mkdir()
    (a / "x.lsm").write_bytes(b"")
    (b / "y.lsm").write_bytes(b"")
    with patch("imajin.agent.state.list_files", return_value=[{"path": str(b / "y.lsm")}]):
        anchor = resolve_session_anchor(extra_paths=[str(a / "x.lsm")])
    assert anchor == a.resolve()


def test_session_anchor_returns_none_when_no_files():
    with patch("imajin.agent.state.list_files", return_value=[]):
        assert resolve_session_anchor() is None
```

- [ ] **Step 2: Run to verify failure**

```
uv run --project /home/jin/py314 pytest tests/test_anchor.py -v
```
Expected: 3 new failures (ImportError of `resolve_session_anchor`).

- [ ] **Step 3: Implement**

Append the `resolve_session_anchor` function to `src/imajin/anchor.py`. Keep the import inside the function so `imajin.anchor` stays import-light.

- [ ] **Step 4: Tests pass**

```
uv run --project /home/jin/py314 pytest tests/test_anchor.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/anchor.py tests/test_anchor.py
git commit -m "feat(anchor): resolve session anchor from registered files"
```

## Task 3: `results_root()` consults anchor

**Files:**
- Modify: `src/imajin/results.py:104-113`
- Test: `tests/test_tools_results.py` (extend) and `tests/test_results_bundle.py` (verify untouched cases still pass)

**Design:** `results_root()` precedence becomes:
1. **Project open** → `project.path / "reports"` (Phase 1 keeps this — it disappears in Phase 3).
2. **Session anchor resolvable** → `<anchor>` (no `reports/` subdir — the user explicitly wants `<timestamp>_<name>/` directly under the data folder).
3. **Fallback** → `user_results_root()` (used only when running with no project AND no registered files; e.g. demo data, headless tests).

Note: when a project is open in Phase 1, the project still wins. Phase 3 removes that branch.

```python
def results_root() -> Path:
    try:
        from imajin.project import current_project
        project = current_project()
    except Exception:
        project = None
    if project is not None:
        return project.path / "reports"

    try:
        from imajin.anchor import resolve_session_anchor
        anchor = resolve_session_anchor()
    except Exception:
        anchor = None
    if anchor is not None:
        return anchor

    return user_results_root()
```

- [ ] **Step 1: Write failing test**

Add to `tests/test_tools_results.py`:

```python
def test_results_root_uses_session_anchor_when_no_project(tmp_path, monkeypatch):
    from imajin import results as _results

    folder = tmp_path / "2026-05-11"
    folder.mkdir()
    fake_file = folder / "img.lsm"
    fake_file.write_bytes(b"")

    monkeypatch.setattr(
        "imajin.project.current_project", lambda: None
    )
    monkeypatch.setattr(
        "imajin.agent.state.list_files",
        lambda: [{"path": str(fake_file)}],
    )
    assert _results.results_root() == folder.resolve()


def test_results_root_falls_back_to_user_root_when_no_anchor(tmp_path, monkeypatch):
    from imajin import results as _results

    monkeypatch.setattr("imajin.project.current_project", lambda: None)
    monkeypatch.setattr("imajin.agent.state.list_files", lambda: [])
    monkeypatch.setattr(
        _results, "user_results_root", lambda: tmp_path / "user_root"
    )
    assert _results.results_root() == tmp_path / "user_root"
```

- [ ] **Step 2: Verify failure**

```
uv run --project /home/jin/py314 pytest tests/test_tools_results.py -k "session_anchor or fallback" -v
```
Expected: anchor case fails (today's code returns `user_results_root` directly).

- [ ] **Step 3: Implement**

Replace `results_root()` in `src/imajin/results.py` with the body shown in **Design** above.

- [ ] **Step 4: All existing results/bundle tests still pass**

```
uv run --project /home/jin/py314 pytest tests/test_tools_results.py tests/test_results_bundle.py -v
```
Expected: all green. If any test relied on the absence of a project to land in `user_results_root`, fix it by clearing `list_files` via monkeypatch.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/results.py tests/test_tools_results.py
git commit -m "feat(results): route results to session anchor folder when no project is open"
```

## Task 4: `create_result_bundle` accepts explicit anchor

**Files:**
- Modify: `src/imajin/results.py:154-193`
- Test: `tests/test_tools_results.py`

**Design:** Allow callers (notably `BatchRecipeRunner`) to override anchor selection without round-tripping through global state. New signature:

```python
def create_result_bundle(
    name: str,
    *,
    kind: str = "single",
    tier: str | None = None,
    metadata: dict[str, Any] | None = None,
    root: Path | str | None = None,
) -> Path:
    ...
    base_root = Path(root) if root is not None else results_root()
    bundle = _unique_subdir(base_root, f"{timestamp}_{slugify_result_name(name)}")
```

Add a small helper:

```python
def _unique_subdir(root: Path, dirname: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = slugify_result_name(dirname)
    candidate = root / base
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = root / f"{base}_{i}"
        if not candidate.exists():
            return candidate
        i += 1
```

`unique_result_dir` keeps its existing behavior (it routes through `results_dir(category)` so it gets the `bundles/` subcategory when no anchor exists). When `root` is supplied we bypass the `bundles/` subdir — the bundle lives directly under the anchor folder.

- [ ] **Step 1: Write failing test**

```python
def test_create_result_bundle_uses_explicit_root(tmp_path):
    from imajin.results import create_result_bundle

    bundle = create_result_bundle("demo", root=tmp_path)
    assert bundle.parent == tmp_path
    assert bundle.name.endswith("_demo")
    assert (bundle / "metadata.json").exists()
```

- [ ] **Step 2: Verify failure**

```
uv run --project /home/jin/py314 pytest tests/test_tools_results.py -k explicit_root -v
```
Expected: TypeError (unexpected kw `root`).

- [ ] **Step 3: Implement** the new signature + `_unique_subdir`. Update the body so when `root` is None, behavior is unchanged: `unique_result_dir("bundles", ...)`.

- [ ] **Step 4: Tests pass; existing bundle tests still pass**

```
uv run --project /home/jin/py314 pytest tests/test_tools_results.py tests/test_results_bundle.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/imajin/results.py tests/test_tools_results.py
git commit -m "feat(results): allow create_result_bundle to target an explicit root"
```

## Task 5: Wire anchor into `BatchRecipeRunner._create_parent_bundle`

**Files:**
- Modify: `src/imajin/tools/batch_runner.py:270-288`
- Test: `tests/test_phase2_workflow.py` (extend with an anchor-routing case)

**Design:** Compute the anchor inside `_create_parent_bundle` from the runner's sample inputs (`resolve_sample_inputs` already returns `file_path` per sample). Pass it as `root=` so the bundle lives directly inside the anchor folder.

```python
def _create_parent_bundle(self) -> Any:
    from imajin.results import create_result_bundle
    from imajin.anchor import resolve_anchor_folder

    sample_paths = []
    for name in self.names:
        info = resolve_sample_inputs(name)
        if info.get("file_path"):
            sample_paths.append(info["file_path"])
    anchor = resolve_anchor_folder(sample_paths)

    return create_result_bundle(
        name=self.recipe.name,
        kind="batch",
        tier="two_tier" if self.domain_strategy is not None else "single_tier",
        metadata={
            "recipe": {
                "name": self.recipe.name,
                "target_channel": self.recipe.target_channel,
                "preprocessing": list(self.recipe.preprocessing or []),
                "segmentation": dict(self.recipe.segmentation or {}),
                "measurement": dict(self.recipe.measurement or {}),
                "domain": dict(self.recipe.domain) if self.recipe.domain else None,
                "cell_diameter_um": self.recipe.cell_diameter_um,
            },
        },
        root=anchor,
    )
```

- [ ] **Step 1: Write failing test**

In `tests/test_phase2_workflow.py` add (mirror existing setup; if there's a fixture that registers files + samples + recipe, reuse it):

```python
def test_batch_bundle_lands_in_anchor_folder(tmp_path, monkeypatch, ...):
    """When two folders are registered, the bundle must land in the alphabetically first."""
    folder_a = tmp_path / "2026-05-09"
    folder_b = tmp_path / "2026-05-10"
    folder_a.mkdir()
    folder_b.mkdir()
    file_a = folder_a / "img_a.tif"
    file_b = folder_b / "img_b.tif"
    # ... write minimal LSM/TIFF fixtures or monkeypatch loader
    # register both files, annotate two samples, create a trivial recipe, run.
    result = run_recipe_on_samples("recipe_name")
    bundle = Path(result["bundle_path"])
    assert bundle.parent == folder_a.resolve()
```

Use the existing batch-runner test fixtures if a real TIFF fixture is unwieldy; the assertion that matters is `bundle.parent == anchor`.

- [ ] **Step 2: Verify failure**

```
uv run --project /home/jin/py314 pytest tests/test_phase2_workflow.py -k anchor_folder -v
```
Expected: bundle still lands under `user_results_root` (test fails on `bundle.parent`).

- [ ] **Step 3: Implement** as in **Design**.

- [ ] **Step 4: All phase2 workflow tests pass**

```
uv run --project /home/jin/py314 pytest tests/test_phase2_workflow.py tests/test_phase3_experiment.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/batch_runner.py tests/test_phase2_workflow.py
git commit -m "feat(batch): route batch bundle to anchor folder of registered samples"
```

## Task 6: Wire anchor into single-image workflows

**Files:**
- Modify: `src/imajin/tools/workflows.py:78-90` (the `own_bundle` branch in `_write_analysis_bundle_outputs`)
- Test: `tests/test_phase2_workflow.py`

**Design:** When a single-image workflow (no parent bundle from a batch) creates its own bundle, derive the anchor from the source layer's file path if present, falling back to `resolve_session_anchor()` and ultimately to `user_results_root` via the unchanged default.

```python
if own_bundle:
    from imajin.anchor import resolve_session_anchor

    file_path = None
    try:
        from imajin.tools.napari_ops import snapshot_layer
        snap = snapshot_layer(target_layer)
        md = snap.metadata if isinstance(snap.metadata, dict) else {}
        file_path = md.get("path") or md.get("source_path")
    except Exception:
        file_path = None
    anchor = resolve_session_anchor(extra_paths=[file_path] if file_path else None)
    bundle_path = create_result_bundle(
        name=f"{target_layer}__{bundle_suffix}",
        kind="single",
        tier=tier,
        metadata={
            "recipe": None,
            "target_channel": target_layer,
            "target_source": target_source,
            "segmentation_method": segmentation_method,
            "analysis_dim": analysis_dim,
        },
        root=anchor,
    )
```

- [ ] **Step 1: Write failing test** mirroring Task 5 but for a single-image flow (`segment_target_objects` from a registered file).

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run workflow tests**

```
uv run --project /home/jin/py314 pytest tests/test_phase2_workflow.py tests/test_tools_segment.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase2_workflow.py
git commit -m "feat(workflows): single-image bundle inherits anchor folder"
```

**Phase 1 stop-and-review checkpoint.** Demo: run an analysis on a folder of LSMs without opening a project; confirm the bundle appears as `<folder>/<timestamp>_<name>/`.

---

# Phase 2 — Bundle metadata schema split + recipe reuse

Goal: separate the portable analysis recipe from the run-specific context, and add a tool that re-imports a previous bundle's recipe into the current session.

## Task 7: Define the new metadata shape

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

**Design:** Top-level keys become:

```jsonc
{
  "schema_version": 2,
  "recipe_params": {
    "name": "...", "target_channel": "...",
    "preprocessing": [...], "segmentation": {...},
    "measurement": {...}, "domain": {...} | null,
    "cell_diameter_um": null
  },
  "run_context": {
    "kind": "batch" | "single",
    "tier": "two_tier" | "single_tier",
    "name": "J20_1234_CaLexA",
    "status": "complete" | "failed" | "in_progress",
    "created_at": "...",
    "folder_set": ["<absolute folder>", ...],   // sorted, dedup'd
    "channel_roles": { "<channel>": "target" | "counterstain" | "ignore", ... },
    "scope_filters": [...],  // optional include= patterns recorded if any
    "samples": [ ... per-sample summaries ... ],
    "n_samples": N, "n_complete": N, "n_failed": N,
    "tables": { "combined": "tables/combined.csv" }
  },
  "environment": {
    "python_version": "...", "imajin_version": "...",
    "deps": { ... }, "git_commit": "..."
  }
}
```

`schema_version: 2` is the cue. Add a normalizer in `read_bundle_metadata` so old (flat) bundles still resolve to the same logical fields when the codebase reads them.

- [ ] **Step 1: Write failing test**

```python
def test_finalize_writes_schema_v2(tmp_path):
    from imajin.results import create_result_bundle
    from imajin.result_bundles import finalize_bundle_metadata

    bundle = create_result_bundle("demo", root=tmp_path, kind="batch", tier="two_tier")
    finalize_bundle_metadata(
        bundle,
        samples=[{"sample_name": "s1", "status": "complete"}],
        status="complete",
        extra={
            "recipe_params": {"name": "demo", "segmentation": {"method": "target_objects"}},
            "run_context_extras": {
                "folder_set": [str(tmp_path)],
                "channel_roles": {"Ch1": "target"},
                "scope_filters": [],
            },
        },
    )
    import json
    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["schema_version"] == 2
    assert meta["recipe_params"]["segmentation"]["method"] == "target_objects"
    assert meta["run_context"]["channel_roles"] == {"Ch1": "target"}
    assert meta["run_context"]["folder_set"] == [str(tmp_path)]
    assert "deps" in meta["environment"]


def test_read_bundle_metadata_normalizes_v1(tmp_path):
    """Old flat bundles still load with logical accessors."""
    from imajin.result_bundles import read_bundle_metadata_normalized

    bundle = tmp_path / "old_bundle"
    bundle.mkdir()
    (bundle / "metadata.json").write_text("""
    {
      "recipe": {"name": "old", "segmentation": {"method": "target_objects"}},
      "kind": "batch", "tier": "two_tier", "name": "old",
      "status": "complete", "samples": []
    }
    """)
    norm = read_bundle_metadata_normalized(bundle)
    assert norm["recipe_params"]["name"] == "old"
    assert norm["run_context"]["kind"] == "batch"
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement**

In `result_bundles.py`:
1. Rewrite `finalize_bundle_metadata` to construct the three-block structure. Accept new optional fields (`recipe_params`, `run_context_extras`) inside `extra`.
2. Add `read_bundle_metadata_normalized(bundle)` that returns the three-block dict regardless of stored shape (uses `read_bundle_metadata` then maps `recipe` → `recipe_params`, top-level run fields → `run_context`).
3. Bump `schema_version` to 2 when writing.

Sketch:

```python
def finalize_bundle_metadata(bundle, *, samples, status, extra=None):
    meta = read_bundle_metadata(bundle)
    seed = dict(meta)  # may be v1-style from create_result_bundle
    extra = extra or {}

    recipe_params = extra.get("recipe_params") or seed.get("recipe") or {}
    run_context_extras = extra.get("run_context_extras") or {}

    environment = {
        k: seed.get(k) for k in ("python_version", "imajin_version", "deps", "git_commit")
        if seed.get(k) is not None
    }

    run_context = {
        "kind": seed.get("kind"),
        "tier": seed.get("tier"),
        "name": seed.get("name"),
        "status": status,
        "created_at": seed.get("created_at"),
        "samples": list(samples),
        "n_samples": len(samples),
        "n_complete": sum(1 for s in samples if s.get("status") == "complete"),
        "n_failed": sum(1 for s in samples if s.get("status") == "failed"),
        "tables": {"combined": "tables/combined.csv"},
        **run_context_extras,
    }

    out = {
        "schema_version": 2,
        "recipe_params": recipe_params,
        "run_context": run_context,
        "environment": environment,
    }
    write_bundle_metadata(bundle, out)
```

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): split metadata into recipe_params/run_context/environment"
```

## Task 8: Batch runner emits run_context

**Files:**
- Modify: `src/imajin/tools/batch_runner.py:_finalize_bundle` (~line 640+) and `_create_parent_bundle`
- Test: `tests/test_phase2_workflow.py`

**Design:** Build `channel_roles`, `folder_set`, and `scope_filters` from session state and pass them through `extra={"run_context_extras": {...}}` to `finalize_bundle_metadata`.

- [ ] **Step 1: Failing test:** registered files in two folders + a channel annotation → bundle's `metadata.json` has the correct `run_context.folder_set` and `run_context.channel_roles`.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement.** Collect:
   - `folder_set`: sorted unique parents from `resolve_sample_inputs(name)['file_path']` across all samples.
   - `channel_roles`: from `list(state.list_channels())`; emit `{channel.name: channel.role}`.
   - `scope_filters`: read from the registered-files include patterns if `state` tracks them (skip if not — leave `[]`).

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/batch_runner.py tests/test_phase2_workflow.py
git commit -m "feat(batch): record folder_set/channel_roles in run_context"
```

## Task 9: Single-image workflow emits run_context

**Files:**
- Modify: `src/imajin/tools/workflows.py:_write_analysis_bundle_outputs`
- Test: `tests/test_phase2_workflow.py`

Mirror Task 8 for the `own_bundle` branch. `samples` becomes a single-entry list; `folder_set` is `[parent_of(file_path)]` when known.

- [ ] **Step 1–5:** TDD cycle, commit `feat(workflows): single-image bundles record run_context`.

## Task 10: `import_recipe_from_bundle` tool

**Files:**
- Create: `src/imajin/tools/recipe_import.py`
- Create: `tests/test_recipe_import.py`
- Modify: `src/imajin/agent/runner.py` — add `import_recipe_from_bundle` to tool allowlist.

**Design:**

```python
# src/imajin/tools/recipe_import.py
from pathlib import Path
from typing import Any

from imajin.agent.state import upsert_recipe
from imajin.result_bundles import read_bundle_metadata_normalized
from imajin.tools.registry import tool


@tool(
    description=(
        "Read a previous result bundle's metadata.json and register its "
        "`recipe_params` as a session recipe. Use this when the user points "
        "you at a prior bundle (e.g. `~/.../20260511_032339_J20_1234_CaLexA`) "
        "and asks to apply the same analysis settings. Sample annotations, "
        "channel roles, and file scope are NOT imported — those must be set "
        "per run."
    )
)
def import_recipe_from_bundle(bundle_path: str, name: str | None = None) -> dict[str, Any]:
    bundle = Path(bundle_path).expanduser().resolve()
    meta = read_bundle_metadata_normalized(bundle)
    rp = meta.get("recipe_params") or {}
    if not rp:
        raise ValueError(f"No recipe_params found in {bundle}/metadata.json")
    recipe_name = name or rp.get("name") or bundle.name
    upsert_recipe(
        name=recipe_name,
        target_channel=rp.get("target_channel"),
        preprocessing=rp.get("preprocessing") or [],
        segmentation=rp.get("segmentation") or {},
        measurement=rp.get("measurement") or {},
        domain=rp.get("domain"),
        cell_diameter_um=rp.get("cell_diameter_um"),
    )
    return {
        "recipe_name": recipe_name,
        "source_bundle": str(bundle),
        "imported": {
            "target_channel": rp.get("target_channel"),
            "segmentation": rp.get("segmentation"),
            "domain": rp.get("domain"),
        },
        "note": (
            "Sample annotations, channel roles, and file scope are not "
            "imported. Run register_files + annotate_sample + annotate_channel "
            "for the current data before calling run_recipe_on_samples."
        ),
    }
```

If `upsert_recipe` doesn't exist with that exact signature, look up the actual recipe-write API in `imajin.agent.state` (search for the existing `get_recipe` callsite; the writer is nearby) and use it.

- [ ] **Step 1: Failing tests** — happy path + missing `recipe_params` → ValueError.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement** + add to tool allowlist in `runner.py`.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/recipe_import.py tests/test_recipe_import.py src/imajin/agent/runner.py
git commit -m "feat(tools): add import_recipe_from_bundle"
```

## Task 11: Prompt guidance for bundle reuse

**Files:**
- Modify: `src/imajin/agent/prompts.py:159-176` (batch-analysis guidance), plus a new short paragraph in the "Intent → pipeline mappings" section.

**Design:** Add to the batch-analysis bullet:

> When the user references a prior bundle (`bundle_path`, `*_project` reference removed) or says "전에 했던 거랑 똑같이"/"이 분석처럼", call `import_recipe_from_bundle(bundle_path=<path>)` first to register the recipe, then ask the user only for the missing per-run pieces: file scope, sample annotation, and channel roles. Do NOT reuse the prior bundle's sample list or channel mapping — those are run-specific.

- [ ] **Step 1: Failing test** — add a small unit test that loads the prompt and asserts the new substring is present (defensive against accidental removal).
- [ ] **Step 2: Implement** the prompt edit.
- [ ] **Step 3: Run prompt tests + smoke test the runner allowlist test.**
- [ ] **Step 4: Commit**

```bash
git add src/imajin/agent/prompts.py tests/...
git commit -m "docs(prompts): guide agent to import_recipe_from_bundle on reuse"
```

**Phase 2 stop-and-review checkpoint.** Demo: run an analysis, inspect `metadata.json` for the new shape; then in a new session ask the agent to "이 번들 설정 그대로 다른 폴더에 적용"하고 동작을 확인합니다.

---

# Phase 3 — Project layer removal

Goal: delete `project.json`/`samples.json`/`recipes.json`/… persistence, the autosave loop, the project tools, the project UI menu, and the `_CURRENT_PROJECT` global. Session state stays in memory; persistence happens only through bundles.

## Task 12: Stop autosaving

**Files:**
- Modify: `src/imajin/agent/state.py:61-110` — delete `_autosave_project` and its callers; replace with no-ops or just inline removal.
- Modify: `src/imajin/agent/execution.py:583-588` — remove `autosave_current_project` call.
- Modify: `src/imajin/tools/batch_runner.py:181-185` — drop the `defer_autosave` context manager.

**Design:** None of these calls should remain. Replace each with a comment-free deletion. Tests that assert autosave fires must be deleted or rewritten.

- [ ] **Step 1: Find every reference**

```
uv run --project /home/jin/py314 pytest -q --collect-only 2>&1 | head -5  # sanity
grep -rn "autosave_current_project\|_autosave_project\|defer_autosave" src tests
```

List every hit; each one is either removed or rewritten.

- [ ] **Step 2: Remove autosave call from execution.py**, delete `_autosave_project` from `state.py`, delete `defer_autosave` usages from `batch_runner.py` and its tests.

- [ ] **Step 3: Run full suite**

```
uv run --project /home/jin/py314 pytest -x
```

Expect breakage in `test_project_persistence.py` (handled in Task 14) and possibly in execution-service tests that asserted autosave side effects. Adjust those tests by removing the autosave assertion (the side effect no longer exists).

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(project): stop autosaving project state on tool completion"
```

## Task 13: Remove project tools

**Files:**
- Delete: `src/imajin/tools/project.py`
- Modify: `src/imajin/agent/runner.py` — remove `create_project`, `save_project`, `load_project` from the tool allowlist; remove their imports.

- [ ] **Step 1: Delete the file, update the runner allowlist, fix any imports.**
- [ ] **Step 2: Run tests; fix.** Existing tests that called these tools should be deleted (project layer is gone, no replacement needed).
- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(tools): remove project create/load/save tools"
```

## Task 14: Strip `project.py` to nothing

**Files:**
- Delete: `src/imajin/project.py`
- Delete: `tests/test_project_persistence.py`
- Modify: `src/imajin/results.py` — drop the `from imajin.project import current_project` branch from `results_root()`.

**Design:** With the project tools, autosave, and UI all gone, `imajin.project` no longer has any users. Confirm with grep, then delete. `results_root()` collapses to:

```python
def results_root() -> Path:
    try:
        from imajin.anchor import resolve_session_anchor
        anchor = resolve_session_anchor()
    except Exception:
        anchor = None
    if anchor is not None:
        return anchor
    return user_results_root()
```

- [ ] **Step 1: Grep for remaining importers**

```
grep -rn "from imajin.project\|imajin\.project\b" src tests
```

If anything remains, fix it (most likely test imports we missed).

- [ ] **Step 2: Delete `project.py` + project-persistence test file**

```
git rm src/imajin/project.py tests/test_project_persistence.py
```

- [ ] **Step 3: Update `results_root`** to the collapsed body above.

- [ ] **Step 4: Full test run**

```
uv run --project /home/jin/py314 pytest
```

Expected: green. If a test relies on `current_project()`, replace it with the appropriate session-state setup.

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: remove imajin.project layer entirely"
```

## Task 15: Remove project UI menu

**Files:**
- Modify: `src/imajin/ui/main.py:23-260` — delete the `_new_project`/`_open_project`/`_save_project` helpers and their menu entries. Keep the rest of the menu intact.

- [ ] **Step 1: Edit the file; verify the menu still loads via a smoke test:**

```
uv run --project /home/jin/py314 pytest tests/test_ui_skeletons.py -v
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor(ui): remove New/Open/Save Project menu actions"
```

## Task 16: Prompt + docs cleanup

**Files:**
- Modify: `src/imajin/agent/prompts.py` — remove any remaining project references; clarify that registered files are session-scoped.
- Modify: `README.md` and `docs/specs/phase5_project_persistence.md` if it documents behaviors that no longer exist (mark deprecated or delete).

- [ ] **Step 1: Sweep** with `grep -rn "project" src/imajin/agent/prompts.py README.md docs/` and update each hit.
- [ ] **Step 2: Commit**

```bash
git commit -m "docs: remove obsolete project-layer references"
```

## Task 17: End-to-end manual demo

**Files:** none (manual verification, but list the script for repeatability).

- [ ] **Step 1:** Start napari with the test data set: `uv run --project /home/jin/py314 python -m imajin`.
- [ ] **Step 2:** Register the `test/` folder, annotate one sample, run a batch recipe.
- [ ] **Step 3:** Confirm:
   - A `<timestamp>_<recipe>/` folder appears inside `test/` (no `_project` subfolder is created, no `Documents/Imajin/results/bundles/` write happens).
   - `metadata.json` inside that folder shows `schema_version: 2`, with `recipe_params`, `run_context`, `environment`.
- [ ] **Step 4:** Close napari, reopen, register a *different* folder, and ask the agent "use the recipe from `<path to first bundle>`". Confirm `import_recipe_from_bundle` runs; sample annotation and channel role prompts still occur; analysis lands in the new folder.
- [ ] **Step 5:** No commit — this task is verification.

**Phase 3 done.** Project layer fully gone; session state ephemeral; results colocated with data.

---

## Self-review checklist (run before handing off)

- **Spec coverage** — Tasks 1–6 cover "anchor folder routing"; Tasks 7–11 cover "metadata two-block split + reuse"; Tasks 12–17 cover "project layer removal". Three requirements ↔ three phases. ✓
- **Placeholders** — Task 5's "minimal LSM/TIFF fixtures or monkeypatch loader" is the only soft spot; resolve at execution time by reading the existing batch fixtures in `tests/test_phase2_workflow.py` and reusing them. Task 8/9 rely on the existing channel-state API; if `state.list_channels` does not exist, use the closest equivalent (`channel_records()` or similar — confirm at execution).
- **Type/name consistency** — `resolve_anchor_folder` (Task 1), `resolve_session_anchor` (Task 2), `create_result_bundle(..., root=...)` (Task 4), `read_bundle_metadata_normalized` (Task 7), `import_recipe_from_bundle` (Task 10). All names used consistently downstream. ✓
- **Cache invariants** — none; this plan touches no caching layer.

## Risks and rollback notes

- Phase 1 silently changes where bundles land for **any** user who is not in a project. If anyone has CI or downstream scripts pointing at `~/Documents/Imajin/results/bundles/`, they will break. Mitigation: announce in the phase-1 commit message and the README; the change is exactly the user's intent.
- Phase 3 deletes `project.json` machinery without a migration tool. Existing `*_project` folders on disk are not touched — they remain readable manually but the app will no longer read them. If we later need to import old project state, write a one-shot CLI; do not re-introduce the project module.
- All three phases ship behind no feature flag — by design, since the user wants the new behavior to be the only behavior.
