# Phase 3 — Experiment, Batch, and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `imajin` capable of registering many imaging files, grouping them into user-confirmed sample/group annotations, applying a single reusable analysis recipe across the batch, and emitting an experiment-level report — without parsing filenames or requiring all files to stay loaded in napari.

**Architecture:** Extend the in-memory session state in `src/imajin/agent/state.py` with new dataclasses (`FileRecord`, `AnalysisRecipe`, `AnalysisRun`) and evolve the existing `SampleEntry` into a Phase-3 `SampleAnnotation` with `sample_id`, `file_ids`, and an `extra` dict for user-confirmed fields. Build seven new `@tool` entries that read/write that state: `register_files`, `annotate_samples`, `list_experiment`, `create_analysis_recipe`, `run_recipe_on_samples`, `summarize_experiment`, `generate_experiment_report`. The batch runner reuses `analyze_target_cells` per sample, attaches `sample_id/sample_name/group/file_id/source_file/source_layer` columns to each measurement table, and records per-sample success/failure in `AnalysisRun` so a failed file never aborts the batch.

**Tech Stack:** Python 3.13+, dataclasses, pandas, pytest, existing `imajin.tools.registry.@tool` framework, existing `analyze_target_cells` workflow from Phase 2.

---

## File Structure

- **Modify** `src/imajin/agent/state.py` — add `FileRecord`, `AnalysisRecipe`, `AnalysisRun`; evolve `SampleEntry` → `SampleAnnotation` (add `sample_id`, `file_ids`, `extra`, make `group` optional); add put/get/list/reset helpers for each new collection.
- **Modify** `src/imajin/tools/experiment.py` — add `register_files`, `annotate_samples`, `list_experiment`, `create_analysis_recipe`, `summarize_experiment` tools. Keep existing `annotate_sample` / `list_sample_annotations` unchanged.
- **Modify** `src/imajin/tools/workflows.py` — add `run_recipe_on_samples` (batch runner that reuses `analyze_target_cells` per sample, attaches sample/group/file columns, records `AnalysisRun`).
- **Modify** `src/imajin/tools/report.py` — add `generate_experiment_report` (markdown/html with overview, sample table, recipe, results, methods, warnings).
- **Modify** `tests/conftest.py` — extend the autouse reset fixture to also clear files/recipes/runs between tests.
- **Create** `tests/test_phase3_experiment.py` — covers all spec test cases (Experiment State, Recipe, Batch Run, Summary, Report).

---

## Conventions for Every Task

- **Run tests with the project venv:** `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v`
- **Commit message style:** Match recent history (`feat: …`, `fix: …`, lowercase), keep first line ≤ 70 chars, no trailing summary.
- **Imports:** Always `from __future__ import annotations` at the top of new modules.
- **No filename parsing:** Tests must explicitly verify that strings like `J41`, `vF`, `midgut`, `R3` from filenames never end up in `group`, `condition`, `tissue`, or `replicate` fields without user confirmation.

---

## Task 1: FileRecord model + storage

**Files:**
- Modify: `src/imajin/agent/state.py:18-30` (add FileRecord dataclass + `_FILES` dict near `_TABLES`)
- Test: `tests/test_phase3_experiment.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase3_experiment.py` with this content:

```python
"""Phase 3: experiment / batch / reporting workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imajin.agent import state


@pytest.fixture(autouse=True)
def _reset_phase3_state():
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()
    state.reset_samples()
    state.reset_tables()
    yield
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()
    state.reset_samples()
    state.reset_tables()


# --- Task 1: FileRecord ------------------------------------------------------

def test_put_file_creates_unloaded_record(tmp_path: Path) -> None:
    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    file_id = state.put_file(
        path=str(p),
        original_name="ctrl_1.lsm",
        file_type="lsm",
        metadata_summary={"axes": "CZYX"},
    )
    assert file_id == "ctrl_1"  # slug of original_name (without extension)
    rec = state.get_file(file_id)
    assert rec.path == str(p)
    assert rec.original_name == "ctrl_1.lsm"
    assert rec.file_type == "lsm"
    assert rec.load_status == "unloaded"
    assert rec.metadata_summary == {"axes": "CZYX"}


def test_put_file_dedups_collisions(tmp_path: Path) -> None:
    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "subdir" / "ctrl_1.lsm"
    a.write_bytes(b"")
    b.parent.mkdir()
    b.write_bytes(b"")

    id_a = state.put_file(path=str(a), original_name="ctrl_1.lsm")
    id_b = state.put_file(path=str(b), original_name="ctrl_1.lsm")

    assert id_a == "ctrl_1"
    assert id_b == "ctrl_1_2"
    assert {f["file_id"] for f in state.list_files()} == {"ctrl_1", "ctrl_1_2"}


def test_list_files_returns_dicts(tmp_path: Path) -> None:
    p = tmp_path / "x.tif"
    p.write_bytes(b"")
    state.put_file(path=str(p), original_name="x.tif", file_type="tif")
    files = state.list_files()
    assert isinstance(files, list)
    assert files[0]["file_id"] == "x"
    assert files[0]["load_status"] == "unloaded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py::test_put_file_creates_unloaded_record -v`

Expected: FAIL with `AttributeError: module 'imajin.agent.state' has no attribute 'reset_files'` (or `put_file`).

- [ ] **Step 3: Implement FileRecord and helpers in state.py**

Insert after the existing `_TABLES` block in `src/imajin/agent/state.py` (around line 17):

```python
import re


def _slugify(name: str) -> str:
    base = name.rsplit(".", 1)[0]  # strip extension only on the rightmost dot
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    return base or "file"


@dataclass
class FileRecord:
    file_id: str
    path: str
    original_name: str
    file_type: str | None = None
    metadata_summary: dict[str, Any] = field(default_factory=dict)
    load_status: str = "unloaded"  # "unloaded" | "loaded" | "failed"
    notes: str | None = None


_FILES: dict[str, FileRecord] = {}


def put_file(
    path: str,
    original_name: str,
    file_type: str | None = None,
    metadata_summary: dict[str, Any] | None = None,
    notes: str | None = None,
    load_status: str = "unloaded",
) -> str:
    base = _slugify(original_name)
    file_id = base
    n = 2
    while file_id in _FILES:
        file_id = f"{base}_{n}"
        n += 1
    _FILES[file_id] = FileRecord(
        file_id=file_id,
        path=path,
        original_name=original_name,
        file_type=file_type,
        metadata_summary=dict(metadata_summary or {}),
        notes=notes,
        load_status=load_status,
    )
    return file_id


def get_file(file_id: str) -> FileRecord:
    if file_id not in _FILES:
        raise KeyError(
            f"File id {file_id!r} not found. Available: {list(_FILES)}"
        )
    return _FILES[file_id]


def list_files() -> list[dict[str, Any]]:
    return [asdict(rec) for rec in _FILES.values()]


def update_file_status(file_id: str, status: str, notes: str | None = None) -> None:
    rec = get_file(file_id)
    rec.load_status = status
    if notes is not None:
        rec.notes = notes


def reset_files() -> None:
    _FILES.clear()
```

- [ ] **Step 4: Run the three Task 1 tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k "task1 or put_file or list_files"` (just the three above)

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/agent/state.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(state): add FileRecord registry for Phase 3 batch workflow

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: SampleAnnotation evolution (sample_id, file_ids, extra, optional group)

**Files:**
- Modify: `src/imajin/agent/state.py:20-29` (rename SampleEntry → SampleAnnotation, add fields, make group optional)
- Modify: `src/imajin/agent/state.py:329-353` (update put_sample / list_samples / get_sample helpers)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 2: SampleAnnotation evolution --------------------------------------

def test_put_sample_accepts_extra_and_optional_group() -> None:
    state.put_sample(
        sample_name="ctrl_1",
        group=None,
        file_ids=["ctrl_1"],
        layers=["ctrl_1_ch0"],
        extra={"genotype": "w1118", "tissue": "midgut"},
    )
    samples = state.list_samples()
    assert len(samples) == 1
    s = samples[0]
    assert s["sample_name"] == "ctrl_1"
    assert s["sample_id"] == "ctrl_1"  # defaults to sample_name
    assert s["group"] is None
    assert s["file_ids"] == ["ctrl_1"]
    assert s["extra"] == {"genotype": "w1118", "tissue": "midgut"}


def test_put_sample_keeps_legacy_files_and_layers() -> None:
    """Existing experiment.annotate_sample() and report.py rely on `files`/`layers`."""
    state.put_sample(
        sample_name="t1",
        group="treatment",
        files=["/data/t1.lsm"],
        layers=["t1_ch0", "t1_ch1"],
    )
    s = state.list_samples()[0]
    assert s["files"] == ["/data/t1.lsm"]
    assert s["layers"] == ["t1_ch0", "t1_ch1"]
    assert s["group"] == "treatment"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py::test_put_sample_accepts_extra_and_optional_group -v`

Expected: FAIL — current `put_sample` raises `ValueError("group must not be empty")` for `None`, and `SampleEntry` has no `extra`/`sample_id`/`file_ids`.

- [ ] **Step 3: Update SampleAnnotation + put_sample**

Replace the existing `SampleEntry`/`_SAMPLES`/`put_sample`/`list_samples`/`get_sample` block in `src/imajin/agent/state.py` (currently lines 20–29 and 329–363) with:

```python
@dataclass
class SampleAnnotation:
    sample_name: str
    sample_id: str = ""  # defaults to sample_name in put_sample
    group: str | None = None
    layers: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    file_ids: list[str] = field(default_factory=list)
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Backward-compat alias so external imports `from state import SampleEntry` keep working.
SampleEntry = SampleAnnotation


_SAMPLES: dict[str, SampleAnnotation] = {}


def put_sample(
    sample_name: str,
    group: str | None = None,
    layers: list[str] | None = None,
    files: list[str] | None = None,
    file_ids: list[str] | None = None,
    notes: str | None = None,
    sample_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    sample_name = sample_name.strip()
    if not sample_name:
        raise ValueError("sample_name must not be empty")
    if group is not None:
        group = group.strip() or None
    sid = (sample_id or sample_name).strip()
    _SAMPLES[sample_name] = SampleAnnotation(
        sample_name=sample_name,
        sample_id=sid,
        group=group,
        layers=list(layers or []),
        files=list(files or []),
        file_ids=list(file_ids or []),
        notes=notes,
        extra=dict(extra or {}),
    )
    return sample_name


def list_samples() -> list[dict[str, Any]]:
    return [asdict(entry) for entry in _SAMPLES.values()]


def get_sample(sample_name: str) -> SampleAnnotation:
    if sample_name not in _SAMPLES:
        raise KeyError(
            f"Sample {sample_name!r} not found. Available: {list(_SAMPLES)}"
        )
    return _SAMPLES[sample_name]


def reset_samples() -> None:
    _SAMPLES.clear()
```

- [ ] **Step 4: Update existing experiment.py annotate_sample to require non-empty group**

Existing tool requires group. Update `src/imajin/tools/experiment.py:annotate_sample` so that an empty string still raises clearly (preserves Phase-2 contract):

```python
def annotate_sample(
    sample_name: str,
    group: str,
    layers: list[str] | None = None,
    files: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if not group or not group.strip():
        raise ValueError("group must not be empty for annotate_sample()")
    normalized_files = [str(Path(f).expanduser()) for f in (files or [])]
    name = put_sample(
        sample_name=sample_name,
        group=group,
        layers=list(layers or []),
        files=normalized_files,
        notes=notes,
    )
    return {
        "sample_name": name,
        "group": group,
        "layers": list(layers or []),
        "files": normalized_files,
        "notes": notes,
    }
```

- [ ] **Step 5: Update Phase-2 sample test to accept the new SampleAnnotation shape**

`tests/test_tools_experiment.py:test_annotate_sample_records_group_metadata` previously asserted the *exact* dict returned by `list_sample_annotations()`. After Task 2, `state.list_samples()` returns dicts with extra keys (`sample_id`, `file_ids`, `extra`). Replace the strict equality with a subset check.

In `tests/test_tools_experiment.py`, replace:

```python
    samples = experiment.list_sample_annotations()
    assert samples == [
        {
            "sample_name": "control_1",
            "group": "control",
            "layers": ["ctrl1_ch0", "ctrl1_ch1"],
            "files": ["/data/control_1.lsm"],
            "notes": "adult gut",
        }
    ]
```

with:

```python
    samples = experiment.list_sample_annotations()
    assert len(samples) == 1
    s = samples[0]
    assert s["sample_name"] == "control_1"
    assert s["group"] == "control"
    assert s["layers"] == ["ctrl1_ch0", "ctrl1_ch1"]
    assert s["files"] == ["/data/control_1.lsm"]
    assert s["notes"] == "adult gut"
    # New Phase-3 fields default to safe values:
    assert s["sample_id"] == "control_1"
    assert s["file_ids"] == []
    assert s["extra"] == {}
```

- [ ] **Step 6: Run new + existing sample tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py tests/test_tools_experiment.py -v`

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/imajin/agent/state.py src/imajin/tools/experiment.py tests/test_phase3_experiment.py tests/test_tools_experiment.py
git commit -m "$(cat <<'EOF'
feat(state): evolve SampleEntry into Phase 3 SampleAnnotation

Adds sample_id, file_ids, and extra; group is now optional. SampleEntry
remains as a backward-compat alias.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: AnalysisRecipe model + storage

**Files:**
- Modify: `src/imajin/agent/state.py` (add AnalysisRecipe + helpers near other models)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 3: AnalysisRecipe ---------------------------------------------------

def test_put_recipe_round_trips() -> None:
    recipe_id = state.put_recipe(
        name="gut_GFP",
        target_channel="green",
        preprocessing=[{"step": "rolling_ball", "radius": 25}],
        segmentation={"tool": "cellpose_sam", "do_3D": True, "diameter": None},
        measurement={"properties": ["area", "centroid", "mean_intensity"]},
        notes="adult midgut R3",
    )
    assert recipe_id == "gut_GFP"
    r = state.get_recipe(recipe_id)
    assert r.target_channel == "green"
    assert r.preprocessing == [{"step": "rolling_ball", "radius": 25}]
    assert r.segmentation["do_3D"] is True
    assert r.measurement["properties"] == ["area", "centroid", "mean_intensity"]
    assert state.list_recipes()[0]["name"] == "gut_GFP"


def test_put_recipe_dedups_by_name() -> None:
    state.put_recipe(name="r1", target_channel="green")
    state.put_recipe(name="r1", target_channel="red")  # overwrite
    rs = state.list_recipes()
    assert len(rs) == 1
    assert rs[0]["target_channel"] == "red"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py::test_put_recipe_round_trips -v`

Expected: FAIL with `AttributeError: module 'imajin.agent.state' has no attribute 'put_recipe'`.

- [ ] **Step 3: Implement AnalysisRecipe**

Append to `src/imajin/agent/state.py` (after the FileRecord section):

```python
@dataclass
class AnalysisRecipe:
    recipe_id: str
    name: str
    target_channel: str | None = None
    preprocessing: list[dict[str, Any]] = field(default_factory=list)
    segmentation: dict[str, Any] = field(default_factory=dict)
    measurement: dict[str, Any] = field(default_factory=dict)
    timecourse: dict[str, Any] | None = None
    colocalization: list[tuple[str, str]] = field(default_factory=list)
    notes: str | None = None


_RECIPES: dict[str, AnalysisRecipe] = {}


def put_recipe(
    name: str,
    target_channel: str | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    segmentation: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    timecourse: dict[str, Any] | None = None,
    colocalization: list[tuple[str, str]] | None = None,
    notes: str | None = None,
) -> str:
    name = name.strip()
    if not name:
        raise ValueError("recipe name must not be empty")
    _RECIPES[name] = AnalysisRecipe(
        recipe_id=name,
        name=name,
        target_channel=target_channel,
        preprocessing=list(preprocessing or []),
        segmentation=dict(segmentation or {}),
        measurement=dict(measurement or {}),
        timecourse=dict(timecourse) if timecourse else None,
        colocalization=list(colocalization or []),
        notes=notes,
    )
    return name


def get_recipe(name: str) -> AnalysisRecipe:
    if name not in _RECIPES:
        raise KeyError(f"Recipe {name!r} not found. Available: {list(_RECIPES)}")
    return _RECIPES[name]


def list_recipes() -> list[dict[str, Any]]:
    return [asdict(r) for r in _RECIPES.values()]


def reset_recipes() -> None:
    _RECIPES.clear()
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py::test_put_recipe_round_trips tests/test_phase3_experiment.py::test_put_recipe_dedups_by_name -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/agent/state.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(state): add AnalysisRecipe registry

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: AnalysisRun model + storage

**Files:**
- Modify: `src/imajin/agent/state.py` (add AnalysisRun + helpers)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 4: AnalysisRun ------------------------------------------------------

def test_put_run_records_status_and_outputs() -> None:
    run_id = state.put_run(
        sample_id="ctrl_1",
        file_id="ctrl_1",
        recipe_id="gut_GFP",
        status="complete",
        table_names=["ctrl_1_measurements"],
        layer_names=["ctrl_1_masks"],
        summary={"n_objects": 42},
    )
    assert run_id  # non-empty
    runs = state.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["sample_id"] == "ctrl_1"
    assert r["recipe_id"] == "gut_GFP"
    assert r["status"] == "complete"
    assert r["summary"] == {"n_objects": 42}


def test_put_run_marks_failed_with_error() -> None:
    run_id = state.put_run(
        sample_id="t_1",
        file_id="t_1",
        recipe_id="gut_GFP",
        status="failed",
        error="cellpose returned zero objects",
    )
    r = state.get_run(run_id)
    assert r.status == "failed"
    assert r.error == "cellpose returned zero objects"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py::test_put_run_records_status_and_outputs -v`

Expected: FAIL — `state.put_run` does not exist.

- [ ] **Step 3: Implement AnalysisRun**

Append to `src/imajin/agent/state.py`:

```python
@dataclass
class AnalysisRun:
    run_id: str
    sample_id: str
    file_id: str
    recipe_id: str
    status: str = "pending"  # "pending" | "running" | "complete" | "failed"
    table_names: list[str] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_RUNS: dict[str, AnalysisRun] = {}
_RUN_COUNTER: list[int] = [0]


def put_run(
    sample_id: str,
    file_id: str,
    recipe_id: str,
    status: str = "pending",
    table_names: list[str] | None = None,
    layer_names: list[str] | None = None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    run_id: str | None = None,
) -> str:
    if run_id is None:
        _RUN_COUNTER[0] += 1
        run_id = f"run_{_RUN_COUNTER[0]:04d}"
    _RUNS[run_id] = AnalysisRun(
        run_id=run_id,
        sample_id=sample_id,
        file_id=file_id,
        recipe_id=recipe_id,
        status=status,
        table_names=list(table_names or []),
        layer_names=list(layer_names or []),
        summary=dict(summary or {}),
        error=error,
    )
    return run_id


def get_run(run_id: str) -> AnalysisRun:
    if run_id not in _RUNS:
        raise KeyError(f"Run {run_id!r} not found. Available: {list(_RUNS)}")
    return _RUNS[run_id]


def list_runs() -> list[dict[str, Any]]:
    return [asdict(r) for r in _RUNS.values()]


def reset_runs() -> None:
    _RUNS.clear()
    _RUN_COUNTER[0] = 0
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k put_run`

Expected: 2 PASSED.

- [ ] **Step 5: Update conftest autouse fixture**

Modify `tests/conftest.py:18-26` to also reset the new collections:

```python
@pytest.fixture(autouse=True)
def _reset_sample_annotations():
    from imajin.agent import state

    state.reset_samples()
    state.reset_channel_annotations()
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()
    yield
    state.reset_samples()
    state.reset_channel_annotations()
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()
```

- [ ] **Step 6: Run the full test suite to make sure nothing regressed**

Run: `uv run --project /home/jin/py314 pytest -x -q --ignore=tests/test_anthropic_integration.py --ignore=tests/test_runner.py`

Expected: All non-network tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/imajin/agent/state.py tests/conftest.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(state): add AnalysisRun registry; reset on test boundaries

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: register_files tool

**Files:**
- Modify: `src/imajin/tools/experiment.py` (add register_files)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 5: register_files ---------------------------------------------------

from imajin.tools import experiment


def test_register_files_creates_records_without_loading(tmp_path: Path) -> None:
    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "ctrl_2.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")

    res = experiment.register_files([str(a), str(b)])
    assert res["n_registered"] == 2
    assert {f["original_name"] for f in res["files"]} == {"ctrl_1.lsm", "ctrl_2.lsm"}
    assert all(f["load_status"] == "unloaded" for f in res["files"])
    # File ids are slugs of the original names.
    assert {f["file_id"] for f in res["files"]} == {"ctrl_1", "ctrl_2"}


def test_register_files_marks_missing_unsupported(tmp_path: Path) -> None:
    real = tmp_path / "ok.lsm"
    real.write_bytes(b"")
    missing = tmp_path / "ghost.lsm"  # not created
    weird = tmp_path / "data.xyz"
    weird.write_bytes(b"")

    res = experiment.register_files([str(real), str(missing), str(weird)])
    by_name = {f["original_name"]: f for f in res["files"]}
    assert by_name["ok.lsm"]["supported"] is True
    assert by_name["ok.lsm"]["exists"] is True
    assert by_name["ghost.lsm"]["exists"] is False
    assert by_name["data.xyz"]["supported"] is False
    assert res["n_unsupported"] == 1
    assert res["n_missing"] == 1


def test_register_files_does_not_parse_filename_into_group(tmp_path: Path) -> None:
    """Spec rule: never silently parse J41/vF/midgut/R3/trailing-numbers."""
    p = tmp_path / "J41 + 1234 vF midgut R3 1.lsm"
    p.write_bytes(b"")
    res = experiment.register_files([str(p)])
    rec = res["files"][0]
    # Tool MAY return a slugified file_id, but MUST NOT auto-fill group/condition/replicate.
    for forbidden in ("group", "condition", "replicate", "tissue"):
        assert forbidden not in rec or rec[forbidden] in (None, "")
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k register_files`

Expected: FAIL — `experiment.register_files` does not exist.

- [ ] **Step 3: Implement register_files**

Append to `src/imajin/tools/experiment.py`:

```python
_SUPPORTED_EXTS = {".lsm", ".czi", ".tif", ".tiff", ".ome.tif", ".ome.tiff"}


def _classify_extension(name: str) -> tuple[bool, str | None]:
    lower = name.lower()
    for ext in sorted(_SUPPORTED_EXTS, key=len, reverse=True):
        if lower.endswith(ext):
            return True, ext.lstrip(".")
    return False, None


@tool(
    description="Register one or more imaging files with the experiment without "
    "loading them into napari. Use this when the user names files or folders to "
    "include in a batch analysis. Returns one record per file with file_id, "
    "supported/missing flags, and any cheap metadata. Filenames are NOT parsed "
    "into condition/replicate/tissue — call annotate_samples for that.",
    phase="3",
)
def register_files(paths: list[str]) -> dict[str, Any]:
    from imajin.agent.state import put_file

    out: list[dict[str, Any]] = []
    n_unsupported = 0
    n_missing = 0
    for raw in paths:
        p = Path(raw).expanduser()
        original_name = p.name
        supported, file_type = _classify_extension(original_name)
        exists = p.exists()
        if not supported:
            n_unsupported += 1
        if not exists:
            n_missing += 1
        file_id = put_file(
            path=str(p.resolve() if exists else p),
            original_name=original_name,
            file_type=file_type,
        )
        out.append(
            {
                "file_id": file_id,
                "path": str(p.resolve() if exists else p),
                "original_name": original_name,
                "file_type": file_type,
                "supported": supported,
                "exists": exists,
                "load_status": "unloaded",
            }
        )
    return {
        "n_registered": len(out),
        "n_supported": len(out) - n_unsupported,
        "n_unsupported": n_unsupported,
        "n_missing": n_missing,
        "files": out,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k register_files`

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/experiment.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(tools): add register_files for batch experiment workflow

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: annotate_samples (bulk)

**Files:**
- Modify: `src/imajin/tools/experiment.py` (add annotate_samples)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 6: annotate_samples (bulk) -----------------------------------------

def test_annotate_samples_bulk_creates_two_groups(tmp_path: Path) -> None:
    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])

    res = experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control", "files": [str(a)]},
            {
                "sample_name": "trt_1",
                "group": "treatment",
                "files": [str(b)],
                "extra": {"genotype": "w1118", "tissue": "midgut"},
            },
        ]
    )
    assert res["n_samples"] == 2
    samples = state.list_samples()
    by_name = {s["sample_name"]: s for s in samples}
    assert by_name["ctrl_1"]["group"] == "control"
    assert by_name["ctrl_1"]["file_ids"] == ["ctrl_1"]  # resolved from path
    assert by_name["trt_1"]["extra"]["genotype"] == "w1118"


def test_annotate_samples_accepts_file_ids_directly(tmp_path: Path) -> None:
    p = tmp_path / "x.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])

    experiment.annotate_samples(
        [{"sample_name": "s1", "group": "g", "file_ids": ["x"]}]
    )
    s = state.list_samples()[0]
    assert s["file_ids"] == ["x"]
    # path was filled in from the FileRecord
    assert s["files"] == [str(p.resolve())]


def test_annotate_samples_does_not_invent_extra_from_filename(tmp_path: Path) -> None:
    """A user passing only sample_name/group/files must not get genotype/tissue/etc.
    autofilled from substrings in the filename."""
    p = tmp_path / "J41 + 1234 vF midgut R3 1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])

    experiment.annotate_samples(
        [{"sample_name": "s1", "group": "control", "files": [str(p)]}]
    )
    s = state.list_samples()[0]
    assert s["extra"] == {}  # nothing inferred
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k annotate_samples`

Expected: FAIL — `experiment.annotate_samples` not defined.

- [ ] **Step 3: Implement annotate_samples**

Append to `src/imajin/tools/experiment.py`:

```python
def _resolve_files_for_sample(
    files: list[str] | None,
    file_ids: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (file_paths, file_ids). Either input can be empty.
    Paths are matched against registered FileRecords; unmatched paths are
    accepted but get no file_id."""
    from imajin.agent.state import _FILES, get_file

    resolved_paths: list[str] = []
    resolved_ids: list[str] = list(file_ids or [])
    by_path = {rec.path: rec for rec in _FILES.values()}

    for raw in files or []:
        p = str(Path(raw).expanduser().resolve())
        rec = by_path.get(p) or by_path.get(str(Path(raw).expanduser()))
        if rec is not None:
            resolved_paths.append(rec.path)
            if rec.file_id not in resolved_ids:
                resolved_ids.append(rec.file_id)
        else:
            resolved_paths.append(p)

    # If user passed file_ids only, fill in paths from the registry.
    for fid in file_ids or []:
        try:
            rec = get_file(fid)
        except KeyError:
            continue
        if rec.path not in resolved_paths:
            resolved_paths.append(rec.path)

    return resolved_paths, resolved_ids


@tool(
    description="Bulk-annotate samples with user-confirmed group/condition/replicate "
    "metadata. Pass a list of dicts, each with sample_name (required), group, files "
    "(paths) or file_ids (registered ids), layers, notes, and extra (a dict of "
    "user-confirmed fields like genotype/tissue/region/replicate). The agent must "
    "never invent these fields from filenames — only store what the user confirmed.",
    phase="3",
)
def annotate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from imajin.agent.state import put_sample

    out: list[dict[str, Any]] = []
    for entry in samples:
        if "sample_name" not in entry:
            raise ValueError("each sample must have a sample_name")
        files, file_ids = _resolve_files_for_sample(
            entry.get("files"), entry.get("file_ids")
        )
        name = put_sample(
            sample_name=entry["sample_name"],
            group=entry.get("group"),
            layers=list(entry.get("layers") or []),
            files=files,
            file_ids=file_ids,
            notes=entry.get("notes"),
            sample_id=entry.get("sample_id"),
            extra=dict(entry.get("extra") or {}),
        )
        out.append(
            {
                "sample_name": name,
                "group": entry.get("group"),
                "file_ids": file_ids,
                "files": files,
                "extra": dict(entry.get("extra") or {}),
            }
        )
    return {"n_samples": len(out), "samples": out}
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k annotate_samples`

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/experiment.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(tools): add annotate_samples bulk tool with file_id resolution

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: list_experiment

**Files:**
- Modify: `src/imajin/tools/experiment.py` (add list_experiment)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 7: list_experiment --------------------------------------------------

def test_list_experiment_returns_all_collections(tmp_path: Path) -> None:
    p = tmp_path / "x.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "s1", "group": "control", "files": [str(p)]}]
    )
    state.put_recipe(name="r1", target_channel="green")
    state.put_run(
        sample_id="s1", file_id="x", recipe_id="r1", status="pending"
    )

    res = experiment.list_experiment()
    assert {f["file_id"] for f in res["files"]} == {"x"}
    assert {s["sample_name"] for s in res["samples"]} == {"s1"}
    assert {r["name"] for r in res["recipes"]} == {"r1"}
    assert res["runs"][0]["sample_id"] == "s1"


def test_list_experiment_handles_empty_state() -> None:
    res = experiment.list_experiment()
    assert res == {"files": [], "samples": [], "recipes": [], "runs": []}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k list_experiment`

Expected: FAIL.

- [ ] **Step 3: Implement list_experiment**

Append to `src/imajin/tools/experiment.py`:

```python
@tool(
    description="Return the current experiment session state: registered files, "
    "sample annotations, analysis recipes, and runs. Use this before batch "
    "analysis or report generation to confirm the experiment shape with the user.",
    phase="3",
)
def list_experiment() -> dict[str, Any]:
    from imajin.agent.state import (
        list_files,
        list_recipes,
        list_runs,
        list_samples,
    )

    return {
        "files": list_files(),
        "samples": list_samples(),
        "recipes": list_recipes(),
        "runs": list_runs(),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k list_experiment`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/experiment.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(tools): add list_experiment overview tool

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: create_analysis_recipe tool

**Files:**
- Modify: `src/imajin/tools/experiment.py` (add create_analysis_recipe)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 8: create_analysis_recipe ------------------------------------------

def test_create_analysis_recipe_stores_full_payload() -> None:
    res = experiment.create_analysis_recipe(
        name="gut_GFP",
        target_channel="green",
        segmentation={"tool": "cellpose_sam", "do_3D": True, "diameter": None},
        measurement={"properties": ["area", "centroid", "mean_intensity"]},
        preprocessing=[{"step": "rolling_ball", "radius": 25}],
    )
    assert res["recipe_id"] == "gut_GFP"
    r = state.get_recipe("gut_GFP")
    assert r.target_channel == "green"
    assert r.segmentation["do_3D"] is True
    assert r.preprocessing == [{"step": "rolling_ball", "radius": 25}]


def test_create_analysis_recipe_minimal_inputs() -> None:
    res = experiment.create_analysis_recipe(name="r2", target_channel="red")
    assert res["recipe_id"] == "r2"
    r = state.get_recipe("r2")
    assert r.segmentation == {}
    assert r.measurement == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k create_analysis_recipe`

Expected: FAIL.

- [ ] **Step 3: Implement create_analysis_recipe**

Append to `src/imajin/tools/experiment.py`:

```python
@tool(
    description="Create or replace a reusable analysis recipe. Captures target "
    "channel, segmentation/measurement/preprocessing settings, and optional "
    "time-course or colocalization parameters so the same pipeline can be applied "
    "across many samples in a batch.",
    phase="3",
)
def create_analysis_recipe(
    name: str,
    target_channel: str | None = None,
    segmentation: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    timecourse: dict[str, Any] | None = None,
    colocalization: list[tuple[str, str]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    from imajin.agent.state import put_recipe

    recipe_id = put_recipe(
        name=name,
        target_channel=target_channel,
        segmentation=segmentation,
        measurement=measurement,
        preprocessing=preprocessing,
        timecourse=timecourse,
        colocalization=colocalization,
        notes=notes,
    )
    return {
        "recipe_id": recipe_id,
        "name": recipe_id,
        "target_channel": target_channel,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k create_analysis_recipe`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/experiment.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(tools): add create_analysis_recipe tool

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: run_recipe_on_samples — single-sample base case

**Files:**
- Modify: `src/imajin/tools/workflows.py` (add run_recipe_on_samples)
- Modify: `src/imajin/tools/measure.py` (extend table to optionally carry sample/group/file columns)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 9: run_recipe_on_samples (single sample) ---------------------------

def _two_label_image() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    img = np.zeros_like(labels, dtype=np.float32)
    img[2:8, 2:8] = 100.0
    img[12:18, 12:18] = 50.0
    return labels, img


def _stub_cellpose(monkeypatch, mask: np.ndarray) -> None:
    from imajin.tools import segment

    class _FakeModel:
        def eval(self, data, **kwargs):  # noqa: ANN001
            return mask, None, None

    monkeypatch.setattr(
        segment, "_get_cellpose_model", lambda *a, **kw: _FakeModel()
    )


def test_run_recipe_on_samples_single_sample_attaches_columns(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    labels, img = _two_label_image()
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [
            {
                "sample_name": "ctrl_1",
                "group": "control",
                "files": [str(p)],
                "layers": ["ctrl_1_ch0"],
            }
        ]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="green",
        segmentation={"tool": "cellpose_sam"},
        measurement={"properties": ["area", "centroid", "mean_intensity"]},
    )
    _stub_cellpose(monkeypatch, labels)

    res = workflows.run_recipe_on_samples(recipe_name="r1")
    assert res["n_samples"] == 1
    assert res["n_complete"] == 1
    assert res["n_failed"] == 0
    run = res["runs"][0]
    assert run["status"] == "complete"
    df = state.get_table(run["table_names"][0])
    # spec-required columns
    for col in (
        "sample_id",
        "sample_name",
        "group",
        "file_id",
        "source_file",
        "source_layer",
    ):
        assert col in df.columns, f"missing required column: {col}"
    assert (df["sample_name"] == "ctrl_1").all()
    assert (df["group"] == "control").all()
    assert (df["file_id"] == "ctrl_1").all()
    assert (df["source_layer"] == "ctrl_1_ch0").all()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_single_sample_attaches_columns -v`

Expected: FAIL — `workflows.run_recipe_on_samples` does not exist.

- [ ] **Step 3: Add the column-attachment helper to state.py**

Append to `src/imajin/agent/state.py`:

```python
def attach_sample_columns_to_table(
    table_name: str,
    sample_id: str,
    sample_name: str,
    group: str | None,
    file_id: str | None,
    source_file: str | None,
    source_layer: str | None,
) -> None:
    """Add identifier columns required by the Phase-3 spec onto an existing table.
    No-op if the table doesn't exist or already has these columns."""
    if table_name not in _TABLES:
        return
    df = _TABLES[table_name].df
    additions = {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "group": group,
        "file_id": file_id,
        "source_file": source_file,
        "source_layer": source_layer,
    }
    for col, value in additions.items():
        if col not in df.columns:
            df[col] = value
    _TABLES[table_name].df = df
    _emit_tables_changed()
```

- [ ] **Step 4: Implement run_recipe_on_samples in workflows.py**

Append to `src/imajin/tools/workflows.py`:

```python
def _resolve_sample_inputs(sample_name: str) -> dict[str, Any]:
    """Pick the layer name + file path the recipe should operate on for one sample."""
    from imajin.agent.state import _FILES, get_sample

    s = get_sample(sample_name)
    layer_name = s.layers[0] if s.layers else None
    file_path: str | None = None
    file_id: str | None = None
    if s.file_ids:
        file_id = s.file_ids[0]
        rec = _FILES.get(file_id)
        if rec is not None:
            file_path = rec.path
    elif s.files:
        file_path = s.files[0]
    return {
        "sample": s,
        "layer_name": layer_name,
        "file_path": file_path,
        "file_id": file_id,
    }


@tool(
    description="Apply a stored analysis recipe to one or more annotated samples. "
    "Iterates samples one by one: resolves the target channel/layer, runs the "
    "Phase-2 analyze_target_cells pipeline, attaches sample/group/file columns to "
    "the resulting measurement table, and records a per-sample AnalysisRun. A "
    "failure on one sample never aborts the batch.",
    phase="3",
    worker=True,
)
def run_recipe_on_samples(
    recipe_name: str,
    sample_names: list[str] | None = None,
) -> dict[str, Any]:
    from imajin.agent.state import (
        attach_sample_columns_to_table,
        get_recipe,
        list_samples,
        put_run,
    )

    recipe = get_recipe(recipe_name)
    if sample_names is None:
        sample_names = [s["sample_name"] for s in list_samples()]
    if not sample_names:
        return {
            "recipe": recipe_name,
            "n_samples": 0,
            "n_complete": 0,
            "n_failed": 0,
            "runs": [],
        }

    seg = recipe.segmentation or {}
    pre_steps = recipe.preprocessing or []
    pre_choice = pre_steps[0]["step"] if pre_steps else None

    runs: list[dict[str, Any]] = []
    n_complete = 0
    n_failed = 0
    for name in sample_names:
        info = _resolve_sample_inputs(name)
        s = info["sample"]
        try:
            result = analyze_target_cells(
                target=recipe.target_channel,
                do_3D=seg.get("do_3D"),
                diameter=seg.get("diameter"),
                preprocess=pre_choice,
            )
        except Exception as exc:  # noqa: BLE001 — bubble error into the run record
            run_id = put_run(
                sample_id=s.sample_id,
                file_id=info["file_id"] or "",
                recipe_id=recipe.recipe_id,
                status="failed",
                error=str(exc),
            )
            runs.append({"run_id": run_id, "status": "failed", "error": str(exc)})
            n_failed += 1
            continue

        if not result.get("ok"):
            run_id = put_run(
                sample_id=s.sample_id,
                file_id=info["file_id"] or "",
                recipe_id=recipe.recipe_id,
                status="failed",
                error=result.get("error", "analysis returned ok=false"),
                summary=result,
            )
            runs.append(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error": result.get("error"),
                }
            )
            n_failed += 1
            continue

        table_name = result.get("table_name")
        if table_name:
            attach_sample_columns_to_table(
                table_name=table_name,
                sample_id=s.sample_id,
                sample_name=s.sample_name,
                group=s.group,
                file_id=info["file_id"],
                source_file=info["file_path"],
                source_layer=result.get("target_channel"),
            )

        run_id = put_run(
            sample_id=s.sample_id,
            file_id=info["file_id"] or "",
            recipe_id=recipe.recipe_id,
            status="complete",
            table_names=[table_name] if table_name else [],
            layer_names=[
                ln
                for ln in (result.get("labels_layer"), result.get("preprocessed_layer"))
                if ln
            ],
            summary={
                "n_objects": result.get("n_objects"),
                "target_channel": result.get("target_channel"),
                "warnings": result.get("warnings", []),
            },
        )
        runs.append(
            {
                "run_id": run_id,
                "status": "complete",
                "sample_name": s.sample_name,
                "table_names": [table_name] if table_name else [],
            }
        )
        n_complete += 1

    return {
        "recipe": recipe_name,
        "n_samples": len(sample_names),
        "n_complete": n_complete,
        "n_failed": n_failed,
        "runs": runs,
    }
```

- [ ] **Step 5: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_single_sample_attaches_columns -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/agent/state.py src/imajin/tools/workflows.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(workflows): run_recipe_on_samples for single-sample batch base case

Attaches sample/group/file/source columns to measurement tables and
records an AnalysisRun per sample.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: run_recipe_on_samples — multi-sample + failure isolation

**Files:**
- Modify: `tests/test_phase3_experiment.py` (add multi-sample tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 10: run_recipe_on_samples (multi-sample, failure isolation) --------

def test_run_recipe_on_samples_multi_sample_one_fails(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    labels, img = _two_label_image()
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    viewer.add_image(np.zeros_like(img), name="trt_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")
    # treatment layer has zero signal; we'll make cellpose return an empty mask for it

    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])
    experiment.annotate_samples(
        [
            {
                "sample_name": "ctrl_1",
                "group": "control",
                "files": [str(a)],
                "layers": ["ctrl_1_ch0"],
            },
            {
                "sample_name": "trt_1",
                "group": "treatment",
                "files": [str(b)],
                "layers": ["trt_1_ch0"],
            },
        ]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="ctrl_1_ch0",  # explicit name; treatment has different layer
        segmentation={"tool": "cellpose_sam"},
        measurement={"properties": ["area", "mean_intensity"]},
    )

    # First sample: cellpose returns labels. Second sample's recipe target
    # ("ctrl_1_ch0") will resolve to the same layer, but we'll force an empty mask
    # on the second call to simulate a sample-level failure.
    call = {"n": 0}

    def _fake_model_factory(*a, **kw):  # noqa: ANN001
        class _FM:
            def eval(self, data, **kwargs):  # noqa: ANN001
                call["n"] += 1
                if call["n"] == 1:
                    return labels, None, None
                return np.zeros_like(labels), None, None

        return _FM()

    from imajin.tools import segment

    monkeypatch.setattr(segment, "_get_cellpose_model", _fake_model_factory)

    res = workflows.run_recipe_on_samples(
        recipe_name="r1", sample_names=["ctrl_1", "trt_1"]
    )
    assert res["n_samples"] == 2
    assert res["n_complete"] == 1
    assert res["n_failed"] == 1
    statuses = [r["status"] for r in res["runs"]]
    assert sorted(statuses) == ["complete", "failed"]

    # Confirm the failed sample is still recorded as an AnalysisRun.
    runs = state.list_runs()
    assert {r["status"] for r in runs} == {"complete", "failed"}
    failed = next(r for r in runs if r["status"] == "failed")
    assert "zero objects" in (failed["error"] or "").lower() or "ok=false" in (
        failed["error"] or ""
    ).lower()


def test_run_recipe_on_samples_no_samples_returns_empty() -> None:
    from imajin.tools import workflows

    state.put_recipe(name="r_empty", target_channel="green")
    res = workflows.run_recipe_on_samples(recipe_name="r_empty", sample_names=[])
    assert res["n_samples"] == 0
    assert res["runs"] == []
```

- [ ] **Step 2: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k run_recipe_on_samples`

Expected: 3 PASSED (the Task-9 test plus the two added here).

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
test(workflows): cover multi-sample batch + per-sample failure isolation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: summarize_experiment

**Files:**
- Modify: `src/imajin/tools/experiment.py` (add summarize_experiment)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 11: summarize_experiment -------------------------------------------

def test_summarize_experiment_sample_and_group_levels() -> None:
    df = pd.DataFrame(
        {
            "label": [1, 2, 1, 2, 1, 2],
            "sample_id": ["c1", "c1", "c2", "c2", "t1", "t1"],
            "sample_name": ["c1", "c1", "c2", "c2", "t1", "t1"],
            "group": ["control", "control", "control", "control", "treatment", "treatment"],
            "mean_intensity": [10.0, 20.0, 12.0, 18.0, 50.0, 60.0],
            "area": [100, 110, 105, 115, 90, 95],
        }
    )
    state.put_table("measurements", df)

    res = experiment.summarize_experiment(measurement="mean_intensity")
    sample_tbl = state.get_table(res["sample_table"])
    assert set(sample_tbl["sample_name"]) == {"c1", "c2", "t1"}
    c1 = sample_tbl[sample_tbl["sample_name"] == "c1"].iloc[0]
    assert c1["mean"] == 15.0  # (10 + 20) / 2
    assert c1["count"] == 2

    group_tbl = state.get_table(res["group_table"])
    assert set(group_tbl["group"]) == {"control", "treatment"}
    ctrl = group_tbl[group_tbl["group"] == "control"].iloc[0]
    # group mean = mean of sample means = (15 + 15) / 2 = 15
    assert ctrl["mean"] == 15.0
    assert ctrl["n_samples"] == 2
    assert ctrl["n_objects"] == 4


def test_summarize_experiment_handles_missing_group() -> None:
    df = pd.DataFrame(
        {
            "label": [1, 2],
            "sample_id": ["s1", "s1"],
            "sample_name": ["s1", "s1"],
            "group": [None, None],
            "mean_intensity": [5.0, 7.0],
        }
    )
    state.put_table("measurements", df)

    res = experiment.summarize_experiment(measurement="mean_intensity")
    group_tbl = state.get_table(res["group_table"])
    # missing group should be reported as a single bucket without crashing
    assert len(group_tbl) >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k summarize_experiment`

Expected: FAIL — `experiment.summarize_experiment` not defined.

- [ ] **Step 3: Implement summarize_experiment**

Append to `src/imajin/tools/experiment.py`:

```python
def _scan_measurement_tables(measurement: str) -> "pd.DataFrame":
    """Concatenate every Phase-3 measurement table that has the requested column
    plus the sample/group identifier columns."""
    import pandas as pd

    from imajin.agent.state import _TABLES

    frames: list[pd.DataFrame] = []
    for entry in _TABLES.values():
        df = entry.df
        if df is None or df.empty:
            continue
        needed = {"sample_name", "sample_id", "group", measurement}
        if not needed.issubset(df.columns):
            continue
        frames.append(df)
    if not frames:
        raise ValueError(
            f"No measurement tables found containing column {measurement!r} "
            "alongside sample_name/sample_id/group. Run a recipe first."
        )
    return pd.concat(frames, ignore_index=True)


@tool(
    description="Aggregate per-object measurements into sample-level and "
    "group-level summary tables. Pass the measurement column name (e.g. "
    "'mean_intensity_green_target', 'area_um2'). Sample-level: count, mean, "
    "median, std, sem per sample. Group-level: mean of sample means, "
    "n_samples, and n_objects per group.",
    phase="3",
)
def summarize_experiment(
    measurement: str,
    group_by: str = "group",
    sample_col: str = "sample_name",
) -> dict[str, Any]:
    import pandas as pd

    from imajin.agent.state import put_table

    df = _scan_measurement_tables(measurement)
    sample_grp = df.groupby(sample_col, dropna=False)[measurement]
    sample_summary = sample_grp.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        sem="sem",
    ).reset_index()

    # Carry the group label on the sample-level table.
    sample_to_group = (
        df[[sample_col, group_by]].drop_duplicates(subset=[sample_col]).set_index(sample_col)
    )
    sample_summary[group_by] = sample_summary[sample_col].map(sample_to_group[group_by])

    group_summary = (
        sample_summary.groupby(group_by, dropna=False)
        .agg(
            n_samples=(sample_col, "nunique"),
            mean=("mean", "mean"),
            median=("median", "mean"),
            std=("std", "mean"),
            sem=("sem", "mean"),
        )
        .reset_index()
    )
    object_counts = df.groupby(group_by, dropna=False)[measurement].size()
    group_summary["n_objects"] = group_summary[group_by].map(object_counts).astype(int)

    sample_table_name = put_table(
        f"summary_sample__{measurement}",
        sample_summary,
        spec={"tool": "summarize_experiment", "measurement": measurement, "level": "sample"},
    )
    group_table_name = put_table(
        f"summary_group__{measurement}",
        group_summary,
        spec={"tool": "summarize_experiment", "measurement": measurement, "level": "group"},
    )
    return {
        "measurement": measurement,
        "sample_table": sample_table_name,
        "group_table": group_table_name,
        "n_samples": int(sample_summary[sample_col].nunique()),
        "n_groups": int(group_summary[group_by].nunique(dropna=False)),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k summarize_experiment`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/experiment.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(tools): add summarize_experiment for sample- and group-level aggregation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: generate_experiment_report

**Files:**
- Modify: `src/imajin/tools/report.py` (add generate_experiment_report)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase3_experiment.py`:

```python
# --- Task 12: generate_experiment_report -------------------------------------

def test_generate_experiment_report_md_includes_all_sections(
    tmp_path: Path, monkeypatch
) -> None:
    from imajin.tools import report

    # Minimal experiment state
    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control", "files": [str(p)]},
            {"sample_name": "trt_1", "group": "treatment"},
        ]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="green",
        segmentation={"tool": "cellpose_sam", "do_3D": True},
        measurement={"properties": ["area", "mean_intensity"]},
    )
    state.put_run(
        sample_id="ctrl_1",
        file_id="ctrl_1",
        recipe_id="r1",
        status="complete",
        summary={"n_objects": 42},
    )
    state.put_run(
        sample_id="trt_1",
        file_id="",
        recipe_id="r1",
        status="failed",
        error="no file registered",
    )

    # Empty provenance log (we only test the experiment-report sections here)
    log_path = tmp_path / "session.jsonl"
    log_path.write_text("", encoding="utf-8")
    from imajin.agent import provenance

    monkeypatch.setattr(provenance, "_LOG_PATH", log_path)

    out = tmp_path / "exp_report.md"
    res = report.generate_experiment_report(str(out), format="md")
    body = out.read_text(encoding="utf-8")

    assert "# Experiment Report" in body
    assert "## Overview" in body
    assert "## Sample Table" in body
    assert "ctrl_1" in body and "trt_1" in body
    assert "## Analysis Recipe" in body
    assert "cellpose_sam" in body
    assert "## Methods" in body  # provenance section is included even if empty
    assert "## Warnings" in body
    assert "trt_1" in body  # failed sample listed in warnings
    assert res["n_samples"] == 2
    assert res["n_failed"] == 1


def test_generate_experiment_report_html_writes_file(tmp_path, monkeypatch) -> None:
    from imajin.tools import report

    log_path = tmp_path / "session.jsonl"
    log_path.write_text("", encoding="utf-8")
    from imajin.agent import provenance

    monkeypatch.setattr(provenance, "_LOG_PATH", log_path)

    out = tmp_path / "exp_report.html"
    res = report.generate_experiment_report(str(out), format="html")
    body = out.read_text(encoding="utf-8")
    assert "<html>" in body
    assert res["format"] == "html"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k generate_experiment_report`

Expected: FAIL — `report.generate_experiment_report` not defined.

- [ ] **Step 3: Implement generate_experiment_report**

Append to `src/imajin/tools/report.py`:

```python
def _render_overview(files, samples, recipes) -> str:
    n_groups = len({s.get("group") for s in samples if s.get("group")})
    return (
        "## Overview\n\n"
        f"- Files registered: {len(files)}\n"
        f"- Samples: {len(samples)}\n"
        f"- Groups: {n_groups}\n"
        f"- Recipes: {len(recipes)}\n"
    )


def _render_sample_table(samples) -> str:
    if not samples:
        return "## Sample Table\n\n_No samples registered._\n"
    lines = ["## Sample Table", "", "| Sample | Group | Files | Notes |", "|---|---|---|---|"]
    for s in samples:
        files = ", ".join(s.get("file_ids") or []) or ", ".join(s.get("files") or [])
        notes = (s.get("notes") or "").replace("|", "/")
        extra = "; ".join(f"{k}={v}" for k, v in (s.get("extra") or {}).items())
        if extra:
            notes = f"{notes} ({extra})" if notes else extra
        lines.append(
            f"| {s.get('sample_name', '?')} | {s.get('group') or '—'} "
            f"| {files or '—'} | {notes or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_recipes(recipes) -> str:
    if not recipes:
        return "## Analysis Recipe\n\n_No recipes defined._\n"
    parts = ["## Analysis Recipe", ""]
    for r in recipes:
        parts.append(f"### {r.get('name')}")
        parts.append(f"- target channel: `{r.get('target_channel') or 'unspecified'}`")
        if r.get("preprocessing"):
            parts.append(f"- preprocessing: `{r['preprocessing']}`")
        if r.get("segmentation"):
            parts.append(f"- segmentation: `{r['segmentation']}`")
        if r.get("measurement"):
            parts.append(f"- measurement: `{r['measurement']}`")
        if r.get("notes"):
            parts.append(f"- notes: {r['notes']}")
        parts.append("")
    return "\n".join(parts)


def _render_runs(runs) -> str:
    if not runs:
        return "## Results\n\n_No runs recorded._\n"
    lines = ["## Results", "", "| Sample | Status | Objects | Tables |", "|---|---|---|---|"]
    for r in runs:
        n_obj = (r.get("summary") or {}).get("n_objects") or "—"
        tables = ", ".join(r.get("table_names") or []) or "—"
        lines.append(
            f"| {r.get('sample_id', '?')} | {r.get('status', '?')} | {n_obj} | {tables} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_warnings(runs) -> str:
    failed = [r for r in runs if r.get("status") == "failed"]
    if not failed:
        return "## Warnings\n\n_No failures recorded._\n"
    lines = ["## Warnings", ""]
    for r in failed:
        lines.append(
            f"- **{r.get('sample_id')}** failed: {r.get('error') or 'unknown error'}"
        )
    lines.append("")
    return "\n".join(lines)


@tool(
    description="Write a Phase-3 experiment-level report to disk (markdown or HTML). "
    "Includes overview, sample table, analysis recipe, per-sample results, the "
    "deterministic Methods paragraph from session provenance, and a warnings "
    "section listing failed samples.",
    phase="3",
)
def generate_experiment_report(
    path: str,
    session_id: str | None = None,
    format: str = "md",
) -> dict[str, Any]:
    from imajin.agent import provenance
    from imajin.agent.state import (
        list_files,
        list_recipes,
        list_runs,
        list_samples,
    )

    if format not in ("md", "html"):
        raise ValueError(f"format must be 'md' or 'html', got {format!r}")

    files = list_files()
    samples = list_samples()
    recipes = list_recipes()
    runs = list_runs()

    records = provenance.read_session(session_id)
    methods_md = _render_methods_markdown(records)

    body = "\n".join(
        [
            "# Experiment Report",
            "",
            _render_overview(files, samples, recipes),
            _render_sample_table(samples),
            _render_recipes(recipes),
            _render_runs(runs),
            methods_md,
            _render_warnings(runs),
        ]
    )

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if format == "md":
        out.write_text(body, encoding="utf-8")
    else:
        out.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>imajin experiment report</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:780px;"
            "margin:2em auto;padding:0 1em;line-height:1.5}</style>"
            f"</head><body><pre>{escape(body)}</pre></body></html>",
            encoding="utf-8",
        )

    return {
        "path": str(out),
        "format": format,
        "n_samples": len(samples),
        "n_groups": len({s.get("group") for s in samples if s.get("group")}),
        "n_files": len(files),
        "n_runs": len(runs),
        "n_failed": sum(1 for r in runs if r.get("status") == "failed"),
        "session_id": session_id or provenance.current_session_id(),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase3_experiment.py -v -k generate_experiment_report`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/report.py tests/test_phase3_experiment.py
git commit -m "$(cat <<'EOF'
feat(report): add generate_experiment_report (Phase 3)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: full-suite regression sweep

**Files:**
- (no code changes; verification only)

- [ ] **Step 1: Run the full fast test suite**

Run: `uv run --project /home/jin/py314 pytest -x -q --ignore=tests/test_anthropic_integration.py --ignore=tests/test_runner.py`

Expected: All tests PASS, including the previously-existing Phase-2 tests (`test_phase2_workflow.py`, `test_tools_experiment.py`, `test_tools_report.py`).

- [ ] **Step 2: Confirm tools are registered for the LLM**

Run:

```bash
uv run --project /home/jin/py314 python -c "
from imajin.tools.registry import iter_tools
phase3 = sorted(t.name for t in iter_tools() if t.phase == '3')
print(phase3)
"
```

Expected output (set equality, order-independent): includes
`register_files`, `annotate_samples`, `list_experiment`, `create_analysis_recipe`, `summarize_experiment`, `run_recipe_on_samples`, `generate_experiment_report`.

- [ ] **Step 3: Final commit if anything was tweaked**

If steps 1–2 required no changes, skip. Otherwise:

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore(phase3): regression sweep

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Criteria Mapping

| Spec criterion | Covered in |
|---|---|
| User can register multiple files | Task 5 (`register_files`) |
| User can annotate files/samples into groups without filename hard-coding | Task 6 (`annotate_samples`) + filename-parsing tests in Tasks 5/6 |
| User can define one analysis recipe | Task 8 (`create_analysis_recipe`) |
| App can apply recipe to multiple samples one by one | Tasks 9–10 (`run_recipe_on_samples`) |
| Result tables contain sample/group/file columns | Task 9 (`attach_sample_columns_to_table`) |
| App can generate sample-level and group-level summaries | Task 11 (`summarize_experiment`) |
| Report includes experiment organization and analysis results | Task 12 (`generate_experiment_report`) |
| Fast test suite passes | Task 13 |

---

## Open-Question Decisions Made in This Plan

- **Storage:** in-memory module-level dicts (matches existing Phase-2 pattern). JSON or SQLite serialization deferred to a later phase.
- **Layer cleanup between samples:** not enforced in v1 — `analyze_target_cells` already adds new layers; deletion can be added behind a flag once we have integration tests with real readers.
- **Group statistics:** count, mean, median, std, sem per sample; mean-of-means + n_samples + n_objects per group.
- **Plots:** deferred to Phase 6 (visualization).
- **Failed-sample retry:** re-run by passing `sample_names=["that_one"]`; previous run record is replaced under the same numeric `run_id` is NOT done — instead a new `run_id` is created. Old failed runs are preserved for the report's Warnings section.
