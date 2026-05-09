# Result Bundle Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every analysis run (batch, two-tier, single-tier; both standalone and inside `run_recipe_on_samples`) deposit its outputs into one uniform bundle directory with a KST-timestamped name, type-first internal layout, and a complete `metadata.json`.

**Architecture:** A new ContextVar-based active-bundle channel lets `run_recipe_on_samples` create a single parent bundle once and have each per-sample `analyze_target_cells` call write into it; standalone calls fall back to creating their own bundle in the same layout. Bundle filesystem primitives live in `src/imajin/results.py`; agent-state-aware writers live in `src/imajin/tools/results.py`.

**Tech Stack:** Python 3.12, pytest, contextvars, importlib.metadata, tifffile, pandas. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-09-result-bundle-organization-design.md`

---

## File Structure

**Create:**
- `tests/test_results_bundle.py` — unit tests for bundle primitives (KST helper, env info, `create_result_bundle` layout)

**Modify:**
- `src/imajin/results.py` — add `_kst_now`, `_collect_env_info`; rewrite `create_result_bundle` body
- `src/imajin/tools/results.py` — add ContextVar + `with_active_bundle` + `current_bundle` + `populate_sample_outputs` + `write_combined_csv` + `finalize_bundle_metadata`; migrate `save_result_bundle` paths
- `src/imajin/tools/workflows.py` — refactor `analyze_target_cells` to use bundle helpers; rewrite `run_recipe_on_samples` bundle lifecycle
- `src/imajin/agent/prompts.py` — one-line update mentioning `bundle_path`
- `tests/test_phase3_experiment.py` — add new bundle integration tests; update existing tests for new layout

**Read-only references during implementation:**
- `docs/superpowers/specs/2026-05-09-result-bundle-organization-design.md` (this plan's spec)
- `src/imajin/agent/state.py:115-176` (AnalysisRecipe + put_recipe / get_recipe)

---

## Task 1: KST timestamp helper

**Files:**
- Modify: `src/imajin/results.py` (add `_kst_now` helper near top)
- Create: `tests/test_results_bundle.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_results_bundle.py`:

```python
from __future__ import annotations

from datetime import timedelta

from imajin.results import _kst_now


def test_kst_now_returns_aware_datetime_with_plus_nine_offset() -> None:
    now = _kst_now()
    assert now.tzinfo is not None
    offset = now.utcoffset()
    assert offset == timedelta(hours=9)


def test_kst_now_strftime_format_matches_bundle_pattern() -> None:
    now = _kst_now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    assert len(stamp) == 15
    assert stamp[8] == "_"
    assert stamp[:4].isdigit()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: 2 errors / failures: `ImportError: cannot import name '_kst_now'`.

- [ ] **Step 3: Implement `_kst_now`**

In `src/imajin/results.py`, replace the top imports block:

```python
from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imajin.paths import is_wsl, normalize_user_path, windows_drive_roots
```

with:

```python
from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from imajin.paths import is_wsl, normalize_user_path, windows_drive_roots


KST = timezone(timedelta(hours=9), name="KST")


def _kst_now() -> datetime:
    """Return current time in KST (UTC+9), used for bundle folder timestamps."""
    return datetime.now(KST)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/results.py tests/test_results_bundle.py
git commit -m "feat(results): add KST timestamp helper for bundle naming"
```

---

## Task 2: Environment info helper

**Files:**
- Modify: `src/imajin/results.py` (append `_collect_env_info` after `_kst_now`)
- Modify: `tests/test_results_bundle.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_results_bundle.py`:

```python
from imajin.results import _collect_env_info


def test_collect_env_info_includes_python_and_imajin_version() -> None:
    info = _collect_env_info()
    assert "python_version" in info
    assert info["python_version"].count(".") >= 1
    assert "imajin_version" in info


def test_collect_env_info_includes_dep_versions() -> None:
    info = _collect_env_info()
    deps = info.get("deps", {})
    assert "tifffile" in deps
    assert "scikit-image" in deps


def test_collect_env_info_git_commit_is_string_or_none() -> None:
    info = _collect_env_info()
    assert "git_commit" in info
    assert info["git_commit"] is None or isinstance(info["git_commit"], str)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: 3 failures: `ImportError: cannot import name '_collect_env_info'`.

- [ ] **Step 3: Implement `_collect_env_info`**

Append to `src/imajin/results.py` after `_kst_now`:

```python
def _git_commit_short() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _collect_env_info() -> dict[str, Any]:
    """Collect package and platform versions for embedding in bundle metadata."""
    import platform
    from importlib import metadata as _metadata

    def _ver(pkg: str) -> str | None:
        try:
            return _metadata.version(pkg)
        except _metadata.PackageNotFoundError:
            return None

    deps_to_record = (
        "cellpose",
        "scikit-image",
        "tifffile",
        "numpy",
        "pandas",
        "napari",
    )
    deps = {pkg: _ver(pkg) for pkg in deps_to_record}
    deps = {k: v for k, v in deps.items() if v is not None}

    return {
        "python_version": platform.python_version(),
        "imajin_version": _ver("imajin"),
        "deps": deps,
        "git_commit": _git_commit_short(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/results.py tests/test_results_bundle.py
git commit -m "feat(results): collect Python/dep/git env info for bundle metadata"
```

---

## Task 3: New bundle layout + KST timestamp + save_result_bundle path migration

**Files:**
- Modify: `src/imajin/results.py:99-117` (`create_result_bundle` body)
- Modify: `src/imajin/tools/results.py:116-122` (single label-write path)
- Modify: `tests/test_results_bundle.py`

This is one task because the layout change in `create_result_bundle` and the path change in `save_result_bundle` must ship together — `save_result_bundle` writes into the directories `create_result_bundle` creates.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_bundle.py`:

```python
import json
import re
from pathlib import Path

from imajin.results import create_result_bundle, read_bundle_metadata


def test_create_result_bundle_uses_kst_timestamp_in_folder_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    bundle = create_result_bundle("demo", kind="single")
    name = bundle.name
    assert re.match(r"^\d{8}_\d{6}_demo$", name), name


def test_create_result_bundle_creates_new_layout_subdirs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    bundle = create_result_bundle("demo", kind="single")
    for sub in ("labels/cells", "labels/domain", "tables", "qc", "stats", "figures"):
        assert (bundle / sub).is_dir(), f"missing subdir: {sub}"
    assert not (bundle / "labels" / "anything.tif").exists()


def test_create_result_bundle_metadata_has_kst_offset_and_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    bundle = create_result_bundle(
        "demo", kind="batch", tier="two_tier", metadata={"recipe": {"name": "demo"}}
    )
    meta = read_bundle_metadata(bundle)
    assert meta["kind"] == "batch"
    assert meta["tier"] == "two_tier"
    assert meta["created_at"].endswith("+09:00")
    assert "imajin_version" in meta
    assert "python_version" in meta
    assert "deps" in meta
    assert meta["recipe"] == {"name": "demo"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: 3 failures (folder pattern, subdir layout, metadata).

- [ ] **Step 3: Rewrite `create_result_bundle`**

Replace `src/imajin/results.py:99-117`:

```python
def create_result_bundle(
    name: str,
    *,
    kind: str = "analysis",
    metadata: dict[str, Any] | None = None,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    bundle = unique_result_dir("bundles", f"{timestamp}_{slugify_result_name(name)}")
    for subdir in ("labels", "tables", "qc", "figures"):
        (bundle / subdir).mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
        "outputs": {},
    }
    write_bundle_metadata(bundle, payload)
    return bundle
```

with:

```python
def create_result_bundle(
    name: str,
    *,
    kind: str = "single",
    tier: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a fresh bundle directory with the standard layout and seed metadata.

    Layout:
        <ts>_<name>/
        ├── metadata.json
        ├── labels/cells/
        ├── labels/domain/
        ├── tables/
        ├── qc/
        ├── stats/
        └── figures/

    `kind` is "single" or "batch"; `tier` is "single_tier" or "two_tier" (or
    None when not yet decided — the caller can update via write_bundle_metadata).
    Extra metadata is merged at the top level of metadata.json.
    """
    now = _kst_now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    bundle = unique_result_dir("bundles", f"{timestamp}_{slugify_result_name(name)}")
    for sub in ("labels/cells", "labels/domain", "tables", "qc", "stats", "figures"):
        (bundle / sub).mkdir(parents=True, exist_ok=True)
    env = _collect_env_info()
    payload: dict[str, Any] = {
        "kind": kind,
        "tier": tier,
        "name": name,
        "status": "in_progress",
        "created_at": now.isoformat(),
        **env,
    }
    extras = dict(metadata or {})
    payload.update(extras)
    write_bundle_metadata(bundle, payload)
    return bundle
```

- [ ] **Step 4: Migrate `save_result_bundle` label path**

In `src/imajin/tools/results.py`, replace:

```python
        out = bundle / "labels" / f"{slugify_result_name(labels_layer)}.tif"
```

with:

```python
        out = bundle / "labels" / "cells" / f"{slugify_result_name(labels_layer)}.tif"
```

- [ ] **Step 5: Run new tests to verify they pass**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Run full suite to surface regressions**

```bash
uv run pytest tests/ -q
```

Expected: any tests that asserted the old `labels/<name>.tif` path will fail. Note them; they get fixed in Task 6 (analyze_target_cells refactor) and Task 16 (existing test updates).

If unrelated regressions appear, stop and investigate. If only label-path-related, that's expected — proceed.

- [ ] **Step 7: Commit**

```bash
git add src/imajin/results.py src/imajin/tools/results.py tests/test_results_bundle.py
git commit -m "feat(results): KST timestamps and tier-aware bundle layout

create_result_bundle now emits labels/cells/, labels/domain/, stats/,
figures/ alongside the existing tables/ and qc/ dirs, names folders
with KST timestamps, and seeds metadata.json with kind, tier, status,
created_at (with +09:00 offset), and a snapshot of env info
(python/imajin/dep versions and git commit). save_result_bundle
writes labels into labels/cells/ to match the new layout."
```

---

## Task 4: Active-bundle ContextVar plumbing

**Files:**
- Modify: `src/imajin/tools/results.py` (add ContextVar + helpers near top)
- Modify: `tests/test_results_bundle.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_bundle.py`:

```python
from imajin.tools.results import current_bundle, with_active_bundle


def test_current_bundle_is_none_by_default() -> None:
    assert current_bundle() is None


def test_with_active_bundle_sets_and_restores(tmp_path) -> None:
    assert current_bundle() is None
    with with_active_bundle(tmp_path) as b:
        assert b == tmp_path
        assert current_bundle() == tmp_path
    assert current_bundle() is None


def test_with_active_bundle_restores_on_exception(tmp_path) -> None:
    try:
        with with_active_bundle(tmp_path):
            assert current_bundle() == tmp_path
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert current_bundle() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_results_bundle.py::test_current_bundle_is_none_by_default tests/test_results_bundle.py::test_with_active_bundle_sets_and_restores tests/test_results_bundle.py::test_with_active_bundle_restores_on_exception -v
```

Expected: `ImportError: cannot import name 'current_bundle' from 'imajin.tools.results'`.

- [ ] **Step 3: Implement ContextVar + helpers**

In `src/imajin/tools/results.py`, replace the top imports block:

```python
from __future__ import annotations

import shutil
import json
from pathlib import Path
from typing import Any

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import get_table, get_table_entry
from imajin.paths import normalize_user_path
from imajin.results import (
    create_result_bundle,
    read_bundle_metadata,
    record_result,
    slugify_result_name,
    unique_result_path,
    write_bundle_metadata,
)
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool
```

with:

```python
from __future__ import annotations

import contextlib
import contextvars
import shutil
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import get_table, get_table_entry
from imajin.paths import normalize_user_path
from imajin.results import (
    create_result_bundle,
    read_bundle_metadata,
    record_result,
    slugify_result_name,
    unique_result_path,
    write_bundle_metadata,
)
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


_active_bundle: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_active_bundle", default=None
)


def current_bundle() -> Path | None:
    """Return the bundle directory currently being populated, or None.

    Set by `with_active_bundle` (used by run_recipe_on_samples to forward
    the parent bundle to per-sample analyze_target_cells calls).
    """
    return _active_bundle.get()


@contextlib.contextmanager
def with_active_bundle(path: Path | str) -> Iterator[Path]:
    """Mark `path` as the current bundle for the duration of the with-block."""
    p = Path(path)
    token = _active_bundle.set(p)
    try:
        yield p
    finally:
        _active_bundle.reset(token)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: all bundle-primitive tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/results.py tests/test_results_bundle.py
git commit -m "feat(results): add active-bundle ContextVar for batch hand-off

run_recipe_on_samples will set the active bundle once at entry; per-
sample analyze_target_cells calls read it via current_bundle() to
write outputs into the parent bundle instead of creating their own."
```

---

## Task 5: populate_sample_outputs helper

**Files:**
- Modify: `src/imajin/tools/results.py` (append helper)
- Modify: `tests/test_results_bundle.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_bundle.py`:

```python
import numpy as np
import tifffile

from imajin.tools.results import populate_sample_outputs


def test_populate_sample_outputs_writes_cells_label(tmp_path, viewer) -> None:
    from imajin.results import create_result_bundle

    monkeypatch_env = {"IMAJIN_RESULTS_DIR": str(tmp_path)}
    import os
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    try:
        bundle = create_result_bundle("demo", kind="single", tier="single_tier")
        viewer.add_labels(np.ones((5, 5), dtype=np.uint16), name="cells_layer")

        out = populate_sample_outputs(
            bundle, sample_slug="s1", labels_cells="cells_layer"
        )
        assert out["labels_cells"] == "labels/cells/s1.tif"
        written = tifffile.imread(bundle / out["labels_cells"])
        assert written.shape == (5, 5)
        assert out["labels_domain"] is None
        assert out["qc_png"] is None
    finally:
        for k in monkeypatch_env:
            os.environ.pop(k, None)


def test_populate_sample_outputs_writes_domain_when_provided(tmp_path, viewer) -> None:
    from imajin.results import create_result_bundle

    import os
    os.environ["IMAJIN_RESULTS_DIR"] = str(tmp_path)
    try:
        bundle = create_result_bundle("demo", kind="single", tier="two_tier")
        viewer.add_labels(np.ones((5, 5), dtype=np.uint16), name="cells_layer")
        viewer.add_labels(np.ones((5, 5), dtype=np.uint16), name="domain_layer")

        out = populate_sample_outputs(
            bundle,
            sample_slug="s1",
            labels_cells="cells_layer",
            labels_domain="domain_layer",
        )
        assert out["labels_cells"] == "labels/cells/s1.tif"
        assert out["labels_domain"] == "labels/domain/s1.tif"
        assert (bundle / out["labels_domain"]).exists()
    finally:
        os.environ.pop("IMAJIN_RESULTS_DIR", None)


def test_populate_sample_outputs_copies_qc_png(tmp_path, viewer) -> None:
    from imajin.results import create_result_bundle

    import os
    os.environ["IMAJIN_RESULTS_DIR"] = str(tmp_path)
    try:
        bundle = create_result_bundle("demo", kind="single", tier="single_tier")
        viewer.add_labels(np.ones((5, 5), dtype=np.uint16), name="cells_layer")
        qc_src = tmp_path / "src_qc.png"
        qc_src.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        out = populate_sample_outputs(
            bundle,
            sample_slug="s1",
            labels_cells="cells_layer",
            qc_png=str(qc_src),
        )
        assert out["qc_png"] == "qc/s1.png"
        assert (bundle / out["qc_png"]).read_bytes() == b"\x89PNG\r\n\x1a\nfake"
    finally:
        os.environ.pop("IMAJIN_RESULTS_DIR", None)


def test_populate_sample_outputs_rejects_collision(tmp_path, viewer) -> None:
    from imajin.results import create_result_bundle
    import pytest

    import os
    os.environ["IMAJIN_RESULTS_DIR"] = str(tmp_path)
    try:
        bundle = create_result_bundle("demo", kind="single", tier="single_tier")
        viewer.add_labels(np.ones((5, 5), dtype=np.uint16), name="cells_layer")
        populate_sample_outputs(bundle, sample_slug="s1", labels_cells="cells_layer")
        with pytest.raises(ValueError, match="already exists"):
            populate_sample_outputs(
                bundle, sample_slug="s1", labels_cells="cells_layer"
            )
    finally:
        os.environ.pop("IMAJIN_RESULTS_DIR", None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_results_bundle.py -v -k populate_sample_outputs
```

Expected: ImportError for `populate_sample_outputs`.

- [ ] **Step 3: Implement `populate_sample_outputs`**

Append to `src/imajin/tools/results.py`:

```python
def _write_label_layer(
    bundle: Path, tier: str, sample_slug: str, layer_name: str
) -> str:
    """Snapshot a label layer and write it to bundle/labels/<tier>/<slug>.tif.

    Returns the path relative to `bundle`.
    Raises ValueError on filename collision within the same bundle.
    """
    import tifffile

    layer = call_on_main(snapshot_layer, layer_name)
    data = _materialize(layer.data)
    labels = data.astype(_label_output_dtype(data), copy=False)
    rel = Path("labels") / tier / f"{sample_slug}.tif"
    out = bundle / rel
    if out.exists():
        raise ValueError(
            f"{rel} already exists in bundle {bundle.name}; "
            "sample_slug collision suspected"
        )
    tifffile.imwrite(out, labels)
    return rel.as_posix()


def _copy_qc_png(bundle: Path, qc_png: str, sample_slug: str) -> str | None:
    """Copy a QC PNG into bundle/qc/<slug>.png. Returns path relative to bundle."""
    src = normalize_user_path(qc_png).resolve()
    if not src.exists():
        return None
    rel = Path("qc") / f"{sample_slug}.png"
    dst = bundle / rel
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return rel.as_posix()


def populate_sample_outputs(
    bundle: Path,
    sample_slug: str,
    *,
    labels_cells: str | None = None,
    labels_domain: str | None = None,
    qc_png: str | None = None,
) -> dict[str, str | None]:
    """Write per-sample outputs into a bundle, returning relative output paths."""
    out: dict[str, str | None] = {
        "labels_cells": None,
        "labels_domain": None,
        "qc_png": None,
    }
    if labels_cells:
        out["labels_cells"] = _write_label_layer(
            bundle, "cells", sample_slug, labels_cells
        )
    if labels_domain:
        out["labels_domain"] = _write_label_layer(
            bundle, "domain", sample_slug, labels_domain
        )
    if qc_png:
        out["qc_png"] = _copy_qc_png(bundle, qc_png, sample_slug)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_results_bundle.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/results.py tests/test_results_bundle.py
git commit -m "feat(results): add populate_sample_outputs bundle writer helper"
```

---

## Task 6: analyze_target_cells single-tier writes new-layout bundle

**Files:**
- Modify: `src/imajin/tools/workflows.py:563-582` (replace existing single-tier `save_result_bundle` block)
- Modify: `tests/test_phase3_experiment.py` (or add to test_phase2_workflow.py — pick whichever has the relevant fixtures)

This task replaces the existing per-call `save_result_bundle` invocation in the single-tier branch with the new helper-based flow. The standalone single-tier behaviour (auto-bundle on every call) is preserved; only the layout changes.

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase2_workflow.py`:

```python
import json
from pathlib import Path


def test_analyze_target_cells_single_tier_writes_new_layout_bundle(
    viewer, tmp_path, monkeypatch
) -> None:
    from imajin.tools.workflows import analyze_target_cells

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    res = analyze_target_cells(target="reporter")
    assert res["ok"] is True
    bundle = Path(res["result_bundle_path"])
    assert bundle.exists()
    assert (bundle / "labels" / "cells" / "reporter.tif").exists()
    assert (bundle / "labels" / "domain").is_dir()
    assert not any((bundle / "labels" / "domain").iterdir())
    assert (bundle / "qc").is_dir()

    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["kind"] == "single"
    assert meta["tier"] == "single_tier"
    assert meta["status"] == "complete"
    assert len(meta["samples"]) == 1
    assert meta["samples"][0]["status"] == "complete"
    assert meta["samples"][0]["outputs"]["labels_cells"] == "labels/cells/reporter.tif"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase2_workflow.py::test_analyze_target_cells_single_tier_writes_new_layout_bundle -v
```

Expected: KeyError or assertion failure on bundle layout (current single-tier writes labels at `labels/<name>.tif`, lacks `samples`/`tier`/`status` fields).

- [ ] **Step 3: Add bundle helpers in workflows.py**

In `src/imajin/tools/workflows.py`, just above the `analyze_target_cells` definition, add:

```python
def _build_sample_summary(
    *,
    sample_name: str,
    status: str,
    error: str | None = None,
    n_cells: int | None = None,
    n_domain_components: int | None = None,
    domain_area_um2: float | None = None,
    qc_warnings: list[str] | None = None,
    outputs: dict[str, str | None] | None = None,
    group: str | None = None,
    file_id: str | None = None,
    source_file: str | None = None,
    source_layer: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_name": sample_name,
        "group": group,
        "file_id": file_id,
        "source_file": source_file,
        "source_layer": source_layer,
        "status": status,
        "error": error,
        "outputs": outputs or {"labels_cells": None, "labels_domain": None, "qc_png": None},
        "summary": {
            "n_cells": n_cells,
            "n_domain_components": n_domain_components,
            "domain_area_um2": domain_area_um2,
            "qc_warnings": list(qc_warnings or []),
        },
    }
```

- [ ] **Step 4: Replace the single-tier bundle block**

In `src/imajin/tools/workflows.py`, find:

```python
    bundle_result: dict[str, Any] | None = None
    if domain_strategy is None:
        try:
            from imajin.tools import results as _results

            bundle_result = _results.save_result_bundle(
                name=f"{target_layer}_{method}_analysis",
                labels_layers=[seg_result["labels_layer"]],
                table_names=[measure_result["table_name"]],
                qc_png_paths=[seg_result.get("qc_png_path")] if seg_result.get("qc_png_path") else [],
                metadata={
                    "target_channel": target_layer,
                    "target_source": resolution.source,
                    "segmentation_method": method,
                    "analysis_dim": "3d" if use_3d else "2d",
                    "labels_layer": seg_result["labels_layer"],
                    "table_name": measure_result["table_name"],
                    "n_objects": int(seg_result.get("n_objects", seg_result.get("n_cells", 0))),
                },
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"result bundle could not be saved: {type(exc).__name__}: {exc}")
```

and replace with:

```python
    bundle_path: Path | None = None
    bundle_outputs: dict[str, str | None] = {
        "labels_cells": None,
        "labels_domain": None,
        "qc_png": None,
    }
    if domain_strategy is None:
        from imajin.results import create_result_bundle, slugify_result_name as _slug
        from imajin.tools.results import (
            current_bundle,
            populate_sample_outputs,
            write_combined_csv,
            finalize_bundle_metadata,
        )

        sample_slug = _slug(target_layer)
        parent = current_bundle()
        own_bundle = parent is None
        if own_bundle:
            bundle_path = create_result_bundle(
                name=target_layer,
                kind="single",
                tier="single_tier",
                metadata={
                    "recipe": None,
                    "target_channel": target_layer,
                    "target_source": resolution.source,
                    "segmentation_method": method,
                    "analysis_dim": "3d" if use_3d else "2d",
                },
            )
        else:
            bundle_path = parent

        try:
            bundle_outputs = populate_sample_outputs(
                bundle_path,
                sample_slug=sample_slug,
                labels_cells=seg_result["labels_layer"],
                qc_png=seg_result.get("qc_png_path"),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"bundle outputs could not be written: {type(exc).__name__}: {exc}"
            )

        if own_bundle:
            sample_summary = _build_sample_summary(
                sample_name=target_layer,
                status="complete",
                n_cells=int(seg_result.get("n_objects", seg_result.get("n_cells", 0))),
                qc_warnings=list(seg_result.get("qc_warnings", [])),
                outputs=bundle_outputs,
                source_layer=target_layer,
            )
            try:
                write_combined_csv(bundle_path, [measure_result["table_name"]])
                finalize_bundle_metadata(
                    bundle_path,
                    samples=[sample_summary],
                    status="complete",
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"bundle could not be finalized: {type(exc).__name__}: {exc}"
                )
```

Then change the return-dict references from `bundle_result.get("bundle_path") if bundle_result else None` and `bundle_result.get("outputs") if bundle_result else {}`. Find:

```python
        "result_bundle_path": bundle_result.get("bundle_path") if bundle_result else None,
        "result_files": bundle_result.get("outputs") if bundle_result else {},
```

and replace with:

```python
        "result_bundle_path": str(bundle_path) if bundle_path is not None else None,
        "result_files": dict(bundle_outputs),
```

- [ ] **Step 5: Add the two finalizer helpers in tools/results.py**

Append to `src/imajin/tools/results.py`:

```python
def write_combined_csv(bundle: Path, table_names: list[str]) -> Path:
    """Concat the given measurement tables and write to bundle/tables/combined.csv."""
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for name in table_names:
        try:
            frame = get_table(name)
        except KeyError:
            continue
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    out = bundle / "tables" / "combined.csv"
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()
    combined.to_csv(out, index=False)
    return out


def finalize_bundle_metadata(
    bundle: Path,
    *,
    samples: list[dict[str, Any]],
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Update bundle/metadata.json with the final samples index and status."""
    meta = read_bundle_metadata(bundle)
    meta["status"] = status
    meta["samples"] = list(samples)
    meta["n_samples"] = len(samples)
    meta["n_complete"] = sum(1 for s in samples if s.get("status") == "complete")
    meta["n_failed"] = sum(1 for s in samples if s.get("status") == "failed")
    meta["tables"] = {"combined": "tables/combined.csv"}
    if extra:
        meta.update(extra)
    write_bundle_metadata(bundle, meta)
```

- [ ] **Step 6: Run the new test**

```bash
uv run pytest tests/test_phase2_workflow.py::test_analyze_target_cells_single_tier_writes_new_layout_bundle -v
```

Expected: pass.

- [ ] **Step 7: Run the full phase2/phase3 suites**

```bash
uv run pytest tests/test_phase2_workflow.py tests/test_phase3_experiment.py -q
```

Expect some existing tests to fail (they assert the old layout; addressed in Task 16). Note specifically which ones; everything else should pass.

- [ ] **Step 8: Commit**

```bash
git add src/imajin/tools/workflows.py src/imajin/tools/results.py tests/test_phase2_workflow.py
git commit -m "feat(workflows): single-tier analyze_target_cells writes new-layout bundle"
```

---

## Task 7: analyze_target_cells two-tier writes bundle

**Files:**
- Modify: `src/imajin/tools/workflows.py` (two-tier branch — `domain_strategy is not None` block)
- Modify: `tests/test_phase2_workflow.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase2_workflow.py`:

```python
def test_analyze_target_cells_two_tier_writes_bundle(
    viewer, tmp_path, monkeypatch
) -> None:
    from imajin.tools.workflows import analyze_target_cells

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    img[120:160, 120:160] += 12.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    res = analyze_target_cells(
        target="reporter",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )
    assert res["ok"] is True
    bundle = Path(res["result_bundle_path"])
    assert bundle.exists()
    assert (bundle / "labels" / "cells" / "reporter.tif").exists()
    assert (bundle / "labels" / "domain" / "reporter.tif").exists()

    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["kind"] == "single"
    assert meta["tier"] == "two_tier"
    assert meta["status"] == "complete"
    assert len(meta["samples"]) == 1
    assert meta["samples"][0]["outputs"]["labels_domain"] == "labels/domain/reporter.tif"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase2_workflow.py::test_analyze_target_cells_two_tier_writes_bundle -v
```

Expected: failure — current two-tier path returns `result_bundle_path=None`.

- [ ] **Step 3: Add the bundle write into the two-tier branch**

In `src/imajin/tools/workflows.py`, find the two-tier return at the end of the `if domain_strategy is not None:` block:

```python
        return {
            "ok": True,
            "target_channel": target_layer,
            "target_source": resolution.source,
            "preprocess": pre_step,
            "preprocessed_layer": pre_record["new_layer"] if pre_record else None,
            "segmentation_method": method,
            "analysis_dim": "3d" if use_3d else "2d",
            "labels_layer": seg_result["labels_layer"],
            "cells_layer": seg_result["labels_layer"],
            "domain_layer": domain_layer,
            "n_domain_components": domain_result["n_components"],
            "domain_area_um2": domain_result["domain_area_um2"],
            "n_cells": int(seg_result.get("n_objects", 0)),
            "tier_table_name": tier_table_name,
            "table_name": measure_result["table_name"],
            "table_columns": measure_result["columns"],
            "qc_png_path": seg_result.get("qc_png_path"),
            "qc_png_error": seg_result.get("qc_png_error"),
            "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
            "voxel_scale": voxel,
            "warnings": warnings + list(domain_result.get("counterstain_warnings", [])),
        }
```

Just **before** the return, insert:

```python
        from imajin.results import create_result_bundle, slugify_result_name as _slug
        from imajin.tools.results import (
            current_bundle,
            populate_sample_outputs,
            write_combined_csv,
            finalize_bundle_metadata,
        )

        sample_slug = _slug(target_layer)
        parent = current_bundle()
        own_bundle = parent is None
        if own_bundle:
            bundle_path = create_result_bundle(
                name=target_layer,
                kind="single",
                tier="two_tier",
                metadata={
                    "recipe": None,
                    "target_channel": target_layer,
                    "target_source": resolution.source,
                    "segmentation_method": method,
                    "analysis_dim": "3d" if use_3d else "2d",
                },
            )
        else:
            bundle_path = parent

        bundle_outputs: dict[str, str | None] = {
            "labels_cells": None,
            "labels_domain": None,
            "qc_png": None,
        }
        try:
            bundle_outputs = populate_sample_outputs(
                bundle_path,
                sample_slug=sample_slug,
                labels_cells=seg_result["labels_layer"],
                labels_domain=domain_layer,
                qc_png=seg_result.get("qc_png_path"),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"bundle outputs could not be written: {type(exc).__name__}: {exc}"
            )

        if own_bundle:
            sample_summary = _build_sample_summary(
                sample_name=target_layer,
                status="complete",
                n_cells=int(seg_result.get("n_objects", 0)),
                n_domain_components=domain_result["n_components"],
                domain_area_um2=domain_result["domain_area_um2"],
                qc_warnings=list(domain_result.get("counterstain_warnings", [])),
                outputs=bundle_outputs,
                source_layer=target_layer,
            )
            try:
                write_combined_csv(bundle_path, [tier_table_name])
                finalize_bundle_metadata(
                    bundle_path,
                    samples=[sample_summary],
                    status="complete",
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"bundle could not be finalized: {type(exc).__name__}: {exc}"
                )
```

Then add to the return dict (in the same block):

```python
            "result_bundle_path": str(bundle_path) if own_bundle else None,
            "result_files": dict(bundle_outputs),
```

(Add these as new keys; do not remove existing keys.)

- [ ] **Step 4: Run the new test**

```bash
uv run pytest tests/test_phase2_workflow.py::test_analyze_target_cells_two_tier_writes_bundle -v
```

Expected: pass.

- [ ] **Step 5: Run all phase2 tests**

```bash
uv run pytest tests/test_phase2_workflow.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase2_workflow.py
git commit -m "feat(workflows): two-tier analyze_target_cells writes domain+cells bundle"
```

---

## Task 8: Both branches honour active parent bundle

**Files:**
- Modify: `tests/test_phase2_workflow.py` (add coverage; logic already accounts for current_bundle())

The code in Tasks 6 and 7 already calls `current_bundle()` and skips own-bundle creation when a parent is set. This task is a regression test that the parent-honouring path actually works without finalizing a separate bundle.

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase2_workflow.py`:

```python
def test_analyze_target_cells_writes_into_active_parent_bundle(
    viewer, tmp_path, monkeypatch
) -> None:
    """When a parent bundle is active, no own-bundle is created."""
    from imajin.results import create_result_bundle
    from imajin.tools.results import with_active_bundle
    from imajin.tools.workflows import analyze_target_cells

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    parent = create_result_bundle(name="parent", kind="batch", tier="single_tier")

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    with with_active_bundle(parent):
        res = analyze_target_cells(target="reporter")

    assert res["ok"] is True
    # Standalone bundle should NOT be created
    assert res["result_bundle_path"] is None
    # Outputs land in parent
    assert (parent / "labels" / "cells" / "reporter.tif").exists()
    # Parent's metadata.json was NOT finalized by the per-sample call
    meta = json.loads((parent / "metadata.json").read_text())
    assert meta["status"] == "in_progress"
    assert "samples" not in meta
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase2_workflow.py::test_analyze_target_cells_writes_into_active_parent_bundle -v
```

Expected: pass already if Tasks 6+7 are correct. **If it fails**, reread the inserted blocks and confirm the `if own_bundle:` guard around `write_combined_csv` and `finalize_bundle_metadata`. The failure's message will pinpoint whether the parent's metadata was incorrectly overwritten or the labels file was not written.

If it passes immediately, that's expected (the behavior is already implemented).

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase2_workflow.py
git commit -m "test(workflows): analyze_target_cells writes into active parent bundle"
```

---

## Task 9: analyze_target_cells exposes primary_table_name

**Files:**
- Modify: `src/imajin/tools/workflows.py` (return dict in both branches)
- Modify: `tests/test_phase2_workflow.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase2_workflow.py`:

```python
def test_analyze_target_cells_returns_primary_table_name_single_tier(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))
    res = analyze_target_cells(target="reporter")
    assert res["primary_table_name"] == res["table_name"]


def test_analyze_target_cells_returns_primary_table_name_two_tier(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))
    res = analyze_target_cells(
        target="reporter",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )
    assert res["primary_table_name"] == res["tier_table_name"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_phase2_workflow.py -k primary_table_name -v
```

Expected: KeyError on `primary_table_name`.

- [ ] **Step 3: Add field to both return dicts**

In `src/imajin/tools/workflows.py`:

In the **single-tier** return (the one with `"table_name": measure_result["table_name"]`), add:

```python
        "primary_table_name": measure_result["table_name"],
```

In the **two-tier** return (the one with `"tier_table_name": tier_table_name`), add:

```python
            "primary_table_name": tier_table_name,
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_phase2_workflow.py -k primary_table_name -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase2_workflow.py
git commit -m "feat(workflows): analyze_target_cells exposes primary_table_name"
```

---

## Task 10: run_recipe_on_samples creates parent bundle and activates context

**Files:**
- Modify: `src/imajin/tools/workflows.py:718` (start of `run_recipe_on_samples`)
- Modify: `tests/test_phase3_experiment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase3_experiment.py`:

```python
def test_run_recipe_on_samples_creates_parent_bundle(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "ctrl_1", "group": "control",
          "files": [str(p)], "layers": ["ctrl_1_ch0"]}]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area", "mean_intensity"]},
    )

    res = workflows.run_recipe_on_samples(recipe_name="r1")

    assert res.get("bundle_path"), "run_recipe_on_samples must return bundle_path"
    bundle = Path(res["bundle_path"])
    assert bundle.is_dir()
    assert bundle.name.startswith(_)  # noqa: F821 — see step
    # Replace assertion line below in next step with explicit timestamp pattern check
```

Replace the comment-only check with the proper assertion (one block instead):

```python
def test_run_recipe_on_samples_creates_parent_bundle(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    import re
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "ctrl_1", "group": "control",
          "files": [str(p)], "layers": ["ctrl_1_ch0"]}]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area", "mean_intensity"]},
    )

    res = workflows.run_recipe_on_samples(recipe_name="r1")

    assert res.get("bundle_path"), "run_recipe_on_samples must return bundle_path"
    bundle = Path(res["bundle_path"])
    assert bundle.is_dir()
    assert re.match(r"^\d{8}_\d{6}_r1$", bundle.name), bundle.name
    # Sample output landed in the parent bundle, not a sibling per-call bundle
    siblings = sorted(bundle.parent.iterdir())
    assert len(siblings) == 1, [s.name for s in siblings]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_creates_parent_bundle -v
```

Expected: fails — current runner doesn't create or return `bundle_path`.

- [ ] **Step 3: Wrap the runner in a parent-bundle context**

In `src/imajin/tools/workflows.py`, find the start of `run_recipe_on_samples`. Just after the early-empty-return:

```python
    if not sample_names:
        return {
            "recipe": recipe_name,
            "n_samples": 0,
            "n_complete": 0,
            "n_failed": 0,
            "runs": [],
        }
```

continue past the existing `mode = ...` and `seg = ...` setup, then insert the parent-bundle creation. Specifically, just before the loop `for index, name in enumerate(sample_names):`:

```python
    from imajin.results import create_result_bundle
    from imajin.tools.results import with_active_bundle

    parent_bundle = create_result_bundle(
        name=recipe.name,
        kind="batch",
        tier="two_tier" if domain_strategy is not None else "single_tier",
        metadata={
            "recipe": {
                "name": recipe.name,
                "target_channel": recipe.target_channel,
                "preprocessing": list(recipe.preprocessing or []),
                "segmentation": dict(recipe.segmentation or {}),
                "measurement": dict(recipe.measurement or {}),
                "domain": dict(recipe.domain) if recipe.domain else None,
                "cell_diameter_um": recipe.cell_diameter_um,
            },
        },
    )
```

Then wrap the `for index, name in enumerate(sample_names):` loop in `with with_active_bundle(parent_bundle):`. The simplest refactor: extract the whole loop body unchanged into the `with` block.

Finally, change the closing return at the end of `run_recipe_on_samples` from:

```python
    return {
        "recipe": recipe_name,
        "n_samples": total,
        "n_complete": n_complete,
        "n_failed": n_failed,
        "cleanup_enabled": cleanup_enabled,
        "runs": runs,
    }
```

to:

```python
    return {
        "recipe": recipe_name,
        "n_samples": total,
        "n_complete": n_complete,
        "n_failed": n_failed,
        "cleanup_enabled": cleanup_enabled,
        "runs": runs,
        "bundle_path": str(parent_bundle),
    }
```

(If your existing return shape differs from the snippet above, find the actual `return {...}` at the end of `run_recipe_on_samples` and add `"bundle_path": str(parent_bundle)`. Don't change other keys.)

- [ ] **Step 4: Run new test**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_creates_parent_bundle -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase3_experiment.py
git commit -m "feat(workflows): run_recipe_on_samples creates parent bundle and activates context"
```

---

## Task 11: Attach sample columns to two-tier tier_table_name

**Files:**
- Modify: `src/imajin/tools/workflows.py` (run_recipe_on_samples per-sample success branch)
- Modify: `tests/test_phase3_experiment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase3_experiment.py`:

```python
def test_run_recipe_on_samples_two_tier_attaches_sample_cols_to_tier_table(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "ctrl_1", "group": "control",
          "files": [str(p)], "layers": ["ctrl_1_ch0"]}]
    )
    experiment.create_analysis_recipe(
        name="two_tier_r",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
        domain={"strategy": "noise_floor", "k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )

    res = workflows.run_recipe_on_samples(recipe_name="two_tier_r")
    assert res["n_complete"] == 1
    run = res["runs"][0]
    table_names = run.get("table_names") or []
    # tier_table_name should be among the run's tables AND have sample columns
    assert table_names
    has_sample_attached = False
    for tname in table_names:
        df = state.get_table(tname)
        if "tier" in df.columns and "sample_name" in df.columns:
            has_sample_attached = True
            assert (df["sample_name"] == "ctrl_1").all()
    assert has_sample_attached, f"tier table missing sample columns; tables={table_names}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_two_tier_attaches_sample_cols_to_tier_table -v
```

Expected: failure — currently only `result.table_name` (cells-only) gets sample columns attached.

- [ ] **Step 3: Attach to tier table when present**

In `src/imajin/tools/workflows.py`, find the per-sample success branch:

```python
            else:
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
```

Replace with:

```python
            else:
                attached_tables: list[str] = []
                for tname_key in ("table_name", "tier_table_name"):
                    tname = result.get(tname_key)
                    if not tname or tname in attached_tables:
                        continue
                    attach_sample_columns_to_table(
                        table_name=tname,
                        sample_id=s.sample_id,
                        sample_name=s.sample_name,
                        group=s.group,
                        file_id=info["file_id"],
                        source_file=info["file_path"],
                        source_layer=result.get("target_channel"),
                    )
                    attached_tables.append(tname)
                table_name = attached_tables[0] if attached_tables else None
```

Then in the same block, change the `put_run(... table_names=[table_name] if table_name else [], ...)` to use `attached_tables` (i.e. all attached, not just the first). Find:

```python
                run_id = put_run(
                    sample_id=s.sample_id,
                    file_id=info["file_id"] or "",
                    recipe_id=recipe.recipe_id,
                    status="complete",
                    table_names=[table_name] if table_name else [],
```

and change `table_names=[table_name] if table_name else []` to `table_names=list(attached_tables)`.

- [ ] **Step 4: Run new test**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_two_tier_attaches_sample_cols_to_tier_table -v
```

Expected: pass.

- [ ] **Step 5: Run all run_recipe tests**

```bash
uv run pytest tests/test_phase3_experiment.py -k run_recipe -v
```

Expected: pre-existing run_recipe tests still pass; new tests pass. Some existing single-tier tests may now have `table_names` of length 1 (which they were before) — verify no `len(table_names) == 1` assertions break unintentionally.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase3_experiment.py
git commit -m "fix(workflows): attach sample columns to tier_table_name in two-tier batch"
```

---

## Task 12: Write tables/combined.csv at end of run_recipe_on_samples

**Files:**
- Modify: `src/imajin/tools/workflows.py` (after sample loop)
- Modify: `tests/test_phase3_experiment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase3_experiment.py`:

```python
def test_run_recipe_on_samples_writes_combined_csv(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    import pandas as pd
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    viewer.add_image(img.copy(), name="trt_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])
    experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control",
             "files": [str(a)], "layers": ["ctrl_1_ch0"]},
            {"sample_name": "trt_1", "group": "treatment",
             "files": [str(b)], "layers": ["trt_1_ch0"]},
        ]
    )
    experiment.create_analysis_recipe(
        name="r_combined",
        target_channel="ctrl_1_ch0",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area", "mean_intensity"]},
    )

    res = workflows.run_recipe_on_samples(recipe_name="r_combined")
    bundle = Path(res["bundle_path"])
    combined = bundle / "tables" / "combined.csv"
    assert combined.exists()

    df = pd.read_csv(combined)
    assert {"sample_name", "group", "file_id"}.issubset(df.columns)
    assert set(df["sample_name"].unique()) == {"ctrl_1", "trt_1"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_writes_combined_csv -v
```

Expected: failure — `tables/combined.csv` not present.

- [ ] **Step 3: Write combined.csv after the loop**

In `src/imajin/tools/workflows.py`, just before the final return at the end of `run_recipe_on_samples`, insert:

```python
    from imajin.tools.results import write_combined_csv

    primary_tables: list[str] = []
    for run in runs:
        for tname in run.get("table_names") or []:
            if tname and tname not in primary_tables:
                primary_tables.append(tname)
    write_combined_csv(parent_bundle, primary_tables)
```

Note: For two-tier batches, `run.table_names` includes both `table_name` and `tier_table_name` (Task 11). For combined.csv we only want ONE per run — the `tier_table_name` for two-tier (long format) or `table_name` for single-tier. Replace the snippet above with:

```python
    from imajin.tools.results import write_combined_csv

    primary_tables: list[str] = []
    for run in runs:
        # Take the LAST attached table per run; for two-tier this is tier_table_name
        # (long format with `tier` column), for single-tier this is table_name.
        names = run.get("table_names") or []
        if names:
            primary_tables.append(names[-1])
    write_combined_csv(parent_bundle, primary_tables)
```

This relies on Task 11's ordering: `table_name` is appended before `tier_table_name`. Verify by reading the implementation.

- [ ] **Step 4: Run new test**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_writes_combined_csv -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase3_experiment.py
git commit -m "feat(workflows): batch writes tables/combined.csv with sample columns"
```

---

## Task 13: Finalize metadata.json with samples + counts + status

**Files:**
- Modify: `src/imajin/tools/workflows.py` (per-sample success/failure branches collect summaries; finalize after loop)
- Modify: `tests/test_phase3_experiment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase3_experiment.py`:

```python
def test_run_recipe_on_samples_finalizes_metadata_with_samples(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "ctrl_1", "group": "control",
          "files": [str(p)], "layers": ["ctrl_1_ch0"]}]
    )
    experiment.create_analysis_recipe(
        name="r_meta",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    res = workflows.run_recipe_on_samples(recipe_name="r_meta")
    bundle = Path(res["bundle_path"])
    meta = json.loads((bundle / "metadata.json").read_text())

    assert meta["kind"] == "batch"
    assert meta["tier"] == "single_tier"
    assert meta["status"] == "complete"
    assert meta["n_samples"] == 1
    assert meta["n_complete"] == 1
    assert meta["n_failed"] == 0
    assert len(meta["samples"]) == 1
    s = meta["samples"][0]
    assert s["sample_name"] == "ctrl_1"
    assert s["group"] == "control"
    assert s["file_id"] == "ctrl_1"
    assert s["status"] == "complete"
    assert s["outputs"]["labels_cells"] == "labels/cells/ctrl_1.tif"
```

Add `import json` at the top of the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_finalizes_metadata_with_samples -v
```

Expected: failure — bundle metadata still says `status="in_progress"` and lacks `samples`.

- [ ] **Step 3: Collect per-sample summaries during the loop**

In `src/imajin/tools/workflows.py`, just after the `runs: list[dict[str, Any]] = []` initialization, add:

```python
    sample_summaries: list[dict[str, Any]] = []
```

In the per-sample failure branch (the `except Exception as exc` block), after the existing `runs.append(...)` add:

```python
            sample_summaries.append(
                _build_sample_summary(
                    sample_name=s.sample_name,
                    status="failed",
                    error=str(exc),
                    group=s.group,
                    file_id=info["file_id"],
                    source_file=info["file_path"],
                    source_layer=info["layer_name"],
                )
            )
```

In the per-sample success branch (`else:` after `if not result.get("ok"):` else), after existing `put_run(...)` and `runs.append(...)`, append:

```python
                sample_summaries.append(
                    _build_sample_summary(
                        sample_name=s.sample_name,
                        status="complete",
                        n_cells=int(result.get("n_cells", result.get("n_objects", 0)) or 0),
                        n_domain_components=result.get("n_domain_components"),
                        domain_area_um2=result.get("domain_area_um2"),
                        qc_warnings=list(result.get("warnings") or []),
                        outputs=dict(result.get("result_files") or {}),
                        group=s.group,
                        file_id=info["file_id"],
                        source_file=info["file_path"],
                        source_layer=result.get("target_channel"),
                    )
                )
```

Also in the **`if not result.get("ok"):` failed branch** (where `failed_sample = True` is set after a non-OK result), add:

```python
                sample_summaries.append(
                    _build_sample_summary(
                        sample_name=s.sample_name,
                        status="failed",
                        error=result.get("error", "analysis returned ok=false"),
                        group=s.group,
                        file_id=info["file_id"],
                        source_file=info["file_path"],
                        source_layer=info["layer_name"],
                    )
                )
```

- [ ] **Step 4: Finalize metadata after the loop**

Replace the `write_combined_csv(parent_bundle, primary_tables)` line you added in Task 12 with:

```python
    from imajin.tools.results import write_combined_csv, finalize_bundle_metadata

    primary_tables: list[str] = []
    for run in runs:
        names = run.get("table_names") or []
        if names:
            primary_tables.append(names[-1])
    write_combined_csv(parent_bundle, primary_tables)
    finalize_bundle_metadata(
        parent_bundle, samples=sample_summaries, status="complete"
    )
```

- [ ] **Step 5: Run new test**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_finalizes_metadata_with_samples -v
```

Expected: pass.

- [ ] **Step 6: Add a failed-sample test**

Append to `tests/test_phase3_experiment.py`:

```python
def test_run_recipe_on_samples_metadata_records_failed_sample(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    viewer.add_image(np.zeros_like(img), name="trt_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])
    experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control",
             "files": [str(a)], "layers": ["ctrl_1_ch0"]},
            {"sample_name": "trt_1", "group": "treatment",
             "files": [str(b)], "layers": ["trt_1_ch0"]},
        ]
    )
    experiment.create_analysis_recipe(
        name="r_fail",
        target_channel="ctrl_1_ch0",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    res = workflows.run_recipe_on_samples(
        recipe_name="r_fail", sample_names=["ctrl_1", "trt_1"]
    )
    bundle = Path(res["bundle_path"])
    meta = json.loads((bundle / "metadata.json").read_text())
    statuses = sorted(s["status"] for s in meta["samples"])
    assert statuses == ["complete", "failed"]
    assert meta["n_complete"] == 1
    assert meta["n_failed"] == 1
    failed = next(s for s in meta["samples"] if s["status"] == "failed")
    assert failed["error"]
    # No labels file written for the failed sample
    failed_slug = failed["sample_name"]
    assert not (bundle / "labels" / "cells" / f"{failed_slug}.tif").exists()
```

- [ ] **Step 7: Run failed-sample test**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_metadata_records_failed_sample -v
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase3_experiment.py
git commit -m "feat(workflows): finalize bundle metadata with samples index and counts"
```

---

## Task 14: Cancellation handling in try/finally

**Files:**
- Modify: `src/imajin/tools/workflows.py` (wrap loop + finalization in try/finally)
- Modify: `tests/test_phase3_experiment.py:752-783` (existing cancellation test) — update for new behaviour

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase3_experiment.py`:

```python
def test_run_recipe_on_samples_cancellation_finalizes_metadata(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows
    from imajin.workers.qt_worker import CancelledError

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    viewer.add_image(img.copy(), name="trt_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])
    experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control",
             "files": [str(a)], "layers": ["ctrl_1_ch0"]},
            {"sample_name": "trt_1", "group": "treatment",
             "files": [str(b)], "layers": ["trt_1_ch0"]},
        ]
    )
    experiment.create_analysis_recipe(
        name="r_cancel",
        target_channel="ctrl_1_ch0",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    call = {"n": 0}
    real = workflows.analyze_target_cells

    def side_effect(*args, **kwargs):
        call["n"] += 1
        if call["n"] == 1:
            return real(*args, **kwargs)
        raise CancelledError("Tool execution cancelled by user.")

    monkeypatch.setattr(workflows, "analyze_target_cells", side_effect)

    with pytest.raises(CancelledError):
        workflows.run_recipe_on_samples(
            recipe_name="r_cancel", sample_names=["ctrl_1", "trt_1"]
        )

    # Bundle exists despite cancellation
    bundles_dir = tmp_path / "bundles"
    bundles = list(bundles_dir.iterdir())
    assert len(bundles) == 1
    bundle = bundles[0]
    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["status"] == "cancelled"
    statuses = [s["status"] for s in meta["samples"]]
    assert statuses == ["complete", "skipped"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_cancellation_finalizes_metadata -v
```

Expected: failure — current behaviour propagates CancelledError without finalizing the bundle.

- [ ] **Step 3: Wrap loop + finalize in try/finally**

In `src/imajin/tools/workflows.py`, refactor `run_recipe_on_samples` so the `with with_active_bundle(parent_bundle):` block uses try/finally. Pseudocode of the structure (replace the existing loop + post-loop block):

```python
    with with_active_bundle(parent_bundle):
        cancelled = False
        try:
            for index, name in enumerate(sample_names):
                # ... existing loop body unchanged ...
                pass
        except CancelledError:
            cancelled = True
            # Mark unprocessed samples as skipped
            processed_names = {s["sample_name"] for s in sample_summaries}
            for name in sample_names:
                if name not in processed_names:
                    try:
                        s2 = state.get_sample(name) if False else None  # see note
                    except Exception:
                        s2 = None
                    sample_summaries.append(
                        _build_sample_summary(
                            sample_name=name,
                            status="skipped",
                            group=None,
                            file_id=None,
                        )
                    )
            raise
        finally:
            from imajin.tools.results import write_combined_csv, finalize_bundle_metadata

            primary_tables: list[str] = []
            for run in runs:
                names = run.get("table_names") or []
                if names:
                    primary_tables.append(names[-1])
            try:
                write_combined_csv(parent_bundle, primary_tables)
                finalize_bundle_metadata(
                    parent_bundle,
                    samples=sample_summaries,
                    status="cancelled" if cancelled else "complete",
                )
            except Exception:
                pass  # never let bundle finalization mask the original error
```

Concretely: locate the `try:` for the existing per-sample work and the post-loop `write_combined_csv` / `finalize_bundle_metadata` lines (added in Tasks 12-13). Wrap the **entire** `for index, name in enumerate(sample_names)` loop and the post-loop finalize calls in an outer `try / except CancelledError / finally`. Move the finalize calls into the `finally` block, parametrize `status` based on whether `CancelledError` fired, append "skipped" entries for unreached samples in the `except CancelledError` block, then `raise`.

For looking up sample metadata of skipped samples (group/file_id), prefer to call `state.get_sample(name)` and tolerate `KeyError`:

```python
            processed_names = {s["sample_name"] for s in sample_summaries}
            for name in sample_names:
                if name in processed_names:
                    continue
                try:
                    s_skip = state.get_sample(name)
                    group = s_skip.group
                    file_id = s_skip.file_ids[0] if s_skip.file_ids else None
                except Exception:
                    group = None
                    file_id = None
                sample_summaries.append(
                    _build_sample_summary(
                        sample_name=name,
                        status="skipped",
                        group=group,
                        file_id=file_id,
                    )
                )
```

(Add `from imajin.agent import state` if not already imported in this function's scope.)

- [ ] **Step 4: Run new test**

```bash
uv run pytest tests/test_phase3_experiment.py::test_run_recipe_on_samples_cancellation_finalizes_metadata -v
```

Expected: pass.

- [ ] **Step 5: Update existing cancellation test**

The pre-existing test at `tests/test_phase3_experiment.py::test_run_recipe_on_samples_propagates_cancellation` asserts `state.list_runs() == []` after cancellation. With the new behaviour, that assertion still holds (no successful sample finalized), but the test should ALSO assert that a bundle exists. Find the existing test and replace its body:

```python
def test_run_recipe_on_samples_propagates_cancellation(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows
    from imajin.workers.qt_worker import CancelledError

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    p = tmp_path / "sample_1.tif"
    p.write_bytes(b"fake")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "sample_1", "group": "g", "file_ids": ["sample_1"]}]
    )
    experiment.create_analysis_recipe(
        name="cancel_recipe",
        target_channel="green",
        segmentation={"method": "intensity_regions"},
    )
    viewer.add_image(
        np.ones((8, 8), dtype=np.float32),
        name="sample_1_green",
        metadata={"source_path": str(p.resolve()), "color": "green"},
    )

    def fake_analyze_target_cells(*args, **kwargs) -> dict[str, object]:
        raise CancelledError("Tool execution cancelled by user.")

    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze_target_cells)

    with pytest.raises(CancelledError):
        workflows.run_recipe_on_samples(recipe_name="cancel_recipe")

    assert state.list_runs() == []
    bundles = list((tmp_path / "bundles").iterdir())
    assert len(bundles) == 1
    meta = json.loads((bundles[0] / "metadata.json").read_text())
    assert meta["status"] == "cancelled"
```

- [ ] **Step 6: Run both cancellation tests**

```bash
uv run pytest tests/test_phase3_experiment.py -k cancel -v
```

Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase3_experiment.py
git commit -m "feat(workflows): cancellation produces a finalized cancelled bundle"
```

---

## Task 15: Agent prompt mention of bundle_path

**Files:**
- Modify: `src/imajin/agent/prompts.py` (the batch-analysis bullet)

- [ ] **Step 1: Update the batch-analysis bullet**

In `src/imajin/agent/prompts.py`, find the bullet that ends with:

```
  Never put 'expression_domain' in the segmentation slot; the runner will reject it.
```

and append after it (within the same bullet) the sentence:

```
  When the batch finishes, `run_recipe_on_samples` returns `bundle_path`, the
  one folder containing every sample's labels/cells/, labels/domain/ (two-tier
  only), tables/combined.csv, qc/, and metadata.json. Cite this path when
  reporting batch outcomes to the user.
```

- [ ] **Step 2: Commit**

```bash
git add src/imajin/agent/prompts.py
git commit -m "docs(prompts): batch-analysis guidance cites bundle_path"
```

---

## Task 16: Update existing tests for new layout

**Files:**
- Modify: `tests/test_phase3_experiment.py` — update `test_run_recipe_on_samples_*` tests that referenced the old per-sample bundle behaviour

The earlier full-suite runs in Tasks 6-7 will have surfaced regressions. Address them here.

- [ ] **Step 1: Run full suite to catalogue current failures**

```bash
uv run pytest tests/ -q 2>&1 | tail -40
```

Expected: a list of failures, all related to the layout migration.

- [ ] **Step 2: Update assertions for each failing test**

Common patterns to migrate:
- Tests that expected `result["result_bundle_path"]` to be a per-sample path inside a sibling bundle: should now read the sample's path inside the parent bundle (`bundle / "labels" / "cells" / "<slug>.tif"`).
- Tests that used the old `labels/<name>.tif` path: change to `labels/cells/<name>.tif`.
- Tests that asserted on `save_result_bundle` outputs containing `"labels": [".../labels/<name>.tif"]`: change to `.../labels/cells/<name>.tif`.

For each failing test, read it, update assertions to match the new layout, and re-run that single test to confirm the green state. Do **not** silence assertions; only adjust paths.

- [ ] **Step 3: Run full suite again**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: migrate existing tests to new bundle layout"
```

---

## Task 17: Final verification

- [ ] **Step 1: Run full test suite end-to-end**

```bash
uv run pytest tests/ -q
```

Expected: all green, zero failures.

- [ ] **Step 2: Spot-check a real batch by hand (optional sanity)**

In a Python REPL or scratch script, exercise the user's actual scenario synthetically (no GUI required):

```python
from pathlib import Path
import os, tempfile, numpy as np

tmp = tempfile.mkdtemp()
os.environ["IMAJIN_RESULTS_DIR"] = tmp

from imajin.agent import state
from imajin.tools import experiment, workflows

# (Set up fake files + samples + recipe; same shape as the failing batch from
# session 242b4fd1b00c.jsonl, but with a synthetic image for each sample.)
# Then run:
res = workflows.run_recipe_on_samples(recipe_name="<name>")
print(res["bundle_path"])
print(sorted(Path(res["bundle_path"]).rglob("*")))
```

Confirm the bundle has the expected layout and `metadata.json` is well-formed.

- [ ] **Step 3: Verify no spurious files left in repo**

```bash
git status
```

Expected: clean working tree.

---

## Self-Review

**Spec coverage check:**
- KST timestamps — Task 1, Task 3 ✓
- New layout (`labels/cells`, `labels/domain`, `tables`, `qc`, `stats`, `figures`) — Task 3 ✓
- `metadata.json` schema (kind, tier, status, recipe, samples, env) — Tasks 3, 13 ✓
- Two-tier `analyze_target_cells` produces a bundle — Task 7 ✓
- Single-tier standalone migrates to new layout — Task 6 ✓
- Batch creates one parent bundle — Task 10 ✓
- Per-sample analyze_target_cells writes into parent bundle — Task 8 ✓
- `tier_table_name` sample-column attach (bug fix) — Task 11 ✓
- `tables/combined.csv` written at end of batch — Task 12 ✓
- Failed sample logged in metadata — Task 13 ✓
- Cancellation finalizes bundle with `status="cancelled"` and skipped samples — Task 14 ✓
- Agent prompt cites `bundle_path` — Task 15 ✓
- `save_result_bundle` tool migrates to new label path — Task 3 ✓
- Old on-disk bundles untouched — implicit (no migration code) ✓

**Type / signature consistency:**
- `current_bundle()`, `with_active_bundle(path)`, `populate_sample_outputs(bundle, sample_slug, ...)`, `write_combined_csv(bundle, table_names)`, `finalize_bundle_metadata(bundle, samples=, status=, extra=)` — names match across tasks.
- `_build_sample_summary` — same kwargs in Task 6 (single-tier), Task 7 (two-tier), Task 13 (runner success/failure), Task 14 (skipped).
- `result_bundle_path`, `result_files`, `primary_table_name` — added consistently to both single- and two-tier return dicts.

**Placeholder check:** No TBD/TODO/"similar to Task N"; all code blocks complete. The only open-ended task is Task 16 (update existing tests), which is necessarily exploratory based on what the suite reports.
