# Unified Analysis Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse all per-analysis outputs into one `<timestamp>_<name>/` bundle folder, drop the flat `figures/`/`stats/`/`segmentation_qc/` fallback folders and `manifest.jsonl`, consolidate stats CSVs into long-format files, and fold table specs / output index / qc warnings into `metadata.json` (schema_version 3).

**Architecture:**
- Add new module-level entry points in `imajin/result_bundles.py`: `start_analysis`, `ensure_active_bundle`, `finalize_analysis`, `bundle_output_path`, `register_output`, `register_table_spec`, `register_stats_rows`.
- A process-global `Path | None` slot (guarded by a lock) replaces the ContextVar-only model so the bundle persists across LLM tool calls.
- Output writers (`tools/figures.py`, `tools/stats.py`, `tools/_segmentation_outputs.py`, `tools/results.py`) call the new helpers. Deprecated `unique_result_path`/`record_result` are removed.

**Tech Stack:** Python 3.12, pytest, pandas, contextvars + threading.Lock, pyobjc/Qt not relevant.

---

## File Structure

**Create:** none.

**Modify:**

| File | Responsibility after change |
|---|---|
| `src/imajin/result_bundles.py` | Single home for bundle lifecycle and write routing. Hosts `start_analysis`, `ensure_active_bundle`, `finalize_analysis`, `bundle_output_path`, `register_output`, `register_table_spec`, `register_stats_rows`, process-global slot, atexit hook, schema_v3 `finalize_bundle_metadata`. |
| `src/imajin/results.py` | Pure-bundle root helpers (`user_results_root`, `results_root`, `slugify_result_name`, `_collect_env_info`, `_kst_now`, `create_result_bundle`, `read_bundle_metadata`, `write_bundle_metadata`). Deprecated fallback API removed. `create_result_bundle` no longer wraps in `bundles/`. |
| `src/imajin/tools/figures.py` | All figure writes routed through `bundle_output_path("figures", filename)` and `register_output("figure", …)`. No flat fallback. |
| `src/imajin/tools/stats.py` | `describe_table`, `compare_groups`, `extract_timecourse_features` write via `register_stats_rows`. No per-`(value_col, level)` filenames. |
| `src/imajin/tools/_segmentation_outputs.py` | `_default_qc_png_path` returns `bundle_output_path("qc", …)`. No anchor-side `segmentation_qc/` folder. |
| `src/imajin/tools/_workflow_outputs.py` | `_remove_copied_standalone_qc` deleted; QC is already in the bundle by the time `populate_sample_outputs` runs. |
| `src/imajin/tools/results.py` | `save_result_bundle` writes via `bundle_output_path`, `register_output`, `register_table_spec`. No `tables/<name>.spec.json`. `_resolve_output_path` simplified. |
| `src/imajin/agent/specialists/...` or new `src/imajin/tools/bundle.py` | Adds two `@tool`-decorated functions `start_analysis` and `finalize_analysis` exposed to the LLM. |

**Test files affected:**

| File | Change |
|---|---|
| `tests/test_results_bundle.py` | Add tests for new lifecycle, schema_v3 fields, ad-hoc reuse. Update tests that read `run_context.tables`, `samples[].outputs`, `qc_warnings`. |
| `tests/test_tools_figures.py` | Assert figure path lives in active bundle's `figures/`. |
| `tests/test_tools_stats.py` | Assert single long-format CSV per source table. |
| `tests/test_tools_results.py` | Assert no `manifest.jsonl`, no `tables/<name>.spec.json`, spec in `metadata.json["table_specs"]`. |
| `tests/test_tools_segment.py` | Assert QC PNG only in `<bundle>/qc/<sample>.png`. |

---

## Task 1: Process-global ad-hoc bundle slot + `ensure_active_bundle`

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_results_bundle.py`:

```python
def test_ensure_active_bundle_lazy_creates_adhoc(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import (
        ensure_active_bundle,
        reset_process_bundle,
        current_bundle,
    )

    reset_process_bundle()
    assert current_bundle() is None

    bundle = ensure_active_bundle()
    assert bundle.parent == tmp_path
    assert bundle.name.endswith("_adhoc")
    assert (bundle / "metadata.json").exists()
    assert current_bundle() == bundle

    # Second call returns the same bundle.
    assert ensure_active_bundle() == bundle

    reset_process_bundle()


def test_ensure_active_bundle_respects_explicit_active_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.results import create_result_bundle
    from imajin.result_bundles import (
        ensure_active_bundle,
        reset_process_bundle,
        with_active_bundle,
    )

    reset_process_bundle()
    explicit = create_result_bundle("named", kind="single")
    with with_active_bundle(explicit):
        assert ensure_active_bundle() == explicit
    # After leaving the context, ad-hoc takes over.
    bundle = ensure_active_bundle()
    assert bundle != explicit
    assert bundle.name.endswith("_adhoc")
    reset_process_bundle()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_ensure_active_bundle_lazy_creates_adhoc tests/test_results_bundle.py::test_ensure_active_bundle_respects_explicit_active_bundle -v`

Expected: FAIL with `ImportError: cannot import name 'ensure_active_bundle' from 'imajin.result_bundles'`.

- [ ] **Step 3: Implement minimal `ensure_active_bundle` and `reset_process_bundle`**

Add to `src/imajin/result_bundles.py` (top-level, after the `_active_sample_slug` definition):

```python
import threading

_process_bundle_lock = threading.Lock()
_process_bundle: Path | None = None


def reset_process_bundle() -> None:
    """Drop the process-global ad-hoc bundle slot. Intended for tests."""
    global _process_bundle
    with _process_bundle_lock:
        _process_bundle = None


def ensure_active_bundle() -> Path:
    """Return the active bundle, creating a process-wide ad-hoc one if needed."""
    global _process_bundle
    ctx_bundle = _active_bundle.get()
    if ctx_bundle is not None:
        return ctx_bundle
    with _process_bundle_lock:
        if _process_bundle is None:
            from imajin.results import create_result_bundle

            _process_bundle = create_result_bundle(
                name="adhoc",
                kind="adhoc",
            )
        return _process_bundle
```

Update `current_bundle()` to also consider the process slot:

```python
def current_bundle() -> Path | None:
    ctx = _active_bundle.get()
    if ctx is not None:
        return ctx
    with _process_bundle_lock:
        return _process_bundle
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_ensure_active_bundle_lazy_creates_adhoc tests/test_results_bundle.py::test_ensure_active_bundle_respects_explicit_active_bundle -v`

Expected: 2 PASSED.

- [ ] **Step 5: Add the autouse fixture for tests**

Add to `tests/test_results_bundle.py`:

```python
@pytest.fixture(autouse=True)
def _reset_process_bundle():
    from imajin.result_bundles import reset_process_bundle

    reset_process_bundle()
    yield
    reset_process_bundle()
```

Run the full test file: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py -v`

Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): add process-global ad-hoc slot and ensure_active_bundle"
```

---

## Task 2: `bundle_output_path` helper

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_results_bundle.py`:

```python
def test_bundle_output_path_creates_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import bundle_output_path, ensure_active_bundle

    out = bundle_output_path("figures", "demo.png")
    bundle = ensure_active_bundle()
    assert out == bundle / "figures" / "demo.png"
    assert out.parent.is_dir()


def test_bundle_output_path_uses_active_named_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.results import create_result_bundle
    from imajin.result_bundles import bundle_output_path, with_active_bundle

    named = create_result_bundle("named", kind="single")
    with with_active_bundle(named):
        out = bundle_output_path("stats", "stuff.csv")
    assert out == named / "stats" / "stuff.csv"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_bundle_output_path_creates_parent tests/test_results_bundle.py::test_bundle_output_path_uses_active_named_bundle -v`

Expected: FAIL with `ImportError: cannot import name 'bundle_output_path' from 'imajin.result_bundles'`.

- [ ] **Step 3: Implement `bundle_output_path`**

Add to `src/imajin/result_bundles.py`:

```python
def bundle_output_path(category: str, filename: str) -> Path:
    """Resolve <bundle>/<category>/<filename>, lazily creating the bundle and parent."""
    bundle = ensure_active_bundle()
    out = bundle / category / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_bundle_output_path_creates_parent tests/test_results_bundle.py::test_bundle_output_path_uses_active_named_bundle -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): add bundle_output_path helper"
```

---

## Task 3: `start_analysis`, `finalize_analysis`, and schema_v3 metadata

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_results_bundle.py`:

```python
def test_start_analysis_creates_named_bundle_in_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import start_analysis, current_bundle
    from imajin.results import read_bundle_metadata

    bundle = start_analysis(name="J20_component1", kind="single")
    assert re.match(r"^\d{8}_\d{6}_J20_component1$", bundle.name)
    meta = read_bundle_metadata(bundle)
    assert meta["schema_version"] == 3
    assert meta["run_context"]["status"] == "in_progress"
    assert meta["run_context"]["name"] == "J20_component1"
    assert meta["run_context"]["kind"] == "single"
    assert current_bundle() == bundle


def test_finalize_analysis_writes_status_and_strips_redacted_fields(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import (
        finalize_analysis,
        start_analysis,
    )
    from imajin.results import read_bundle_metadata

    bundle = start_analysis(name="demo", kind="single")
    samples = [
        {
            "sample_name": "s1",
            "status": "complete",
            "summary": {
                "n_cells": 5,
                "qc_warnings": ["should be dropped"],
            },
            "outputs": {"labels_cells": "labels/cells/s1.tif"},
        }
    ]
    finalize_analysis(status="complete", samples=samples)

    meta = read_bundle_metadata(bundle)
    assert meta["schema_version"] == 3
    rc = meta["run_context"]
    assert rc["status"] == "complete"
    assert rc["finalized_at"] is not None
    assert rc["n_samples"] == 1
    assert rc["n_complete"] == 1
    sample = rc["samples"][0]
    assert "qc_warnings" not in sample.get("summary", {})
    assert "outputs" not in sample
    assert "tables" not in rc  # `run_context.tables` shorthand removed.
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_start_analysis_creates_named_bundle_in_progress tests/test_results_bundle.py::test_finalize_analysis_writes_status_and_strips_redacted_fields -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `start_analysis` and `finalize_analysis`**

Add to `src/imajin/result_bundles.py`:

```python
def start_analysis(
    name: str,
    *,
    kind: str = "single",
    tier: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a named bundle, write seed metadata.json (schema_v3, status='in_progress'),
    and set it as the active bundle for the calling context."""
    from imajin.results import create_result_bundle

    bundle = create_result_bundle(
        name=name,
        kind=kind,
        tier=tier,
        metadata=metadata,
    )
    # Promote the bundle into the process-global slot so cross-call tool writes
    # share it without a containing with-block.
    global _process_bundle
    with _process_bundle_lock:
        _process_bundle = bundle
    return bundle


def finalize_analysis(
    *,
    status: str = "complete",
    samples: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Finalize the currently active bundle. Writes schema_v3 metadata.json with the
    final status and clears the process slot."""
    bundle = ensure_active_bundle()
    finalize_bundle_metadata(
        bundle,
        samples=list(samples or []),
        status=status,
        extra=dict(extra or {}),
    )
    reset_process_bundle()
    return bundle
```

Update `finalize_bundle_metadata` (same file) to:
- emit `schema_version: 3`
- include `finalized_at`
- strip `qc_warnings` from each `samples[i].summary`
- strip `outputs` from each `samples[i]`
- omit `tables` from `run_context` (the `{"combined": "tables/combined.csv"}` shorthand is gone)

Replace the body of `finalize_bundle_metadata`:

```python
def finalize_bundle_metadata(
    bundle: Path,
    *,
    samples: list[dict[str, Any]],
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    seed = read_bundle_metadata(bundle)
    normalized = _normalize_bundle_metadata(seed)
    extra = dict(extra or {})
    run_context_extras = dict(extra.pop("run_context_extras", {}) or {})

    recipe_params = (
        extra.pop("recipe_params", None)
        or normalized.get("recipe_params")
        or {}
    )
    environment = {
        **dict(normalized.get("environment") or {}),
        **dict(extra.pop("environment", {}) or {}),
    }
    samples_list = [_redact_sample(s) for s in samples]
    run_context = {
        **dict(normalized.get("run_context") or {}),
        "status": status,
        "finalized_at": _kst_now_iso(),
        "samples": samples_list,
        "n_samples": len(samples_list),
        "n_complete": sum(1 for s in samples_list if s.get("status") == "complete"),
        "n_failed": sum(1 for s in samples_list if s.get("status") == "failed"),
        **run_context_extras,
        **extra,
    }
    # Drop fields that schema_v3 no longer carries.
    run_context.pop("tables", None)

    write_bundle_metadata(
        bundle,
        {
            "schema_version": 3,
            "recipe_params": dict(recipe_params),
            "run_context": run_context,
            "environment": environment,
            "table_specs": dict(seed.get("table_specs") or {}),
            "outputs": list(seed.get("outputs") or []),
        },
    )


def _redact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    out.pop("outputs", None)  # filesystem mirror removed
    summary = dict(out.get("summary") or {})
    summary.pop("qc_warnings", None)
    out["summary"] = summary
    return out


def _kst_now_iso() -> str:
    from imajin.results import _kst_now

    return _kst_now().isoformat()
```

Make sure `read_bundle_metadata` is imported at the top of `result_bundles.py` (it already is).

- [ ] **Step 4: Run the tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_start_analysis_creates_named_bundle_in_progress tests/test_results_bundle.py::test_finalize_analysis_writes_status_and_strips_redacted_fields -v`

Expected: 2 PASSED.

- [ ] **Step 5: Update the existing finalize-related test that expects schema_v2 to expect schema_v3**

In `tests/test_results_bundle.py`, search for any test that asserts `schema_version == 2`. If found, update to `3` and adjust the assertions to match the new shape. Run the full file to find regressions:

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py -v`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): start_analysis, finalize_analysis, schema_v3 metadata"
```

---

## Task 4: `register_output` (outputs index in metadata.json)

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_results_bundle.py`:

```python
def test_register_output_appends_to_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import (
        bundle_output_path,
        register_output,
        start_analysis,
    )
    from imajin.results import read_bundle_metadata

    bundle = start_analysis(name="demo")
    p = bundle_output_path("figures", "x.png")
    p.write_bytes(b"\x89PNG\r\n")
    register_output("figure", p, {"source": "test"})

    meta = read_bundle_metadata(bundle)
    outputs = meta["outputs"]
    assert len(outputs) == 1
    entry = outputs[0]
    assert entry["kind"] == "figure"
    assert entry["path"] == "figures/x.png"
    assert entry["metadata"] == {"source": "test"}
    assert entry["created_at"]


def test_register_output_rejects_path_outside_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import register_output, start_analysis

    start_analysis(name="demo")
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(b"")
    with pytest.raises(ValueError, match="outside the active bundle"):
        register_output("figure", outside, None)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_register_output_appends_to_metadata tests/test_results_bundle.py::test_register_output_rejects_path_outside_bundle -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `register_output`**

Add to `src/imajin/result_bundles.py`:

```python
from datetime import UTC, datetime as _datetime  # at the top if not already present


def register_output(
    kind: str,
    path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append an entry to the active bundle's metadata.json `outputs` index.

    `path` must already live inside the active bundle; it is recorded as a
    POSIX-relative path. Writes are flushed immediately so consumers can read
    a partial bundle.
    """
    bundle = ensure_active_bundle()
    target = Path(path).resolve()
    bundle_resolved = bundle.resolve()
    try:
        rel = target.relative_to(bundle_resolved)
    except ValueError as exc:
        raise ValueError(
            f"output {target} is outside the active bundle {bundle_resolved}"
        ) from exc

    record = {
        "kind": kind,
        "path": rel.as_posix(),
        "created_at": _datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
    }
    seed = read_bundle_metadata(bundle)
    outputs = list(seed.get("outputs") or [])
    outputs.append(record)
    seed["outputs"] = outputs
    if "schema_version" not in seed:
        seed["schema_version"] = 3
    write_bundle_metadata(bundle, seed)
```

- [ ] **Step 4: Run the tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_register_output_appends_to_metadata tests/test_results_bundle.py::test_register_output_rejects_path_outside_bundle -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): register_output appends to metadata.json"
```

---

## Task 5: `register_table_spec`

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing test**

```python
def test_register_table_spec_merges_into_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import register_table_spec, start_analysis
    from imajin.results import read_bundle_metadata

    bundle = start_analysis(name="demo")
    register_table_spec("measurements", {"tool": "measure_table", "value_cols": ["mean_intensity"]})
    register_table_spec("ratios", {"tool": "derive_ratio", "source": "measurements"})

    meta = read_bundle_metadata(bundle)
    assert meta["table_specs"]["measurements"]["tool"] == "measure_table"
    assert meta["table_specs"]["ratios"]["source"] == "measurements"
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_register_table_spec_merges_into_metadata -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `register_table_spec`**

```python
def register_table_spec(table_name: str, spec: dict[str, Any]) -> None:
    bundle = ensure_active_bundle()
    seed = read_bundle_metadata(bundle)
    table_specs = dict(seed.get("table_specs") or {})
    table_specs[str(table_name)] = dict(spec)
    seed["table_specs"] = table_specs
    if "schema_version" not in seed:
        seed["schema_version"] = 3
    write_bundle_metadata(bundle, seed)
```

- [ ] **Step 4: Run the test**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_register_table_spec_merges_into_metadata -v`

Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): register_table_spec writes into metadata.json"
```

---

## Task 6: `register_stats_rows` (long-format CSV writer)

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_register_stats_rows_describe_long_format(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import register_stats_rows, start_analysis

    bundle = start_analysis(name="demo")
    register_stats_rows(
        kind="describe",
        table="measurements",
        rows=[
            {"value_col": "mean_intensity", "level": "object",
             "sample_aggregation": "", "group": "control",
             "n": 200, "mean": 1.1, "median": 1.05},
        ],
    )
    register_stats_rows(
        kind="describe",
        table="measurements",
        rows=[
            {"value_col": "max_intensity", "level": "object",
             "sample_aggregation": "", "group": "control",
             "n": 200, "mean": 5.2, "median": 5.0},
        ],
    )

    df = pd.read_csv(bundle / "stats" / "describe__measurements.csv")
    assert set(df["value_col"]) == {"mean_intensity", "max_intensity"}
    assert len(df) == 2


def test_register_stats_rows_overwrites_same_key(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import register_stats_rows, start_analysis

    bundle = start_analysis(name="demo")
    register_stats_rows(
        kind="describe",
        table="measurements",
        rows=[
            {"value_col": "mean_intensity", "level": "object",
             "sample_aggregation": "", "group": "control", "n": 200, "mean": 1.0},
        ],
    )
    register_stats_rows(
        kind="describe",
        table="measurements",
        rows=[
            {"value_col": "mean_intensity", "level": "object",
             "sample_aggregation": "", "group": "control", "n": 200, "mean": 1.5},
        ],
    )

    df = pd.read_csv(bundle / "stats" / "describe__measurements.csv")
    assert len(df) == 1
    assert df.iloc[0]["mean"] == 1.5


def test_register_stats_rows_compare_separate_file(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import register_stats_rows, start_analysis

    bundle = start_analysis(name="demo")
    register_stats_rows(
        kind="compare",
        table="measurements",
        rows=[
            {"value_col": "mean_intensity", "test": "welch_ttest",
             "data_level": "sample", "p_value": 0.02},
        ],
    )
    df = pd.read_csv(bundle / "stats" / "compare__measurements.csv")
    assert df.iloc[0]["test"] == "welch_ttest"
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_register_stats_rows_describe_long_format tests/test_results_bundle.py::test_register_stats_rows_overwrites_same_key tests/test_results_bundle.py::test_register_stats_rows_compare_separate_file -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `register_stats_rows`**

Add to `src/imajin/result_bundles.py`:

```python
# Per-kind dedup keys; rows with the same key replace earlier rows.
_STATS_KEY_FIELDS = {
    "describe": ("value_col", "level", "sample_aggregation", "group"),
    "compare": ("value_col", "test", "data_level", "group_a", "group_b"),
    "timecourse_features": ("value_col", "sample_name", "label"),
}


def register_stats_rows(
    *,
    kind: str,
    table: str,
    rows: list[dict[str, Any]],
) -> None:
    """Merge stats rows into `<bundle>/stats/<kind>__<table>.csv` (long format).

    Rows are deduplicated by the kind-specific key fields; later rows replace
    earlier ones with the same key. The destination CSV is rewritten on every
    call so partial bundles are readable.
    """
    if kind not in _STATS_KEY_FIELDS:
        raise ValueError(f"unsupported stats kind {kind!r}")
    if not rows:
        return

    import pandas as pd

    bundle = ensure_active_bundle()
    target = bundle / "stats" / f"{kind}__{slugify_result_name(table)}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)
    if target.exists():
        existing = pd.read_csv(target)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    key_cols = [c for c in _STATS_KEY_FIELDS[kind] if c in combined.columns]
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)

    combined.to_csv(target, index=False)
    register_output(
        f"stats_{kind}",
        target,
        {"table": table, "n_rows": int(len(combined))},
    )
```

`slugify_result_name` is imported from `imajin.results`; ensure it's in scope.

- [ ] **Step 4: Run the tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_register_stats_rows_describe_long_format tests/test_results_bundle.py::test_register_stats_rows_overwrites_same_key tests/test_results_bundle.py::test_register_stats_rows_compare_separate_file -v`

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): register_stats_rows long-format CSV writer"
```

---

## Task 7: Drop `bundles/` wrapper from `create_result_bundle`

**Files:**
- Modify: `src/imajin/results.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing test**

Replace the existing test of the bundle parent layout (or add new):

```python
def test_create_result_bundle_writes_directly_under_root(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.results import create_result_bundle

    bundle = create_result_bundle("demo", kind="single")
    assert bundle.parent == tmp_path
    assert not (tmp_path / "bundles").exists()
```

If the existing test `test_create_result_bundle_uses_kst_timestamp_in_folder_name` already runs against `bundles/`, update its assertion to expect the bundle to be directly under `tmp_path`.

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_create_result_bundle_writes_directly_under_root -v`

Expected: FAIL — the bundle still lands under `bundles/`.

- [ ] **Step 3: Replace `unique_result_dir("bundles", …)` with a direct-root variant**

In `src/imajin/results.py` `create_result_bundle`:

Replace this block:

```python
    if root is not None:
        bundle = _unique_subdir(Path(root), f"{timestamp}_{slugify_result_name(name)}")
    else:
        bundle = unique_result_dir("bundles", f"{timestamp}_{slugify_result_name(name)}")
```

with:

```python
    if root is None:
        root = user_results_root()
    bundle = _unique_subdir(Path(root), f"{timestamp}_{slugify_result_name(name)}")
```

`_unique_subdir` already exists in this file.

- [ ] **Step 4: Run the test**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_create_result_bundle_writes_directly_under_root -v`

Expected: PASSED.

- [ ] **Step 5: Run the full bundle test file**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py -v`

Expected: all PASS. Fix any regression in tests that hard-coded `bundles/` in the path.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/results.py tests/test_results_bundle.py
git commit -m "refactor(bundles): drop bundles/ wrapper from create_result_bundle"
```

---

## Task 8: Migrate `tools/figures.py` to `bundle_output_path` + `register_output`

**Files:**
- Modify: `src/imajin/tools/figures.py`
- Test: `tests/test_tools_figures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools_figures.py`:

```python
def test_figure_writes_into_active_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from imajin.agent import state
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.results import read_bundle_metadata
    from imajin.tools import figures

    reset_process_bundle()
    bundle = start_analysis(name="figtest")
    state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
            }
        ),
        spec={"tool": "test"},
    )

    res = figures.plot_group_distribution("measurements", "mean_intensity")

    out = Path(res["path"])
    assert out.parent == bundle / "figures"
    assert out.exists()
    outputs = read_bundle_metadata(bundle)["outputs"]
    assert any(o["kind"] == "figure" and o["path"] == f"figures/{out.name}" for o in outputs)

    reset_process_bundle()
```

Add `from pathlib import Path` to the test file imports if absent.

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_figures.py::test_figure_writes_into_active_bundle -v`

Expected: FAIL — the figure still routes via `unique_result_path` so the `outputs` index will be empty.

- [ ] **Step 3: Rewire `figures.py`**

In `src/imajin/tools/figures.py`:

Replace the import line:

```python
from imajin.results import record_result, slugify_result_name, unique_result_path
```

with:

```python
from imajin.result_bundles import bundle_output_path, register_output
from imajin.results import slugify_result_name
```

Replace `_figure_path`:

```python
def _figure_path(stem: str, output_path: str | None, fmt: str) -> Path:
    suffix = fmt.lower().lstrip(".")
    if output_path:
        out = normalize_user_path(output_path).resolve()
        if not out.suffix:
            out = out.with_suffix(f".{suffix}")
        return out
    filename = f"{slugify_result_name(stem)}.{suffix}"
    return bundle_output_path("figures", filename)
```

Replace `_save_figure`:

```python
def _save_figure(fig: Any, out: Path, *, dpi: int, metadata: dict[str, Any]) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", transparent=False)
    register_output("figure", out, metadata)
    return str(out)
```

- [ ] **Step 4: Run the tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_figures.py -v`

Expected: all PASS. The new test should pass; existing tests that pass `output_path=…` still write to the explicit path.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/figures.py tests/test_tools_figures.py
git commit -m "refactor(figures): route default outputs through bundle_output_path"
```

---

## Task 9: Migrate `tools/stats.py` to `register_stats_rows`

**Files:**
- Modify: `src/imajin/tools/stats.py`
- Test: `tests/test_tools_stats.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools_stats.py`:

```python
def test_describe_table_writes_long_format(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from imajin.agent import state
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.tools import stats

    reset_process_bundle()
    bundle = start_analysis(name="stattest")
    state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
                "max_intensity": [3.0, 3.1, 4.5, 4.6],
            }
        ),
        spec={"tool": "test"},
    )

    stats.describe_table("measurements", "mean_intensity")
    stats.describe_table("measurements", "max_intensity")

    df = pd.read_csv(bundle / "stats" / "describe__measurements.csv")
    assert {"mean_intensity", "max_intensity"} <= set(df["value_col"])
    # Object-level rows for both groups, both value_cols → 4 rows minimum.
    object_rows = df[df["level"] == "object"]
    assert len(object_rows) >= 4
    # No flat per-value_col stats files.
    assert not any(p.name.startswith("stats_object__") for p in (bundle / "stats").iterdir())
    reset_process_bundle()


def test_compare_groups_writes_long_format(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from imajin.agent import state
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.tools import stats

    reset_process_bundle()
    bundle = start_analysis(name="cmptest")
    state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
            }
        ),
        spec={"tool": "test"},
    )

    stats.compare_groups("measurements", "mean_intensity")
    df = pd.read_csv(bundle / "stats" / "compare__measurements.csv")
    assert df.iloc[0]["value_col"] == "mean_intensity"
    assert df.iloc[0]["p_value"] < 0.05
    reset_process_bundle()
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_stats.py::test_describe_table_writes_long_format tests/test_tools_stats.py::test_compare_groups_writes_long_format -v`

Expected: FAIL — the existing implementation writes `stats_object__measurements__mean_intensity.csv` etc.

- [ ] **Step 3: Rewire `stats.py`**

In `src/imajin/tools/stats.py`:

Replace the import:

```python
from imajin.results import record_result, slugify_result_name, unique_result_path
```

with:

```python
from imajin.result_bundles import register_stats_rows
from imajin.results import slugify_result_name
```

Delete `_stats_csv_path` and `_write_stats_csv` entirely.

In `describe_table` (the function around line 359):

Replace the `_write_stats_csv` calls with two `register_stats_rows` calls. The object/sample DataFrames are already computed (`object_desc`, `sample_desc`). Convert each to row dicts and add `value_col`, `level`, `sample_aggregation`:

```python
    object_rows = object_desc.to_dict(orient="records")
    for row in object_rows:
        row.update({
            "value_col": value_col,
            "level": "object",
            "sample_aggregation": "",
        })
    register_stats_rows(kind="describe", table=table_name, rows=object_rows)

    sample_rows_for_csv: list[dict[str, Any]] = []
    if sample_df is not None:
        sample_rows_for_csv = sample_desc.to_dict(orient="records")
        for row in sample_rows_for_csv:
            row.update({
                "value_col": value_col,
                "level": "sample",
                "sample_aggregation": "mean",
            })
        register_stats_rows(kind="describe", table=table_name, rows=sample_rows_for_csv)

    object_csv = None  # legacy field; long-format file lives in bundle/stats.
    sample_csv = None
```

In `compare_groups` (around line 614 — locate the `_write_stats_csv` call with the `stats_compare__…` stem):

Replace it with:

```python
    rows = result_df.to_dict(orient="records")
    for row in rows:
        row["value_col"] = value_col
    register_stats_rows(kind="compare", table=table_name, rows=rows)
    csv_path = None  # legacy field
```

In `extract_timecourse_features` (around line 1015):

Replace `_write_stats_csv(...)` with:

```python
    rows = features.to_dict(orient="records")
    for row in rows:
        row["value_col"] = value_col
    register_stats_rows(kind="timecourse_features", table=table_name, rows=rows)
    csv_path = None
```

The three functions return dicts that include `csv_path`; downstream consumers should still tolerate `None` (the long-format file is discoverable via `<bundle>/stats/`). If any downstream code relies on the specific path string, the planning of follow-up Task 13 covers wiring an alternate accessor — but for now `None` is acceptable because the bundle layout is the source of truth.

- [ ] **Step 4: Run the tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_stats.py -v`

Expected: all PASS. Existing tests that asserted file names like `stats_object__measurements__mean_intensity.csv` must be updated to read `<bundle>/stats/describe__measurements.csv` and filter by `value_col`.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/stats.py tests/test_tools_stats.py
git commit -m "refactor(stats): consolidate per-value_col CSVs into long-format files"
```

---

## Task 10: Migrate `tools/_segmentation_outputs.py`

**Files:**
- Modify: `src/imajin/tools/_segmentation_outputs.py`
- Test: `tests/test_tools_segment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools_segment.py` (model on existing segmentation tests):

```python
def test_segment_qc_writes_only_to_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.tools._segmentation_outputs import _default_qc_png_path

    reset_process_bundle()
    bundle = start_analysis(name="seg")
    path = _default_qc_png_path("foo_labels")
    assert path.parent == bundle / "qc"
    assert path.name == "foo_labels.png"
    # No flat segmentation_qc folder anywhere under the results root.
    assert not (tmp_path / "segmentation_qc").exists()
    reset_process_bundle()
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_segment.py::test_segment_qc_writes_only_to_bundle -v`

Expected: FAIL — `_default_qc_png_path` currently routes through `unique_result_path("segmentation_qc", …)` or anchor-side fallback.

- [ ] **Step 3: Simplify `_default_qc_png_path`**

In `src/imajin/tools/_segmentation_outputs.py`:

Replace the import:

```python
from imajin.results import record_result, unique_result_path
```

with:

```python
from imajin.result_bundles import bundle_output_path, register_output
```

Replace the entire `_default_qc_png_path` function with:

```python
def _default_qc_png_path(labels_layer: str, source_layer: Any | None = None) -> Path:
    return bundle_output_path("qc", f"{_slug(labels_layer)}.png")
```

In `_save_qc_png` replace the `record_result(...)` call with `register_output("qc_png", path, {...})` (signature equivalent).

`_anchor`/`_source_path_from_layer` helpers are now unused by this module; leave them in place for other callers, or remove if grep shows none.

- [ ] **Step 4: Run the test file**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_segment.py -v`

Expected: all PASS. Update any existing test that asserts the QC PNG lives at `segmentation_qc/<labels>.png` (it now lives at `<bundle>/qc/<labels>.png`).

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/_segmentation_outputs.py tests/test_tools_segment.py
git commit -m "refactor(segment): write QC PNG directly into bundle/qc"
```

---

## Task 11: Remove standalone-QC cleanup from `_workflow_outputs.py`

**Files:**
- Modify: `src/imajin/tools/_workflow_outputs.py`
- Test: covered by existing `tests/test_phase2_workflow.py` and `tests/test_results_bundle.py`

- [ ] **Step 1: Inspect callers and verify QC is already in the bundle**

`segment_cells_*` tools now write QC to `<bundle>/qc/<slug>.png` (Task 10). The workflow's `populate_sample_outputs.copy_qc_png` may receive the bundle-relative path; if `src == dst`, copy is a no-op. The cleanup hook deletes the original — but with the new flow, the original IS the destination. Removing the cleanup is safe but we must also tweak `copy_qc_png` to handle the "already inside bundle" case cleanly.

- [ ] **Step 2: Remove `_remove_copied_standalone_qc` and its call site**

In `src/imajin/tools/_workflow_outputs.py`:

Delete the function `_remove_copied_standalone_qc` and remove the surrounding try/except block in `_write_analysis_bundle_outputs` that calls it (lines around 138-147 in the current file).

In `src/imajin/result_bundles.py` `copy_qc_png`, change the rename logic so an already-in-bundle QC PNG is renamed (or symlinked) to `qc/<sample_slug>.png` if the names differ, instead of copying:

```python
def copy_qc_png(bundle: Path, qc_png: str, sample_slug: str) -> str | None:
    src = normalize_user_path(qc_png).resolve()
    if not src.exists():
        return None
    rel = Path("qc") / f"{sample_slug}.png"
    dst = (bundle / rel).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src == dst:
        return rel.as_posix()
    try:
        bundle_resolved = bundle.resolve()
        if src.is_relative_to(bundle_resolved):
            src.rename(dst)
            return rel.as_posix()
    except AttributeError:
        # Python < 3.9 fallback; not relevant here but kept defensive.
        pass
    shutil.copy2(src, dst)
    return rel.as_posix()
```

- [ ] **Step 3: Run the workflow tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_phase2_workflow.py tests/test_results_bundle.py tests/test_tools_segment.py -v`

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/imajin/tools/_workflow_outputs.py src/imajin/result_bundles.py
git commit -m "refactor(workflow): drop standalone-QC cleanup; rename in place when in bundle"
```

---

## Task 12: Migrate `tools/results.py` `save_result_bundle`

**Files:**
- Modify: `src/imajin/tools/results.py`
- Test: `tests/test_tools_results.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools_results.py`:

```python
def test_save_result_bundle_writes_table_spec_into_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from imajin.agent import state
    from imajin.result_bundles import reset_process_bundle
    from imajin.results import read_bundle_metadata
    from imajin.tools.results import save_result_bundle

    reset_process_bundle()
    state.put_table(
        "measurements",
        pd.DataFrame({"label": [1, 2], "mean_intensity": [0.1, 0.2]}),
        spec={"tool": "measure_test", "layer": "cells"},
    )

    out = save_result_bundle(name="b1", table_names=["measurements"])
    bundle = Path(out["bundle_path"])

    # No per-table spec.json file any more.
    assert not (bundle / "tables" / "measurements.spec.json").exists()
    # Spec moved into metadata.json.
    meta = read_bundle_metadata(bundle)
    assert meta["table_specs"]["measurements"]["tool"] == "measure_test"
    reset_process_bundle()


def test_save_result_bundle_does_not_write_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools.results import save_result_bundle

    reset_process_bundle()
    save_result_bundle(name="b2")
    assert not (tmp_path / "manifest.jsonl").exists()
    reset_process_bundle()
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_results.py::test_save_result_bundle_writes_table_spec_into_metadata tests/test_tools_results.py::test_save_result_bundle_does_not_write_manifest -v`

Expected: FAIL — `tables/measurements.spec.json` exists and/or `manifest.jsonl` is created.

- [ ] **Step 3: Rewire `save_result_bundle` and `_resolve_output_path`**

In `src/imajin/tools/results.py`:

Replace imports:

```python
from imajin.results import (
    create_result_bundle,
    read_bundle_metadata,
    record_result,
    slugify_result_name,
    unique_result_path,
    write_bundle_metadata,
)
```

with:

```python
from imajin.result_bundles import (
    bundle_output_path,
    register_output,
    register_table_spec,
)
from imajin.results import (
    create_result_bundle,
    read_bundle_metadata,
    slugify_result_name,
    write_bundle_metadata,
)
```

Replace `_resolve_output_path`:

```python
def _resolve_output_path(
    path: str | None,
    *,
    category: str,
    filename: str,
) -> Path:
    if path:
        return normalize_user_path(path).resolve()
    return bundle_output_path(category, filename)
```

Update `save_labels` so its `_resolve_output_path` call matches the new signature (it currently passes `root=…`). Replace the call site (around line 113):

```python
    out = _resolve_output_path(
        path,
        category="labels",
        filename=f"{slugify_result_name(labels_layer)}.tif",
    )
```

In `save_result_bundle`:

- Delete the block that writes `tables/<name>.spec.json` (the four lines starting at `spec_path = bundle / "tables" / f"…spec.json"`). Replace with:

```python
        register_table_spec(table_name, get_table_entry(table_name).spec)
```

- Replace each `record_result(...)` call at the bottom with `register_output(...)`. The single trailing `record_result("result_bundle", bundle, {...})` becomes:

```python
    register_output(
        "result_bundle",
        bundle / "metadata.json",
        {
            "name": name,
            "n_labels": len(outputs["labels"]),
            "n_tables": len(outputs["tables"]),
            "n_qc": len(outputs["qc"]),
            "n_figures": len(outputs["figures"]),
        },
    )
```

- In `save_labels`, replace `record_result("labels_tiff", out, {...})` with `register_output("labels_tiff", out, {...})`.

- [ ] **Step 4: Run the test file**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_results.py -v`

Expected: all PASS. Update legacy tests that asserted `tables/<name>.spec.json` exists or that `manifest.jsonl` exists.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/results.py tests/test_tools_results.py
git commit -m "refactor(results): move table specs into metadata.json, drop spec.json"
```

---

## Task 13: Remove deprecated API from `results.py`

**Files:**
- Modify: `src/imajin/results.py`
- Test: covered by `tests/test_results_bundle.py`, `tests/test_tools_results.py`, `tests/test_tools_segment.py`

- [ ] **Step 1: Verify no callers remain**

Run: `grep -rn "unique_result_path\|unique_result_dir\|results_dir\|record_result\|_manifest_root" src/imajin/ tests/`

Expected: the only matches are inside `src/imajin/results.py` (definitions) and any tests that still import them. Update those tests to use the new API or remove the assertions.

If any matches appear outside `results.py`, fix them in a follow-up step before deletion.

- [ ] **Step 2: Delete the deprecated functions and constants**

In `src/imajin/results.py`, remove:

- `_RESULT_CATEGORY_DIRS` constant
- `results_dir`
- `unique_result_path`
- `unique_result_dir`
- `_manifest_root`
- `record_result`

Keep: `user_results_root`, `results_root`, `slugify_result_name`, `_unique_subdir`, `create_result_bundle`, `write_bundle_metadata`, `read_bundle_metadata`, `_collect_env_info`, `_kst_now`, `_windows_documents_dir`, `_git_commit_short`.

- [ ] **Step 3: Run the full test suite**

Run: `uv run --project /home/jin/py314 pytest tests/ -x --ignore=tests/test_anthropic_integration.py`

Expected: all PASS. Fix any remaining importers.

- [ ] **Step 4: Commit**

```bash
git add src/imajin/results.py
git commit -m "refactor(results): remove flat-path fallback and manifest.jsonl helpers"
```

---

## Task 14: Expose `start_analysis` / `finalize_analysis` as LLM tools

**Files:**
- Create: `src/imajin/tools/bundle.py`
- Modify: `src/imajin/tools/__init__.py` (if it exists; otherwise the module gets imported via the existing tool autodiscovery)
- Test: `tests/test_tools_results.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools_results.py`:

```python
def test_start_and_finalize_analysis_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle
    from imajin.results import read_bundle_metadata
    from imajin.tools.bundle import finalize_analysis, start_analysis

    reset_process_bundle()
    res = start_analysis(name="J20_component1")
    bundle = Path(res["bundle_path"])
    assert (bundle / "metadata.json").exists()
    assert read_bundle_metadata(bundle)["run_context"]["status"] == "in_progress"

    finalize = finalize_analysis()
    assert finalize["status"] == "complete"
    assert read_bundle_metadata(bundle)["run_context"]["status"] == "complete"
    reset_process_bundle()
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_results.py::test_start_and_finalize_analysis_tools -v`

Expected: FAIL — `imajin.tools.bundle` does not exist.

- [ ] **Step 3: Create `src/imajin/tools/bundle.py`**

```python
from __future__ import annotations

from typing import Any

from imajin.result_bundles import (
    finalize_analysis as _finalize_analysis,
    start_analysis as _start_analysis,
)
from imajin.tools.registry import tool


@tool(
    description=(
        "Open a new analysis bundle. All subsequent figure, stats, QC, and table "
        "outputs in this task land inside <root>/<timestamp>_<name>/. Call this at "
        "the start of a user task so the bundle has a meaningful name; otherwise "
        "an ad-hoc bundle is opened lazily on the first output."
    ),
    phase="0",
)
def start_analysis(
    name: str,
    *,
    kind: str = "single",
    tier: str | None = None,
) -> dict[str, Any]:
    bundle = _start_analysis(name=name, kind=kind, tier=tier)
    return {
        "bundle_path": str(bundle),
        "metadata_path": str(bundle / "metadata.json"),
        "name": name,
    }


@tool(
    description=(
        "Finalize the currently active analysis bundle. Writes the final "
        "metadata.json status and clears the process slot so the next analysis "
        "starts fresh. Safe to call even if no bundle has been opened (it will "
        "lazily create one)."
    ),
    phase="0",
)
def finalize_analysis(
    status: str = "complete",
) -> dict[str, Any]:
    bundle = _finalize_analysis(status=status)
    return {
        "bundle_path": str(bundle),
        "status": status,
    }
```

If `src/imajin/tools/__init__.py` enumerates explicit submodule imports, append `from imajin.tools import bundle  # noqa: F401` to register the tools. Otherwise the existing tool autoloading picks it up.

Check how other tool modules are loaded:

Run: `grep -rn "from imajin.tools import\|imajin.tools\\.\(figures\|stats\|segment\)" src/imajin/`

Expected output guides which import to add (commonly in `src/imajin/agent/specialists/__init__.py` or a registry bootstrap module).

- [ ] **Step 4: Run the test**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_results.py::test_start_and_finalize_analysis_tools -v`

Expected: PASSED.

- [ ] **Step 5: Verify tools are reachable from the registry**

```python
uv run --project /home/jin/py314 python -c "from imajin.tools import bundle; from imajin.tools.registry import get_tool; print(get_tool('start_analysis').description[:60]); print(get_tool('finalize_analysis').description[:60])"
```

Expected: descriptions print.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/tools/bundle.py tests/test_tools_results.py
git commit -m "feat(tools): expose start_analysis and finalize_analysis as LLM tools"
```

---

## Task 15: Atexit finalizer for the process-global ad-hoc bundle

**Files:**
- Modify: `src/imajin/result_bundles.py`
- Test: `tests/test_results_bundle.py`

- [ ] **Step 1: Write the failing test**

```python
def test_atexit_finalizes_adhoc_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import (
        _run_atexit_finalize,
        ensure_active_bundle,
        reset_process_bundle,
    )
    from imajin.results import read_bundle_metadata

    reset_process_bundle()
    bundle = ensure_active_bundle()
    assert read_bundle_metadata(bundle)["run_context"]["status"] == "in_progress"

    _run_atexit_finalize()  # simulate process exit
    assert read_bundle_metadata(bundle)["run_context"]["status"] == "complete"
    reset_process_bundle()
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_atexit_finalizes_adhoc_bundle -v`

Expected: FAIL — `_run_atexit_finalize` does not exist.

- [ ] **Step 3: Implement the finalizer**

Add to `src/imajin/result_bundles.py`:

```python
import atexit


def _run_atexit_finalize() -> None:
    global _process_bundle
    with _process_bundle_lock:
        bundle = _process_bundle
    if bundle is None:
        return
    try:
        finalize_bundle_metadata(bundle, samples=[], status="complete")
    finally:
        with _process_bundle_lock:
            _process_bundle = None


atexit.register(_run_atexit_finalize)
```

- [ ] **Step 4: Run the test**

Run: `uv run --project /home/jin/py314 pytest tests/test_results_bundle.py::test_atexit_finalizes_adhoc_bundle -v`

Expected: PASSED.

- [ ] **Step 5: Final full-suite run**

Run: `uv run --project /home/jin/py314 pytest tests/ -x --ignore=tests/test_anthropic_integration.py`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/result_bundles.py tests/test_results_bundle.py
git commit -m "feat(bundles): atexit finalizer for process-global ad-hoc bundle"
```

---

## Notes for the implementer

- `uv run --project /home/jin/py314` is the project's required Python runner per `CLAUDE.md`. Never invoke `python`, `python3`, or `pip` directly.
- The repo enables `from __future__ import annotations` everywhere; new modules should keep that convention.
- The KST timestamp helper `_kst_now` lives in `imajin.results`; reuse it rather than constructing UTC strings ad-hoc.
- `pytest tests/test_anthropic_integration.py` is excluded above because it hits a live API and may be skipped in CI; include it in manual runs if your environment allows.
- Existing dirty files in the working tree (`src/imajin/tools/figures.py`, `tools/stats.py`, …) at the start of this branch are unrelated changes left in place by the user. The plan above modifies overlapping files; the implementer should review those uncommitted edits and either integrate or stash them before starting Task 1.
