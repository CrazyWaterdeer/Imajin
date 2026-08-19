from __future__ import annotations

import json
import re
from datetime import timedelta

import numpy as np
import pytest
import tifffile

from imajin.results import (
    _collect_env_info,
    _kst_now,
    create_result_bundle,
    read_bundle_metadata,
)
from imajin.result_bundles import (
    finalize_bundle_metadata,
    read_bundle_metadata_normalized,
)
from imajin.tools.results import (
    current_bundle,
    populate_sample_outputs,
    with_active_bundle,
)


@pytest.fixture(autouse=True)
def _reset_process_bundle():
    from imajin.result_bundles import reset_process_bundle

    reset_process_bundle()
    yield
    reset_process_bundle()


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


def test_create_result_bundle_uses_kst_timestamp_in_folder_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    bundle = create_result_bundle("demo", kind="single")
    name = bundle.name
    assert re.match(r"^\d{8}_\d{6}_demo$", name), name


def test_create_result_bundle_creates_subdirs_lazily(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    bundle = create_result_bundle("demo", kind="single")
    for sub in ("labels/cells", "labels/domain", "tables", "qc", "stats", "figures"):
        assert not (bundle / sub).exists(), f"unexpected empty subdir: {sub}"
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


def test_create_result_bundle_framework_fields_win_over_caller_metadata(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    bundle = create_result_bundle(
        "demo",
        kind="single",
        tier="single_tier",
        metadata={
            "status": "complete",       # should NOT override
            "kind": "batch",            # should NOT override
            "imajin_version": "FAKE",   # should NOT override
            "recipe": {"name": "demo"}, # should pass through
        },
    )
    meta = read_bundle_metadata(bundle)
    assert meta["status"] == "in_progress"
    assert meta["kind"] == "single"
    assert meta["imajin_version"] != "FAKE"
    assert meta["recipe"] == {"name": "demo"}


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


def test_populate_sample_outputs_writes_cells_label(tmp_path, viewer, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
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


def test_populate_sample_outputs_writes_domain_when_provided(tmp_path, viewer, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
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


def test_populate_sample_outputs_copies_qc_png(tmp_path, viewer, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
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


def test_populate_sample_outputs_rejects_collision(tmp_path, viewer, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    bundle = create_result_bundle("demo", kind="single", tier="single_tier")
    viewer.add_labels(np.ones((5, 5), dtype=np.uint16), name="cells_layer")
    populate_sample_outputs(bundle, sample_slug="s1", labels_cells="cells_layer")
    with pytest.raises(ValueError, match="already exists"):
        populate_sample_outputs(
            bundle, sample_slug="s1", labels_cells="cells_layer"
        )


def test_finalize_writes_schema_v3(tmp_path) -> None:
    bundle = create_result_bundle("demo", root=tmp_path, kind="batch", tier="two_tier")

    finalize_bundle_metadata(
        bundle,
        samples=[{"sample_name": "s1", "status": "complete"}],
        status="complete",
        extra={
            "recipe_params": {
                "name": "demo",
                "segmentation": {"method": "target_objects"},
            },
            "run_context_extras": {
                "folder_set": [str(tmp_path)],
                "channel_roles": {"Ch1": "target"},
                "scope_filters": [],
            },
        },
    )

    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["schema_version"] == 3
    assert meta["recipe_params"]["segmentation"]["method"] == "target_objects"
    assert meta["run_context"]["kind"] == "batch"
    assert meta["run_context"]["tier"] == "two_tier"
    assert meta["run_context"]["status"] == "complete"
    assert meta["run_context"]["channel_roles"] == {"Ch1": "target"}
    assert meta["run_context"]["folder_set"] == [str(tmp_path)]
    assert "deps" in meta["environment"]
    assert "tables" not in meta["run_context"]


def test_ensure_active_bundle_lazy_creates_adhoc(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import (
        ensure_active_bundle,
        current_bundle,
    )

    assert current_bundle() is None

    bundle = ensure_active_bundle()
    assert bundle.parent == tmp_path
    assert bundle.name.endswith("_adhoc")
    assert (bundle / "metadata.json").exists()
    assert current_bundle() == bundle

    # Second call returns the same bundle.
    assert ensure_active_bundle() == bundle


def test_ensure_active_bundle_respects_explicit_active_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.results import create_result_bundle
    from imajin.result_bundles import (
        ensure_active_bundle,
        with_active_bundle,
    )

    explicit = create_result_bundle("named", kind="single")
    with with_active_bundle(explicit):
        assert ensure_active_bundle() == explicit
    # After leaving the context, ad-hoc takes over.
    bundle = ensure_active_bundle()
    assert bundle != explicit
    assert bundle.name.endswith("_adhoc")


def test_read_bundle_metadata_normalizes_v1(tmp_path) -> None:
    bundle = tmp_path / "old_bundle"
    bundle.mkdir()
    (bundle / "metadata.json").write_text(
        """
        {
          "recipe": {"name": "old", "segmentation": {"method": "target_objects"}},
          "kind": "batch", "tier": "two_tier", "name": "old",
          "status": "complete", "samples": []
        }
        """,
        encoding="utf-8",
    )

    norm = read_bundle_metadata_normalized(bundle)

    assert norm["schema_version"] == 2
    assert norm["recipe_params"]["name"] == "old"
    assert norm["run_context"]["kind"] == "batch"
    assert norm["run_context"]["tier"] == "two_tier"
    assert norm["run_context"]["status"] == "complete"


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


def test_finalize_after_start_preserves_kind_tier_name(tmp_path, monkeypatch):
    """start_analysis -> finalize_analysis must retain bundle provenance fields."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import finalize_analysis, start_analysis
    from imajin.results import read_bundle_metadata

    bundle = start_analysis(name="provenance_demo", kind="single", tier="two_tier")
    finalize_analysis(status="complete", samples=[{"sample_name": "s1", "status": "complete"}])

    meta = read_bundle_metadata(bundle)
    rc = meta["run_context"]
    assert rc["kind"] == "single"
    assert rc["tier"] == "two_tier"
    assert rc["name"] == "provenance_demo"
    assert rc["created_at"]  # ISO timestamp preserved


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


def test_create_result_bundle_writes_directly_under_root(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.results import create_result_bundle

    bundle = create_result_bundle("demo", kind="single")
    assert bundle.parent == tmp_path
    assert not (tmp_path / "bundles").exists()


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


def test_workflow_bundle_is_promoted_to_process_slot(tmp_path, monkeypatch):
    """After _write_analysis_bundle_outputs creates a bundle, subsequent
    register_output / register_stats_rows calls must target the same bundle,
    not a separate ad-hoc bundle."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import (
        current_bundle,
        promote_to_process_bundle,
        reset_process_bundle,
    )
    from imajin.results import create_result_bundle, read_bundle_metadata

    reset_process_bundle()
    # Simulate what the workflow does internally.
    bundle = create_result_bundle("simulated_workflow", kind="single")
    promote_to_process_bundle(bundle)

    # A subsequent tool that uses ensure_active_bundle() sees the workflow bundle.
    assert current_bundle() == bundle

    from imajin.result_bundles import bundle_output_path, register_output

    fig = bundle_output_path("figures", "x.png")
    fig.write_bytes(b"\x89PNG\r\n")
    register_output("figure", fig, {})

    meta = read_bundle_metadata(bundle)
    assert any(o["path"] == "figures/x.png" for o in meta["outputs"])
    # No separate adhoc bundle should have been created.
    adhoc_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.endswith("_adhoc")]
    assert adhoc_dirs == []
    reset_process_bundle()


def test_finalize_does_not_erase_recorded_samples(tmp_path, monkeypatch):
    """A second finalize with no samples must not wipe the per-sample record.

    finalize_analysis() and the atexit hook both pass samples=[]; before this
    guard either one erased the whole run_context.samples list that per-file
    analyses had accumulated.
    """
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import finalize_bundle_metadata
    from imajin.results import create_result_bundle, read_bundle_metadata

    bundle = create_result_bundle("session", kind="single")
    finalize_bundle_metadata(
        bundle,
        samples=[{"sample_name": "file_a", "status": "complete", "summary": {"n_cells": 16}}],
        status="complete",
    )
    finalize_bundle_metadata(bundle, samples=[], status="complete")

    run_context = read_bundle_metadata(bundle)["run_context"]
    assert [s["sample_name"] for s in run_context["samples"]] == ["file_a"]
    assert run_context["n_samples"] == 1
    assert run_context["n_complete"] == 1


def test_finalize_merges_samples_by_name(tmp_path, monkeypatch):
    """Sequential per-file finalizes accumulate; a re-run of one file replaces it."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import finalize_bundle_metadata
    from imajin.results import create_result_bundle, read_bundle_metadata

    bundle = create_result_bundle("session", kind="single")
    finalize_bundle_metadata(
        bundle,
        samples=[{"sample_name": "file_a", "status": "complete", "summary": {"n_cells": 16}}],
        status="in_progress",
    )
    finalize_bundle_metadata(
        bundle,
        samples=[{"sample_name": "file_b", "status": "complete", "summary": {"n_cells": 61}}],
        status="in_progress",
    )
    # Re-analysing file_a replaces its record rather than duplicating it.
    finalize_bundle_metadata(
        bundle,
        samples=[{"sample_name": "file_a", "status": "complete", "summary": {"n_cells": 20}}],
        status="complete",
    )

    run_context = read_bundle_metadata(bundle)["run_context"]
    assert [s["sample_name"] for s in run_context["samples"]] == ["file_a", "file_b"]
    assert run_context["samples"][0]["summary"]["n_cells"] == 20
    assert run_context["n_samples"] == 2


def test_atexit_close_preserves_outputs_and_samples(tmp_path, monkeypatch):
    """The atexit hook is status-only: it must not drop outputs/table_specs/samples."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import (
        _run_atexit_finalize,
        bundle_output_path,
        finalize_bundle_metadata,
        promote_to_process_bundle,
        register_output,
        reset_process_bundle,
    )
    from imajin.results import create_result_bundle, read_bundle_metadata

    reset_process_bundle()
    bundle = create_result_bundle("session", kind="single")
    promote_to_process_bundle(bundle)
    finalize_bundle_metadata(
        bundle,
        samples=[{"sample_name": "file_a", "status": "complete", "summary": {"n_cells": 16}}],
        status="in_progress",
    )
    fig = bundle_output_path("figures", "x.png")
    fig.write_bytes(b"\x89PNG\r\n")
    register_output("figure", fig, {})

    _run_atexit_finalize()  # simulate process exit

    meta = read_bundle_metadata(bundle)
    assert meta["run_context"]["status"] == "complete"
    assert [s["sample_name"] for s in meta["run_context"]["samples"]] == ["file_a"]
    assert any(o["path"] == "figures/x.png" for o in meta["outputs"])
    assert "sample_index" in meta
    reset_process_bundle()


def test_finalize_analysis_on_empty_slot_creates_nothing(tmp_path, monkeypatch):
    """Closing a session that never opened a bundle must not mint an orphan."""
    root = tmp_path / "empty_root"
    root.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(root))
    from imajin.result_bundles import finalize_analysis, reset_process_bundle

    reset_process_bundle()
    assert finalize_analysis() is None
    assert list(root.iterdir()) == []


def test_register_output_warns_when_bundle_is_already_closed(tmp_path, monkeypatch):
    """A finalized bundle that keeps accepting writes is a bug signal."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pytest

    from imajin.result_bundles import (
        bundle_output_path,
        finalize_bundle_metadata,
        promote_to_process_bundle,
        register_output,
        reset_process_bundle,
    )
    from imajin.results import create_result_bundle

    reset_process_bundle()
    bundle = create_result_bundle("closed", kind="single")
    promote_to_process_bundle(bundle)
    finalize_bundle_metadata(bundle, samples=[], status="complete")

    late = bundle_output_path("qc", "late.png")
    late.write_bytes(b"\x89PNG\r\n")
    with pytest.warns(RuntimeWarning, match="closed bundle"):
        register_output("qc_png", late, {})
    reset_process_bundle()


def test_start_analysis_roots_at_the_session_anchor(tmp_path, monkeypatch):
    """The session bundle lands next to the data, like every other bundle creator."""
    raw = tmp_path / "raw"
    raw.mkdir()
    # Point the fallback at a DECOY: if start_analysis still hard-coded
    # user_results_root() it would short-circuit on this env var and pass.
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "fallback"))

    from imajin import session as state
    from imajin.result_bundles import reset_process_bundle, start_analysis

    reset_process_bundle()
    source = raw / "a.lsm"
    source.write_bytes(b"stub")
    state.put_file(str(source), "a.lsm")

    bundle = start_analysis("session")
    assert bundle.parent == raw.resolve()
    assert not (tmp_path / "fallback").exists()
    reset_process_bundle()


def test_start_analysis_falls_back_to_results_root_without_session_files(
    tmp_path, monkeypatch
):
    root = tmp_path / "results"
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(root))
    from imajin.result_bundles import reset_process_bundle, start_analysis

    reset_process_bundle()
    bundle = start_analysis("session")
    assert bundle.parent == root
    reset_process_bundle()
