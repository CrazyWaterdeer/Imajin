from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile

from imajin.results import (
    _collect_env_info,
    _kst_now,
    create_result_bundle,
    read_bundle_metadata,
)
from imajin.tools.results import (
    current_bundle,
    populate_sample_outputs,
    with_active_bundle,
)


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
