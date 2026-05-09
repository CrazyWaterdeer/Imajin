from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

from imajin.results import (
    _collect_env_info,
    _kst_now,
    create_result_bundle,
    read_bundle_metadata,
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
