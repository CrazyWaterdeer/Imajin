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
