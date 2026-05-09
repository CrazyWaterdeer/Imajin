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
