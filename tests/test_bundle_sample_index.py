from __future__ import annotations

from imajin.result_bundles import read_sample_index, record_sample_index_entry
from imajin.results import create_result_bundle


def _bundle_with_file(tmp_path, name="a.lsm"):
    anchor = tmp_path / "data"
    anchor.mkdir(exist_ok=True)
    (anchor / name).write_bytes(b"x")
    bundle = create_result_bundle(name="t", kind="batch", root=tmp_path / "out")
    return bundle, anchor


def test_record_and_read_sample_index(tmp_path):
    bundle, anchor = _bundle_with_file(tmp_path)
    entry = record_sample_index_entry(
        bundle,
        source_file=str(anchor / "a.lsm"),
        anchor=str(anchor),
        method="target_objects",
        mode="single",
        status="complete",
        table="tbl",
    )
    assert entry["key"] == "a.lsm"  # anchor-relative
    idx = read_sample_index(bundle)
    assert idx["legacy_inferred"] is False
    assert [e["key"] for e in idx["entries"]] == ["a.lsm"]
    assert idx["input_anchor"] and idx["entries"][0]["sample_slug"].startswith("a_")


def test_record_is_idempotent_upsert(tmp_path):
    bundle, anchor = _bundle_with_file(tmp_path)
    record_sample_index_entry(bundle, source_file=str(anchor / "a.lsm"), anchor=str(anchor))
    record_sample_index_entry(
        bundle, source_file=str(anchor / "a.lsm"), anchor=str(anchor), table="tbl2"
    )
    idx = read_sample_index(bundle)
    assert len(idx["entries"]) == 1  # same key upserted, not duplicated
    assert idx["entries"][0]["table"] == "tbl2"


def test_zero_object_sample_still_recorded(tmp_path):
    # A completed sample with no measurement table is still "complete" in the index,
    # so resume treats it as done — not pending (the review's Codex #3).
    bundle, anchor = _bundle_with_file(tmp_path, name="b.lsm")
    record_sample_index_entry(
        bundle, source_file=str(anchor / "b.lsm"), anchor=str(anchor), status="complete", table=None
    )
    idx = read_sample_index(bundle)
    assert idx["entries"][0]["status"] == "complete"


def test_legacy_bundle_is_flagged(tmp_path):
    bundle = create_result_bundle(name="t", kind="batch", root=tmp_path / "out")
    idx = read_sample_index(bundle)  # no samples index written
    assert idx["legacy_inferred"] is True
    assert idx["entries"] == []
