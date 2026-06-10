"""Characterization tests for the session-state module.

Pins the *observable* behaviour of the free-function session API through its
public surface only — never asserting that a particular module global exists.
This is the safety net for the de-globalization refactor: it must pass on the
current code unchanged and survive Phase 1 (internals → ``current_session()``)
without edits.

Imports target ``imajin.agent.state`` for now; Phase 2 retargets them to
``imajin.session``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imajin.agent import state
from imajin.agent.state import AnalysisSession


@pytest.fixture
def fresh_session():
    """Run a test against a brand-new session and restore the original after.

    Gives complete isolation for tables and table-listeners, which the autouse
    conftest fixture deliberately does *not* reset.
    """
    original = state.current_session()
    state.set_current_session(AnalysisSession())
    try:
        yield
    finally:
        state.set_current_session(original)


def _demo_df() -> pd.DataFrame:
    return pd.DataFrame({"label": [1, 2], "area": [10.0, 20.0]})


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def test_put_file_returns_slug_id_and_round_trips() -> None:
    file_id = state.put_file("/data/My Image.lsm", "My Image.lsm", file_type="lsm")

    assert file_id == "My_Image"
    rec = state.get_file(file_id)
    assert rec.path == "/data/My Image.lsm"
    assert rec.original_name == "My Image.lsm"
    assert rec.file_type == "lsm"
    assert rec.load_status == "unloaded"


def test_put_file_suffixes_duplicate_original_name() -> None:
    first = state.put_file("/a/img.lsm", "img.lsm")
    second = state.put_file("/b/img.lsm", "img.lsm")

    assert first == "img"
    assert second == "img_2"
    assert {f["file_id"] for f in state.list_files()} == {"img", "img_2"}


def test_update_file_status_mutates_status_and_notes() -> None:
    file_id = state.put_file("/a/img.lsm", "img.lsm")

    state.update_file_status(file_id, "loaded", notes="ok")

    rec = state.get_file(file_id)
    assert rec.load_status == "loaded"
    assert rec.notes == "ok"


def test_iter_and_list_files_reflect_inserts_and_reset() -> None:
    state.put_file("/a/img.lsm", "img.lsm")
    state.put_file("/b/other.lsm", "other.lsm")

    assert len(state.iter_file_records()) == 2
    assert len(state.list_files()) == 2

    state.reset_files()
    assert state.iter_file_records() == []
    assert state.list_files() == []


def test_get_file_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        state.get_file("nope")


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #
def test_put_recipe_round_trips() -> None:
    name = state.put_recipe("seg_recipe", target_channel="green", review_mode="interactive")

    assert name == "seg_recipe"
    r = state.get_recipe(name)
    assert r.target_channel == "green"
    assert r.review_mode == "interactive"
    assert [x["name"] for x in state.list_recipes()] == ["seg_recipe"]

    state.reset_recipes()
    assert state.list_recipes() == []


def test_put_recipe_rejects_bad_review_mode() -> None:
    with pytest.raises(ValueError):
        state.put_recipe("bad", review_mode="weird")


# --------------------------------------------------------------------------- #
# runs (incl. run-counter increment)
# --------------------------------------------------------------------------- #
def test_put_run_round_trips_and_increments_counter() -> None:
    first = state.put_run(sample_id="s1", file_id="f1", recipe_id="r1")
    second = state.put_run(sample_id="s2", file_id="f2", recipe_id="r1")

    assert first == "run_0001"
    assert second == "run_0002"
    assert state.get_run(first).sample_id == "s1"
    assert {r["run_id"] for r in state.list_runs()} == {"run_0001", "run_0002"}


def test_reset_runs_clears_and_resets_counter() -> None:
    state.put_run(sample_id="s1", file_id="f1", recipe_id="r1")
    state.reset_runs()

    assert state.list_runs() == []
    assert state.put_run(sample_id="s2", file_id="f2", recipe_id="r1") == "run_0001"


# --------------------------------------------------------------------------- #
# qc records
# --------------------------------------------------------------------------- #
def test_put_qc_record_round_trips() -> None:
    src = state.put_qc_record("img.lsm", status="warning", warnings=["dim"])

    assert src == "img.lsm"
    rec = state.get_qc_record("img.lsm")
    assert rec.status == "warning"
    assert rec.warnings == ["dim"]
    assert [q["source"] for q in state.list_qc_records()] == ["img.lsm"]

    state.reset_qc_records()
    assert state.list_qc_records() == []


def test_put_qc_record_rejects_bad_status() -> None:
    with pytest.raises(ValueError):
        state.put_qc_record("x", status="nonsense")


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #
def test_put_sample_round_trips() -> None:
    name = state.put_sample("sampleA", group="ctrl", layers=["L1"], sample_id="A")

    assert name == "sampleA"
    s = state.get_sample("sampleA")
    assert s.sample_id == "A"
    assert s.group == "ctrl"
    assert s.layers == ["L1"]
    assert [x["sample_name"] for x in state.list_samples()] == ["sampleA"]

    state.reset_samples()
    assert state.list_samples() == []


def test_put_sample_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        state.put_sample("   ")


# --------------------------------------------------------------------------- #
# tables + listeners
# --------------------------------------------------------------------------- #
def test_put_table_round_trips_and_suffixes_collision(fresh_session) -> None:
    df = _demo_df()
    first = state.put_table("demo", df)
    second = state.put_table("demo", _demo_df())

    assert first == "demo"
    assert second == "demo_1"
    assert state.get_table("demo").equals(df)
    assert set(state.list_tables()) == {"demo", "demo_1"}


def test_update_table_replaces_dataframe(fresh_session) -> None:
    state.put_table("demo", _demo_df())
    replacement = pd.DataFrame({"label": [9], "area": [99.0]})

    state.update_table("demo", replacement)

    assert state.get_table("demo").equals(replacement)


def test_get_table_unknown_raises_keyerror(fresh_session) -> None:
    with pytest.raises(KeyError):
        state.get_table("missing")


def test_on_tables_changed_fires_on_put_and_is_idempotent(fresh_session) -> None:
    calls: list[int] = []
    cb = lambda: calls.append(1)  # noqa: E731
    state.on_tables_changed(cb)
    state.on_tables_changed(cb)  # registering the same callable again is a no-op

    state.put_table("demo", _demo_df())

    assert calls == [1]


def test_reset_tables_clears_and_notifies(fresh_session) -> None:
    state.put_table("demo", _demo_df())
    calls: list[int] = []
    state.on_tables_changed(lambda: calls.append(1))

    state.reset_tables()

    assert state.list_tables() == []
    assert calls == [1]


def test_bulk_state_update_coalesces_to_single_fire(fresh_session) -> None:
    calls: list[int] = []
    state.on_tables_changed(lambda: calls.append(1))

    with state.bulk_state_update("bulk"):
        state.put_table("a", _demo_df())
        state.put_table("b", _demo_df())
        state.put_table("c", _demo_df())
        # nothing should have fired while still inside the block
        assert calls == []

    # exactly one notification after the block exits
    assert calls == [1]
    assert set(state.list_tables()) == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# session isolation
# --------------------------------------------------------------------------- #
def test_session_swap_isolates_state() -> None:
    original = state.current_session()
    try:
        state.reset_tables()
        state.put_file("/a/a.lsm", "a.lsm")
        state.put_table("t", _demo_df())
        old = state.current_session()

        new = AnalysisSession()
        state.set_current_session(new)

        # the new session sees none of the old session's state
        assert state.list_files() == []
        assert state.list_tables() == []

        # mutating the new session does not reach back into the old object
        state.put_file("/b/b.lsm", "b.lsm")
        assert "a" in old.files
        assert "b" not in old.files
        assert "b" in new.files
        assert "a" not in new.files
    finally:
        state.set_current_session(original)
        state.reset_tables()


def test_reset_session_yields_empty_session() -> None:
    original = state.current_session()
    try:
        state.put_file("/a/a.lsm", "a.lsm")

        session = state.reset_session()

        assert isinstance(session, AnalysisSession)
        assert state.current_session() is session
        assert state.list_files() == []
        assert state.list_tables() == []
    finally:
        state.set_current_session(original)


# --------------------------------------------------------------------------- #
# snapshot / restore round-trip (all six restorable families)
# --------------------------------------------------------------------------- #
def _full_payload() -> dict:
    return {
        "files": [
            {
                "file_id": "img",
                "path": "/p/img.lsm",
                "original_name": "img.lsm",
                "file_type": "lsm",
                "load_status": "loaded",
            }
        ],
        "samples": [
            {"sample_name": "s1", "sample_id": "s1", "group": "ctrl", "layers": ["L"]}
        ],
        "channels": [
            {"layer_name": "L", "role": "target", "color": "green", "marker": "GFP"}
        ],
        "recipes": [{"name": "r1", "target_channel": "green"}],
        "runs": [
            {
                "run_id": "run_0003",
                "sample_id": "s1",
                "file_id": "img",
                "recipe_id": "r1",
                "status": "complete",
            }
        ],
        "qc_records": [{"source": "img", "status": "pass"}],
    }


def test_restore_then_snapshot_round_trips_all_families(fresh_session) -> None:
    state.restore_session_state(**_full_payload())

    snap = state.snapshot_session_state()

    assert [f["file_id"] for f in snap["files"]] == ["img"]
    assert snap["files"][0]["load_status"] == "loaded"
    assert [s["sample_name"] for s in snap["samples"]] == ["s1"]
    assert [c["layer_name"] for c in snap["channels"]] == ["L"]
    assert snap["channels"][0]["role"] == "target"
    assert [r["name"] for r in snap["recipes"]] == ["r1"]
    assert [r["run_id"] for r in snap["runs"]] == ["run_0003"]
    assert [q["source"] for q in snap["qc_records"]] == ["img"]


def test_restore_clear_existing_replaces_prior_state(fresh_session) -> None:
    state.put_file("/old/old.lsm", "old.lsm")

    state.restore_session_state(**_full_payload())  # clear_existing=True default

    assert {f["file_id"] for f in state.list_files()} == {"img"}


def test_restore_continues_run_counter(fresh_session) -> None:
    # The restore path pins the run counter forward to the highest restored id;
    # a subsequently-generated run must continue from there, not restart at 1.
    state.restore_session_state(
        runs=[{"run_id": "run_0007", "sample_id": "s", "file_id": "f", "recipe_id": "r"}]
    )

    next_run = state.put_run(sample_id="s", file_id="f", recipe_id="r")

    assert next_run == "run_0008"


# --------------------------------------------------------------------------- #
# channel helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("GFP", "green"),
        ("far red", "ir"),
        ("DAPI", "uv"),
        ("mCherry", "red"),
        (None, None),
        ("definitely-not-a-color", None),
    ],
)
def test_canonical_channel_color(value, expected) -> None:
    assert state.canonical_channel_color(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("primary", "target"),
        ("counter", "counterstain"),
        ("exclude", "ignore"),
        (None, "unknown"),
    ],
)
def test_canonical_channel_role(value, expected) -> None:
    assert state.canonical_channel_role(value) == expected


def test_canonical_channel_role_unknown_raises() -> None:
    with pytest.raises(ValueError):
        state.canonical_channel_role("not-a-role")


# --------------------------------------------------------------------------- #
# channel annotations + resolution (need a viewer with stub layers)
# --------------------------------------------------------------------------- #
def test_put_channel_annotation_round_trips(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="reporter")

    resolved = state.put_channel_annotation(
        "reporter", role="target", color="green", marker="GCaMP"
    )

    assert resolved == "reporter"
    [entry] = state.list_channel_annotations()
    assert entry["layer_name"] == "reporter"
    assert entry["role"] == "target"
    assert entry["color"] == "green"

    state.reset_channel_annotations()
    assert state.list_channel_annotations() == []


def test_resolve_target_channel_infers_single_image_layer(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="only_layer")

    res = state.resolve_target_channel()

    assert res.layer == "only_layer"
    assert res.source == "inference"


def test_resolve_target_channel_uses_confirmed_annotation(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="green_ch")
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="other_ch")
    state.put_channel_annotation("green_ch", role="target", color="green")

    res = state.resolve_target_channel()

    assert res.layer == "green_ch"
    assert res.source == "annotation"


def test_resolve_target_channel_ambiguous_raises(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="img_a")
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="img_b")

    with pytest.raises(state.AmbiguousChannelError):
        state.resolve_target_channel()


def test_resolve_target_channel_refuses_counterstain_query(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="nuc_dapi")
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="signal")
    state.put_channel_annotation("nuc_dapi", role="counterstain", color="uv")

    # querying a counterstain-annotated channel must not auto-select it
    with pytest.raises(state.AmbiguousChannelError):
        state.resolve_target_channel("uv")


def test_resolve_layer_name_via_annotation(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="reporter_ch1")
    state.put_channel_annotation("reporter_ch1", role="target", color="green", marker="GCaMP")

    assert state.resolve_layer_name("green") == "reporter_ch1"
    assert state.resolve_layer_name("GCaMP") == "reporter_ch1"
