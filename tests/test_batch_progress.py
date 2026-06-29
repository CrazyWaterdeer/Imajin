"""Batch-progress ledger + get_batch_progress tool (no viewer needed)."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_session():
    from imajin import session as state

    state.reset_runs()
    state.current_session().files.clear()
    yield
    state.reset_runs()
    state.current_session().files.clear()


def test_no_progress_returns_none() -> None:
    from imajin.agent.context import batch_progress_data, summarize_batch_progress

    assert summarize_batch_progress() is None
    d = batch_progress_data()
    assert d["analysed"] == [] and d["pending"] == [] and d["failed"] == []
    assert d["universe_known"] is False and d["next_pending"] is None


def test_complete_run_unregistered_marks_pending_unknown() -> None:
    from imajin import session as state
    from imajin.agent.context import batch_progress_data, summarize_batch_progress

    state.put_run(
        sample_id="rectum 1",
        file_id="/data/rectum1.lsm",
        recipe_id="interactive:target_objects:two_tier",
        status="complete",
        table_names=["rectum1_two_tier"],
    )
    s = summarize_batch_progress()
    assert s is not None
    assert "analysed 1" in s and "rectum1_two_tier" in s and "pending unknown" in s
    d = batch_progress_data()
    assert len(d["analysed"]) == 1 and d["analysed"][0]["table"] == "rectum1_two_tier"
    assert d["universe_known"] is False


def test_registered_pending_and_keyspace_normalization() -> None:
    from imajin import session as state
    from imajin.agent.context import batch_progress_data, summarize_batch_progress

    fid_a = state.put_file("/data/a.lsm", "a")
    state.put_file("/data/b.lsm", "b")
    # A batch-style run keyed by the REGISTERED file id must normalise to a's path
    # key so file a counts as analysed and file b as pending (Codex keyspace bug).
    state.put_run(
        sample_id="a", file_id=fid_a, recipe_id="recipe_1",
        status="complete", table_names=["a_tbl"],
    )
    d = batch_progress_data()
    assert d["universe_known"] is True and d["n_universe"] == 2
    assert len(d["analysed"]) == 1 and len(d["pending"]) == 1
    assert d["next_pending"] is not None
    s = summarize_batch_progress()
    assert "analysed 1/2" in s and "pending 1" in s


def test_failed_only_shown_separately() -> None:
    from imajin import session as state
    from imajin.agent.context import batch_progress_data

    state.put_run(
        sample_id="x", file_id="/data/x.lsm",
        recipe_id="interactive:target_objects:single", status="failed",
    )
    d = batch_progress_data()
    assert len(d["failed"]) == 1 and len(d["analysed"]) == 0


def test_char_cap_enforced() -> None:
    from imajin import session as state
    from imajin.agent.context import summarize_batch_progress

    for i in range(50):
        state.put_run(
            sample_id=f"s{i}", file_id=f"/data/file_{i}_{'x' * 40}.lsm",
            recipe_id="interactive:target_objects:single", status="complete",
            table_names=[f"tbl_{i}"],
        )
    s = summarize_batch_progress(max_chars=400)
    assert s is not None and len(s) <= 400


def test_get_batch_progress_tool() -> None:
    from imajin import session as state
    from imajin.tools.experiment import get_batch_progress

    state.put_file("/data/a.lsm", "a")
    state.put_run(
        sample_id="a", file_id="a", recipe_id="r",
        status="complete", table_names=["t"],
    )
    out = get_batch_progress()
    assert out["universe_known"] is True
    assert len(out["analysed"]) == 1 and out["next_pending"] is None
