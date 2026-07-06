from __future__ import annotations

from imajin.analysis.resume import file_signature, rel_key, sample_slug_for


def test_rel_key_is_anchor_relative_and_forward_slashed():
    assert rel_key("/data/exp/a.lsm", "/data/exp") == "a.lsm"
    assert rel_key("/data/exp/sub/a.lsm", "/data/exp") == "sub/a.lsm"


def test_rel_key_matches_across_mounts():
    # The same file under a shared anchor keys identically regardless of the
    # absolute root — WSL vs Windows — which is what lets resume match cross-machine.
    wsl = rel_key("/mnt/d/exp/a.lsm", "/mnt/d/exp")
    win = rel_key("D:/exp/a.lsm", "D:/exp")
    assert wsl == win == "a.lsm"


def test_rel_key_outside_anchor_uses_dotdot():
    assert rel_key("/data/other/a.lsm", "/data/exp") == "../other/a.lsm"


def test_sample_slug_deterministic_and_distinct():
    same = "a.lsm"
    other = "sub/a.lsm"  # same stem, different key
    assert sample_slug_for(same) == sample_slug_for(same)  # idempotent
    assert sample_slug_for(same) != sample_slug_for(other)  # no collision on stem
    assert sample_slug_for(same).startswith("a_")


def test_file_signature(tmp_path):
    assert file_signature(tmp_path / "nope.lsm") is None
    f = tmp_path / "x.txt"
    f.write_text("hi")
    sig = file_signature(f)
    assert sig is not None and sig["size"] == 2 and "mtime" in sig
