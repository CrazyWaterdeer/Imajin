from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from imajin.anchor import resolve_anchor_folder, resolve_session_anchor


def test_returns_none_for_empty_input():
    assert resolve_anchor_folder([]) is None


def test_single_file_returns_its_parent(tmp_path: Path):
    f = tmp_path / "a.lsm"
    f.write_bytes(b"")
    assert resolve_anchor_folder([f]) == tmp_path.absolute()


def test_multi_folder_returns_alphabetically_first(tmp_path: Path):
    a = tmp_path / "2026-05-09"
    b = tmp_path / "2026-05-10"
    a.mkdir()
    b.mkdir()
    (a / "x.lsm").write_bytes(b"")
    (b / "y.lsm").write_bytes(b"")
    anchor = resolve_anchor_folder([b / "y.lsm", a / "x.lsm"])
    assert anchor == a.absolute()


def test_case_insensitive_sort(tmp_path: Path):
    upper = tmp_path / "Zeta"
    lower = tmp_path / "alpha"
    upper.mkdir()
    lower.mkdir()
    (upper / "u.lsm").write_bytes(b"")
    (lower / "l.lsm").write_bytes(b"")
    anchor = resolve_anchor_folder([upper / "u.lsm", lower / "l.lsm"])
    assert anchor == lower.absolute()


def test_ignores_empty_strings():
    assert resolve_anchor_folder(["", None]) is None  # type: ignore[list-item]


def test_dot_path_resolves_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    anchor = resolve_anchor_folder(["./img.lsm"])
    assert anchor == tmp_path.absolute()


def test_session_anchor_uses_registered_files(tmp_path):
    a = tmp_path / "alpha"
    a.mkdir()
    file_a = a / "x.lsm"
    file_a.write_bytes(b"")
    with patch("imajin.agent.state.list_files", return_value=[{"path": str(file_a)}]):
        assert resolve_session_anchor() == a.absolute()


def test_session_anchor_merges_extra_paths(tmp_path):
    a = tmp_path / "alpha"
    b = tmp_path / "beta"
    a.mkdir()
    b.mkdir()
    (a / "x.lsm").write_bytes(b"")
    (b / "y.lsm").write_bytes(b"")
    with patch("imajin.agent.state.list_files", return_value=[{"path": str(b / "y.lsm")}]):
        anchor = resolve_session_anchor(extra_paths=[str(a / "x.lsm")])
    assert anchor == a.absolute()


def test_session_anchor_returns_none_when_no_files():
    with patch("imajin.agent.state.list_files", return_value=[]):
        assert resolve_session_anchor() is None
