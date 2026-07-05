from __future__ import annotations

import errno
from pathlib import Path

from imajin.paths import (
    _safe_is_dir,
    normalize_user_path,
    windows_drive_roots,
    windows_file_dialog_locations,
)


def test_normalize_windows_drive_path_for_wsl() -> None:
    path = normalize_user_path(r"C:\Users\Jin\Downloads\sample.lsm")
    assert path == Path("/mnt/c/Users/Jin/Downloads/sample.lsm")


def test_normalize_windows_drive_path_with_forward_slashes() -> None:
    path = normalize_user_path("D:/data/experiment/sample.ome.tif")
    assert path == Path("/mnt/d/data/experiment/sample.ome.tif")


def test_normalize_linux_path_is_unchanged() -> None:
    path = normalize_user_path("/home/jin/Imajin/data.tif")
    assert path == Path("/home/jin/Imajin/data.tif")


def test_normalize_quoted_windows_directory() -> None:
    path = normalize_user_path(r'"C:\Users\Jin\Documents\School\GIST\Lab\Project\test"')
    assert path == Path("/mnt/c/Users/Jin/Documents/School/GIST/Lab/Project/test")


def test_normalize_windows_file_uri() -> None:
    path = normalize_user_path("file:///C:/Users/Jin/Downloads/sample%201.lsm")
    assert path == Path("/mnt/c/Users/Jin/Downloads/sample 1.lsm")


def test_normalize_wsl_localhost_unc_path() -> None:
    path = normalize_user_path(
        r"\\wsl.localhost\Ubuntu\home\jin\New Folder\sample.lsm"
    )
    assert path == Path("/home/jin/New Folder/sample.lsm")


def test_normalize_wsl_dollar_unc_path() -> None:
    path = normalize_user_path(r"\\wsl$\Ubuntu\home\jin\New Folder\sample.lsm")
    assert path == Path("/home/jin/New Folder/sample.lsm")


def test_normalize_wsl_file_uri() -> None:
    path = normalize_user_path("file://wsl.localhost/Ubuntu/home/jin/sample%201.lsm")
    assert path == Path("/home/jin/sample 1.lsm")


def test_safe_is_dir_swallows_dead_mount_oserror() -> None:
    class _DeadMount:
        def is_dir(self) -> bool:
            raise OSError(errno.ENODEV, "No such device")

    assert _safe_is_dir(_DeadMount()) is False  # type: ignore[arg-type]


def _patch_fake_mnt(monkeypatch, children: dict[str, object]) -> None:
    """Present a synthetic ``/mnt`` whose named children behave per ``children``.

    Each value is ``True`` (live directory), ``False`` (not a directory), or an
    ``OSError`` instance to raise from ``is_dir`` (a dead mount).
    """
    mnt = Path("/mnt")
    real_exists, real_iterdir, real_is_dir = Path.exists, Path.iterdir, Path.is_dir

    def fake_exists(self: Path) -> bool:
        return True if self == mnt else real_exists(self)

    def fake_iterdir(self: Path):
        if self == mnt:
            return iter([mnt / name for name in children])
        return real_iterdir(self)

    def fake_is_dir(self: Path) -> bool:
        if self.parent == mnt and self.name in children:
            outcome = children[self.name]
            if isinstance(outcome, OSError):
                raise outcome
            return bool(outcome)
        return real_is_dir(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    monkeypatch.setattr(Path, "is_dir", fake_is_dir)


def test_windows_drive_roots_skips_dead_mount(monkeypatch) -> None:
    _patch_fake_mnt(
        monkeypatch,
        {"c": True, "e": OSError(errno.ENODEV, "No such device"), "wslg": True},
    )
    # 'wslg' is >1 char so it is not a drive; 'e' is dead and must be skipped, not raise.
    assert windows_drive_roots() == [Path("/mnt/c")]


def test_windows_file_dialog_locations_survives_dead_mount(monkeypatch) -> None:
    _patch_fake_mnt(
        monkeypatch, {"c": True, "e": OSError(errno.ENODEV, "No such device")}
    )
    # No Users/ subtree on the fake live drive, so this is just the live root —
    # the point is that enumeration returns instead of crashing on /mnt/e.
    assert Path("/mnt/c") in windows_file_dialog_locations()


def test_windows_file_dialog_locations_excludes_junction_profiles(monkeypatch) -> None:
    mnt = Path("/mnt")
    drive = mnt / "c"
    users = drive / "Users"
    real_user = users / "Jin"
    junctions = [users / "All Users", users / "Default User", users / "Public"]
    known_dirs = {mnt, drive, users, real_user, *junctions}
    real_iterdir, real_is_dir, real_exists = Path.iterdir, Path.is_dir, Path.exists

    def fake_exists(self: Path) -> bool:
        return True if self == mnt else real_exists(self)

    def fake_iterdir(self: Path):
        if self == mnt:
            return iter([drive])
        if self == users:
            return iter([*junctions, real_user])
        return real_iterdir(self)

    def fake_is_dir(self: Path) -> bool:
        return True if self in known_dirs else real_is_dir(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    monkeypatch.setattr(Path, "is_dir", fake_is_dir)

    names = {p.name for p in windows_file_dialog_locations()}
    assert "Jin" in names  # a real profile is offered
    assert names.isdisjoint({"All Users", "Default User", "Public"})  # junctions are not
