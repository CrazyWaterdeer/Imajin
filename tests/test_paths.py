from __future__ import annotations

from pathlib import Path

from imajin.paths import normalize_user_path


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
