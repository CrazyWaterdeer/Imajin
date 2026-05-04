from __future__ import annotations

from pathlib import Path

from imajin.agent import state
from imajin.tools import experiment


def test_annotate_sample_records_group_metadata() -> None:
    state.reset_samples()

    res = experiment.annotate_sample(
        sample_name="control_1",
        group="control",
        layers=["ctrl1_ch0", "ctrl1_ch1"],
        files=["/data/control_1.lsm"],
        notes="adult gut",
    )

    assert res["sample_name"] == "control_1"
    assert res["group"] == "control"
    samples = experiment.list_sample_annotations()
    assert len(samples) == 1
    s = samples[0]
    assert s["sample_name"] == "control_1"
    assert s["group"] == "control"
    assert s["layers"] == ["ctrl1_ch0", "ctrl1_ch1"]
    assert s["files"] == ["/data/control_1.lsm"]
    assert s["notes"] == "adult gut"
    # New Phase-3 fields default to safe values:
    assert s["sample_id"] == "control_1"
    assert s["file_ids"] == []
    assert s["extra"] == {}

    state.reset_samples()


def test_register_files_accepts_windows_style_paths() -> None:
    state.reset_files()

    res = experiment.register_files(
        [r"C:\Users\Jin\Documents\School\GIST\Lab\Project\test\sample.lsm"]
    )

    [record] = res["files"]
    assert record["path"] == (
        "/mnt/c/Users/Jin/Documents/School/GIST/Lab/Project/test/sample.lsm"
    )
    assert record["original_name"] == "sample.lsm"
    assert record["supported"] is True
    assert record["exists"] is False
    assert res["n_missing"] == 1

    state.reset_files()


def test_register_files_expands_directory_inputs(tmp_path: Path) -> None:
    state.reset_files()
    (tmp_path / "control_1.lsm").write_bytes(b"")
    (tmp_path / "control_2.czi").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore me")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested_1.lsm").write_bytes(b"")

    res = experiment.register_files([str(tmp_path)])

    assert res["n_registered"] == 2
    assert res["n_supported"] == 2
    assert res["n_input_dirs"] == 1
    assert res["n_scanned_dirs"] == 1
    assert res["n_ignored_non_image"] == 1
    assert res["directories"][0]["n_found"] == 2
    assert {f["original_name"] for f in res["files"]} == {
        "control_1.lsm",
        "control_2.czi",
    }

    state.reset_files()


def test_register_files_can_filter_folder_by_user_scope_text(tmp_path: Path) -> None:
    state.reset_files()
    (tmp_path / "10012 + 1234 vF saline injected midgut R3 1.lsm").write_bytes(b"")
    (tmp_path / "2966 + 1234 vF venerose injected hindgut rectum 1.lsm").write_bytes(b"")
    (tmp_path / "2966+1234 vF saline injected midgut R3 2.lsm").write_bytes(b"")

    res = experiment.register_files([str(tmp_path)], include=["2966 + 1234"])

    assert res["n_registered"] == 2
    assert res["n_discarded_by_filter"] == 1
    assert {record["original_name"] for record in res["files"]} == {
        "2966 + 1234 vF venerose injected hindgut rectum 1.lsm",
        "2966+1234 vF saline injected midgut R3 2.lsm",
    }

    state.reset_files()


def test_filter_registered_files_returns_matching_registry_records(tmp_path: Path) -> None:
    state.reset_files()
    (tmp_path / "10012 + 1234 vF saline injected midgut R3 1.lsm").write_bytes(b"")
    (tmp_path / "2966 + 1234 vF venerose injected hindgut rectum 1.lsm").write_bytes(b"")
    (tmp_path / "2966 + 1234 vF saline injected midgut R3 2.lsm").write_bytes(b"")
    experiment.register_files([str(tmp_path)])

    res = experiment.filter_registered_files(
        include=["2966 + 1234"], exclude=["venerose"]
    )

    assert res["n_registered"] == 3
    assert res["n_matched"] == 1
    assert res["representative_file"]["original_name"] == (
        "2966 + 1234 vF saline injected midgut R3 2.lsm"
    )
    assert res["representative_path"].endswith(
        "2966 + 1234 vF saline injected midgut R3 2.lsm"
    )

    state.reset_files()


def test_filter_registered_files_no_match_has_no_representative(tmp_path: Path) -> None:
    state.reset_files()
    (tmp_path / "10012 + 1234 vF saline injected midgut R3 1.lsm").write_bytes(b"")
    experiment.register_files([str(tmp_path)])

    res = experiment.filter_registered_files(include=["2966 + 1234"])

    assert res["n_matched"] == 0
    assert res["files"] == []
    assert res["representative_file"] is None
    assert res["representative_path"] is None

    state.reset_files()


def test_list_registered_files_is_paginated(tmp_path: Path) -> None:
    state.reset_files()
    for idx in range(12):
        (tmp_path / f"sample_{idx:02d}.lsm").write_bytes(b"")
    experiment.register_files([str(tmp_path)])

    first = experiment.list_registered_files(limit=5)
    second = experiment.list_registered_files(offset=first["next_offset"], limit=5)

    assert first["n_matched"] == 12
    assert first["has_more"] is True
    assert first["next_offset"] == 5
    assert len(first["files"]) == 5
    assert second["offset"] == 5
    assert len(second["files"]) == 5

    state.reset_files()


def test_register_files_can_scan_directories_recursively(tmp_path: Path) -> None:
    state.reset_files()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested_1.ome.tif").write_bytes(b"")

    res = experiment.register_files([str(tmp_path)], recursive=True)

    assert res["n_registered"] == 1
    assert res["n_scanned_dirs"] == 2
    assert res["files"][0]["original_name"] == "nested_1.ome.tif"

    state.reset_files()
