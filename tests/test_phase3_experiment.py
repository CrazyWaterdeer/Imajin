"""Phase 3: experiment / batch / reporting workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imajin.agent import state


@pytest.fixture(autouse=True)
def _reset_phase3_state():
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()
    state.reset_samples()
    state.reset_tables()
    yield
    state.reset_files()
    state.reset_recipes()
    state.reset_runs()
    state.reset_samples()
    state.reset_tables()


# --- Task 1: FileRecord ------------------------------------------------------

def test_put_file_creates_unloaded_record(tmp_path: Path) -> None:
    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    file_id = state.put_file(
        path=str(p),
        original_name="ctrl_1.lsm",
        file_type="lsm",
        metadata_summary={"axes": "CZYX"},
    )
    assert file_id == "ctrl_1"
    rec = state.get_file(file_id)
    assert rec.path == str(p)
    assert rec.original_name == "ctrl_1.lsm"
    assert rec.file_type == "lsm"
    assert rec.load_status == "unloaded"
    assert rec.metadata_summary == {"axes": "CZYX"}


def test_put_file_dedups_collisions(tmp_path: Path) -> None:
    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "subdir" / "ctrl_1.lsm"
    a.write_bytes(b"")
    b.parent.mkdir()
    b.write_bytes(b"")

    id_a = state.put_file(path=str(a), original_name="ctrl_1.lsm")
    id_b = state.put_file(path=str(b), original_name="ctrl_1.lsm")

    assert id_a == "ctrl_1"
    assert id_b == "ctrl_1_2"
    assert {f["file_id"] for f in state.list_files()} == {"ctrl_1", "ctrl_1_2"}


def test_list_files_returns_dicts(tmp_path: Path) -> None:
    p = tmp_path / "x.tif"
    p.write_bytes(b"")
    state.put_file(path=str(p), original_name="x.tif", file_type="tif")
    files = state.list_files()
    assert isinstance(files, list)
    assert files[0]["file_id"] == "x"
    assert files[0]["load_status"] == "unloaded"


# --- Task 2: SampleAnnotation evolution --------------------------------------

def test_put_sample_accepts_extra_and_optional_group() -> None:
    state.put_sample(
        sample_name="ctrl_1",
        group=None,
        file_ids=["ctrl_1"],
        layers=["ctrl_1_ch0"],
        extra={"genotype": "w1118", "tissue": "midgut"},
    )
    samples = state.list_samples()
    assert len(samples) == 1
    s = samples[0]
    assert s["sample_name"] == "ctrl_1"
    assert s["sample_id"] == "ctrl_1"  # defaults to sample_name
    assert s["group"] is None
    assert s["file_ids"] == ["ctrl_1"]
    assert s["extra"] == {"genotype": "w1118", "tissue": "midgut"}


def test_put_sample_keeps_legacy_files_and_layers() -> None:
    """Existing experiment.annotate_sample() and report.py rely on `files`/`layers`."""
    state.put_sample(
        sample_name="t1",
        group="treatment",
        files=["/data/t1.lsm"],
        layers=["t1_ch0", "t1_ch1"],
    )
    s = state.list_samples()[0]
    assert s["files"] == ["/data/t1.lsm"]
    assert s["layers"] == ["t1_ch0", "t1_ch1"]
    assert s["group"] == "treatment"


def test_put_sample_falls_back_to_name_for_whitespace_sample_id() -> None:
    state.put_sample(sample_name="abc", sample_id="   ")
    s = state.list_samples()[0]
    assert s["sample_id"] == "abc"


def test_render_samples_handles_none_group(tmp_path, monkeypatch) -> None:
    """Samples with group=None should render under 'unassigned', not 'None'."""
    from imajin.agent import provenance
    from imajin.tools import report

    state.put_sample(sample_name="ctrl_1", group=None)

    log_path = tmp_path / "session.jsonl"
    log_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(provenance, "_LOG_PATH", log_path)

    out = tmp_path / "r.md"
    report.generate_report(str(out), format="md")
    body = out.read_text(encoding="utf-8")
    assert "**unassigned**" in body
    assert "**None**" not in body


# --- Task 3: AnalysisRecipe ---------------------------------------------------

def test_put_recipe_round_trips() -> None:
    recipe_id = state.put_recipe(
        name="gut_GFP",
        target_channel="green",
        preprocessing=[{"step": "rolling_ball", "radius": 25}],
        segmentation={"tool": "cellpose_sam", "do_3D": True, "diameter": None},
        measurement={"properties": ["area", "centroid", "mean_intensity"]},
        notes="adult midgut R3",
    )
    assert recipe_id == "gut_GFP"
    r = state.get_recipe(recipe_id)
    assert r.target_channel == "green"
    assert r.preprocessing == [{"step": "rolling_ball", "radius": 25}]
    assert r.segmentation["do_3D"] is True
    assert r.measurement["properties"] == ["area", "centroid", "mean_intensity"]
    assert state.list_recipes()[0]["name"] == "gut_GFP"


def test_put_recipe_dedups_by_name() -> None:
    state.put_recipe(name="r1", target_channel="green")
    state.put_recipe(name="r1", target_channel="red")  # overwrite
    rs = state.list_recipes()
    assert len(rs) == 1
    assert rs[0]["target_channel"] == "red"
