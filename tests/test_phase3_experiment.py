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


# --- Task 4: AnalysisRun ------------------------------------------------------

def test_put_run_records_status_and_outputs() -> None:
    run_id = state.put_run(
        sample_id="ctrl_1",
        file_id="ctrl_1",
        recipe_id="gut_GFP",
        status="complete",
        table_names=["ctrl_1_measurements"],
        layer_names=["ctrl_1_masks"],
        summary={"n_objects": 42},
    )
    assert run_id  # non-empty
    runs = state.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["sample_id"] == "ctrl_1"
    assert r["recipe_id"] == "gut_GFP"
    assert r["status"] == "complete"
    assert r["summary"] == {"n_objects": 42}


def test_put_run_marks_failed_with_error() -> None:
    run_id = state.put_run(
        sample_id="t_1",
        file_id="t_1",
        recipe_id="gut_GFP",
        status="failed",
        error="cellpose returned zero objects",
    )
    r = state.get_run(run_id)
    assert r.status == "failed"
    assert r.error == "cellpose returned zero objects"


# --- Task 5: register_files ---------------------------------------------------

from imajin.tools import experiment  # if not already imported


def test_register_files_creates_records_without_loading(tmp_path: Path) -> None:
    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "ctrl_2.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")

    res = experiment.register_files([str(a), str(b)])
    assert res["n_registered"] == 2
    assert {f["original_name"] for f in res["files"]} == {"ctrl_1.lsm", "ctrl_2.lsm"}
    assert all(f["load_status"] == "unloaded" for f in res["files"])
    assert {f["file_id"] for f in res["files"]} == {"ctrl_1", "ctrl_2"}


def test_register_files_marks_missing_unsupported(tmp_path: Path) -> None:
    real = tmp_path / "ok.lsm"
    real.write_bytes(b"")
    missing = tmp_path / "ghost.lsm"  # not created
    weird = tmp_path / "data.xyz"
    weird.write_bytes(b"")

    res = experiment.register_files([str(real), str(missing), str(weird)])
    by_name = {f["original_name"]: f for f in res["files"]}
    assert by_name["ok.lsm"]["supported"] is True
    assert by_name["ok.lsm"]["exists"] is True
    assert by_name["ghost.lsm"]["exists"] is False
    assert by_name["data.xyz"]["supported"] is False
    assert res["n_unsupported"] == 1
    assert res["n_missing"] == 1


def test_register_files_does_not_parse_filename_into_group(tmp_path: Path) -> None:
    """Spec rule: never silently parse J41/vF/midgut/R3/trailing-numbers."""
    p = tmp_path / "J41 + 1234 vF midgut R3 1.lsm"
    p.write_bytes(b"")
    res = experiment.register_files([str(p)])
    rec = res["files"][0]
    for forbidden in ("group", "condition", "replicate", "tissue"):
        assert forbidden not in rec or rec[forbidden] in (None, "")


# --- Task 6: annotate_samples (bulk) -----------------------------------------

def test_annotate_samples_bulk_creates_two_groups(tmp_path: Path) -> None:
    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])

    res = experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control", "files": [str(a)]},
            {
                "sample_name": "trt_1",
                "group": "treatment",
                "files": [str(b)],
                "extra": {"genotype": "w1118", "tissue": "midgut"},
            },
        ]
    )
    assert res["n_samples"] == 2
    samples = state.list_samples()
    by_name = {s["sample_name"]: s for s in samples}
    assert by_name["ctrl_1"]["group"] == "control"
    assert by_name["ctrl_1"]["file_ids"] == ["ctrl_1"]
    assert by_name["trt_1"]["extra"]["genotype"] == "w1118"


def test_annotate_samples_accepts_file_ids_directly(tmp_path: Path) -> None:
    p = tmp_path / "x.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])

    experiment.annotate_samples(
        [{"sample_name": "s1", "group": "g", "file_ids": ["x"]}]
    )
    s = state.list_samples()[0]
    assert s["file_ids"] == ["x"]
    assert s["files"] == [str(p.resolve())]


def test_annotate_samples_does_not_invent_extra_from_filename(tmp_path: Path) -> None:
    """A user passing only sample_name/group/files must not get genotype/tissue/etc.
    autofilled from substrings in the filename."""
    p = tmp_path / "J41 + 1234 vF midgut R3 1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])

    experiment.annotate_samples(
        [{"sample_name": "s1", "group": "control", "files": [str(p)]}]
    )
    s = state.list_samples()[0]
    assert s["extra"] == {}


# --- Task 7: list_experiment --------------------------------------------------

def test_list_experiment_returns_all_collections(tmp_path: Path) -> None:
    p = tmp_path / "x.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "s1", "group": "control", "files": [str(p)]}]
    )
    state.put_recipe(name="r1", target_channel="green")
    state.put_run(
        sample_id="s1", file_id="x", recipe_id="r1", status="pending"
    )

    res = experiment.list_experiment()
    assert {f["file_id"] for f in res["files"]} == {"x"}
    assert {s["sample_name"] for s in res["samples"]} == {"s1"}
    assert {r["name"] for r in res["recipes"]} == {"r1"}
    assert res["runs"][0]["sample_id"] == "s1"


def test_list_experiment_handles_empty_state() -> None:
    res = experiment.list_experiment()
    assert res == {"files": [], "samples": [], "recipes": [], "runs": []}


# --- Task 8: create_analysis_recipe ------------------------------------------

def test_create_analysis_recipe_stores_full_payload() -> None:
    res = experiment.create_analysis_recipe(
        name="gut_GFP",
        target_channel="green",
        segmentation={"tool": "cellpose_sam", "do_3D": True, "diameter": None},
        measurement={"properties": ["area", "centroid", "mean_intensity"]},
        preprocessing=[{"step": "rolling_ball", "radius": 25}],
    )
    assert res["recipe_id"] == "gut_GFP"
    r = state.get_recipe("gut_GFP")
    assert r.target_channel == "green"
    assert r.segmentation["do_3D"] is True
    assert r.preprocessing == [{"step": "rolling_ball", "radius": 25}]


def test_create_analysis_recipe_minimal_inputs() -> None:
    res = experiment.create_analysis_recipe(name="r2", target_channel="red")
    assert res["recipe_id"] == "r2"
    r = state.get_recipe("r2")
    assert r.segmentation == {}
    assert r.measurement == {}


# --- Task 9: run_recipe_on_samples (single sample) ---------------------------

def _two_label_image() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    img = np.zeros_like(labels, dtype=np.float32)
    img[2:8, 2:8] = 100.0
    img[12:18, 12:18] = 50.0
    return labels, img


def _stub_cellpose(monkeypatch, mask: np.ndarray) -> None:
    from imajin.tools import segment

    class _FakeModel:
        def eval(self, data, **kwargs):  # noqa: ANN001
            return mask, None, None

    monkeypatch.setattr(
        segment, "_get_cellpose_model", lambda *a, **kw: _FakeModel()
    )


def test_run_recipe_on_samples_single_sample_attaches_columns(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    labels, img = _two_label_image()
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [
            {
                "sample_name": "ctrl_1",
                "group": "control",
                "files": [str(p)],
                "layers": ["ctrl_1_ch0"],
            }
        ]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="green",
        segmentation={"tool": "cellpose_sam"},
        measurement={"properties": ["area", "centroid", "mean_intensity"]},
    )
    _stub_cellpose(monkeypatch, labels)

    res = workflows.run_recipe_on_samples(recipe_name="r1")
    assert res["n_samples"] == 1
    assert res["n_complete"] == 1
    assert res["n_failed"] == 0
    run = res["runs"][0]
    assert run["status"] == "complete"
    df = state.get_table(run["table_names"][0])
    for col in (
        "sample_id",
        "sample_name",
        "group",
        "file_id",
        "source_file",
        "source_layer",
    ):
        assert col in df.columns, f"missing required column: {col}"
    assert (df["sample_name"] == "ctrl_1").all()
    assert (df["group"] == "control").all()
    assert (df["file_id"] == "ctrl_1").all()
    assert (df["source_layer"] == "ctrl_1_ch0").all()


# --- Task 10: run_recipe_on_samples (multi-sample, failure isolation) --------

def test_run_recipe_on_samples_multi_sample_one_fails(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    labels, img = _two_label_image()
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    viewer.add_image(np.zeros_like(img), name="trt_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])
    experiment.annotate_samples(
        [
            {
                "sample_name": "ctrl_1",
                "group": "control",
                "files": [str(a)],
                "layers": ["ctrl_1_ch0"],
            },
            {
                "sample_name": "trt_1",
                "group": "treatment",
                "files": [str(b)],
                "layers": ["trt_1_ch0"],
            },
        ]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="ctrl_1_ch0",
        segmentation={"tool": "cellpose_sam"},
        measurement={"properties": ["area", "mean_intensity"]},
    )

    call = {"n": 0}

    def _fake_model_factory(*a, **kw):  # noqa: ANN001
        class _FM:
            def eval(self, data, **kwargs):  # noqa: ANN001
                call["n"] += 1
                if call["n"] == 1:
                    return labels, None, None
                return np.zeros_like(labels), None, None

        return _FM()

    from imajin.tools import segment

    monkeypatch.setattr(segment, "_get_cellpose_model", _fake_model_factory)

    res = workflows.run_recipe_on_samples(
        recipe_name="r1", sample_names=["ctrl_1", "trt_1"]
    )
    assert res["n_samples"] == 2
    assert res["n_complete"] == 1
    assert res["n_failed"] == 1
    statuses = [r["status"] for r in res["runs"]]
    assert sorted(statuses) == ["complete", "failed"]

    runs = state.list_runs()
    assert {r["status"] for r in runs} == {"complete", "failed"}
    failed = next(r for r in runs if r["status"] == "failed")
    err = (failed["error"] or "").lower()
    assert "zero objects" in err or "ok=false" in err


def test_run_recipe_on_samples_no_samples_returns_empty() -> None:
    from imajin.tools import workflows

    state.put_recipe(name="r_empty", target_channel="green")
    res = workflows.run_recipe_on_samples(recipe_name="r_empty", sample_names=[])
    assert res["n_samples"] == 0
    assert res["runs"] == []


# --- Task 11: summarize_experiment -------------------------------------------

def test_summarize_experiment_sample_and_group_levels() -> None:
    df = pd.DataFrame(
        {
            "label": [1, 2, 1, 2, 1, 2],
            "sample_id": ["c1", "c1", "c2", "c2", "t1", "t1"],
            "sample_name": ["c1", "c1", "c2", "c2", "t1", "t1"],
            "group": ["control", "control", "control", "control", "treatment", "treatment"],
            "mean_intensity": [10.0, 20.0, 12.0, 18.0, 50.0, 60.0],
            "area": [100, 110, 105, 115, 90, 95],
        }
    )
    state.put_table("measurements", df)

    res = experiment.summarize_experiment(measurement="mean_intensity")
    sample_tbl = state.get_table(res["sample_table"])
    assert set(sample_tbl["sample_name"]) == {"c1", "c2", "t1"}
    c1 = sample_tbl[sample_tbl["sample_name"] == "c1"].iloc[0]
    assert c1["mean"] == 15.0
    assert c1["count"] == 2

    group_tbl = state.get_table(res["group_table"])
    assert set(group_tbl["group"]) == {"control", "treatment"}
    ctrl = group_tbl[group_tbl["group"] == "control"].iloc[0]
    assert ctrl["mean"] == 15.0  # mean of sample means (15, 15)
    assert ctrl["n_samples"] == 2
    assert ctrl["n_objects"] == 4


def test_summarize_experiment_handles_missing_group() -> None:
    df = pd.DataFrame(
        {
            "label": [1, 2],
            "sample_id": ["s1", "s1"],
            "sample_name": ["s1", "s1"],
            "group": [None, None],
            "mean_intensity": [5.0, 7.0],
        }
    )
    state.put_table("measurements", df)

    res = experiment.summarize_experiment(measurement="mean_intensity")
    group_tbl = state.get_table(res["group_table"])
    assert len(group_tbl) >= 1
