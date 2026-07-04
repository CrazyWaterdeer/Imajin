"""Phase 3: experiment / batch / reporting workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imajin import session as state


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
    assert "ctrl_1_ch0" in viewer.layers
    assert "ctrl_1_ch0_masks" not in viewer.layers
    assert res["cleanup_enabled"] is True
    assert run["cleanup_removed_layers"]


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

    monkeypatch.setattr("imajin.tools._segmentation_io._get_cellpose_model", _fake_model_factory)

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


def test_run_recipe_on_samples_auto_loads_sample_local_target_and_cleans_layers(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import files, workflows

    viewer.add_image(
        np.ones((8, 8), dtype=np.float32),
        name="representative_green",
        metadata={"color": "green"},
    )
    a = tmp_path / "ctrl_1.tif"
    b = tmp_path / "trt_1.tif"
    a.write_bytes(b"fake")
    b.write_bytes(b"fake")
    registered = experiment.register_files([str(a), str(b)])
    by_name = {rec["original_name"]: rec for rec in registered["files"]}
    experiment.annotate_samples(
        [
            {
                "sample_name": "ctrl_1",
                "group": "control",
                "file_ids": [by_name["ctrl_1.tif"]["file_id"]],
            },
            {
                "sample_name": "trt_1",
                "group": "treatment",
                "file_ids": [by_name["trt_1.tif"]["file_id"]],
            },
        ]
    )
    experiment.create_analysis_recipe(
        name="green_batch",
        target_channel="green",
        segmentation={"tool": "target_objects"},
    )

    targets: list[str] = []

    def fake_load_file(path: str) -> dict[str, object]:
        layer_name = f"{Path(path).stem}_green"
        viewer.add_image(
            np.zeros((8, 8), dtype=np.float32),
            name=layer_name,
            metadata={"color": "green"},
        )
        return {"layer_names": [layer_name]}

    def fake_analyze_target_cells(target: str | None = None, **kwargs) -> dict[str, object]:
        assert target is not None
        targets.append(target)
        labels_layer = f"{target}_masks"
        table_name = f"{target}_measurements"
        viewer.add_labels(np.ones((8, 8), dtype=np.uint16), name=labels_layer)
        state.put_table(table_name, pd.DataFrame({"label": [1]}))
        return {
            "ok": True,
            "target_channel": target,
            "labels_layer": labels_layer,
            "preprocessed_layer": None,
            "table_name": table_name,
            "n_objects": 1,
            "object_unit": "object_or_roi",
            "segmentation_method": "target_objects",
            "analysis_dim": "2d",
            "warnings": [],
        }

    monkeypatch.setattr(files, "load_file", fake_load_file)
    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze_target_cells)

    res = workflows.run_recipe_on_samples(recipe_name="green_batch")

    assert res["n_complete"] == 2
    assert targets == ["ctrl_1_green", "trt_1_green"]
    assert [layer.name for layer in viewer.layers] == ["representative_green"]
    assert all(run["cleanup_removed_layers"] for run in res["runs"])


def test_run_recipe_on_samples_cleans_already_loaded_file_layers(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    p = tmp_path / "sample_1.tif"
    p.write_bytes(b"fake")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "sample_1", "group": "g", "file_ids": ["sample_1"]}]
    )
    experiment.create_analysis_recipe(
        name="green_batch",
        target_channel="green",
        segmentation={"tool": "target_objects"},
    )
    viewer.add_image(
        np.ones((8, 8), dtype=np.float32),
        name="sample_1_green",
        metadata={"source_path": str(p.resolve()), "color": "green"},
    )

    def fake_analyze_target_cells(target: str | None = None, **kwargs) -> dict[str, object]:
        labels_layer = f"{target}_labels"
        table_name = f"{target}_measurements"
        viewer.add_labels(np.ones((8, 8), dtype=np.uint16), name=labels_layer)
        state.put_table(table_name, pd.DataFrame({"label": [1]}))
        return {
            "ok": True,
            "target_channel": target,
            "labels_layer": labels_layer,
            "preprocessed_layer": None,
            "table_name": table_name,
            "n_objects": 1,
            "object_unit": "object_or_roi",
            "segmentation_method": "target_objects",
            "analysis_dim": "2d",
            "warnings": [],
        }

    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze_target_cells)

    res = workflows.run_recipe_on_samples(recipe_name="green_batch")

    assert res["n_complete"] == 1
    assert [layer.name for layer in viewer.layers] == []
    assert "sample_1_green" in res["runs"][0]["cleanup_removed_layers"]


def test_run_recipe_on_samples_projection_recipe_uses_projected_target(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import files, workflows

    p = tmp_path / "sample_1.tif"
    p.write_bytes(b"fake")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "sample_1", "group": "g", "file_ids": ["sample_1"]}]
    )
    experiment.create_analysis_recipe(
        name="projected_green",
        target_channel="green",
        segmentation={"tool": "target_objects", "do_3D": True},
        measurement={"projection": "mean", "axis": "z"},
    )

    def fake_load_file(path: str) -> dict[str, object]:
        viewer.add_image(
            np.ones((3, 8, 8), dtype=np.float32),
            name="sample_1_green",
            metadata={
                "source_path": str(Path(path).resolve()),
                "color": "green",
                "axes": "ZYX",
            },
        )
        return {"layer_names": ["sample_1_green"], "already_loaded": False}

    targets: list[str] = []

    def fake_analyze_target_cells(target: str | None = None, **kwargs) -> dict[str, object]:
        assert target is not None
        targets.append(target)
        assert kwargs["do_3D"] is False
        assert viewer.layers[target].data.shape == (8, 8)
        labels_layer = f"{target}_labels"
        table_name = f"{target}_measurements"
        viewer.add_labels(np.ones((8, 8), dtype=np.uint16), name=labels_layer)
        state.put_table(table_name, pd.DataFrame({"label": [1]}))
        return {
            "ok": True,
            "target_channel": target,
            "labels_layer": labels_layer,
            "preprocessed_layer": None,
            "table_name": table_name,
            "n_objects": 1,
            "object_unit": "object_or_roi",
            "segmentation_method": "target_objects",
            "analysis_dim": "2d",
            "warnings": [],
        }

    monkeypatch.setattr(files, "load_file", fake_load_file)
    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze_target_cells)

    res = workflows.run_recipe_on_samples(recipe_name="projected_green")

    assert res["n_complete"] == 1
    assert targets == ["sample_1_green_avg_z"]
    assert [layer.name for layer in viewer.layers] == []
    removed = res["runs"][0]["cleanup_removed_layers"]
    assert "sample_1_green" in removed
    assert "sample_1_green_avg_z" in removed


def test_run_recipe_on_samples_accepts_tool_name_method_and_projection_preprocess(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import files, workflows

    p = tmp_path / "sample_1.tif"
    p.write_bytes(b"fake")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "sample_1", "group": "g", "file_ids": ["sample_1"]}]
    )
    experiment.create_analysis_recipe(
        name="calexa_like_recipe",
        target_channel="Ch2-T1",
        preprocessing=[{"step": "average_projection", "axis": "z"}],
        segmentation={"method": "segment_intensity_regions", "threshold_method": "otsu"},
    )

    def fake_load_file(path: str) -> dict[str, object]:
        viewer.add_image(
            np.ones((3, 8, 8), dtype=np.float32),
            name="Ch2-T1",
            metadata={"source_path": str(Path(path).resolve()), "axes": "ZYX"},
        )
        return {"layer_names": ["Ch2-T1"], "already_loaded": False}

    calls: list[dict[str, object]] = []

    def fake_analyze_target_cells(target: str | None = None, **kwargs) -> dict[str, object]:
        calls.append({"target": target, **kwargs})
        assert target == "Ch2-T1_avg_z"
        assert kwargs["preprocess"] is None
        assert kwargs["segmentation_method"] == "segment_intensity_regions"
        assert kwargs["do_3D"] is False
        labels_layer = f"{target}_labels"
        table_name = f"{target}_measurements"
        viewer.add_labels(np.ones((8, 8), dtype=np.uint16), name=labels_layer)
        state.put_table(table_name, pd.DataFrame({"label": [1]}))
        return {
            "ok": True,
            "target_channel": target,
            "labels_layer": labels_layer,
            "preprocessed_layer": None,
            "table_name": table_name,
            "n_objects": 1,
            "object_unit": "object_or_roi",
            "segmentation_method": "intensity_regions",
            "analysis_dim": "2d",
            "warnings": [],
        }

    monkeypatch.setattr(files, "load_file", fake_load_file)
    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze_target_cells)

    res = workflows.run_recipe_on_samples(recipe_name="calexa_like_recipe")

    assert res["n_complete"] == 1
    assert calls


def test_run_recipe_on_samples_propagates_cancellation(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows
    from imajin.workers.qt_worker import CancelledError

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    p = tmp_path / "sample_1.tif"
    p.write_bytes(b"fake")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "sample_1", "group": "g", "file_ids": ["sample_1"]}]
    )
    experiment.create_analysis_recipe(
        name="cancel_recipe",
        target_channel="green",
        segmentation={"method": "intensity_regions"},
    )
    viewer.add_image(
        np.ones((8, 8), dtype=np.float32),
        name="sample_1_green",
        metadata={"source_path": str(p.resolve()), "color": "green"},
    )

    def fake_analyze_target_cells(*args, **kwargs) -> dict[str, object]:
        raise CancelledError("Tool execution cancelled by user.")

    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze_target_cells)

    with pytest.raises(CancelledError):
        workflows.run_recipe_on_samples(recipe_name="cancel_recipe")

    assert state.list_runs() == []
    bundles = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.endswith("_cancel_recipe")
    ]
    assert len(bundles) == 1
    meta = json.loads((bundles[0] / "metadata.json").read_text())
    assert meta["schema_version"] == 3
    assert meta["run_context"]["status"] == "cancelled"


def test_run_recipe_on_samples_no_samples_returns_empty() -> None:
    from imajin.tools import workflows

    state.put_recipe(name="r_empty", target_channel="green")
    res = workflows.run_recipe_on_samples(recipe_name="r_empty", sample_names=[])
    assert res["n_samples"] == 0
    assert res["runs"] == []
    assert res["bundle_path"] is None


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


# --- Task 12: generate_experiment_report -------------------------------------

def test_generate_experiment_report_md_includes_all_sections(
    tmp_path: Path, monkeypatch
) -> None:
    from imajin.tools import report

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control", "files": [str(p)]},
            {"sample_name": "trt_1", "group": "treatment"},
        ]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="green",
        segmentation={"tool": "cellpose_sam", "do_3D": True},
        measurement={"properties": ["area", "mean_intensity"]},
    )
    state.put_run(
        sample_id="ctrl_1",
        file_id="ctrl_1",
        recipe_id="r1",
        status="complete",
        summary={"n_objects": 42},
    )
    state.put_run(
        sample_id="trt_1",
        file_id="",
        recipe_id="r1",
        status="failed",
        error="no file registered",
    )

    log_path = tmp_path / "session.jsonl"
    log_path.write_text("", encoding="utf-8")
    from imajin.agent import provenance
    monkeypatch.setattr(provenance, "_LOG_PATH", log_path)

    out = tmp_path / "exp_report.md"
    res = report.generate_experiment_report(str(out), format="md")
    body = out.read_text(encoding="utf-8")

    assert "# Experiment Report" in body
    assert "## Overview" in body
    assert "## Sample Table" in body
    assert "ctrl_1" in body and "trt_1" in body
    assert "## Analysis Recipe" in body
    assert "cellpose_sam" in body
    assert "## Methods" in body
    assert "## Warnings" in body
    assert "trt_1" in body  # failed sample listed in warnings
    assert res["n_samples"] == 2
    assert res["n_failed"] == 1


def test_generate_experiment_report_html_writes_file(tmp_path, monkeypatch) -> None:
    from imajin.tools import report

    log_path = tmp_path / "session.jsonl"
    log_path.write_text("", encoding="utf-8")
    from imajin.agent import provenance
    monkeypatch.setattr(provenance, "_LOG_PATH", log_path)

    out = tmp_path / "exp_report.html"
    res = report.generate_experiment_report(str(out), format="html")
    body = out.read_text(encoding="utf-8")
    assert "<html>" in body
    assert res["format"] == "html"


def test_generate_experiment_report_includes_statistics_tables(tmp_path, monkeypatch) -> None:
    from imajin.tools import report, stats

    df = pd.DataFrame(
        {
            "sample_name": ["ctrl_1", "ctrl_2", "trt_1", "trt_2"],
            "group": ["control", "control", "treatment", "treatment"],
            "mean_intensity_reporter": [10.0, 11.0, 18.0, 19.0],
        }
    )
    table = state.put_table("measurements", df, spec={"tool": "test"})
    stats.describe_table(table, "mean_intensity_reporter", save_csv=False)
    stats.compare_groups(table, "mean_intensity_reporter", save_csv=False)

    log_path = tmp_path / "session.jsonl"
    log_path.write_text("", encoding="utf-8")
    from imajin.agent import provenance
    monkeypatch.setattr(provenance, "_LOG_PATH", log_path)

    out = tmp_path / "exp_report.md"
    report.generate_experiment_report(str(out), format="md")
    body = out.read_text(encoding="utf-8")
    assert "## Statistics" in body
    assert "mean_intensity_reporter" in body
    assert "p_value" in body


def test_create_analysis_recipe_passes_through_domain(viewer) -> None:
    from imajin.session import get_recipe, reset_recipes
    from imajin.tools.experiment import create_analysis_recipe

    reset_recipes()
    create_analysis_recipe(
        name="calexa_recipe",
        target_channel="green",
        cell_diameter_um=15.0,
        domain={"strategy": "noise_floor", "k_mad": 5.0},
    )

    rec = get_recipe("calexa_recipe")
    assert rec.cell_diameter_um == 15.0
    assert rec.domain == {"strategy": "noise_floor", "k_mad": 5.0}


# --- Validation: segmentation method ----------------------------------------

def test_create_analysis_recipe_rejects_expression_domain_in_segmentation_slot() -> None:
    """`expression_domain` belongs in `domain`, not `segmentation`."""
    with pytest.raises(ValueError, match="segmentation"):
        experiment.create_analysis_recipe(
            name="bad",
            target_channel="green",
            segmentation={"method": "expression_domain", "k_mad": 5.0},
        )


def test_create_analysis_recipe_rejects_invalid_method_in_tool_key() -> None:
    with pytest.raises(ValueError, match="segmentation"):
        experiment.create_analysis_recipe(
            name="bad",
            target_channel="green",
            segmentation={"tool": "noise_floor"},
        )


@pytest.mark.parametrize(
    "method",
    [
        "cellpose_sam",
        "cellpose",
        "target_objects",
        "intensity_regions",
        "segment_intensity_regions",
    ],
)
def test_create_analysis_recipe_accepts_known_methods_and_aliases(method: str) -> None:
    state.reset_recipes()
    experiment.create_analysis_recipe(
        name="r",
        target_channel="green",
        segmentation={"method": method},
    )


def test_create_analysis_recipe_rejects_unknown_domain_strategy() -> None:
    with pytest.raises(ValueError, match="domain"):
        experiment.create_analysis_recipe(
            name="bad_domain",
            target_channel="green",
            segmentation={"method": "intensity_regions"},
            domain={"strategy": "kmeans"},
        )


# --- Runner forwards recipe.domain to analyze_target_cells ------------------

def _setup_single_sample_with_layer(viewer, tmp_path: Path, sample_name: str) -> Path:
    layer = f"{sample_name}_ch0"
    viewer.add_image(
        np.zeros((20, 20), dtype=np.float32), name=layer, scale=(0.5, 0.5)
    )
    state.put_channel_annotation(layer, role="target", color="green")
    p = tmp_path / f"{sample_name}.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [
            {
                "sample_name": sample_name,
                "group": "control",
                "files": [str(p)],
                "layers": [layer],
            }
        ]
    )
    return p


def _stub_analyze_target_cells(monkeypatch, captured: dict[str, object]) -> None:
    from imajin.tools import workflows

    def fake_analyze(target=None, **kwargs):
        captured["target"] = target
        captured.update(kwargs)
        return {
            "ok": True,
            "target_channel": target,
            "labels_layer": None,
            "preprocessed_layer": None,
            "table_name": None,
            "n_objects": 0,
            "object_unit": "cells",
            "segmentation_method": kwargs.get("segmentation_method"),
            "analysis_dim": "2d",
            "warnings": [],
        }

    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze)


def test_run_recipe_on_samples_forwards_recipe_domain_to_two_tier(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    _setup_single_sample_with_layer(viewer, tmp_path, "ctrl_1")
    experiment.create_analysis_recipe(
        name="two_tier",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area", "mean_intensity"]},
        domain={"strategy": "noise_floor", "k_mad": 5.0, "min_area_um2": 1.0},
    )

    captured: dict[str, object] = {}
    _stub_analyze_target_cells(monkeypatch, captured)

    res = workflows.run_recipe_on_samples(recipe_name="two_tier")

    assert res["n_complete"] == 1
    assert captured.get("domain_strategy") == "noise_floor"
    assert captured.get("domain_options") == {"k_mad": 5.0, "min_area_um2": 1.0}
    assert captured.get("segmentation_method") == "intensity_regions"


def test_run_recipe_on_samples_accepts_expression_domain_alias_in_recipe_domain(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    """`recipe.domain.method = 'expression_domain'` translates to the noise_floor strategy."""
    from imajin.tools import workflows

    _setup_single_sample_with_layer(viewer, tmp_path, "ctrl_1")
    experiment.create_analysis_recipe(
        name="two_tier_alias",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
        domain={"method": "expression_domain", "k_mad": 5.0},
    )

    captured: dict[str, object] = {}
    _stub_analyze_target_cells(monkeypatch, captured)

    workflows.run_recipe_on_samples(recipe_name="two_tier_alias")

    assert captured.get("domain_strategy") == "noise_floor"
    assert captured.get("domain_options") == {"k_mad": 5.0}


def test_run_recipe_on_samples_no_domain_stays_single_tier(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    _setup_single_sample_with_layer(viewer, tmp_path, "ctrl_1")
    experiment.create_analysis_recipe(
        name="single_tier",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    captured: dict[str, object] = {}
    _stub_analyze_target_cells(monkeypatch, captured)

    workflows.run_recipe_on_samples(recipe_name="single_tier")

    assert captured.get("domain_strategy") is None
    assert captured.get("domain_options") is None


def test_run_recipe_on_samples_creates_parent_bundle(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    import re
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "ctrl_1", "group": "control",
          "files": [str(p)], "layers": ["ctrl_1_ch0"]}]
    )
    experiment.create_analysis_recipe(
        name="r1",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area", "mean_intensity"]},
    )

    res = workflows.run_recipe_on_samples(recipe_name="r1")

    assert res.get("bundle_path"), "run_recipe_on_samples must return bundle_path"
    bundle = Path(res["bundle_path"])
    assert bundle.is_dir()
    assert re.match(r"^\d{8}_\d{6}_r1$", bundle.name), bundle.name
    # Sample output landed in the parent bundle, not a sibling per-call bundle
    sibling_bundles = [
        p
        for p in sorted(bundle.parent.iterdir())
        if p.is_dir() and re.match(r"^\d{8}_\d{6}_", p.name)
    ]
    assert sibling_bundles == [bundle], [s.name for s in sibling_bundles]


def test_batch_parent_bundle_lands_directly_in_anchor_folder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from imajin.tools.batch_runner import BatchRecipeRunner

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "fallback"))

    folder_a = tmp_path / "2026-05-09"
    folder_b = tmp_path / "2026-05-10"
    folder_a.mkdir()
    folder_b.mkdir()
    file_a = folder_a / "a.lsm"
    file_b = folder_b / "b.lsm"
    file_a.write_bytes(b"")
    file_b.write_bytes(b"")

    experiment.register_files([str(file_b), str(file_a)])
    experiment.annotate_samples(
        [
            {"sample_name": "b", "group": "treated", "files": [str(file_b)]},
            {"sample_name": "a", "group": "control", "files": [str(file_a)]},
        ]
    )
    experiment.create_analysis_recipe(
        name="anchor_recipe",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    runner = BatchRecipeRunner("anchor_recipe", sample_names=["b", "a"])
    runner.recipe = state.get_recipe("anchor_recipe")
    runner.names = ["b", "a"]
    runner.domain_strategy = None

    bundle = runner._create_parent_bundle()

    assert bundle.parent == folder_a.resolve()
    assert bundle.name.endswith("_anchor_recipe")


def test_run_recipe_on_samples_two_tier_attaches_sample_cols_to_tier_table(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "ctrl_1", "group": "control",
          "files": [str(p)], "layers": ["ctrl_1_ch0"]}]
    )
    experiment.create_analysis_recipe(
        name="two_tier_r",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
        domain={"strategy": "noise_floor", "k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )

    res = workflows.run_recipe_on_samples(recipe_name="two_tier_r")
    assert res["n_complete"] == 1
    run = res["runs"][0]
    table_names = run.get("table_names") or []
    assert table_names
    has_sample_attached = False
    for tname in table_names:
        df = state.get_table(tname)
        if "tier" in df.columns and "sample_name" in df.columns:
            has_sample_attached = True
            assert (df["sample_name"] == "ctrl_1").all()
    assert has_sample_attached, f"tier table missing sample columns; tables={table_names}"


def test_run_recipe_on_samples_writes_combined_csv(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    viewer.add_image(img.copy(), name="trt_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])
    experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control",
             "files": [str(a)], "layers": ["ctrl_1_ch0"]},
            {"sample_name": "trt_1", "group": "treatment",
             "files": [str(b)], "layers": ["trt_1_ch0"]},
        ]
    )
    experiment.create_analysis_recipe(
        name="r_combined",
        target_channel="ctrl_1_ch0",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area", "mean_intensity"]},
    )

    res = workflows.run_recipe_on_samples(recipe_name="r_combined")
    bundle = Path(res["bundle_path"])
    combined = bundle / "tables" / "combined.csv"
    assert combined.exists()

    df = pd.read_csv(combined)
    assert {"sample_name", "group", "file_id"}.issubset(df.columns)
    assert set(df["sample_name"].unique()) == {"ctrl_1", "trt_1"}
    stats_files = list((bundle / "stats").glob("*.csv"))
    assert any("mean_intensity" in p.name for p in stats_files)
    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["run_context"]["statistics_outputs"]


def test_run_recipe_on_samples_finalizes_metadata_with_samples(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    p = tmp_path / "ctrl_1.lsm"
    p.write_bytes(b"")
    experiment.register_files([str(p)])
    experiment.annotate_samples(
        [{"sample_name": "ctrl_1", "group": "control",
          "files": [str(p)], "layers": ["ctrl_1_ch0"]}]
    )
    experiment.create_analysis_recipe(
        name="r_meta",
        target_channel="green",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    res = workflows.run_recipe_on_samples(recipe_name="r_meta")
    bundle = Path(res["bundle_path"])
    meta = json.loads((bundle / "metadata.json").read_text())
    run_context = meta["run_context"]

    assert meta["schema_version"] == 3
    assert run_context["kind"] == "batch"
    assert run_context["tier"] == "single_tier"
    assert run_context["status"] == "complete"
    assert run_context["n_samples"] == 1
    assert run_context["n_complete"] == 1
    assert run_context["n_failed"] == 0
    assert len(run_context["samples"]) == 1
    s = run_context["samples"][0]
    assert s["sample_name"] == "ctrl_1"
    assert s["group"] == "control"
    assert s["file_id"] == "ctrl_1"
    assert s["status"] == "complete"
    assert "outputs" not in s


def test_run_recipe_on_samples_records_folder_set_and_channel_roles(
    viewer,
    monkeypatch,
    tmp_path: Path,
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "fallback"))

    folder_a = tmp_path / "2026-05-09"
    folder_b = tmp_path / "2026-05-10"
    folder_a.mkdir()
    folder_b.mkdir()
    file_a = folder_a / "ctrl_1.lsm"
    file_b = folder_b / "trt_1.lsm"
    file_a.write_bytes(b"")
    file_b.write_bytes(b"")

    viewer.add_image(np.ones((8, 8), dtype=np.float32), name="ctrl_layer")
    viewer.add_image(np.ones((8, 8), dtype=np.float32), name="trt_layer")
    state.put_channel_annotation("ctrl_layer", role="target", color="green")
    state.put_channel_annotation("trt_layer", role="counterstain", color="uv")

    experiment.register_files([str(file_b), str(file_a)])
    experiment.annotate_samples(
        [
            {
                "sample_name": "ctrl_1",
                "group": "control",
                "files": [str(file_a)],
                "layers": ["ctrl_layer"],
            },
            {
                "sample_name": "trt_1",
                "group": "treated",
                "files": [str(file_b)],
                "layers": ["trt_layer"],
            },
        ]
    )
    experiment.create_analysis_recipe(
        name="r_context",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    def fake_analyze_target_cells(*args, **kwargs):
        return {
            "ok": True,
            "target_channel": kwargs.get("target"),
            "segmentation_method": "intensity_regions",
            "analysis_dim": "2d",
            "n_objects": 1,
            "n_cells": 1,
            "result_files": {},
            "warnings": [],
        }

    monkeypatch.setattr(workflows, "analyze_target_cells", fake_analyze_target_cells)

    res = workflows.run_recipe_on_samples(recipe_name="r_context")
    meta = json.loads((Path(res["bundle_path"]) / "metadata.json").read_text())
    run_context = meta["run_context"]

    assert run_context["folder_set"] == [
        str(folder_a.resolve()),
        str(folder_b.resolve()),
    ]
    assert run_context["channel_roles"] == {
        "ctrl_layer": "target",
        "trt_layer": "counterstain",
    }
    assert run_context["scope_filters"] == []


def test_run_recipe_on_samples_metadata_records_failed_sample(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
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
            {"sample_name": "ctrl_1", "group": "control",
             "files": [str(a)], "layers": ["ctrl_1_ch0"]},
            {"sample_name": "trt_1", "group": "treatment",
             "files": [str(b)], "layers": ["trt_1_ch0"]},
        ]
    )
    experiment.create_analysis_recipe(
        name="r_fail",
        target_channel="ctrl_1_ch0",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    # Force the second sample to fail by stubbing analyze_target_cells; the
    # recipe locks target to ctrl_1_ch0 for both samples, so we pivot on the
    # active sample slug (set by run_recipe_on_samples per sample).
    from imajin.tools import workflows as _wf
    from imajin.tools.results import current_sample_slug
    real_analyze = _wf.analyze_target_cells

    def _flaky_analyze(*args, **kwargs):
        if current_sample_slug() == "trt_1":
            raise RuntimeError("synthetic failure for trt_1")
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(_wf, "analyze_target_cells", _flaky_analyze)

    res = workflows.run_recipe_on_samples(
        recipe_name="r_fail", sample_names=["ctrl_1", "trt_1"]
    )
    bundle = Path(res["bundle_path"])
    meta = json.loads((bundle / "metadata.json").read_text())
    run_context = meta["run_context"]
    statuses = sorted(s["status"] for s in run_context["samples"])
    assert statuses == ["complete", "failed"]
    assert run_context["n_complete"] == 1
    assert run_context["n_failed"] == 1
    failed = next(s for s in run_context["samples"] if s["status"] == "failed")
    assert failed["error"]
    # No labels file written for the failed sample
    failed_slug = failed["sample_name"]
    assert not (bundle / "labels" / "cells" / f"{failed_slug}.tif").exists()


def test_run_recipe_on_samples_cancellation_finalizes_metadata(
    viewer, monkeypatch, tmp_path: Path
) -> None:
    from imajin.tools import workflows
    from imajin.workers.qt_worker import CancelledError

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))

    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    viewer.add_image(img, name="ctrl_1_ch0", scale=(0.5, 0.5))
    viewer.add_image(img.copy(), name="trt_1_ch0", scale=(0.5, 0.5))
    state.put_channel_annotation("ctrl_1_ch0", role="target", color="green")

    a = tmp_path / "ctrl_1.lsm"
    b = tmp_path / "trt_1.lsm"
    a.write_bytes(b"")
    b.write_bytes(b"")
    experiment.register_files([str(a), str(b)])
    experiment.annotate_samples(
        [
            {"sample_name": "ctrl_1", "group": "control",
             "files": [str(a)], "layers": ["ctrl_1_ch0"]},
            {"sample_name": "trt_1", "group": "treatment",
             "files": [str(b)], "layers": ["trt_1_ch0"]},
        ]
    )
    experiment.create_analysis_recipe(
        name="r_cancel",
        target_channel="ctrl_1_ch0",
        segmentation={"tool": "intensity_regions"},
        measurement={"properties": ["area"]},
    )

    call = {"n": 0}
    real = workflows.analyze_target_cells

    def side_effect(*args, **kwargs):
        call["n"] += 1
        if call["n"] == 1:
            return real(*args, **kwargs)
        raise CancelledError("Tool execution cancelled by user.")

    monkeypatch.setattr(workflows, "analyze_target_cells", side_effect)

    with pytest.raises(CancelledError):
        workflows.run_recipe_on_samples(
            recipe_name="r_cancel", sample_names=["ctrl_1", "trt_1"]
        )

    bundles = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.endswith("_r_cancel")
    ]
    assert len(bundles) == 1
    bundle = bundles[0]
    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["schema_version"] == 3
    assert meta["run_context"]["status"] == "cancelled"
    statuses = [s["status"] for s in meta["run_context"]["samples"]]
    assert statuses == ["complete", "skipped"]
