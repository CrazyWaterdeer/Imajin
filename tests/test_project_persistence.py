from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def reset_project_state():
    from imajin.agent import state
    from imajin.agent.execution import get_execution_service

    state.reset_tables()
    get_execution_service().clear_jobs()
    yield
    state.reset_tables()
    get_execution_service().clear_jobs()


def test_create_project_writes_initial_layout(tmp_path: Path) -> None:
    from imajin.project import create_project

    root = tmp_path / "experiment.imajin"
    res = create_project(root, name="experiment")

    assert res["name"] == "experiment"
    assert (root / "project.json").exists()
    assert (root / "files.json").exists()
    assert (root / "samples.json").exists()
    assert (root / "channels.json").exists()
    assert (root / "recipes.json").exists()
    assert (root / "runs.json").exists()
    assert (root / "qc.json").exists()
    assert (root / "jobs.json").exists()
    assert (root / "tables" / "index.json").exists()
    assert (root / "provenance").is_dir()

    payload = json.loads((root / "project.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["project_id"].startswith("proj_")


def test_save_and_load_project_round_trip(tmp_path: Path, viewer) -> None:
    from imajin.agent import provenance, state
    from imajin.agent.execution import get_execution_service
    from imajin.config import Settings
    from imajin.project import create_project, load_project, save_project

    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"fake image bytes")

    viewer.add_image([[1, 2], [3, 4]], name="green", scale=(0.5, 0.5))
    file_id = state.put_file(
        path=str(raw),
        original_name=raw.name,
        file_type="tif",
        metadata_summary={"axes": "YX"},
    )
    state.put_sample(
        sample_name="control 1",
        group="control",
        layers=["green"],
        file_ids=[file_id],
        extra={"tissue": "midgut"},
    )
    state.put_channel_annotation(
        "green",
        role="target",
        color="green",
        marker="GCaMP",
        biological_target="gut cells",
    )
    state.put_recipe(
        name="cell recipe",
        target_channel="green",
        segmentation={"do_3D": False, "diameter": 20},
    )
    state.put_run(
        sample_id="control 1",
        file_id=file_id,
        recipe_id="cell recipe",
        status="complete",
        table_names=["measurements"],
        summary={"n_objects": 2},
        run_id="run_0007",
    )
    state.put_qc_record(
        "measurements",
        status="pass",
        warnings=[],
        metrics={"n_rows": 2},
        reviewed_by_user=True,
        notes="checked",
    )
    state.set_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_id": ["control 1", "control 1"],
                "sample_name": ["control 1", "control 1"],
                "group": ["control", "control"],
                "file_id": [file_id, file_id],
                "mean_intensity": [1.0, 2.0],
            }
        ),
        spec={"op": "measure_intensity"},
    )
    provenance.start_session(
        driver="manual",
        settings=Settings(data_dir=tmp_path / "appdata"),
    )
    provenance.record_call("dummy", {}, {"ok": True}, 0.01, ok=True)

    service = get_execution_service()
    service.submit_workflow("already_done", lambda: "ok", wait=True)

    root = tmp_path / "experiment.imajin"
    create_project(root, name="roundtrip")
    saved = save_project()

    assert saved["n_files"] == 1
    assert saved["n_samples"] == 1
    assert saved["n_channels"] == 1
    assert saved["n_recipes"] == 1
    assert saved["n_runs"] == 1
    assert saved["n_qc_records"] == 1
    assert saved["n_tables"] == 1
    assert saved["n_jobs"] == 1
    assert len(list((root / "provenance").glob("*.jsonl"))) == 1

    state.reset_files()
    state.reset_samples()
    state.reset_channel_annotations()
    state.reset_recipes()
    state.reset_runs()
    state.reset_qc_records()
    state.reset_tables()
    service.clear_jobs()

    loaded = load_project(root)

    assert loaded["warnings"] == []
    assert state.list_files()[0]["file_id"] == file_id
    assert state.list_samples()[0]["group"] == "control"
    assert state.list_channel_annotations()[0]["marker"] == "GCaMP"
    assert state.list_recipes()[0]["name"] == "cell recipe"
    assert state.list_runs()[0]["run_id"] == "run_0007"
    assert state.get_qc_record("measurements").notes == "checked"
    restored = state.get_table("measurements")
    assert list(restored["mean_intensity"]) == [1.0, 2.0]
    assert service.list_jobs()[0].workflow_name == "already_done"


def test_load_project_marks_missing_file(tmp_path: Path) -> None:
    from imajin.agent import state
    from imajin.project import create_project, load_project, save_project

    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"fake image bytes")
    file_id = state.put_file(path=str(raw), original_name=raw.name, file_type="tif")

    root = tmp_path / "missing.imajin"
    create_project(root)
    save_project()
    raw.unlink()

    state.reset_files()
    loaded = load_project(root)

    assert loaded["warnings"]
    [record] = state.list_files()
    assert record["file_id"] == file_id
    assert record["load_status"] == "missing"


def test_relink_file_updates_missing_record(tmp_path: Path) -> None:
    from imajin.agent import state
    from imajin.project import create_project, relink_file

    old = tmp_path / "old.tif"
    new = tmp_path / "new.tif"
    new.write_bytes(b"new image bytes")
    file_id = state.put_file(path=str(old), original_name=old.name, file_type="tif")

    create_project(tmp_path / "relink.imajin")
    result = relink_file(file_id, new)

    assert result["exists"] is True
    assert state.list_files()[0]["path"] == str(new.resolve())
    assert state.list_files()[0]["load_status"] == "unloaded"


def test_future_schema_version_fails_clearly(tmp_path: Path) -> None:
    from imajin.project import create_project, load_project

    root = tmp_path / "future.imajin"
    create_project(root)
    payload = json.loads((root / "project.json").read_text())
    payload["schema_version"] = 999
    (root / "project.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="newer than supported"):
        load_project(root)


def test_project_tools_are_registered(tmp_path: Path) -> None:
    from imajin.tools import call_tool

    root = tmp_path / "tool.imajin"
    created = call_tool("create_project", path=str(root), name="via tool")
    saved = call_tool("save_project")
    status = call_tool("project_status")
    loaded = call_tool("load_project", path=str(root))

    assert created["name"] == "via tool"
    assert saved["path"] == str(root.resolve())
    assert status["open"] is True
    assert loaded["name"] == "via tool"


def test_project_tool_name_argument_works_through_execution_service(
    qapp, tmp_path: Path
) -> None:
    from imajin.agent.execution import ToolExecutionService
    from imajin.agent.qt_tool_runner import MainThreadToolRunner

    # Regression: several dispatch layers have their own "name" parameter.
    # Tool inputs named "name" must still reach create_project.
    runner = MainThreadToolRunner()
    service = ToolExecutionService()
    root = tmp_path / "service.imajin"

    result = service.call_tool_blocking(
        "create_project",
        {"path": str(root), "name": "via service"},
        source="manual",
        tool_caller=runner.call,
    )

    assert result["name"] == "via service"


def test_project_autosaves_session_annotations(tmp_path: Path) -> None:
    from imajin.agent import state
    from imajin.project import create_project, project_status

    root = tmp_path / "autosave.imajin"
    create_project(root)

    state.put_sample("control 1", group="control", extra={"tissue": "gut"})
    state.set_table(
        "measurements",
        pd.DataFrame({"sample_name": ["control 1"], "mean_intensity": [12.5]}),
        spec={"op": "measure_intensity"},
    )

    samples = json.loads((root / "samples.json").read_text())
    tables = json.loads((root / "tables" / "index.json").read_text())
    status = project_status()

    assert samples[0]["sample_name"] == "control 1"
    assert tables[0]["name"] == "measurements"
    assert (root / tables[0]["path"]).exists()
    assert status["n_samples"] == 1
    assert status["n_tables"] == 1
    assert status["last_autosave"]["ok"] is True


def test_autosave_is_suspended_during_load(tmp_path: Path) -> None:
    from imajin.agent import state
    from imajin.project import create_project, load_project

    root = tmp_path / "load-suspend.imajin"
    create_project(root)
    state.put_sample("before", group="control")

    original_project_json = (root / "project.json").read_text()
    loaded = load_project(root)

    assert loaded["n_samples"] == 1
    assert (root / "project.json").read_text() == original_project_json
