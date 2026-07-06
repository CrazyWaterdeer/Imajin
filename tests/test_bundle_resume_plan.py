from __future__ import annotations

from imajin.result_bundles import record_sample_index_entry
from imajin.results import create_result_bundle, read_bundle_metadata, write_bundle_metadata
from imajin.tools.bundle_resume import plan_resume, read_result_bundle


def _make_bundle(data_dir, analysed, recipe_params=None):
    """A batch-style bundle under ``data_dir`` with the given analysed files recorded."""
    bundle = create_result_bundle(name="run", kind="batch", root=data_dir)
    if recipe_params:
        meta = read_bundle_metadata(bundle)
        meta["recipe_params"] = recipe_params
        write_bundle_metadata(bundle, meta)
    for name in analysed:
        (data_dir / name).write_bytes(b"x")
        record_sample_index_entry(
            bundle,
            source_file=str(data_dir / name),
            anchor=str(data_dir),
            method="target_objects",
            mode="single",
            status="complete",
        )
    return bundle


def test_plan_resume_diffs_analysed_and_pending(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_bundle(data, ["a.lsm", "b.lsm"], recipe_params={"target_channel": "green"})
    (data / "c.lsm").write_bytes(b"x")  # on disk, not analysed → pending

    plan = plan_resume(str(data))
    assert plan["status"] == "ok"
    assert plan["analysed"] == ["a.lsm", "b.lsm"]
    assert plan["pending"] == ["c.lsm"]
    assert plan["missing"] == []
    assert plan["recipe_params"]["target_channel"] == "green"  # recipe recovered


def test_plan_resume_reports_missing_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_bundle(data, ["a.lsm", "gone.lsm"])
    (data / "gone.lsm").unlink()  # analysed in bundle but no longer on disk

    plan = plan_resume(str(data))
    assert plan["missing"] == ["gone.lsm"]
    assert plan["pending"] == []


def test_plan_resume_no_bundle(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.lsm").write_bytes(b"x")
    assert plan_resume(str(data))["status"] == "no_bundle"


def test_plan_resume_multiple_bundles_requires_choice(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_bundle(data, ["a.lsm"])
    _make_bundle(data, ["b.lsm"])
    plan = plan_resume(str(data))
    assert plan["status"] == "multiple_bundles"
    assert len(plan["bundles"]) == 2


def test_read_result_bundle_recovers_recipe_and_keys(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    bundle = _make_bundle(
        data, ["a.lsm"], recipe_params={"segmentation": {"method": "target_objects"}}
    )
    info = read_result_bundle(str(bundle))
    assert info["analysed_keys"] == ["a.lsm"]
    assert info["recipe_params"]["segmentation"]["method"] == "target_objects"
    assert info["legacy_inferred"] is False
