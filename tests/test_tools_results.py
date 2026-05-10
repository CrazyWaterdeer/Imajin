from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from imajin.agent import state
from imajin.tools import results


def test_save_labels_writes_tiff_to_results_root(viewer, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:8, 3:9] = 1
    viewer.add_labels(labels, name="target_objects")

    res = results.save_labels("target_objects")

    out = tmp_path / "results" / "labels" / "target_objects.tif"
    assert res["path"] == str(out)
    assert out.exists()
    saved = tifffile.imread(out)
    np.testing.assert_array_equal(saved, labels.astype(np.uint16))
    assert (tmp_path / "results" / "manifest.jsonl").exists()


def test_save_result_bundle_collects_labels_tables_and_qc(
    viewer, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    viewer.add_labels(labels, name="objects")
    state.put_table(
        "measurements",
        pd.DataFrame({"label": [1], "mean_intensity": [42.0]}),
        spec={"tool": "measure_intensity", "labels_layer": "objects"},
    )
    qc = tmp_path / "qc.png"
    qc.write_bytes(b"fake-png")

    res = results.save_result_bundle(
        name="sample analysis",
        labels_layers=["objects"],
        table_names=["measurements"],
        qc_png_paths=[str(qc)],
        metadata={"sample": "sample_1"},
    )

    bundle = tmp_path / "results" / "bundles"
    assert res["bundle_path"].startswith(str(bundle))
    metadata = json.loads((Path(res["bundle_path"]) / "metadata.json").read_text())
    assert metadata["sample"] == "sample_1"
    outputs = res["outputs"]
    assert len(outputs["labels"]) == 1
    assert len(outputs["tables"]) == 1
    assert len(outputs["qc"]) == 1
    assert tifffile.imread(outputs["labels"][0]).max() == 1


def test_results_root_uses_session_anchor_when_no_project(tmp_path, monkeypatch):
    from imajin import results as _results

    folder = tmp_path / "2026-05-11"
    folder.mkdir()
    fake_file = folder / "img.lsm"
    fake_file.write_bytes(b"")

    monkeypatch.setattr(
        "imajin.agent.state.list_files",
        lambda: [{"path": str(fake_file)}],
    )
    assert _results.results_root() == folder.absolute()


def test_results_root_falls_back_to_user_root_when_no_anchor(tmp_path, monkeypatch):
    from imajin import results as _results

    monkeypatch.setattr("imajin.agent.state.list_files", lambda: [])
    monkeypatch.setattr(_results, "user_results_root", lambda: tmp_path / "user_root")
    assert _results.results_root() == tmp_path / "user_root"


def test_create_result_bundle_uses_explicit_root(tmp_path):
    from imajin.results import create_result_bundle

    bundle = create_result_bundle("demo", root=tmp_path)
    assert bundle.parent == tmp_path
    assert bundle.name.endswith("_demo")
    assert (bundle / "metadata.json").exists()
    # The standard layout is created
    for sub in ("labels/cells", "labels/domain", "tables", "qc", "stats", "figures"):
        assert (bundle / sub).is_dir()


def test_record_result_keeps_manifest_out_of_anchor_folder(tmp_path, monkeypatch):
    """`manifest.jsonl` must never land in the raw-data anchor folder."""
    from imajin import results as _results

    anchor = tmp_path / "2026-05-11"
    anchor.mkdir()
    fake_file = anchor / "img.lsm"
    fake_file.write_bytes(b"")
    user_root = tmp_path / "user_root"

    monkeypatch.setattr(
        "imajin.agent.state.list_files",
        lambda: [{"path": str(fake_file)}],
    )
    monkeypatch.setattr(_results, "user_results_root", lambda: user_root)

    # Sanity: results_root would point at the anchor folder
    assert _results.results_root() == anchor.absolute()

    # But record_result must drop the manifest at user_root, NOT at anchor
    _results.record_result("test_kind", fake_file)
    assert (user_root / "manifest.jsonl").exists()
    assert not (anchor / "manifest.jsonl").exists()
