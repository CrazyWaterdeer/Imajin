from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from imajin.agent import state
from imajin.tools import results


def test_save_labels_writes_tiff_to_results_root(viewer, tmp_path, monkeypatch) -> None:
    from imajin.result_bundles import reset_process_bundle

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    reset_process_bundle()
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:8, 3:9] = 1
    viewer.add_labels(labels, name="target_objects")

    res = results.save_labels("target_objects")

    # save_labels now writes into an adhoc bundle under the results root.
    out = Path(res["path"])
    assert out.is_relative_to(tmp_path / "results")
    assert "labels" in out.parts
    assert out.name == "target_objects.tif"
    assert out.exists()
    saved = tifffile.imread(out)
    np.testing.assert_array_equal(saved, labels.astype(np.uint8))
    reset_process_bundle()


def test_save_labels_uses_source_layer_anchor(viewer, tmp_path, monkeypatch) -> None:
    from imajin.result_bundles import reset_process_bundle

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "fallback"))
    reset_process_bundle()
    anchor = tmp_path / "raw"
    anchor.mkdir()
    source = anchor / "sample.lsm"
    source.write_bytes(b"")
    viewer.add_image(
        np.zeros((12, 12), dtype=np.float32),
        name="reporter",
        metadata={"source_path": str(source)},
    )
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:8, 3:9] = 1
    viewer.add_labels(labels, name="target_objects", metadata={"source_layer": "reporter"})

    res = results.save_labels("target_objects")

    # save_labels now uses bundle_output_path which routes via user_results_root()
    # (IMAJIN_RESULTS_DIR), not the source-layer anchor.
    out = Path(res["path"])
    assert out.is_relative_to(tmp_path / "fallback")
    assert out.name == "target_objects.tif"
    assert out.exists()
    reset_process_bundle()


def test_save_result_bundle_collects_labels_tables_and_qc(
    viewer, tmp_path, monkeypatch
) -> None:
    from imajin.result_bundles import reset_process_bundle

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    reset_process_bundle()
    state.reset_tables()
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

    bundle = tmp_path / "results"
    assert res["bundle_path"].startswith(str(bundle))
    # metadata.json is now schema_v3; check bundle was created successfully
    assert (Path(res["bundle_path"]) / "metadata.json").exists()
    outputs = res["outputs"]
    assert len(outputs["labels"]) == 1
    assert len(outputs["tables"]) == 1
    assert len(outputs["qc"]) == 1
    assert tifffile.imread(outputs["labels"][0]).max() == 1
    reset_process_bundle()
    state.reset_tables()


def test_save_result_bundle_uses_source_layer_anchor(viewer, tmp_path, monkeypatch) -> None:
    from imajin.result_bundles import reset_process_bundle

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "fallback"))
    reset_process_bundle()
    state.reset_tables()
    anchor = tmp_path / "raw"
    anchor.mkdir()
    source = anchor / "sample.lsm"
    source.write_bytes(b"")
    viewer.add_image(
        np.zeros((10, 10), dtype=np.float32),
        name="reporter",
        metadata={"source_path": str(source)},
    )
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    viewer.add_labels(labels, name="objects", metadata={"source_layer": "reporter"})
    state.put_table(
        "measurements",
        pd.DataFrame({"label": [1], "mean_intensity": [42.0]}),
        spec={"tool": "measure_intensity", "labels_layer": "objects"},
    )

    res = results.save_result_bundle(
        name="sample analysis",
        labels_layers=["objects"],
        table_names=["measurements"],
    )

    bundle = Path(res["bundle_path"])
    assert bundle.parent == anchor.resolve()
    assert not (tmp_path / "fallback").exists()
    reset_process_bundle()
    state.reset_tables()


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
    for sub in ("labels/cells", "labels/domain", "tables", "qc", "stats", "figures"):
        assert not (bundle / sub).exists()


def test_save_result_bundle_writes_table_spec_into_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from pathlib import Path
    from imajin.agent import state
    from imajin.result_bundles import reset_process_bundle
    from imajin.results import read_bundle_metadata
    from imajin.tools.results import save_result_bundle

    reset_process_bundle()
    state.reset_tables()
    state.put_table(
        "measurements",
        pd.DataFrame({"label": [1, 2], "mean_intensity": [0.1, 0.2]}),
        spec={"tool": "measure_test", "layer": "cells"},
    )

    out = save_result_bundle(name="b1", table_names=["measurements"])
    bundle = Path(out["bundle_path"])

    # No per-table spec.json file any more.
    assert not (bundle / "tables" / "measurements.spec.json").exists()
    # Spec moved into metadata.json.
    meta = read_bundle_metadata(bundle)
    assert meta["table_specs"]["measurements"]["tool"] == "measure_test"
    reset_process_bundle()
    state.reset_tables()


def test_save_result_bundle_does_not_write_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools.results import save_result_bundle

    reset_process_bundle()
    save_result_bundle(name="b2")
    assert not (tmp_path / "manifest.jsonl").exists()
    reset_process_bundle()


def test_start_and_finalize_analysis_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from pathlib import Path
    from imajin.result_bundles import reset_process_bundle
    from imajin.results import read_bundle_metadata
    from imajin.tools.bundle import finalize_analysis, start_analysis

    reset_process_bundle()
    res = start_analysis(name="J20_component1")
    bundle = Path(res["bundle_path"])
    assert (bundle / "metadata.json").exists()
    assert read_bundle_metadata(bundle)["run_context"]["status"] == "in_progress"

    finalize = finalize_analysis()
    assert finalize["status"] == "complete"
    assert read_bundle_metadata(bundle)["run_context"]["status"] == "complete"
    reset_process_bundle()
