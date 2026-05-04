from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from imajin.agent import state
from imajin.project import create_project
from imajin.tools import results


def test_save_labels_writes_tiff_to_project_reports(viewer, tmp_path) -> None:
    create_project(tmp_path / "project")
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:8, 3:9] = 1
    viewer.add_labels(labels, name="target_objects")

    res = results.save_labels("target_objects")

    out = tmp_path / "project" / "reports" / "labels" / "target_objects.tif"
    assert res["path"] == str(out)
    assert out.exists()
    saved = tifffile.imread(out)
    np.testing.assert_array_equal(saved, labels.astype(np.uint16))
    assert (tmp_path / "project" / "reports" / "manifest.jsonl").exists()


def test_save_result_bundle_collects_labels_tables_and_qc(viewer, tmp_path) -> None:
    create_project(tmp_path / "project")
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

    bundle = tmp_path / "project" / "reports" / "bundles"
    assert res["bundle_path"].startswith(str(bundle))
    metadata = json.loads((Path(res["bundle_path"]) / "metadata.json").read_text())
    assert metadata["metadata"]["sample"] == "sample_1"
    outputs = res["outputs"]
    assert len(outputs["labels"]) == 1
    assert len(outputs["tables"]) == 1
    assert len(outputs["qc"]) == 1
    assert tifffile.imread(outputs["labels"][0]).max() == 1
