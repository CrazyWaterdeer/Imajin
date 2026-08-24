from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from imajin import session as state
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


def test_save_result_bundle_appends_to_active_analysis_bundle(
    viewer, tmp_path, monkeypatch
) -> None:
    from imajin.result_bundles import reset_process_bundle, start_analysis

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    reset_process_bundle()
    state.reset_tables()
    state.put_table("m1", pd.DataFrame({"label": [1], "area": [10.0]}))
    state.put_table("m2", pd.DataFrame({"label": [1], "area": [20.0]}))

    started = start_analysis("session")
    r1 = results.save_result_bundle(name="file1", table_names=["m1"])
    r2 = results.save_result_bundle(name="file2", table_names=["m2"])

    # sequential per-file saves collect into the one started bundle, not a folder each
    assert r1["reused"] is True and r2["reused"] is True
    assert r1["bundle_path"] == str(started)
    assert r2["bundle_path"] == str(started)
    tables = sorted(p.name for p in (started / "tables").glob("*.csv"))
    assert tables == ["m1.csv", "m2.csv"]
    reset_process_bundle()
    state.reset_tables()


def test_save_result_bundle_creates_then_reuses_without_start_analysis(
    viewer, tmp_path, monkeypatch
) -> None:
    from imajin.result_bundles import reset_process_bundle

    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    reset_process_bundle()
    state.reset_tables()
    state.put_table("a", pd.DataFrame({"label": [1], "area": [1.0]}))
    state.put_table("b", pd.DataFrame({"label": [1], "area": [2.0]}))

    # no start_analysis: the first save creates + promotes a bundle, the next reuses it
    first = results.save_result_bundle(name="fileA", table_names=["a"])
    second = results.save_result_bundle(name="fileB", table_names=["b"])
    assert first["reused"] is False
    assert second["reused"] is True
    assert second["bundle_path"] == first["bundle_path"]

    # new_bundle=True is the escape hatch for a genuinely separate folder
    forced = results.save_result_bundle(name="fileC", table_names=["b"], new_bundle=True)
    assert forced["reused"] is False
    assert forced["bundle_path"] != first["bundle_path"]
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
        "imajin.session.list_files",
        lambda: [{"path": str(fake_file)}],
    )
    assert _results.results_root() == folder.absolute()


def test_results_root_falls_back_to_user_root_when_no_anchor(tmp_path, monkeypatch):
    from imajin import results as _results

    monkeypatch.setattr("imajin.session.list_files", lambda: [])
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
    from imajin import session as state
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


def test_save_result_bundle_outputs_index_is_list_of_records(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from pathlib import Path
    from imajin import session as state
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
    out = save_result_bundle(name="idx", table_names=["measurements"])
    bundle = Path(out["bundle_path"])

    meta = read_bundle_metadata(bundle)
    outputs_index = meta["outputs"]
    assert isinstance(outputs_index, list)
    for entry in outputs_index:
        assert isinstance(entry, dict)
        assert "kind" in entry and "path" in entry
    # Table CSV and the result_bundle record should both be indexed.
    kinds = {e["kind"] for e in outputs_index}
    assert "table_csv" in kinds
    assert "result_bundle" in kinds
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


def test_save_result_bundle_reuse_of_one_layer_is_idempotent(
    viewer, tmp_path, monkeypatch
):
    """Re-saving the same in-memory layer updates one file, it does not pile up."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import results as results_tools

    reset_process_bundle()
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    viewer.add_labels(labels, name="Ch2-T1_objects")

    res1 = results_tools.save_result_bundle("a", labels_layers=["Ch2-T1_objects"])
    bundle = Path(res1["bundle_path"])
    results_tools.save_result_bundle("a", labels_layers=["Ch2-T1_objects"])

    written = sorted(p.name for p in (bundle / "labels" / "cells").iterdir())
    assert written == ["Ch2-T1_objects.tif"]
    reset_process_bundle()


def test_save_result_bundle_keeps_each_source_files_labels(
    viewer, tmp_path, monkeypatch
):
    """Two FILES segmenting to the same layer name must both keep their labels.

    Every file in a folder loads as `Ch2-T1` and segments to `Ch2-T1_objects`,
    so keying the destination on the layer name sends the second file's save
    straight onto the first file's TIFF. Identity has to come from the file.
    """
    import tifffile

    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path / "results"))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools
    from imajin.tools import results as results_tools

    reset_process_bundle()
    bundle = Path(bundle_tools.start_analysis("session")["bundle_path"])

    written_max = {}
    for stem, n_objects in (("rectum_1", 1), ("rectum_2", 2)):
        source = raw / f"{stem}.lsm"
        source.write_bytes(b"stub")
        labels = np.zeros((8, 8), dtype=np.uint16)
        for i in range(n_objects):
            labels[i * 3 : i * 3 + 2, 1:3] = i + 1

        viewer.layers.clear()
        # Both files load under the SAME names, as a real instrument emits.
        viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Ch2-T1")
        viewer.layers["Ch2-T1"].metadata["source_path"] = str(source)
        viewer.add_labels(labels, name="Ch2-T1_objects")
        viewer.layers["Ch2-T1_objects"].metadata["source_layer"] = "Ch2-T1"

        results_tools.save_result_bundle(stem, labels_layers=["Ch2-T1_objects"])
        written_max[stem] = n_objects

    files = sorted(p.name for p in (bundle / "labels" / "cells").iterdir())
    assert files == ["rectum_1.tif", "rectum_2.tif"]
    for stem, expected in written_max.items():
        actual = int(tifffile.imread(bundle / "labels" / "cells" / f"{stem}.tif").max())
        assert actual == expected, f"{stem}.tif holds another file's labels"
    reset_process_bundle()


def test_save_result_bundle_keeps_distinct_qc_images(viewer, tmp_path, monkeypatch):
    """Two QC PNGs with the same basename from different sources both survive."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import results as results_tools

    reset_process_bundle()
    src_a = tmp_path / "a" / "qc.png"
    src_b = tmp_path / "b" / "qc.png"
    for src, payload in ((src_a, b"AAAA"), (src_b, b"BBBB")):
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"\x89PNG\r\n" + payload)

    res = results_tools.save_result_bundle("file_a", qc_png_paths=[str(src_a)])
    bundle = Path(res["bundle_path"])
    results_tools.save_result_bundle("file_b", qc_png_paths=[str(src_b)])

    written = sorted(p.name for p in (bundle / "qc").iterdir())
    assert written == ["qc.png", "qc_2.png"]
    assert (bundle / "qc" / "qc.png").read_bytes().endswith(b"AAAA")
    assert (bundle / "qc" / "qc_2.png").read_bytes().endswith(b"BBBB")
    reset_process_bundle()


def test_save_result_bundle_reports_that_an_appended_name_is_not_applied(
    viewer, tmp_path, monkeypatch
):
    """Appending keeps the folder's name; the caller must be told, not misled."""
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle
    from imajin.tools import bundle as bundle_tools
    from imajin.tools import results as results_tools

    reset_process_bundle()
    bundle_tools.start_analysis("session_name")

    res = results_tools.save_result_bundle("mF_rectum_1")
    assert res["reused"] is True
    assert res["name_applied"] is False
    assert res["bundle_name"] == "session_name"
    reset_process_bundle()


def test_save_result_bundle_keeps_caller_metadata_when_appending(
    viewer, tmp_path, monkeypatch
):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    from imajin.result_bundles import reset_process_bundle
    from imajin.results import read_bundle_metadata
    from imajin.tools import bundle as bundle_tools
    from imajin.tools import results as results_tools

    reset_process_bundle()
    bundle_tools.start_analysis("session_name")
    res = results_tools.save_result_bundle("s", metadata={"roi": "hindgut"})

    seed = read_bundle_metadata(Path(res["bundle_path"]))
    assert seed["recipe_params"]["roi"] == "hindgut"
    reset_process_bundle()
