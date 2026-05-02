from __future__ import annotations

import numpy as np
import pytest

from imajin.agent import state
from imajin.tools import measure


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


def _two_label_image():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    img_a = np.zeros_like(labels, dtype=np.float32)
    img_a[2:8, 2:8] = 100.0
    img_a[12:18, 12:18] = 50.0
    img_b = np.zeros_like(labels, dtype=np.float32)
    img_b[2:8, 2:8] = 10.0
    img_b[12:18, 12:18] = 200.0
    return labels, img_a, img_b


def test_measure_intensity_two_layers(viewer) -> None:
    labels, a, b = _two_label_image()
    viewer.add_labels(labels, name="masks")
    viewer.add_image(a, name="ch_red")
    viewer.add_image(b, name="ch_green")

    res = measure.measure_intensity(
        labels_layer="masks",
        image_layers=["ch_red", "ch_green"],
        properties=["label", "area", "mean_intensity"],
    )

    assert res["n_rows"] == 2
    assert "mean_intensity_ch_red" in res["columns"]
    assert "mean_intensity_ch_green" in res["columns"]

    df = state.get_table(res["table_name"])
    label_to_red = dict(zip(df["label"], df["mean_intensity_ch_red"]))
    label_to_green = dict(zip(df["label"], df["mean_intensity_ch_green"]))
    assert label_to_red[1] == pytest.approx(100.0)
    assert label_to_red[2] == pytest.approx(50.0)
    assert label_to_green[1] == pytest.approx(10.0)
    assert label_to_green[2] == pytest.approx(200.0)


def test_refresh_measurement_picks_up_label_edit(viewer) -> None:
    labels, a, _ = _two_label_image()
    lbl_layer = viewer.add_labels(labels.copy(), name="masks")
    viewer.add_image(a, name="ch")

    res = measure.measure_intensity(
        labels_layer="masks", image_layers=["ch"], properties=["label", "area"]
    )
    assert state.get_table(res["table_name"]).shape[0] == 2

    new_labels = labels.copy()
    new_labels[new_labels == 2] = 0
    lbl_layer.data = new_labels

    refreshed = measure.refresh_measurement(res["table_name"])
    assert refreshed["n_rows"] == 1
    assert refreshed["delta_rows"] == -1


def test_filter_table_pandas_query(viewer) -> None:
    labels, a, _ = _two_label_image()
    viewer.add_labels(labels, name="masks")
    viewer.add_image(a, name="ch")
    res = measure.measure_intensity(labels_layer="masks", image_layers=["ch"])

    f = measure.filter_table(res["table_name"], "mean_intensity_ch > 75")
    assert f["n_rows"] == 1


def test_summarize_table_mean(viewer) -> None:
    labels, a, b = _two_label_image()
    viewer.add_labels(labels, name="masks")
    viewer.add_image(a, name="ch_red")
    viewer.add_image(b, name="ch_green")
    res = measure.measure_intensity(
        labels_layer="masks", image_layers=["ch_red", "ch_green"]
    )

    summary = measure.summarize_table(res["table_name"], op="mean")
    assert "mean_intensity_ch_red" in summary["values"]
    assert summary["values"]["mean_intensity_ch_red"] == pytest.approx(75.0)
    assert summary["values"]["mean_intensity_ch_green"] == pytest.approx(105.0)


def test_measure_intensity_rejects_shape_mismatch(viewer) -> None:
    labels, a, _ = _two_label_image()
    viewer.add_labels(labels, name="masks")
    viewer.add_image(a[:10, :10], name="too_small")
    with pytest.raises(ValueError, match="shape mismatch"):
        measure.measure_intensity(labels_layer="masks", image_layers=["too_small"])


def test_measure_intensity_over_time_static_rois(viewer) -> None:
    labels, a, _ = _two_label_image()
    series = np.stack([a, a * 2, a * 3], axis=0)
    viewer.add_labels(labels, name="rois")
    viewer.add_image(series, name="gcamp", metadata={"axes": "TYX"})

    res = measure.measure_intensity_over_time(
        labels_layer="rois",
        image_layer="gcamp",
        properties=["label", "area", "mean_intensity"],
    )

    assert res["n_timepoints"] == 3
    assert res["n_labels"] == 2
    assert res["n_rows"] == 6
    df = state.get_table(res["table_name"])
    label1 = df[df["label"] == 1].sort_values("time")
    assert label1["mean_intensity"].tolist() == pytest.approx([100.0, 200.0, 300.0])


def test_measure_intensity_over_time_dynamic_labels(viewer) -> None:
    labels, a, _ = _two_label_image()
    labels_t = np.stack([labels, np.where(labels == 2, 0, labels)], axis=0)
    series = np.stack([a, a * 2], axis=0)
    viewer.add_labels(labels_t, name="tracked_rois")
    viewer.add_image(series, name="calexa", metadata={"axes": "TYX"})

    res = measure.measure_intensity_over_time("tracked_rois", "calexa")

    assert res["n_timepoints"] == 2
    df = state.get_table(res["table_name"])
    assert df[df["time"] == 0]["label"].tolist() == [1, 2]
    assert df[df["time"] == 1]["label"].tolist() == [1]


def test_measure_intensity_over_time_rejects_shape_mismatch(viewer) -> None:
    viewer.add_labels(np.ones((8, 8), dtype=np.int32), name="rois")
    viewer.add_image(np.zeros((3, 10, 10), dtype=np.float32), name="movie")

    with pytest.raises(ValueError, match="shape mismatch"):
        measure.measure_intensity_over_time("rois", "movie")
