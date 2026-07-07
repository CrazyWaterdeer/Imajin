from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imajin import session as state
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


def test_region_column_from_label_names_metadata(viewer) -> None:
    labels, a, _ = _two_label_image()
    viewer.add_labels(labels, name="parts", metadata={"label_names": {1: "inside", 2: "outside"}})
    viewer.add_image(a, name="ch_red")

    res = measure.measure_intensity(labels_layer="parts", image_layers=["ch_red"])
    assert "region" in res["columns"]
    df = state.get_table(res["table_name"])
    label_to_region = dict(zip(df["label"], df["region"]))
    assert label_to_region[1] == "inside"
    assert label_to_region[2] == "outside"


def test_no_region_column_without_label_names(viewer) -> None:
    labels, a, _ = _two_label_image()
    viewer.add_labels(labels, name="masks")  # no label_names metadata
    viewer.add_image(a, name="ch_red")

    res = measure.measure_intensity(labels_layer="masks", image_layers=["ch_red"])
    assert "region" not in res["columns"]


def test_region_column_accepts_string_keys(viewer) -> None:
    # A bundle round-trip through JSON turns {1: ...} into {"1": ...}.
    labels, a, _ = _two_label_image()
    viewer.add_labels(labels, name="parts", metadata={"label_names": {"1": "inside", "2": "outside"}})
    viewer.add_image(a, name="ch_red")

    res = measure.measure_intensity(labels_layer="parts", image_layers=["ch_red"])
    df = state.get_table(res["table_name"])
    assert dict(zip(df["label"], df["region"]))[1] == "inside"


def test_region_column_partial_mapping_keeps_rows(viewer) -> None:
    labels, a, _ = _two_label_image()
    viewer.add_labels(labels, name="parts", metadata={"label_names": {1: "inside"}})
    viewer.add_image(a, name="ch_red")

    res = measure.measure_intensity(labels_layer="parts", image_layers=["ch_red"])
    df = state.get_table(res["table_name"])
    assert res["n_rows"] == 2  # unmapped label 2 is not dropped
    label_to_region = dict(zip(df["label"], df["region"]))
    assert label_to_region[1] == "inside"
    assert pd.isna(label_to_region[2])


def test_region_column_survives_refresh(viewer) -> None:
    labels, a, _ = _two_label_image()
    viewer.add_labels(labels, name="parts", metadata={"label_names": {1: "inside", 2: "outside"}})
    viewer.add_image(a, name="ch_red")

    res = measure.measure_intensity(labels_layer="parts", image_layers=["ch_red"])
    refreshed = measure.refresh_measurement(res["table_name"])
    assert "region" in refreshed["columns"]
    df = state.get_table(res["table_name"])
    assert dict(zip(df["label"], df["region"]))[2] == "outside"


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


def test_combine_tables_concats_with_explicit_labels() -> None:
    state.put_table("rep1", pd.DataFrame({"red": [80.0, 79.0], "region": ["inside", "outside"]}))
    state.put_table("rep2", pd.DataFrame({"red": [58.0, 61.0], "region": ["inside", "outside"]}))
    state.put_table("rep3", pd.DataFrame({"red": [76.0, 83.0], "region": ["inside", "outside"]}))

    res = measure.combine_tables(
        ["rep1", "rep2", "rep3"],
        new_table_name="coloc_combined",
        labels=["rep1", "rep2", "rep3"],
    )

    assert res["table_name"] == "coloc_combined"
    assert res["n_rows"] == 6
    assert res["n_sources"] == 3
    df = state.get_table("coloc_combined")
    assert list(df["sample_name"]) == ["rep1", "rep1", "rep2", "rep2", "rep3", "rep3"]
    # ready for paired inside/outside analysis: 3 samples x 2 regions
    assert df.groupby(["sample_name", "region"]).ngroups == 6


def test_combine_tables_defaults_label_to_source_table_name() -> None:
    state.put_table("t_a", pd.DataFrame({"v": [1.0]}))
    state.put_table("t_b", pd.DataFrame({"v": [2.0]}))

    res = measure.combine_tables(["t_a", "t_b"])

    df = state.get_table(res["table_name"])
    assert list(df["sample_name"]) == ["t_a", "t_b"]
    assert res["labels"] == ["t_a", "t_b"]


def test_combine_tables_unions_columns_with_nan_fill() -> None:
    state.put_table("with_area", pd.DataFrame({"v": [1.0], "area": [10.0]}))
    state.put_table("no_area", pd.DataFrame({"v": [2.0]}))

    res = measure.combine_tables(["with_area", "no_area"], labels=["s1", "s2"])

    df = state.get_table(res["table_name"])
    assert "area" in df.columns
    assert df.loc[df["sample_name"] == "s2", "area"].isna().all()
    assert res["columns_not_in_all_sources"] == ["area"]


def test_combine_tables_guards_existing_label_column() -> None:
    state.put_table("already", pd.DataFrame({"v": [1.0], "sample_name": ["x"]}))
    with pytest.raises(ValueError, match="already has"):
        measure.combine_tables(["already"])
    # explicit labels signal intent and override the guard
    res = measure.combine_tables(["already"], labels=["forced"])
    assert list(state.get_table(res["table_name"])["sample_name"]) == ["forced"]


def test_combine_tables_rejects_mismatched_labels_length() -> None:
    state.put_table("t1", pd.DataFrame({"v": [1.0]}))
    state.put_table("t2", pd.DataFrame({"v": [2.0]}))
    with pytest.raises(ValueError, match="one label per table"):
        measure.combine_tables(["t1", "t2"], labels=["only_one"])


def test_combine_tables_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        measure.combine_tables([])


def test_combine_tables_missing_table_raises() -> None:
    state.put_table("real", pd.DataFrame({"v": [1.0]}))
    with pytest.raises(KeyError):
        measure.combine_tables(["real", "ghost"])


def test_combined_table_feeds_paired_compare_groups() -> None:
    from imajin.tools import stats

    # 3 replicates, several inside/outside objects each (object-level rows)
    for rep, (ins, out) in {
        "rep1": (80.0, 79.0),
        "rep2": (58.0, 61.0),
        "rep3": (76.0, 83.0),
    }.items():
        state.put_table(
            rep,
            pd.DataFrame(
                {
                    "red": [ins, ins + 2, out, out + 2],
                    "region": ["inside", "inside", "outside", "outside"],
                }
            ),
        )

    combined = measure.combine_tables(
        ["rep1", "rep2", "rep3"], labels=["rep1", "rep2", "rep3"]
    )["table_name"]

    res = stats.compare_groups(
        combined,
        value_col="red",
        group_col="region",
        test="wilcoxon",
        save_csv=False,
    )
    assert "wilcoxon" in res["test"]
    # paired mode aggregates the per-object rows to one value per specimen
    assert res["data_level"] == "sample"
    assert isinstance(res["p_value"], float)


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


def test_measure_projected_intensity_uses_average_projection(viewer) -> None:
    labels, a, _ = _two_label_image()
    stack = np.stack([a, a * 2, a * 3], axis=0)
    viewer.add_labels(labels, name="rois", scale=(0.5, 0.5))
    viewer.add_image(stack, name="calexa", scale=(1.0, 0.5, 0.5), metadata={"axes": "ZYX"})

    res = measure.measure_projected_intensity(
        labels_layer="rois",
        image_layer="calexa",
        projection="mean",
        properties=["label", "area", "mean_intensity"],
    )

    assert res["projection"] == "mean"
    assert res["projected_layer"] == "calexa_avg_z"
    df = state.get_table(res["table_name"])
    label1 = df[df["label"] == 1].iloc[0]
    assert label1["mean_intensity_calexa_avg_z"] == pytest.approx(200.0)
    assert "area_um2" in df.columns


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
    viewer.add_image(
        np.zeros((3, 10, 10), dtype=np.float32),
        name="movie",
        metadata={"axes": "TYX"},
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        measure.measure_intensity_over_time("rois", "movie")


def test_measure_intensity_over_time_requires_time_axis_metadata_or_argument(viewer) -> None:
    viewer.add_labels(np.ones((8, 8), dtype=np.int32), name="rois")
    viewer.add_image(np.zeros((3, 8, 8), dtype=np.float32), name="stack_without_axes")

    with pytest.raises(ValueError, match="do not include a time axis"):
        measure.measure_intensity_over_time("rois", "stack_without_axes")

    res = measure.measure_intensity_over_time(
        "rois",
        "stack_without_axes",
        time_axis=0,
    )
    assert res["n_timepoints"] == 3
