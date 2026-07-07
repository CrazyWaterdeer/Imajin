from __future__ import annotations

import pandas as pd
import pytest

from imajin import session as state
from imajin.tools import figures


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


@pytest.fixture(autouse=True)
def _require_matplotlib():
    pytest.importorskip("matplotlib")


def test_plot_group_distribution_writes_svg(tmp_path) -> None:
    table = state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
            }
        ),
        spec={"tool": "test"},
    )
    out = tmp_path / "distribution.svg"

    res = figures.plot_group_distribution(
        table,
        "mean_intensity",
        output_path=str(out),
    )

    assert res["path"] == str(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert res["data_level"] == "sample"
    assert res["plot_data_table"] in state.list_tables()
    assert res["p_value"] < 0.05
    assert res["stats_result_table"] in state.list_tables()


def test_plot_timecourse_writes_svg(tmp_path) -> None:
    table = state.put_table(
        "timecourse",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c1", "c1", "t1", "t1", "t1"],
                "group": ["control", "control", "control", "treated", "treated", "treated"],
                "label": [1, 1, 1, 1, 1, 1],
                "time_index": [0, 1, 2, 0, 1, 2],
                "delta_f": [0.0, 0.1, 0.2, 0.0, 0.5, 1.0],
            }
        ),
        spec={"tool": "test"},
    )
    out = tmp_path / "timecourse.svg"

    res = figures.plot_timecourse(
        table,
        value_col="delta_f",
        output_path=str(out),
    )

    assert res["path"] == str(out)
    assert out.exists()
    assert res["unit_level"] == "sample"
    assert res["plot_data_table"] in state.list_tables()


def test_plot_scatter_writes_svg(tmp_path) -> None:
    table = state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "ch1": [1.0, 2.0, 3.0, 4.0],
                "ch2": [2.0, 4.0, 6.0, 8.0],
            }
        ),
        spec={"tool": "test"},
    )
    out = tmp_path / "scatter.svg"

    res = figures.plot_scatter(table, "ch1", "ch2", output_path=str(out))

    assert res["path"] == str(out)
    assert out.exists()
    assert res["pearson_r"] == pytest.approx(1.0)
    assert res["pearson_p_value"] == pytest.approx(0.0)
    assert res["fit_slope"] == pytest.approx(2.0)


def test_figure_writes_into_active_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAJIN_RESULTS_DIR", str(tmp_path))
    import pandas as pd
    from pathlib import Path
    from imajin import session as state
    from imajin.result_bundles import reset_process_bundle, start_analysis
    from imajin.results import read_bundle_metadata
    from imajin.tools import figures

    reset_process_bundle()
    bundle = start_analysis(name="figtest")
    state.put_table(
        "measurements",
        pd.DataFrame(
            {
                "sample_name": ["c1", "c2", "t1", "t2"],
                "group": ["control", "control", "treated", "treated"],
                "mean_intensity": [1.0, 1.2, 2.5, 2.8],
            }
        ),
        spec={"tool": "test"},
    )

    res = figures.plot_group_distribution("measurements", "mean_intensity")

    out = Path(res["path"])
    assert out.parent == bundle / "figures"
    assert out.exists()
    outputs = read_bundle_metadata(bundle)["outputs"]
    assert any(o["kind"] == "figure" and o["path"] == f"figures/{out.name}" for o in outputs)

    reset_process_bundle()


def _dist_table(name: str = "m") -> str:
    return state.put_table(
        name,
        pd.DataFrame(
            {
                "sample_name": [f"c{i}" for i in range(4)] + [f"t{i}" for i in range(4)],
                "group": ["control"] * 4 + ["treated"] * 4,
                "mean_intensity": [1.0, 1.2, 0.9, 1.1, 2.5, 2.8, 2.6, 2.7],
            }
        ),
        spec={"tool": "test"},
    )


@pytest.mark.parametrize("kind", ["box", "bar", "violin", "dots"])
def test_plot_group_distribution_kind_variants(tmp_path, kind) -> None:
    table = _dist_table()
    out = tmp_path / f"{kind}.svg"
    res = figures.plot_group_distribution(table, "mean_intensity", kind=kind, output_path=str(out))
    assert res["path"] == str(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_plot_paired_lines_and_style_options(tmp_path) -> None:
    rows = []
    for i in range(5):  # same samples in both groups -> paired connecting lines
        rows += [
            {"sample_name": f"s{i}", "group": "inside", "mean_intensity": 5.0 + i},
            {"sample_name": f"s{i}", "group": "outside", "mean_intensity": 3.0 + i},
        ]
    table = state.put_table("io", pd.DataFrame(rows), spec={"tool": "test"})
    out = tmp_path / "paired.png"
    res = figures.plot_group_distribution(
        table, "mean_intensity", group_col="group", kind="dots", paired=True,
        palette=["#3E6DB5", "#E08214"], ymin=0.0, ymax=12.0, zero_baseline=True,
        jitter=0.05, point_size=30.0, format="png", output_path=str(out),
    )
    assert res["path"] == str(out)
    assert out.exists()


def test_plot_posthoc_brackets_for_three_groups(tmp_path) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    rows = []
    for g, mu in [("a", 10.0), ("b", 12.0), ("c", 20.0)]:
        for i in range(7):
            rows.append({"sample_name": f"{g}{i}", "group": g, "mean_intensity": float(mu + rng.normal(0, 1.5))})
    table = state.put_table("three", pd.DataFrame(rows), spec={"tool": "test"})
    out = tmp_path / "three.svg"
    res = figures.plot_group_distribution(
        table, "mean_intensity", group_col="group", log_y=True, output_path=str(out)
    )
    assert len(res["groups"]) == 3
    assert out.exists()


def test_new_palette_and_font_options(tmp_path) -> None:
    assert figures._PALETTE[0] == "#636867"  # control = slate grey
    assert figures._PALETTE[1] == "#DA4E42"  # coral red
    table = _dist_table()

    sans = tmp_path / "sans.svg"
    figures.plot_group_distribution(table, "mean_intensity", output_path=str(sans))
    assert "Noto Sans" in sans.read_text(encoding="utf-8")

    serif = tmp_path / "serif.svg"
    figures.plot_group_distribution(table, "mean_intensity", font="serif", output_path=str(serif))
    assert "Noto Serif" in serif.read_text(encoding="utf-8")


def test_pretty_label_scientific_formatting() -> None:
    P = figures._pretty_label
    assert P("mean_intensity") == "Mean Intensity"
    assert P("mean_intensity_GFP") == "Mean Intensity GFP"   # acronym kept uppercase
    assert P("dff") == "ΔF/F₀"
    assert P("area_um2") == "Area (µm²)"                      # trailing unit -> parens
    assert P("time_s") == "Time (s)"
    assert P("outside_green") == "Outside Green"
    assert P("Ch2-T2") == "Ch2-T2"                            # intentional mixed case kept
    assert "_" not in P("some_raw_column_name")


def test_figure_labels_have_no_raw_underscores(tmp_path) -> None:
    table = state.put_table(
        "m",
        pd.DataFrame({"sample_name": ["a0", "a1", "b0", "b1"],
                      "group": ["green_coloc", "green_coloc", "outside_green", "outside_green"],
                      "mean_intensity": [1.0, 2.0, 3.0, 4.0]}),
        spec={"tool": "test"},
    )
    out = tmp_path / "lbl.svg"
    figures.plot_group_distribution(table, "mean_intensity", output_path=str(out))
    svg = out.read_text(encoding="utf-8")
    assert "Mean Intensity" in svg and "Outside Green" in svg
    assert "mean_intensity" not in svg and "outside_green" not in svg
