from __future__ import annotations

from pathlib import Path

from imajin import session as state
from imajin.analysis.metadata_validation import validate_acquisition_metadata
from imajin.io.metadata import read_metadata_summary
from imajin.tools import experiment, workflows


def _summary(
    green: dict[str, object],
    counterstain: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "metadata_read_mode": "metadata_only",
        "n_channels": 2,
        "channel_names": ["GFP", "DAPI"],
        "channel_metadata": [
            {"name": "GFP", "color": "green", **green},
            {"name": "DAPI", "color": "uv", **(counterstain or {})},
        ],
    }


def test_read_metadata_summary_uses_metadata_only_path(tiny_ome_tiff: Path) -> None:
    summary = read_metadata_summary(tiny_ome_tiff)

    assert summary["metadata_read_mode"] == "metadata_only"
    assert summary["axes"] == "CZYX"
    assert summary["shape"] == (3, 5, 64, 64)
    assert summary["channel_metadata"][0]["bit_depth"] == 16


def test_validate_acquisition_metadata_detects_target_mismatch_only() -> None:
    records = [
        {
            "path": "/data/a.lsm",
            "metadata_summary": _summary(
                {
                    "laser_intensity": 5.0,
                    "detector_gain": 600,
                    "bit_depth": 16,
                    "pinhole_size": 1.0,
                },
                {"laser_intensity": 1.0, "detector_gain": 100},
            ),
        },
        {
            "path": "/data/b.lsm",
            "metadata_summary": _summary(
                {
                    "laser_intensity": 8.0,
                    "detector_gain": 600,
                    "bit_depth": 16,
                    "pinhole_size": 1.0,
                },
                {"laser_intensity": 9.0, "detector_gain": 900},
            ),
        },
    ]

    result = validate_acquisition_metadata(
        records,
        target_channel="green",
        analysis_kind="intensity",
        strict_missing=True,
    )

    assert result["status"] == "fail"
    assert [item["setting"] for item in result["mismatches"]] == ["laser_intensity"]
    assert result["n_channels_checked"] == 2


def test_validate_acquisition_metadata_skips_intensity_settings_for_area() -> None:
    records = [
        {"path": "/data/a.lsm", "metadata_summary": _summary({"laser_intensity": 5})},
        {"path": "/data/b.lsm", "metadata_summary": _summary({"laser_intensity": 8})},
    ]

    result = validate_acquisition_metadata(
        records,
        target_channel="green",
        analysis_kind="area",
        strict_missing=True,
    )

    assert result["ok"] is True
    assert result["settings_checked"] == []


def test_run_recipe_on_samples_stops_before_analysis_on_metadata_mismatch(
    monkeypatch,
) -> None:
    state.put_file(
        path="/data/a.lsm",
        original_name="a.lsm",
        file_type="lsm",
        metadata_summary=_summary(
            {
                "laser_intensity": 5.0,
                "detector_gain": 600,
                "bit_depth": 16,
                "pinhole_size": 1.0,
            }
        ),
    )
    state.put_file(
        path="/data/b.lsm",
        original_name="b.lsm",
        file_type="lsm",
        metadata_summary=_summary(
            {
                "laser_intensity": 8.0,
                "detector_gain": 600,
                "bit_depth": 16,
                "pinhole_size": 1.0,
            }
        ),
    )
    experiment.annotate_samples(
        [
            {"sample_name": "a", "group": "g", "file_ids": ["a"]},
            {"sample_name": "b", "group": "g", "file_ids": ["b"]},
        ]
    )
    experiment.create_analysis_recipe(
        name="calexa",
        target_channel="green",
        segmentation={"method": "intensity_regions"},
    )

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("analysis should not run after metadata mismatch")

    monkeypatch.setattr(workflows, "analyze_target_cells", fail_if_called)

    result = workflows.run_recipe_on_samples(recipe_name="calexa")

    assert result["n_complete"] == 0
    assert result["n_failed"] == 2
    assert result["bundle_path"] is None
    assert result["metadata_validation"]["status"] == "fail"
