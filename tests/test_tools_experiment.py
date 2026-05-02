from __future__ import annotations

from imajin.agent import state
from imajin.tools import experiment


def test_annotate_sample_records_group_metadata() -> None:
    state.reset_samples()

    res = experiment.annotate_sample(
        sample_name="control_1",
        group="control",
        layers=["ctrl1_ch0", "ctrl1_ch1"],
        files=["/data/control_1.lsm"],
        notes="adult gut",
    )

    assert res["sample_name"] == "control_1"
    assert res["group"] == "control"
    samples = experiment.list_sample_annotations()
    assert len(samples) == 1
    s = samples[0]
    assert s["sample_name"] == "control_1"
    assert s["group"] == "control"
    assert s["layers"] == ["ctrl1_ch0", "ctrl1_ch1"]
    assert s["files"] == ["/data/control_1.lsm"]
    assert s["notes"] == "adult gut"
    # New Phase-3 fields default to safe values:
    assert s["sample_id"] == "control_1"
    assert s["file_ids"] == []
    assert s["extra"] == {}

    state.reset_samples()
