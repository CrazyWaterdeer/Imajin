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
    assert samples == [
        {
            "sample_name": "control_1",
            "group": "control",
            "layers": ["ctrl1_ch0", "ctrl1_ch1"],
            "files": ["/data/control_1.lsm"],
            "notes": "adult gut",
        }
    ]

    state.reset_samples()
