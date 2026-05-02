from __future__ import annotations

import numpy as np
import pytest

from imajin.agent import state
from imajin.tools import channels


def test_annotate_channel_canonicalizes_far_red(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="sample_cy5")

    res = channels.annotate_channel(
        layer="sample_cy5",
        role="primary",
        color="far red",
        marker="Cy5",
        biological_target="gut cells",
    )

    assert res["role"] == "target"
    assert res["color"] == "ir"
    [entry] = channels.list_channel_annotations_tool()
    assert entry["layer_name"] == "sample_cy5"
    assert entry["role"] == "target"
    assert entry["color"] == "ir"


def test_resolve_channel_uses_annotation_and_get_layer_alias(viewer) -> None:
    layer = viewer.add_image(np.ones((8, 8), dtype=np.uint16), name="reporter_ch1")
    channels.annotate_channel(
        layer="reporter_ch1",
        role="target",
        color="green",
        marker="GCaMP",
    )

    assert channels.resolve_channel("green")["layer"] == "reporter_ch1"
    assert channels.resolve_channel("GCaMP")["layer"] == "reporter_ch1"
    assert state.get_layer("green") is layer


def test_resolve_channel_infers_common_marker_from_layer_name(viewer) -> None:
    viewer.add_image(np.ones((8, 8), dtype=np.uint16), name="brain_DAPI")

    assert channels.resolve_channel("UV")["layer"] == "brain_DAPI"
    assert state.get_layer("DAPI").name == "brain_DAPI"


def test_resolve_channel_rejects_ambiguous_color(viewer) -> None:
    viewer.add_image(np.ones((8, 8), dtype=np.uint16), name="sample_GFP_a")
    viewer.add_image(np.ones((8, 8), dtype=np.uint16), name="sample_GFP_b")

    with pytest.raises(KeyError, match="ambiguous"):
        channels.resolve_channel("green")


def test_resolve_channel_uses_file_wavelength_metadata(viewer) -> None:
    shared_metadata = {
        "channel_names": ["ch0", "ch1", "ch2", "ch3"],
        "channel_metadata": [
            {"name": "DAPI", "color": "uv", "emission_wavelength_nm": 460},
            {"name": "GCaMP", "color": "green", "excitation_wavelength_nm": 488},
            {"name": "mCherry", "color": "red", "emission_wavelength_nm": 610},
            {"name": "Cy5", "color": "ir", "emission_wavelength_nm": 670},
        ],
    }
    for name in shared_metadata["channel_names"]:
        viewer.add_image(
            np.ones((8, 8), dtype=np.uint16),
            name=name,
            metadata=shared_metadata,
        )

    assert channels.resolve_channel("UV")["layer"] == "ch0"
    assert channels.resolve_channel("green")["layer"] == "ch1"
    assert channels.resolve_channel("red")["layer"] == "ch2"
    assert channels.resolve_channel("far red")["layer"] == "ch3"
