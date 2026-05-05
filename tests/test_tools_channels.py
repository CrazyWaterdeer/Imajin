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


def test_annotate_counterstain_sets_gray_colormap(viewer) -> None:
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="TOPRO")

    res = channels.annotate_channel(
        layer="TOPRO",
        role="counterstain",
        color="far red",
        marker="TOPRO",
    )

    assert res["role"] == "counterstain"
    assert res["color"] == "ir"
    assert viewer.layers["TOPRO"].colormap.name == "gray"


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


def test_resolve_channel_preserves_fiji_style_channel_numbers(viewer) -> None:
    shared_metadata = {
        "channel_names": ["Ch1", "Ch2"],
        "channel_metadata": [
            {"name": "Ch1", "dye_name": "Alexa Fluor 488", "color": "green"},
            {"name": "Ch2", "dye_name": "Alexa Fluor 633", "color": "ir"},
        ],
    }
    for name in shared_metadata["channel_names"]:
        viewer.add_image(
            np.ones((8, 8), dtype=np.uint16),
            name=name,
            metadata=shared_metadata,
        )

    assert channels.resolve_channel("GFP")["layer"] == "Ch1"
    assert channels.resolve_channel("green")["layer"] == "Ch1"
    assert channels.resolve_channel("far red")["layer"] == "Ch2"


def test_detect_counterstain_returns_annotated_topro(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools
    from imajin.agent.state import put_channel_annotation

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="ch3_633")
    put_channel_annotation(
        layer_name="ch3_633",
        role="counterstain",
        marker="topro",
    )

    result = channels_tools.detect_counterstain_channel()

    assert result["counterstain_layer"] == "ch3_633"
    assert result["counterstain_marker"] == "topro"
    assert result["is_nuclear"] is True
    assert result["confidence"] == "annotated"
    assert result["needs_user_confirmation"] is False


def test_detect_counterstain_marks_non_nuclear_phalloidin(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools
    from imajin.agent.state import put_channel_annotation

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="ch_actin")
    put_channel_annotation(
        layer_name="ch_actin",
        role="counterstain",
        marker="phalloidin",
    )

    result = channels_tools.detect_counterstain_channel()
    assert result["counterstain_marker"] == "phalloidin"
    assert result["is_nuclear"] is False


def test_detect_counterstain_infers_from_633_layer_name(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Ch3-633")
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Ch1-488")

    result = channels_tools.detect_counterstain_channel()

    assert result["counterstain_layer"] == "Ch3-633"
    assert result["confidence"] == "inferred"
    assert result["needs_user_confirmation"] is True
    assert result["is_nuclear"] is None
    assert "Ch3-633" in result["candidate_layers"]


def test_detect_counterstain_returns_none_when_absent(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Ch1-488")

    result = channels_tools.detect_counterstain_channel()
    assert result["counterstain_layer"] is None
    assert result["confidence"] == "none"
    assert result["needs_user_confirmation"] is False
    assert result["candidate_layers"] == []


def test_detect_counterstain_filters_by_sample_layers(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools
    from imajin.agent.state import put_channel_annotation, put_sample

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="A_topro")
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="B_topro")
    put_channel_annotation(layer_name="A_topro", role="counterstain", marker="topro")
    put_channel_annotation(layer_name="B_topro", role="counterstain", marker="topro")

    put_sample(sample_name="sampleA", layers=["A_topro"])

    result = channels_tools.detect_counterstain_channel(sample_name="sampleA")
    assert result["counterstain_layer"] == "A_topro"
