from __future__ import annotations

import numpy as np
import pytest

from imajin.io.lsm import _channel_metadata, _channel_names, _voxel_size_um
from imajin.io.memory import array_nbytes, should_load_into_memory


def test_voxel_size_converts_meters_to_microns() -> None:
    meta = {"VoxelSizeX": 1e-7, "VoxelSizeY": 2e-7, "VoxelSizeZ": 5e-7}
    vz, vy, vx = _voxel_size_um(meta)
    assert abs(vx - 0.1) < 1e-9
    assert abs(vy - 0.2) < 1e-9
    assert abs(vz - 0.5) < 1e-9


def test_voxel_size_defaults_when_missing() -> None:
    vz, vy, vx = _voxel_size_um({})
    assert (vz, vy, vx) == (1.0, 1.0, 1.0)


def test_channel_names_from_color_names() -> None:
    meta = {"ChannelColors": {"ColorNames": ["DAPI", "GFP", "RFP"]}}
    assert _channel_names(meta) == ["DAPI", "GFP", "RFP"]


def test_channel_names_from_scan_information_tracks() -> None:
    meta = {
        "ScanInformation": {
            "Tracks": [
                {"DataChannels": [{"Name": "Track1-Ch1"}]},
                {"DataChannels": [{"Name": "Track2-Ch1"}, {"Name": "Track2-Ch2"}]},
            ]
        }
    }
    assert _channel_names(meta) == ["Track1-Ch1", "Track2-Ch1", "Track2-Ch2"]


def test_channel_names_empty_when_no_metadata() -> None:
    assert _channel_names({}) == []


def test_channel_metadata_from_lsm_scan_wavelengths() -> None:
    meta = {
        "ScanInformation": {
            "Tracks": [
                {
                    "IlluminationChannels": [{"Wavelength": 4.88e-7}],
                    "DataChannels": [{"Name": "GCaMP"}],
                },
                {
                    "DataChannels": [
                        {"Name": "mCherry", "EmissionWavelength": 610},
                        {"Name": "Cy5", "EmissionWavelength": 670},
                    ]
                },
            ]
        }
    }

    metadata = _channel_metadata(meta)

    assert [m["name"] for m in metadata] == ["GCaMP", "mCherry", "Cy5"]
    assert [m["color"] for m in metadata] == ["green", "red", "ir"]
    assert metadata[0]["excitation_wavelength_nm"] == pytest.approx(488)


def test_array_nbytes_uses_shape_and_dtype() -> None:
    assert array_nbytes((2, 3, 4), np.uint16) == 48


def test_should_load_into_memory_when_headroom_available() -> None:
    assert should_load_into_memory(5 * 1024**3, available_bytes=8 * 1024**3)


def test_should_not_load_into_memory_when_headroom_missing() -> None:
    assert not should_load_into_memory(5 * 1024**3, available_bytes=6 * 1024**3)
