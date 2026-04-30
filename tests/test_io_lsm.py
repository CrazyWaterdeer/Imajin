from __future__ import annotations

from imajin.io.lsm import _channel_names, _voxel_size_um


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
