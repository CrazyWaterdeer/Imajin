from __future__ import annotations

import pytest

from imajin.io.channel_metadata import color_from_wavelengths, wavelength_nm
from imajin.io.ome import _parse_ome_xml


def test_wavelength_units_normalize_to_nm() -> None:
    assert wavelength_nm(488) == pytest.approx(488)
    assert wavelength_nm(0.488) == pytest.approx(488)
    assert wavelength_nm(4.88e-7) == pytest.approx(488)


def test_wavelengths_map_to_common_confocal_colors() -> None:
    assert color_from_wavelengths(excitation_nm=405) == "uv"
    assert color_from_wavelengths(excitation_nm=488) == "green"
    assert color_from_wavelengths(excitation_nm=561) == "red"
    assert color_from_wavelengths(excitation_nm=640) == "ir"
    assert color_from_wavelengths(emission_nm=460) == "uv"
    assert color_from_wavelengths(emission_nm=520) == "green"
    assert color_from_wavelengths(emission_nm=610) == "red"
    assert color_from_wavelengths(emission_nm=670) == "ir"


def test_parse_ome_xml_extracts_channel_wavelengths() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image ID="Image:0">
        <Pixels PhysicalSizeX="0.2" PhysicalSizeY="0.3" PhysicalSizeZ="0.7">
          <Channel ID="Channel:0" Name="DAPI" EmissionWavelength="460" />
          <Channel ID="Channel:1" Name="GCaMP" ExcitationWavelength="488" />
          <Channel ID="Channel:2" Name="Cy5" EmissionWavelength="670" />
        </Pixels>
      </Image>
    </OME>
    """

    voxel, names, metadata = _parse_ome_xml(xml)

    assert voxel == (0.7, 0.3, 0.2)
    assert names == ["DAPI", "GCaMP", "Cy5"]
    assert [m["color"] for m in metadata] == ["uv", "green", "ir"]
    assert metadata[0]["emission_wavelength_nm"] == pytest.approx(460)
    assert metadata[1]["excitation_wavelength_nm"] == pytest.approx(488)
