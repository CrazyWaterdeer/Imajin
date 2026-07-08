from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from imajin.analysis import deconvolution as dc
from imajin.tools import preprocess


def _points_image(shape=(64, 64), centers=((16, 16), (40, 48), (30, 20))):
    img = np.zeros(shape, dtype=np.float32)
    for c in centers:
        img[c] = 1000.0
    return img


def _sharpness(x):
    return float(x.max()) / (float(x.mean()) + 1e-9)


def test_diffraction_sigmas_axial_worse_than_lateral():
    lat, ax = dc.diffraction_sigmas_um(numerical_aperture=1.4, wavelength_um=0.52, refractive_index=1.515)
    assert lat > 0 and ax > 0
    assert ax > lat  # confocal axial resolution is worse


def test_build_psf_normalized_and_anisotropic():
    psf2d, sig2d = dc.build_psf((0.2, 0.2), 2, lateral_sigma_um=0.3, axial_sigma_um=None)
    assert psf2d.ndim == 2
    assert np.isclose(psf2d.sum(), 1.0)
    psf3d, sig3d = dc.build_psf((1.0, 0.2, 0.2), 3, lateral_sigma_um=0.3, axial_sigma_um=0.9)
    assert psf3d.ndim == 3
    assert np.isclose(psf3d.sum(), 1.0)
    # anisotropic voxels + worse axial ⇒ different per-axis sigma
    assert sig3d[1] != sig3d[0]


def test_resolve_wavelength_from_param_and_metadata():
    assert dc.resolve_wavelength_nm({}, 640.0) == (640.0, "parameter")
    val, src = dc.resolve_wavelength_nm({"emission_wavelength_nm": 600}, None)
    assert val == 600.0 and src.startswith("metadata")
    assert dc.resolve_wavelength_nm({}, None) == (None, None)


def test_deconvolve_sharpens_blurred_image(viewer):
    orig = _points_image()
    blurred = gaussian_filter(orig, sigma=2.0)
    viewer.add_image(blurred, name="blur", scale=(1.0, 1.0))

    res = preprocess.deconvolve("blur", iterations=25, psf="gaussian", lateral_sigma_um=2.0)

    deconv = viewer.layers[res["new_layer"]].data
    assert _sharpness(deconv) > _sharpness(blurred)  # energy re-concentrated
    assert res["psf_shape"] and isinstance(res["psf_shape"], tuple)


def test_deconvolve_theoretical_reads_metadata_wavelength(viewer):
    blurred = gaussian_filter(_points_image(), sigma=1.5)
    viewer.add_image(blurred, name="ch_meta", scale=(0.2, 0.2), metadata={"emission_wavelength_nm": 600})

    res = preprocess.deconvolve("ch_meta", iterations=5, psf="theoretical", numerical_aperture=1.4)

    assert not any("wavelength not found" in w for w in res["warnings"])
    assert viewer.layers[res["new_layer"]].metadata["emission_wavelength_nm"] == 600.0


def test_deconvolve_theoretical_warns_without_wavelength(viewer):
    viewer.add_image(gaussian_filter(_points_image(), 1.5), name="ch_nowl", scale=(0.2, 0.2))
    res = preprocess.deconvolve("ch_nowl", iterations=3, psf="theoretical")
    assert any("assumed 520 nm" in w for w in res["warnings"])


def test_deconvolve_gaussian_requires_sigma(viewer):
    viewer.add_image(_points_image(), name="need_sigma", scale=(1.0, 1.0))
    with pytest.raises(ValueError, match="lateral_sigma_um"):
        preprocess.deconvolve("need_sigma", psf="gaussian")


def test_deconvolve_3d_runs_and_reports_axial(viewer):
    vol = np.zeros((12, 32, 32), dtype=np.float32)
    vol[6, 16, 16] = 1000.0
    vol[4, 10, 20] = 1000.0
    viewer.add_image(gaussian_filter(vol, sigma=(1.0, 2.0, 2.0)), name="vol3d", scale=(1.0, 0.3, 0.3))

    res = preprocess.deconvolve("vol3d", iterations=5, psf="theoretical", emission_wavelength_nm=520)

    assert res["new_layer"] in viewer.layers
    assert res["axial_sigma_um"] is not None
    assert len(res["psf_shape"]) == 3
