"""Deconvolution numerics: PSF synthesis + Richardson-Lucy.

A confocal image is the true fluorophore distribution convolved with the
microscope point-spread function (PSF), which is anisotropic — worse along z
than laterally. Deconvolution estimates the sharp distribution by reversing that
blur, tightening puncta, recovering axial resolution and contrast so downstream
segmentation / spot detection / colocalization improve.

The PSF here is a **Gaussian approximation** whose lateral and axial widths come
from diffraction theory (NA, emission wavelength, refractive index) — light and
metadata-driven, not a full Gibson-Lanni model. Pass explicit sigmas, or a
measured-bead PSF later, when a rigorous PSF is needed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def diffraction_sigmas_um(
    numerical_aperture: float, wavelength_um: float, refractive_index: float
) -> tuple[float, float]:
    """(lateral, axial) Gaussian sigma in µm from diffraction theory.

    Lateral FWHM ≈ 0.51 λ / NA; axial FWHM ≈ 0.88 λ / (n − √(n² − NA²)).
    Returns the widths converted from FWHM to Gaussian sigma.
    """
    na = float(numerical_aperture)
    lam = float(wavelength_um)
    n = float(refractive_index)
    if na <= 0 or lam <= 0 or n <= 0:
        raise ValueError("numerical_aperture, wavelength and refractive_index must be > 0")
    na = min(na, n - 1e-6)  # NA cannot exceed the immersion index
    fwhm_lateral = 0.51 * lam / na
    fwhm_axial = 0.88 * lam / (n - math.sqrt(max(n * n - na * na, 1e-9)))
    return fwhm_lateral * _FWHM_TO_SIGMA, fwhm_axial * _FWHM_TO_SIGMA


def _gaussian_kernel(sigma_vox: tuple[float, ...], max_radius: int) -> np.ndarray:
    radii = [int(min(max_radius, max(1, math.ceil(3.0 * s)))) for s in sigma_vox]
    axes = [np.arange(-r, r + 1) for r in radii]
    grids = np.meshgrid(*axes, indexing="ij")
    exponent = np.zeros_like(grids[0], dtype=float)
    for a, s in enumerate(sigma_vox):
        exponent = exponent + (grids[a] ** 2) / (2.0 * s * s)
    psf = np.exp(-exponent)
    total = psf.sum()
    return psf / total if total > 0 else psf


def build_psf(
    spacing_um: tuple[float, ...],
    ndim: int,
    lateral_sigma_um: float,
    axial_sigma_um: float | None,
    max_radius: int = 15,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Anisotropic Gaussian PSF kernel (sums to 1) for a 2D/3D image.

    Sigmas are converted from µm to per-axis voxels using ``spacing_um`` and
    floored at 0.5 vox (a PSF narrower than a voxel is meaningless).
    """
    if ndim == 2:
        sy, sx = spacing_um
        sigma_vox = (lateral_sigma_um / sy, lateral_sigma_um / sx)
    elif ndim == 3:
        sz, sy, sx = spacing_um
        ax = axial_sigma_um if axial_sigma_um else lateral_sigma_um
        sigma_vox = (ax / sz, lateral_sigma_um / sy, lateral_sigma_um / sx)
    else:
        raise ValueError("deconvolution supports 2D or 3D images")
    sigma_vox = tuple(max(0.5, float(s)) for s in sigma_vox)
    return _gaussian_kernel(sigma_vox, max_radius), sigma_vox


def richardson_lucy_deconvolve(
    image: np.ndarray, psf: np.ndarray, iterations: int
) -> np.ndarray:
    """Richardson-Lucy deconvolution, preserving the original intensity range.

    The image is normalised to [0, 1] for the (non-negative, Poisson-suited)
    iteration and rescaled back, so quantitative intensities stay comparable.
    """
    from skimage.restoration import richardson_lucy

    img = image.astype(np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return image.astype(np.float32)
    norm = (img - lo) / (hi - lo)
    out = richardson_lucy(norm, psf, num_iter=int(iterations), clip=False)
    out = np.clip(out, 0.0, None)
    return (out * (hi - lo) + lo).astype(np.float32)


def resolve_wavelength_nm(
    metadata: dict[str, Any] | None, override: float | None
) -> tuple[float | None, str | None]:
    """Emission wavelength (nm): an explicit override wins, else parse it from
    layer metadata. Returns (value, source) or (None, None)."""
    if override:
        return float(override), "parameter"
    from imajin.io.channel_metadata import wavelength_nm

    md = metadata or {}
    for key in (
        "emission_wavelength_nm", "emission_wavelength",
        "wavelength_nm", "wavelength",
        "excitation_wavelength_nm", "excitation_wavelength",
    ):
        if key in md:
            w = wavelength_nm(md[key])
            if w:
                return float(w), f"metadata:{key}"
    return None, None
