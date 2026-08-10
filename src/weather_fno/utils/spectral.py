"""
Radially-averaged 2D power spectrum -- the standard diagnostic for
comparing spectral content between fields (e.g. forecast vs ground truth,
or the same field at different resolutions), independent of exact grid
size.

This is a PLANAR approximation (flat 2D FFT on the lat/lon grid, not a
proper spherical-harmonic treatment) -- reasonable for a baseline
diagnostic and standard practice in ML weather forecasting papers for
exactly this kind of "is the model blurry / injecting noise" check, but
not a rigorous spherical spectral analysis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def radial_power_spectrum(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D power spectrum of a real-valued (H, W) field, radially averaged
    over wavenumber magnitude.

    Wavenumber is expressed in "cycles per full domain" (k=1 means one
    full wave spanning the whole grid) rather than cycles-per-gridpoint --
    this is what makes spectra from DIFFERENT resolutions directly
    comparable on the same x-axis (the physical domain, e.g. 360 degrees
    of longitude, is the same regardless of how many gridpoints sample
    it), and it's also the same convention an FNO's own `n_modes` uses, so
    the model's own mode cutoff can be marked directly on the same plot.

    Returns:
        (wavenumber, power) -- both 1D arrays. power[i] is the mean
        squared FFT magnitude of all 2D frequency components whose
        magnitude falls in [wavenumber[i], wavenumber[i] + 1).

    Uses norm="forward" (FFT itself divided by N=H*W) rather than numpy's
    default unnormalised transform. This matters specifically because this
    function is meant to compare spectra across DIFFERENT grid sizes (e.g.
    a 1440x721 store vs a 240x121 one): for a single-frequency component
    of fixed physical amplitude A, the default unnormalised FFT's peak
    magnitude scales with N (so squared power scales with N^2); even
    norm="ortho" (dividing by sqrt(N)) leaves a residual factor of N in
    the squared magnitude. Only norm="forward" gives a peak power that's
    independent of how many gridpoints sample the same physical field --
    verified directly against a synthetic single-wavenumber field sampled
    at two different grid sizes (see tests alongside this module).
    """
    h, w = field.shape
    fft = np.fft.fft2(field - field.mean(), norm="forward")
    power_2d = np.abs(fft) ** 2

    ky = np.fft.fftfreq(h) * h  # cycles per full domain (not per gridpoint)
    kx = np.fft.fftfreq(w) * w
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    k_max = min(h, w) // 2
    k_bins = np.arange(0, k_max)
    power_radial = np.zeros(len(k_bins))
    for i, k in enumerate(k_bins):
        mask = (k_mag >= k) & (k_mag < k + 1)
        if mask.any():
            power_radial[i] = power_2d[mask].mean()

    return k_bins, power_radial
