"""
Radially-averaged 2D power spectrum -- a standard diagnostic for comparing
spectral content between fields (forecast vs ground truth, or the same
field at different resolutions). This is a flat 2D FFT on the lat/lon
grid, not a proper spherical-harmonic transform -- an approximation, but
good enough for spotting things like "is the model blurry".
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def radial_power_spectrum(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D power spectrum of a real-valued (H, W) field, averaged radially
    over wavenumber magnitude.

    Wavenumber is in "cycles per full domain" (k=1 = one full wave across
    the whole grid), not cycles per gridpoint -- this makes spectra from
    DIFFERENT resolutions comparable on the same x-axis, and matches the
    convention an FNO's own n_modes uses.

    Uses norm="forward" (FFT divided by N=H*W) instead of numpy's default.
    This matters because we compare spectra across different grid sizes:
    with the default (unnormalised) FFT, the same physical signal sampled
    on a bigger grid gets a bigger squared magnitude just from having more
    points -- norm="forward" cancels that out, so power at a given
    wavenumber only depends on the actual signal, not the grid size.

    Returns:
        (wavenumber, power) -- both 1D arrays. power[i] is the mean
        squared FFT magnitude of all frequency components whose magnitude
        falls in [wavenumber[i], wavenumber[i] + 1).
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
