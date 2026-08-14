"""
Preprocessing for inference-time data (any configured InferenceTarget:
native full-resolution, 1.5deg, or the coarse store itself).

Separate module from `data/preprocessing.py` specifically for the one
thing that's genuinely different at inference time: relative humidity may
need to be DERIVED from specific humidity + temperature rather than read
directly (some stores provide specific humidity as the prognostic
variable instead). The axis-flip and normalisation steps below are NOT
reimplemented here — they delegate straight to `data/preprocessing.py`,
since flipping/normalising an array doesn't actually depend on where that
array came from; keeping two copies of the same three-line function
around was pure duplication with nothing inference-specific about it.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from weather_fno.data.preprocessing import flip_axes, normalise


def compute_relative_humidity(
    specific_humidity: np.ndarray,
    temperature: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    """
    Derive relative humidity (fraction, 0-1) from specific humidity and
    temperature on a pressure-level surface.

    pressure_hpa is a constant, not a field — since specific_humidity and
    temperature are already given ON a pressure-level surface (500 or 850
    hPa), that level number IS the pressure at every gridpoint, so there's
    no separate pressure field to load from the store.

    Formula (standard meteorological approximation):
      1. Saturation vapour pressure from temperature (Tetens' formula).
      2. Actual vapour pressure from specific humidity:
         e = q * p / (0.622 + 0.378 * q)
      3. Relative humidity = e / e_sat

    Output is a FRACTION (0-1), matching the convention every source that
    provides relative_humidity directly (the coarse training store,
    1p5deg) already uses -- confirmed live against those stores'
    real values (2026-08-14). This used to multiply by 100 and return a
    percentage instead, which silently fed native_highres's derived
    r500/r850 into the model at ~100x the scale the training
    normalisation stats (fit on the coarse store's 0-1 values) expect --
    see CODE_REFERENCE.md.

    Args:
        specific_humidity: kg/kg. CONFIRMED (2026-08-14) directly against
            the native store's own `units` attribute ("kg kg**-1").
        temperature: Kelvin. CONFIRMED (2026-08-14) directly against the
            native store's own `units` attribute ("K") -- Tetens' formula
            below converts to Celsius internally assuming Kelvin input.
        pressure_hpa: the pressure level these fields sit on, e.g. 500 or
            850 (spec.level from the ChannelSpec calling this).

    Returns:
        Relative humidity as a fraction (0-1), same shape as inputs.
    """
    t_celsius = temperature - 273.15
    e_sat = 6.1078 * np.power(10.0, (7.5 * t_celsius) / (237.3 + t_celsius))

    q = specific_humidity
    e = q * pressure_hpa / (0.622 + 0.378 * q)

    rh = e / e_sat
    return np.clip(rh, 0.0, 1.0)


def flip_axes_inference(arr: np.ndarray, flip_lat: bool, flip_lon: bool) -> np.ndarray:
    """Thin alias for data/preprocessing.py::flip_axes, kept as a separate
    name in inference/ call sites for readability (makes it clear at a
    glance which store's own flip settings are being applied)."""
    return flip_axes(arr, flip_lat, flip_lon)


def normalise_for_inference(arr: np.ndarray, train_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Normalise inference-time data using the SAME stats computed on the
    training split — never re-fit stats on inference data. Thin wrapper
    around data/preprocessing.py::normalise (which returns (array, stats)
    since it can also FIT stats) — this always applies existing stats and
    returns just the array, which is all every inference call site needs.
    """
    return normalise(arr, stats=train_stats)[0]
