"""
Preprocessing specific to inference-time data. The one real difference
from training data: some inference stores don't provide relative humidity
directly, so it has to be derived from specific humidity + temperature.
Axis-flip and normalisation are just thin wrappers around
data/preprocessing.py -- no need to duplicate those, they work the same
regardless of where the array came from.
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
    temperature on a pressure-level surface, using Tetens' formula for
    saturation vapour pressure.

    pressure_hpa is a constant, not a field: specific_humidity and
    temperature are already given ON a pressure level (500 or 850 hPa), so
    that level number is the pressure everywhere in the field.

    Must return a FRACTION (0-1), not a percentage -- that's the
    convention every store that provides relative_humidity directly
    already uses, and the model was trained on that convention.

    Args:
        specific_humidity: kg/kg.
        temperature: Kelvin.
        pressure_hpa: the pressure level these fields are on (e.g. 500).

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
