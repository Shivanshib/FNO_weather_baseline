"""
Preprocessing for higher-resolution inference data.

Separate from `data/preprocessing.py` because the inference-time source is a
different (higher-resolution) GCS store, may have different raw variables
available, and needs relative humidity DERIVED rather than read directly
(the store provides specific humidity as a prognostic variable, not
relative humidity — unlike the training channels, which use relative
humidity per FourCastNet's variable set).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_relative_humidity(
    specific_humidity: np.ndarray,
    temperature: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    """
    Derive relative humidity (%) from specific humidity and temperature on
    a pressure-level surface.

    pressure_hpa is a constant, not a field — since specific_humidity and
    temperature are already given ON a pressure-level surface (500 or 850
    hPa), that level number IS the pressure at every gridpoint, so there's
    no separate pressure field to load from the store.

    Formula (standard meteorological approximation):
      1. Saturation vapour pressure from temperature (Tetens' formula).
      2. Actual vapour pressure from specific humidity:
         e = q * p / (0.622 + 0.378 * q)
      3. Relative humidity = 100 * e / e_sat

    Args:
        specific_humidity: kg/kg. TODO confirm this against the store —
            some sources give g/kg instead, which would need /1000 first.
        temperature: Kelvin. TODO confirm — Tetens' formula below converts
            to Celsius internally assuming Kelvin input.
        pressure_hpa: the pressure level these fields sit on, e.g. 500 or
            850 (spec.level from the ChannelSpec calling this).

    Returns:
        Relative humidity as a percentage (0-100), same shape as inputs.
    """
    t_celsius = temperature - 273.15
    e_sat = 6.1078 * np.power(10.0, (7.5 * t_celsius) / (237.3 + t_celsius))

    q = specific_humidity
    e = q * pressure_hpa / (0.622 + 0.378 * q)

    rh = 100.0 * e / e_sat
    return np.clip(rh, 0.0, 100.0)


def flip_axes_inference(arr: np.ndarray, flip_lat: bool, flip_lon: bool) -> np.ndarray:
    """Same idea as data/preprocessing.py's flip_axes, kept separate in case
    the higher-resolution store needs different corrections."""
    if flip_lat:
        arr = arr[..., ::-1, :]
    if flip_lon:
        arr = arr[..., :, ::-1]
    return np.ascontiguousarray(arr)


def normalise_for_inference(arr: np.ndarray, train_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Normalise inference-time data using the SAME stats computed on the
    training split — never re-fit stats on inference data.
    """
    mean = train_stats["mean"].reshape(1, -1, 1, 1)
    std = train_stats["std"].reshape(1, -1, 1, 1)
    return (arr - mean) / std
