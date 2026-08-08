"""
Preprocessing for higher-resolution inference data.

Separate from `data/preprocessing.py` because the inference-time source is a
different (higher-resolution) GCS store, may have different raw variables
available, and needs specific humidity DERIVED rather than read directly.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_specific_humidity(
    relative_humidity: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """
    Derive specific humidity from relative humidity, temperature and
    pressure at the highest inference resolution.

    TODO: confirm units coming out of the higher-res store before filling
    this in — this needs:
      1. Saturation vapour pressure from temperature (e.g. Tetens' or
         Clausius-Clapeyron approximation).
      2. Actual vapour pressure = relative_humidity * saturation_vapour_pressure.
      3. Specific humidity q = 0.622 * e / (pressure - 0.378 * e)
         (standard meteorological approximation).

    Args:
        relative_humidity: fraction in [0, 1] or percent — confirm which.
        temperature: TODO confirm units (K vs C).
        pressure: TODO confirm units (Pa vs hPa) and whether this is
            surface pressure or per-level pressure.

    Returns:
        Specific humidity array, same shape as inputs.
    """
    raise NotImplementedError(
        "Fill in once the higher-resolution store's raw variables, units "
        "and pressure-level structure are confirmed."
    )


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
