"""
Preprocessing specific to inference-time data. Axis-flip, normalisation,
and relative-humidity derivation are just thin re-exports of
data/preprocessing.py (relative humidity derivation moved there once
TRAINING gained the same need -- see gcs_dataset.py -- not just inference)
-- no need to duplicate those, they work the same regardless of where the
array came from. Kept importable from here too so existing call sites
(inference/predict.py) don't need to change.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from weather_fno.data.preprocessing import compute_relative_humidity, flip_axes, normalise

__all__ = ["compute_relative_humidity", "flip_axes_inference", "normalise_for_inference"]


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
