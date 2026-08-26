"""
Normalisation, axis-flip, and relative-humidity-derivation helpers, used
both for training data and inference data. Kept separate from the Dataset
classes so they can be tested on their own, without needing GCS access.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


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

    Used both at inference time (inference/predict.py::load_inference_data,
    for a store like native_highres that only provides specific humidity)
    and at TRAINING time (data/gcs_dataset.py, when data.gcs_bucket_path
    points at such a store) -- the same physical quantity needs deriving
    the same way regardless of which pipeline is asking for it.

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


def flip_axes(arr: np.ndarray, flip_lat: bool, flip_lon: bool) -> np.ndarray:
    """
    Flip spatial axes of an array shaped (T, C, H, W), H=latitude,
    W=longitude. Different stores order their lat/lon axes differently
    (e.g. north-to-south vs south-to-north) — flip_lat/flip_lon correct
    for that per store. See configs/baseline_fno.yaml for which stores
    need which flip.
    """
    if flip_lat:
        arr = arr[..., ::-1, :]
    if flip_lon:
        arr = arr[..., :, ::-1]
    return np.ascontiguousarray(arr)


def normalise(
    arr: np.ndarray, stats: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Per-channel standardisation: (x - mean) / std.

    If `stats` is None, computes mean/std from `arr` itself (use this ONLY
    on the training split). Otherwise applies the given stats (use this for
    val/test/inference so they're normalised identically to train).

    Args:
        arr: shape (T, C, H, W)
        stats: optional {"mean": (C,), "std": (C,)} computed elsewhere.

    Returns:
        (normalised_array, stats_used)
    """
    if stats is None:
        mean = arr.mean(axis=(0, 2, 3))
        std = arr.std(axis=(0, 2, 3))
        std = np.where(std < 1e-8, 1.0, std)  # guard against dead channels
        stats = {"mean": mean, "std": std}

    mean = stats["mean"].reshape(1, -1, 1, 1)
    std = stats["std"].reshape(1, -1, 1, 1)
    return (arr - mean) / std, stats


def denormalise(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of `normalise` — needed before computing physical-unit
    metrics or plotting real fields."""
    mean = stats["mean"].reshape(1, -1, 1, 1)
    std = stats["std"].reshape(1, -1, 1, 1)
    return arr * std + mean
