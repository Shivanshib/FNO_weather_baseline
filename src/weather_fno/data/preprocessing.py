"""
Normalisation and axis-flip helpers, used both for training data and
inference data. Kept separate from the Dataset classes so they can be
tested on their own, without needing GCS access.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


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
