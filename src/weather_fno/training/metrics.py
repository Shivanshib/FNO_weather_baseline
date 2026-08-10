"""
Weather-forecasting metrics.

Only latitude-weighted MSE is implemented as a working placeholder for now —
fill in the rest (RMSE, ACC, per-variable breakdowns, etc.) once you've
confirmed exactly which ones you need. All of WeatherBench2's deterministic
metrics use the same area-weighting idea, so lat_weights() below is the
piece the rest will build on.
"""

from __future__ import annotations

import numpy as np
import torch


def lat_weights(lat_degrees: np.ndarray) -> torch.Tensor:
    """
    Area weights for an equiangular lat/lon grid, following the WeatherBench2
    / FourCastNet convention: weight proportional to cos(latitude),
    normalised to mean 1 over the latitude axis. Needed because grid cells
    shrink towards the poles on an equiangular grid — without this, polar
    regions are over-weighted relative to their true area.

    Args:
        lat_degrees: 1D array of latitude values in degrees, shape (H,).

    Returns:
        Tensor of shape (H,) — multiply elementwise (broadcast over the
        latitude axis) into any per-gridpoint error before averaging.
    """
    lat_rad = np.deg2rad(lat_degrees)
    w = np.cos(lat_rad)
    w = w / w.mean()
    return torch.from_numpy(w).float()


def lat_weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Latitude-weighted MSE.

    Args:
        pred, target: shape (B, C, H, W)
        weights: shape (H,), from lat_weights()

    Returns:
        Scalar tensor.
    """
    w = weights.view(1, 1, -1, 1).to(pred.device)
    return (w * (pred - target) ** 2).mean()


def lat_weighted_rmse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Latitude-weighted RMSE — same weighting as lat_weighted_mse, reported
    in the data's actual units instead of squared units. More interpretable
    for comparing against published baselines."""
    return torch.sqrt(lat_weighted_mse(pred, target, weights))


def lat_weighted_rmse_per_channel(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Same as lat_weighted_rmse but keeps the channel dimension instead of
    averaging over it — one RMSE value per channel, in that channel's own
    physical units. Different channels have very different natural
    magnitudes (e.g. geopotential in the thousands vs temperature in the
    hundreds), so per-channel is what you actually want for verification
    plots/scorecards rather than one number averaged across all of them.

    Args:
        pred, target: shape (B, C, H, W)
        weights: shape (H,)

    Returns:
        Tensor of shape (C,).
    """
    w = weights.view(1, 1, -1, 1).to(pred.device)
    mse_per_channel = (w * (pred - target) ** 2).mean(dim=(0, 2, 3))
    return torch.sqrt(mse_per_channel)


# TODO: fill in once channels/levels are finalised —
#   - lat_weighted_acc   (anomaly correlation coefficient, needs a climatology)
