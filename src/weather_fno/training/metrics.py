"""
Latitude-weighted error metrics used for both the training loss and
evaluation. Grid cells shrink towards the poles on an equiangular lat/lon
grid, so errors there need down-weighting or the poles dominate the score.
"""

from __future__ import annotations

import numpy as np
import torch


def lat_weights(lat_degrees: np.ndarray) -> torch.Tensor:
    """
    Area weight per latitude row: proportional to cos(latitude), scaled to
    average 1. Standard WeatherBench2/FourCastNet convention.

    Args:
        lat_degrees: 1D latitude values in degrees, shape (H,).

    Returns:
        Tensor (H,) -- broadcast-multiply into a per-gridpoint error before
        averaging.
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
    Same as lat_weighted_rmse, but one value per channel instead of
    averaged over all of them -- needed since channels have very different
    natural scales (e.g. geopotential vs temperature), so a single
    combined number isn't very meaningful for scorecards/plots.

    Args:
        pred, target: shape (B, C, H, W)
        weights: shape (H,)

    Returns:
        Tensor of shape (C,).
    """
    w = weights.view(1, 1, -1, 1).to(pred.device)
    mse_per_channel = (w * (pred - target) ** 2).mean(dim=(0, 2, 3))
    return torch.sqrt(mse_per_channel)


def lat_weighted_acc(
    pred: torch.Tensor, target: torch.Tensor, climatology: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Anomaly Correlation Coefficient, per channel, latitude-weighted --
    the (area-weighted) Pearson correlation between the forecast's anomaly
    (pred - climatology) and the true anomaly (target - climatology).
    Standard WeatherBench2/FourCastNet convention: 1 = perfect anomaly
    correlation, 0 = no skill beyond climatology, so it directly answers
    "is the model adding anything beyond just knowing the season" in a way
    plain RMSE can't (a model that's just slightly-blurred climatology can
    still have a low RMSE while having ACC near 0).

    Args:
        pred, target, climatology: shape (B, C, H, W) -- climatology is
            THIS forecast's own valid-time climatological field (from
            data/climatology.py::query_climatology, looked up per lead
            time before calling this).
        weights: shape (H,)

    Returns:
        Tensor of shape (C,), each value in [-1, 1].
    """
    w = weights.view(1, 1, -1, 1).to(pred.device)
    pred_anom = pred - climatology
    target_anom = target - climatology
    numerator = (w * pred_anom * target_anom).sum(dim=(0, 2, 3))
    denominator = torch.sqrt(
        (w * pred_anom ** 2).sum(dim=(0, 2, 3)) * (w * target_anom ** 2).sum(dim=(0, 2, 3))
    )
    return numerator / denominator
