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


# Signed latitude bands (degrees, ascending) rather than |lat| bands --
# hemispheres genuinely differ (e.g. storm tracks, land/ocean fraction),
# so folding them together would hide asymmetries. Degree-based, not
# equal-row-count, so bands mean the same physical thing regardless of
# which target grid (64x32 coarse, 240x121, 1440x721 native) they're
# applied to.
DEFAULT_LAT_BAND_EDGES = (-90, -60, -20, 20, 60, 90)
DEFAULT_LAT_BAND_LABELS = ("60S-90S", "20S-60S", "20S-20N", "20N-60N", "60N-90N")


def lat_banded_rmse_per_channel(
    pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, lat_values: np.ndarray,
    band_edges=DEFAULT_LAT_BAND_EDGES,
) -> np.ndarray:
    """
    Same lat-weighted RMSE as lat_weighted_rmse_per_channel, computed
    separately WITHIN each latitude band instead of pooled over the whole
    globe. A single global RMSE can't tell you whether error is spread
    evenly or concentrated at the poles / in the tropics -- this does.

    Args:
        pred, target: shape (B, C, H, W)
        weights: shape (H,), from lat_weights() -- the usual per-row
            cos(latitude) area weight, applied within each band instead
            of globally (so within-band weighting is still correct even
            though the band itself isn't area-equal).
        lat_values: shape (H,), real latitude in degrees for each row of
            pred/target, same row order as weights.
        band_edges: ascending degree boundaries defining len(band_edges)-1
            bands (default DEFAULT_LAT_BAND_EDGES/LABELS above).

    Returns:
        np.ndarray shape (n_bands, C). A band with no rows in lat_values
        (e.g. too few latitude rows for this grid to hit every band) is
        NaN there rather than raising.
    """
    n_bands = len(band_edges) - 1
    n_channels = pred.shape[1]
    out = np.full((n_bands, n_channels), np.nan, dtype=np.float32)
    for b in range(n_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        # Last band is inclusive at both ends so lat=90 isn't dropped;
        # every other band is half-open to avoid double-counting a row
        # that sits exactly on a shared boundary.
        if b == n_bands - 1:
            mask = (lat_values >= lo) & (lat_values <= hi)
        else:
            mask = (lat_values >= lo) & (lat_values < hi)
        if not mask.any():
            continue
        mask_t = torch.from_numpy(mask)
        out[b] = lat_weighted_rmse_per_channel(pred[:, :, mask_t], target[:, :, mask_t],
                                                weights[mask_t]).numpy()
    return out


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
