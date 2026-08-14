"""
Every plot the project produces, one function each:
  plot_history            train/val loss curves, linear or log scale (scripts/train.py, smoke_test.py)
  plot_rmse_vs_lead_time  model vs persistence RMSE scorecard (scripts/evaluate.py)
  plot_forecast_maps      ground truth / forecast / error map grid (scripts/evaluate.py)
  plot_power_spectrum     radial power spectrum comparison (notebooks/plot_forecast_maps.ipynb)
Plus recentre_longitude, a shared helper (not a plot itself) used before
imshow-based map plots -- see its own docstring below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display available on a headless SSH session
import matplotlib.pyplot as plt
from matplotlib import cm

from weather_fno.config import ChannelSpec


def recentre_longitude(*fields: np.ndarray, lon: np.ndarray) -> tuple:
    """
    Reorder one or more (..., W) fields plus their shared 1D longitude
    coordinate from the raw store's ascending [0, 360) convention into the
    standard Atlantic-centred [-180, 180) convention used by recognisable
    world maps (and by notebooks/data_checks.ipynb's xarray/cartopy plots,
    which re-project by real coordinate value regardless of convention).

    Every store this project uses provides longitude ascending 0 -> ~360
    (confirmed directly against GCS -- see flip_lon comments in
    configs/baseline_fno.yaml), and the data pipeline correctly keeps that
    raw order throughout training and inference: flip_lon=false everywhere
    is right, not a bug, and the FNO itself is translation-invariant
    across this circular axis so training is unaffected either way. But
    plotting that raw order directly via imshow's extent produces a
    Greenwich/Africa-centred, Pacific-split map -- technically correct,
    just not the layout most people recognise (e.g. Australia lands
    left-of-centre instead of on the right). This function only reorders
    columns for DISPLAY, never the arrays used for training/inference.

    Args:
        *fields: one or more arrays sharing `lon` as their last axis.
        lon: that shared 1D longitude coordinate, ascending 0 -> ~360.

    Returns:
        (*recentred_fields, recentred_lon) -- same number of fields given,
        plus the recentred longitude array, in that order.
    """
    lon = np.asarray(lon)
    lon_shifted = ((lon + 180) % 360) - 180
    order = np.argsort(lon_shifted)
    recentred_fields = tuple(field[..., order] for field in fields)
    return (*recentred_fields, lon_shifted[order])


def plot_history(history: dict, out_path: str, run_name: str = "", log_scale: bool = False) -> None:
    """
    Train/val loss vs epoch. With log_scale=True, plots the y-axis on a log
    scale instead -- useful once loss has dropped enough that the linear
    version flattens out and hides ongoing improvement.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train", color=cm.viridis(0.2))
    ax.plot(history["val_loss"], label="val", color=cm.viridis(0.7))
    ax.set_xlabel("epoch")
    ax.set_ylabel("lat-weighted MSE")
    title = f"Training history {run_name}".strip()
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both" if log_scale else "major")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_vs_lead_time(
    result: dict,
    channels: List[ChannelSpec],
    headline_channels: Sequence[str],
    out_path: str,
    title: str = "",
) -> None:
    """
    Small multi-panel "scorecard" — one panel per headline channel, model
    RMSE vs a persistence baseline (repeat the initial condition
    unchanged) over lead time. Separate panels rather than one shared axis
    because different channels have very different natural magnitudes
    (e.g. geopotential in the thousands vs temperature in the hundreds) —
    a shared y-axis would flatten all but the largest-magnitude channel.

    Args:
        result: one target's result dict from inference/evaluate.py
            (needs "lead_hours", "model_rmse" (n_steps, C),
            "persistence_rmse" (n_steps, C)).
        channels: cfg.data.channels — used to map headline_channels'
            short_names to column indices into model_rmse/persistence_rmse.
        headline_channels: short_names to plot, e.g. ["t2m", "z500"].
    """
    idx_by_short_name = {c.short_name: i for i, c in enumerate(channels)}
    missing = [c for c in headline_channels if c not in idx_by_short_name]
    if missing:
        raise KeyError(f"headline_channels not found in channels: {missing}")

    lead_days = result["lead_hours"] / 24
    n = len(headline_channels)
    ncols = min(n, 2)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)

    for ax, short_name in zip(axes.flat, headline_channels):
        idx = idx_by_short_name[short_name]
        ax.plot(lead_days, result["model_rmse"][:, idx], label="model",
                 color=cm.viridis(0.2), marker="o", markersize=3)
        ax.plot(lead_days, result["persistence_rmse"][:, idx], label="persistence",
                 color=cm.viridis(0.7), marker="o", markersize=3, linestyle="--")
        ax.set_title(short_name)
        ax.set_xlabel("lead time (days)")
        ax.set_ylabel("lat-weighted RMSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_maps(
    result: dict,
    channels: List[ChannelSpec],
    channel_short_name: str,
    lat: np.ndarray,
    lon: np.ndarray,
    lead_step_indices: Sequence[int],
    out_path: str,
    title: str = "",
) -> None:
    """
    Grid of (ground truth | forecast | error) map panels, one row per
    selected lead time, for a single channel. Ground truth/forecast share
    a colour scale (viridis — a sequential field, not diverging) so they're
    visually comparable; error gets its own diverging colour scale
    (RdBu_r) centred at zero, since error is fundamentally a different
    kind of quantity (signed, meaningfully centred at 0) that viridis
    would misrepresent.

    The shared vmin/vmax is deliberately taken from GROUND TRUTH ONLY, not
    forecast — ground truth is real, physically-bounded weather data, so
    its own range is always a sensible scale. If vmin/vmax were also
    stretched to fit the forecast (as an earlier version of this function
    did), a single badly-diverged forecast value (e.g. a baseline model
    producing implausible values at a resolution it wasn't trained for)
    would blow out the colour scale for EVERY panel, including ground
    truth, making even correct, normally-varying ground truth data look
    like a flat, washed-out block. Forecast values outside the ground
    truth's range simply saturate to the colour scale's boundary colour
    (matplotlib's default `imshow` behaviour for out-of-range values with
    vmin/vmax set) — which is actually more informative here, since
    saturated regions directly show where and how much the forecast has
    left the physically plausible range, rather than hiding it by
    rescaling everything to fit.

    Args:
        result: one target's result dict from inference/evaluate.py
            (needs "forecast" and "ground_truth", both (n_steps, C, H, W)).
        channel_short_name: which channel to plot, by short_name.
        lat, lon: that target's real coordinate arrays (for axis extent —
            see inference/evaluate.py::get_target_lat_values).
        lead_step_indices: which lead-time steps (0-indexed into
            forecast/ground_truth) to show, one row each.
    """
    idx_by_short_name = {c.short_name: i for i, c in enumerate(channels)}
    ch_idx = idx_by_short_name[channel_short_name]

    forecast = result["forecast"][:, ch_idx]
    ground_truth = result["ground_truth"][:, ch_idx]
    lead_hours = result["lead_hours"]

    # Display-only reorder to the recognisable -180/180 map convention --
    # see recentre_longitude's docstring; the underlying data is untouched.
    forecast, ground_truth, lon = recentre_longitude(forecast, ground_truth, lon=lon)

    vmin = ground_truth[lead_step_indices].min()
    vmax = ground_truth[lead_step_indices].max()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    nrows = len(lead_step_indices)
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 3.2 * nrows), squeeze=False)

    for row, step in enumerate(lead_step_indices):
        gt_field = ground_truth[step]
        fc_field = forecast[step]
        error = fc_field - gt_field
        err_abs_max = np.abs(error).max()

        ax_gt, ax_fc, ax_err = axes[row]
        im_gt = ax_gt.imshow(gt_field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                              extent=extent, aspect="auto")
        im_fc = ax_fc.imshow(fc_field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                              extent=extent, aspect="auto")
        im_err = ax_err.imshow(error, origin="lower", cmap="RdBu_r",
                                vmin=-err_abs_max, vmax=err_abs_max, extent=extent, aspect="auto")

        lead_label = f"+{lead_hours[step]}h ({lead_hours[step] / 24:.1f}d)"
        ax_gt.set_ylabel(lead_label)
        if row == 0:
            ax_gt.set_title("ground truth")
            ax_fc.set_title("forecast")
            ax_err.set_title("error (forecast - truth)")
        for ax in (ax_gt, ax_fc, ax_err):
            ax.set_xticks([])
            ax.set_yticks([]) if ax is not ax_gt else None

        fig.colorbar(im_fc, ax=[ax_gt, ax_fc], fraction=0.023, pad=0.01)
        fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.01)

    fig.suptitle(f"{title} — {channel_short_name}".strip(" —"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_power_spectrum(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str,
    title: str = "",
    n_modes_cutoff: Optional[float] = None,
) -> None:
    """
    Log-log plot of one or more radial power spectra (see
    utils/spectral.py::radial_power_spectrum) on the same axes, for
    comparing spectral content — e.g. forecast vs ground truth at a given
    lead time, or the same field across different resolutions.

    Args:
        spectra: {label: (wavenumber, power)} — one line per label.
        n_modes_cutoff: if given, draws a vertical reference line at this
            wavenumber — pass the FNO's own n_modes value to show exactly
            where the model stops retaining spectral content, directly on
            the same plot as the data's actual spectral shape.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (k, power) in spectra.items():
        # Skip k=0 (the DC/mean component) -- not meaningful on a log axis
        # and would dominate the y-range without adding information.
        ax.loglog(k[1:], power[1:], label=label)

    if n_modes_cutoff is not None:
        ax.axvline(n_modes_cutoff, color="gray", linestyle="--",
                   label=f"FNO mode cutoff (k={n_modes_cutoff})")

    ax.set_xlabel("wavenumber (cycles per domain)")
    ax.set_ylabel("power")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
