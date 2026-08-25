"""
Every plot the project produces, one function each:
  plot_history            train/val loss curves
  plot_rmse_vs_lead_time  model vs persistence (vs climatology, if
                          available) RMSE scorecard
  plot_acc_vs_lead_time   model anomaly correlation coefficient (ACC)
                          scorecard, with a skill-threshold reference line
  plot_forecast_maps      ground truth / forecast / error map grid
  plot_power_spectrum     radial power spectrum comparison
Plus recentre_longitude, a shared helper used before map plots.
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
    coordinate from the store's raw ascending [0, 360) convention into the
    standard Atlantic-centred [-180, 180) convention used by recognisable
    world maps.

    This is DISPLAY ONLY -- training/inference always use the raw [0, 360)
    order (flip_lon=false everywhere is correct, not a bug: the FNO is
    translation-invariant across this circular axis). Plotting the raw
    order directly would put Australia left-of-centre instead of on the
    right, which is technically correct but not how people expect a map
    to look.

    Args:
        *fields: one or more arrays sharing `lon` as their last axis.
        lon: shared 1D longitude coordinate, ascending 0 -> ~360.

    Returns:
        (*recentred_fields, recentred_lon).
    """
    lon = np.asarray(lon)
    lon_shifted = ((lon + 180) % 360) - 180
    order = np.argsort(lon_shifted)
    recentred_fields = tuple(field[..., order] for field in fields)
    return (*recentred_fields, lon_shifted[order])


def _plot_loss_panel(ax, epochs, train_loss, val_loss, label: str, log_scale: bool) -> None:
    ax.plot(epochs, train_loss, label="train", color=cm.viridis(0.2))
    ax.plot(epochs, val_loss, label="val", color=cm.viridis(0.7))
    ax.set_xlabel("epoch")
    ax.set_ylabel("lat-weighted MSE")
    ax.set_title(label)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both" if log_scale else "major")


def plot_history(history: dict, out_path: str, run_name: str = "", log_scale: bool = False,
                  pretrain_epochs: Optional[int] = None) -> None:
    """
    Train/val loss vs epoch. log_scale=True is useful once loss has
    dropped enough that the linear plot flattens out.

    pretrain_epochs (cfg.training.epochs): if given AND history actually
    contains epochs past it (i.e. fine-tuning ran), splits into two
    side-by-side panels -- pretrain and fine-tune -- instead of one
    continuous line. This isn't just cosmetic: fine-tuning sums TWO
    steps' loss (Trainer._run_epoch's n_future_steps branch) while
    pretraining logs a single step's loss, so the two phases are on
    genuinely different scales and a shared axis makes fine-tuning's own
    (real, meaningful) progress within its own phase hard to read once
    it's dwarfed by the jump at the boundary. Omit (or leave None) to get
    the old single-panel behaviour -- e.g. smoke_test.py's runs, which
    force finetune_epochs=0 and so never have a boundary to split on.
    """
    train_loss, val_loss = history["train_loss"], history["val_loss"]
    n_epochs = len(train_loss)
    split = pretrain_epochs is not None and n_epochs > pretrain_epochs

    title = f"Training history {run_name}".strip()
    if log_scale:
        title += " (log scale)"

    if not split:
        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_loss_panel(ax, range(n_epochs), train_loss, val_loss, "", log_scale)
        fig.suptitle(title)
    else:
        fig, (ax_pre, ax_fine) = plt.subplots(1, 2, figsize=(13, 4))
        pre_epochs = range(pretrain_epochs)
        fine_epochs = range(pretrain_epochs, n_epochs)
        _plot_loss_panel(ax_pre, pre_epochs, train_loss[:pretrain_epochs],
                          val_loss[:pretrain_epochs], "pretrain (1-step)", log_scale)
        _plot_loss_panel(ax_fine, fine_epochs, train_loss[pretrain_epochs:],
                          val_loss[pretrain_epochs:], "fine-tune (2-step, summed loss)", log_scale)
        fig.suptitle(title)
        fig.tight_layout()

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
    Small multi-panel scorecard: model RMSE vs a persistence baseline (and
    a climatology baseline, if available) over lead time, one panel per
    headline channel. Separate panels because different channels have
    very different scales (e.g. geopotential vs temperature) -- a shared
    axis would flatten all but the largest one.

    Args:
        result: one target's result dict from inference/evaluate.py
            (needs "lead_hours", "model_rmse" (n_steps, C),
            "persistence_rmse" (n_steps, C); "climatology_rmse"
            (n_steps, C) plotted too if present -- coarse-grid targets
            only, see data/climatology.py).
        channels: cfg.data.channels, used to map headline_channels'
            short_names to column indices.
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

    has_climatology = "climatology_rmse" in result
    for ax, short_name in zip(axes.flat, headline_channels):
        idx = idx_by_short_name[short_name]
        ax.plot(lead_days, result["model_rmse"][:, idx], label="model",
                 color=cm.viridis(0.2), marker="o", markersize=3)
        ax.plot(lead_days, result["persistence_rmse"][:, idx], label="persistence",
                 color=cm.viridis(0.7), marker="o", markersize=3, linestyle="--")
        if has_climatology:
            ax.plot(lead_days, result["climatology_rmse"][:, idx], label="climatology",
                     color=cm.viridis(0.95), marker="o", markersize=3, linestyle=":")
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


def plot_acc_vs_lead_time(
    result: dict,
    channels: List[ChannelSpec],
    headline_channels: Sequence[str],
    out_path: str,
    title: str = "",
    skill_threshold: float = 0.6,
) -> None:
    """
    Same layout as plot_rmse_vs_lead_time, but for the model's Anomaly
    Correlation Coefficient (training/metrics.py::lat_weighted_acc)
    instead of RMSE -- one panel per headline channel, model_acc vs lead
    time, with a horizontal reference line at `skill_threshold`. ACC=0.6
    is the conventional threshold below which a forecast is considered to
    have lost useful skill (WeatherBench2/operational NWP convention).

    Args:
        result: one target's result dict from inference/evaluate.py --
            needs "lead_hours" and "model_acc" (n_steps, C). Only present
            for targets climatology was computed for (coarse-grid only,
            see data/climatology.py) -- callers should skip this plot
            entirely if "model_acc" isn't in the result.
        channels, headline_channels: same as plot_rmse_vs_lead_time.
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
        ax.plot(lead_days, result["model_acc"][:, idx], label="model",
                 color=cm.viridis(0.2), marker="o", markersize=3)
        ax.axhline(skill_threshold, color="gray", linestyle="--",
                   label=f"skill threshold ({skill_threshold})")
        ax.set_title(short_name)
        ax.set_xlabel("lead time (days)")
        ax.set_ylabel("ACC")
        ax.set_ylim(-0.05, 1.05)
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
    selected lead time, for a single channel. Ground truth and forecast
    share a sequential colour scale (viridis); error gets its own
    diverging scale (RdBu_r) centred at zero, since error is signed.

    The shared colour scale's vmin/vmax come from GROUND TRUTH ONLY, never
    from the forecast: if a badly diverged forecast value stretched the
    scale too, every panel (including correct ground truth) would look
    like a flat, washed-out block. Forecast values outside that range just
    saturate to the scale's boundary colour instead, which actually shows
    where the forecast has left the plausible range.

    Args:
        result: one target's result dict from inference/evaluate.py
            (needs "forecast" and "ground_truth", both (n_steps, C, H, W)).
        channel_short_name: which channel to plot, by short_name.
        lat, lon: that target's real coordinate arrays, from
            inference/evaluate.py::get_target_lat_lon.
        lead_step_indices: which lead-time steps (0-indexed) to show.
    """
    idx_by_short_name = {c.short_name: i for i, c in enumerate(channels)}
    ch_idx = idx_by_short_name[channel_short_name]

    forecast = result["forecast"][:, ch_idx]
    ground_truth = result["ground_truth"][:, ch_idx]
    lead_hours = result["lead_hours"]

    # Display-only reorder -- see recentre_longitude's docstring.
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
        ax_gt.imshow(gt_field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
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
            if ax is not ax_gt:
                ax.set_yticks([])  # keep y-ticks only on the leftmost column

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
    utils/spectral.py::radial_power_spectrum) on the same axes -- e.g.
    forecast vs ground truth, or the same field at different resolutions.

    Args:
        spectra: {label: (wavenumber, power)} -- one line per label.
        n_modes_cutoff: if given, draws a vertical line at this wavenumber
            (pass the FNO's n_modes) to show where the model's spectral
            content cuts off.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (k, power) in spectra.items():
        # Skip k=0 (the mean) -- not meaningful on a log axis.
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
