"""Plot training history (loss curves) and forecast verification plots."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display available on a headless SSH session
import matplotlib.pyplot as plt
from matplotlib import cm

from weather_fno.config import ChannelSpec


def plot_history(history: dict, out_path: str, run_name: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train", color=cm.viridis(0.2))
    ax.plot(history["val_loss"], label="val", color=cm.viridis(0.7))
    ax.set_xlabel("epoch")
    ax.set_ylabel("lat-weighted MSE")
    ax.set_title(f"Training history {run_name}".strip())
    ax.legend()
    ax.grid(alpha=0.3)

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

    vmin = min(forecast[lead_step_indices].min(), ground_truth[lead_step_indices].min())
    vmax = max(forecast[lead_step_indices].max(), ground_truth[lead_step_indices].max())
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
