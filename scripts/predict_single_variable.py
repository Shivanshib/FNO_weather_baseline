"""
Run the full 7-day (28-step) autoregressive forecast for a SINGLE variable,
on every configured inference target, and save the per-lead-time forecast
AND ground truth maps -- not just aggregate metrics -- so they can be
plotted flexibly afterward (see notebooks/plot_forecast_maps.ipynb) without
needing GPU or network access again.

Deliberately restricted to one variable to keep saved file sizes small and
predictable: the full 20-channel array at native 1440x721 resolution would
be several GB (see CODE_REFERENCE.md's disk-quota history) -- one
channel's worth of maps across all 28 lead times is a much safer ~100-200MB
even at native resolution, and gets further reduced via compression.

The model still runs on all 20 channels internally at every step (the
autoregressive rollout needs the full multivariate state to step forward
correctly) -- only the SAVED output is restricted to one channel.

Usage:
    python scripts/predict_single_variable.py --config configs/baseline_fno.yaml --channel t2m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from weather_fno.config import load_config
from weather_fno.data.preprocessing import denormalise
from weather_fno.inference.evaluate import get_target_lat_lon
from weather_fno.inference.predict import load_inference_data, load_trained_model, rollout
from weather_fno.inference.preprocessing import normalise_for_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_fno.yaml")
    parser.add_argument("--channel", type=str, default="t2m",
                         help="short_name of the single channel to forecast, e.g. t2m, z500, mslp")
    args = parser.parse_args()

    cfg = load_config(args.config)

    channel_idx = next((i for i, c in enumerate(cfg.data.channels) if c.short_name == args.channel), None)
    if channel_idx is None:
        available = [c.short_name for c in cfg.data.channels]
        raise ValueError(f"--channel '{args.channel}' not found. Available: {available}")

    train_cache_path = cfg.data.stats_cache_path.replace("normalisation_stats", "train_cache")
    cached = np.load(train_cache_path)
    train_stats = {"mean": cached["mean"], "std": cached["std"]}

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, device)

    n_steps = cfg.inference.forecast_lead_steps
    out_dir = Path(cfg.inference.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in cfg.inference.targets:
        print(f"[{target.name}] fetching {n_steps + 1} timesteps and running the full "
              f"{n_steps}-step ({n_steps * 6}h / {n_steps * 6 / 24:.1f}d) autoregressive rollout...")

        arr = load_inference_data(cfg, target, n_timesteps=n_steps + 1)
        initial_condition = arr[0, channel_idx]  # (H, W)
        ground_truth = arr[1:, channel_idx]      # (n_steps, H, W)

        arr_norm = normalise_for_inference(arr, train_stats)
        x0 = torch.from_numpy(arr_norm[0:1]).float()
        predictions_norm = rollout(model, x0, n_steps, device)
        # Denormalise BEFORE slicing to one channel -- denormalise needs
        # the full per-channel mean/std broadcast, not a single-channel one.
        forecast_full = denormalise(predictions_norm, train_stats)
        forecast = forecast_full[:, channel_idx]  # (n_steps, H, W)

        lat, lon = get_target_lat_lon(cfg, target)
        lead_hours = np.arange(1, n_steps + 1) * 6

        out_path = out_dir / f"{target.name}_{args.channel}_forecast_maps.npz"
        np.savez_compressed(
            out_path,
            initial_condition=initial_condition,
            ground_truth=ground_truth,
            forecast=forecast,
            lead_hours=lead_hours,
            lat=lat,
            lon=lon,
            channel_short_name=args.channel,
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"[{target.name}] saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
