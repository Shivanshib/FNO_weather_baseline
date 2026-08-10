"""
Score a trained model's autoregressive forecast against ground truth.

Separate from predict.py because this needs GROUND TRUTH at every lead
time (not just an initial condition) to compare against — predict.py stays
focused on "produce a forecast", this module is "how good was it".

Deliberately keeps the full forecast/ground-truth arrays in memory only,
never writing them to disk: for the native full-resolution target, a
single (forecast_lead_steps, 20, 721, 1440) float32 array is already ~2GB,
and writing that (twice — forecast AND ground truth) risks exactly the
kind of disk-quota failure the training pipeline hit earlier. Only the
small per-lead-time metrics (a few KB) and the resulting plot images get
persisted — see scripts/evaluate.py.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from weather_fno.config import Config, InferenceTarget
from weather_fno.data.io import open_dataset
from weather_fno.data.preprocessing import denormalise
from weather_fno.inference.predict import load_inference_data, load_trained_model, rollout
from weather_fno.inference.preprocessing import normalise_for_inference
from weather_fno.training.metrics import lat_weighted_rmse_per_channel, lat_weights


def get_target_lat_lon(cfg: Config, target: InferenceTarget):
    """
    Real lat/lon coordinate values for `target`'s own grid, flipped to
    match the row/column order load_inference_data actually produces for
    that target (same reasoning as GCSWeatherDataset.lat_values for the
    training grid) — a metadata-only read, no array data downloaded.

    Returns (lat_values, lon_values).
    """
    ds = open_dataset(target.gcs_bucket_path)
    lat_values = ds[cfg.data.lat_dim].values
    lon_values = ds[cfg.data.lon_dim].values
    if target.flip_lat:
        lat_values = lat_values[::-1]
    if target.flip_lon:
        lon_values = lon_values[::-1]
    return lat_values, lon_values


def evaluate_target(
    cfg: Config,
    target: InferenceTarget,
    model,
    train_stats: Dict[str, np.ndarray],
    weights: torch.Tensor,
    device,
) -> dict:
    """
    Fetches an initial condition PLUS forecast_lead_steps of ground truth
    for `target`, runs the same autoregressive rollout run_inference uses,
    and scores it against ground truth AND a persistence baseline
    (repeating the initial condition unchanged at every lead time — the
    standard "is the model adding any skill at all beyond doing nothing"
    check) at every lead time, per channel.

    Returns a dict with lead_hours, model_rmse (n_steps, C),
    persistence_rmse (n_steps, C), plus the in-memory arrays needed for
    plotting (initial_condition, forecast, ground_truth — physical units,
    NOT persisted to disk by this function).
    """
    n_steps = cfg.inference.forecast_lead_steps
    arr = load_inference_data(cfg, target, n_timesteps=n_steps + 1)  # (n_steps+1, C, H, W)
    initial_condition = arr[0:1]
    ground_truth = arr[1:]

    arr_norm = normalise_for_inference(arr, train_stats)
    x0 = torch.from_numpy(arr_norm[0:1]).float()
    predictions_norm = rollout(model, x0, n_steps, device)
    forecast = denormalise(predictions_norm, train_stats)

    n_channels = len(cfg.data.channels)
    model_rmse = np.zeros((n_steps, n_channels), dtype=np.float32)
    persistence_rmse = np.zeros((n_steps, n_channels), dtype=np.float32)

    initial_t = torch.from_numpy(initial_condition).float()
    for step in range(n_steps):
        gt_step = torch.from_numpy(ground_truth[step:step + 1]).float()
        pred_step = torch.from_numpy(forecast[step:step + 1]).float()
        model_rmse[step] = lat_weighted_rmse_per_channel(pred_step, gt_step, weights).numpy()
        # Persistence: "predict no change from the initial condition" —
        # computed directly against initial_t rather than materialising a
        # (n_steps, C, H, W) repeated array, which would double memory use
        # for no reason.
        persistence_rmse[step] = lat_weighted_rmse_per_channel(initial_t, gt_step, weights).numpy()

    return {
        "lead_hours": np.arange(1, n_steps + 1) * 6,
        "model_rmse": model_rmse,
        "persistence_rmse": persistence_rmse,
        "initial_condition": initial_condition,
        "forecast": forecast,
        "ground_truth": ground_truth,
    }


def run_evaluation(cfg: Config, train_stats: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Evaluates every target in cfg.inference.targets against ground
    truth, using the same trained checkpoint for all of them. Returns
    {target.name: result} — see evaluate_target for the result shape."""
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, device)

    results: Dict[str, dict] = {}
    for target in cfg.inference.targets:
        print(f"[{target.name}] fetching {cfg.inference.forecast_lead_steps + 1} timesteps "
              f"(1 initial condition + {cfg.inference.forecast_lead_steps} ground-truth) "
              f"and running the rollout...")
        lat, _ = get_target_lat_lon(cfg, target)
        weights = lat_weights(lat)
        results[target.name] = evaluate_target(cfg, target, model, train_stats, weights, device)

        final_model_rmse = results[target.name]["model_rmse"][-1].mean()
        final_persistence_rmse = results[target.name]["persistence_rmse"][-1].mean()
        print(f"  done — mean RMSE across all channels at final lead time: "
              f"model={final_model_rmse:.4g}, persistence={final_persistence_rmse:.4g}")

    return results
