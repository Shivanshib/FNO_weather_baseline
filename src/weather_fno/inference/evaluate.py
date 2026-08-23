"""
Score a trained model's autoregressive forecast against ground truth.

Separate from predict.py because scoring needs GROUND TRUTH at every lead
time, not just an initial condition -- predict.py stays focused on
producing a forecast, this module is about how good it was.

Keeps the full forecast/ground-truth arrays in memory only, never writing
them to disk -- a single native-resolution array is already ~2GB, so
saving that (twice) risks a disk-quota problem for no benefit. Only the
small per-lead-time metrics and the resulting plots get saved
(scripts/evaluate.py does that).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from weather_fno.config import Config, InferenceTarget
from weather_fno.data.io import open_dataset
from weather_fno.data.preprocessing import denormalise
from weather_fno.inference.predict import load_inference_data, load_trained_model, rollout
from weather_fno.inference.preprocessing import normalise_for_inference
from weather_fno.training.metrics import lat_weighted_rmse_per_channel, lat_weights


def get_target_lat_lon(cfg: Config, target: InferenceTarget):
    """Real lat/lon coordinate arrays for `target`'s grid, flipped to
    match the row/column order load_inference_data actually produces.
    Metadata-only -- no array data downloaded. Returns (lat_values, lon_values)."""
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
    Fetch an initial condition plus forecast_lead_steps of ground truth
    for `target`, run the autoregressive rollout, and score it against
    ground truth AND a persistence baseline (repeat the initial condition
    unchanged -- the standard "is the model better than doing nothing"
    check) at every lead time, per channel.

    Returns a dict with lead_hours, model_rmse (n_steps, C),
    persistence_rmse (n_steps, C), plus the in-memory arrays needed for
    plotting (initial_condition, forecast, ground_truth -- physical units).
    """
    # 1. Fetch initial condition + real ground truth for every lead time.
    n_steps = cfg.inference.forecast_lead_steps
    arr = load_inference_data(cfg, target, n_timesteps=n_steps + 1,
                               start_date=cfg.inference.start_date)  # (n_steps+1, C, H, W)
    initial_condition = arr[0:1]
    ground_truth = arr[1:]

    # 2. Autoregressive rollout from that same initial condition.
    arr_norm = normalise_for_inference(arr, train_stats)
    x0 = torch.from_numpy(arr_norm[0:1]).float()
    predictions_norm = rollout(model, x0, n_steps, device, target_mode=cfg.model.target_mode)
    forecast = denormalise(predictions_norm, train_stats)

    # 3. Score model AND persistence against ground truth, per lead time.
    n_channels = len(cfg.data.channels)
    model_rmse = np.zeros((n_steps, n_channels), dtype=np.float32)
    persistence_rmse = np.zeros((n_steps, n_channels), dtype=np.float32)

    initial_t = torch.from_numpy(initial_condition).float()
    for step in range(n_steps):
        gt_step = torch.from_numpy(ground_truth[step:step + 1]).float()
        pred_step = torch.from_numpy(forecast[step:step + 1]).float()
        model_rmse[step] = lat_weighted_rmse_per_channel(pred_step, gt_step, weights).numpy()
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
    """Evaluate every target in cfg.inference.targets against ground
    truth, using the same trained checkpoint for all of them. Returns
    {target.name: result} -- see evaluate_target for the result shape."""
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
