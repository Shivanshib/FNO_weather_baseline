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

from typing import Dict, Optional

import numpy as np
import torch

from weather_fno.config import Config, InferenceTarget
from weather_fno.data.climatology import query_climatology
from weather_fno.data.io import open_dataset
from weather_fno.data.preprocessing import denormalise
from weather_fno.inference.predict import load_inference_data, load_trained_model, rollout
from weather_fno.inference.preprocessing import normalise_for_inference
from weather_fno.training.metrics import lat_weighted_acc, lat_weighted_rmse_per_channel, lat_weights


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
    climatology: Optional[dict] = None,
) -> dict:
    """
    Fetch an initial condition plus forecast_lead_steps of ground truth
    for `target`, run the autoregressive rollout, and score it against
    ground truth AND a persistence baseline (repeat the initial condition
    unchanged -- the standard "is the model better than doing nothing"
    check) at every lead time, per channel.

    climatology (optional, from data/climatology.py::compute_climatology
    or its cached .npz, coarse-grid only -- see that module's docstring
    for why): if given, ALSO scores a climatology baseline (RMSE) and the
    model's anomaly correlation coefficient (ACC) against it, at every
    lead time, per channel. If None, those two keys are simply absent
    from the result -- callers (scripts/evaluate.py) handle that by just
    not plotting them, not by erroring.

    Returns a dict with lead_hours, model_rmse (n_steps, C),
    persistence_rmse (n_steps, C), optionally climatology_rmse (n_steps, C)
    and model_acc (n_steps, C), plus the in-memory arrays needed for
    plotting (initial_condition, forecast, ground_truth -- physical units).
    """
    # 1. Fetch initial condition + real ground truth for every lead time.
    n_steps = cfg.inference.forecast_lead_steps
    arr, time_values = load_inference_data(cfg, target, n_timesteps=n_steps + 1,
                                            start_date=cfg.inference.start_date,
                                            return_time=True)  # (n_steps+1, C, H, W)
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
    climatology_rmse = np.zeros((n_steps, n_channels), dtype=np.float32) if climatology is not None else None
    model_acc = np.zeros((n_steps, n_channels), dtype=np.float32) if climatology is not None else None

    if climatology is not None:
        # One climatology field per lead time's own real valid date (NOT
        # the initial condition's date) -- ground_truth[step] is
        # time_values[step + 1] (index 0 is the initial condition).
        climatology_fields = query_climatology(climatology, time_values[1:])

    initial_t = torch.from_numpy(initial_condition).float()
    for step in range(n_steps):
        gt_step = torch.from_numpy(ground_truth[step:step + 1]).float()
        pred_step = torch.from_numpy(forecast[step:step + 1]).float()
        model_rmse[step] = lat_weighted_rmse_per_channel(pred_step, gt_step, weights).numpy()
        persistence_rmse[step] = lat_weighted_rmse_per_channel(initial_t, gt_step, weights).numpy()

        if climatology is not None:
            clim_step = torch.from_numpy(climatology_fields[step:step + 1]).float()
            climatology_rmse[step] = lat_weighted_rmse_per_channel(clim_step, gt_step, weights).numpy()
            model_acc[step] = lat_weighted_acc(pred_step, gt_step, clim_step, weights).numpy()

    result = {
        "lead_hours": np.arange(1, n_steps + 1) * 6,
        "model_rmse": model_rmse,
        "persistence_rmse": persistence_rmse,
        "initial_condition": initial_condition,
        "forecast": forecast,
        "ground_truth": ground_truth,
    }
    if climatology is not None:
        result["climatology_rmse"] = climatology_rmse
        result["model_acc"] = model_acc
    return result


def run_evaluation(
    cfg: Config, train_stats: Dict[str, np.ndarray], climatology: Optional[dict] = None
) -> Dict[str, dict]:
    """
    Evaluate every target in cfg.inference.targets against ground truth,
    using the same trained checkpoint for all of them. Returns
    {target.name: result} -- see evaluate_target for the result shape.

    climatology (optional) is only ever computed at the TRAINING grid's
    resolution (coarse-only, see data/climatology.py) -- it only gets
    applied to a target whose own resolution matches cfg.data.resolution
    (in practice, the "coarse" target), never to native_highres/1p5deg,
    since a mismatched grid shape would be meaningless (or crash) there.
    Targets it doesn't apply to still get model_rmse/persistence_rmse as
    normal, just without climatology_rmse/model_acc.
    """
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, device)

    results: Dict[str, dict] = {}
    for target in cfg.inference.targets:
        print(f"[{target.name}] fetching {cfg.inference.forecast_lead_steps + 1} timesteps "
              f"(1 initial condition + {cfg.inference.forecast_lead_steps} ground-truth) "
              f"and running the rollout...")
        lat, _ = get_target_lat_lon(cfg, target)
        weights = lat_weights(lat)
        target_climatology = climatology if target.resolution == cfg.data.resolution else None
        results[target.name] = evaluate_target(cfg, target, model, train_stats, weights, device,
                                                 climatology=target_climatology)

        final_model_rmse = results[target.name]["model_rmse"][-1].mean()
        final_persistence_rmse = results[target.name]["persistence_rmse"][-1].mean()
        msg = (f"  done — mean RMSE across all channels at final lead time: "
               f"model={final_model_rmse:.4g}, persistence={final_persistence_rmse:.4g}")
        if target_climatology is not None:
            final_climatology_rmse = results[target.name]["climatology_rmse"][-1].mean()
            final_acc = results[target.name]["model_acc"][-1].mean()
            msg += f", climatology={final_climatology_rmse:.4g}, ACC={final_acc:.3f}"
        print(msg)

    return results
