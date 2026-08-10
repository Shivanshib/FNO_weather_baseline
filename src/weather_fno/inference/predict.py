"""
Run a multi-step (autoregressive) forecast on higher-resolution data with a
model trained at 64x32.

FNOs are discretization-invariant by construction (the spectral convolution
operates on Fourier modes, not raw grid points), so the same trained model
can in principle be evaluated directly on a higher-resolution grid without
retraining — accuracy at the new resolution should still be validated
against something, though, since this is only a baseline model.

Since the model is trained on single 6-hour steps, a 1-week forecast means
running it forecast_lead_steps times, feeding each prediction back in as
the next input (autoregressive rollout) — not one forward pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from weather_fno.config import Config, InferenceTarget
from weather_fno.data.io import open_dataset
from weather_fno.data.preprocessing import denormalise
from weather_fno.inference.preprocessing import (
    compute_relative_humidity,
    flip_axes_inference,
    normalise_for_inference,
)
from weather_fno.models.fno_baseline import build_model
from weather_fno.utils.checkpoint import load_checkpoint


def load_inference_data(cfg: Config, target: InferenceTarget, n_timesteps: int = 1) -> np.ndarray:
    """
    Open one inference-time GCS store and build a (T, C, H, W) array
    matching cfg.data.channels, in the same channel order used for
    training.

    Only `n_timesteps` are ever pulled (default 1 — just the initial
    condition for the autoregressive rollout in run_inference), sliced
    BEFORE pulling any values out of zarr — reading every timestep first
    (as an earlier version of this function did) is infeasible for a
    multi-decade, full-resolution store. Pass a larger n_timesteps (e.g.
    forecast_lead_steps + 1) to also pull ground-truth timesteps for
    scoring a forecast against, as inference/evaluate.py does — still
    bounded and deliberate, never "the whole store".

    Relative humidity (r500, r850) isn't available directly in every
    inference store — target.derive_relative_humidity switches on deriving
    it per pressure level from specific humidity and temperature at that
    same level instead. Both derived channels (500 and 850 hPa) are
    handled here, since each has a different ChannelSpec with a different
    .level. Stores that provide relative_humidity directly (like the
    coarse training store) should leave derive_relative_humidity=False.
    """
    ds = open_dataset(target.gcs_bucket_path)
    ds = ds.isel(time=slice(0, n_timesteps))

    channel_arrays = []
    for spec in cfg.data.channels:
        if spec.name == "relative_humidity" and target.derive_relative_humidity:
            q = ds["specific_humidity"].sel(level=spec.level)
            t = ds["temperature"].sel(level=spec.level)
            q = q.transpose("time", cfg.data.lat_dim, cfg.data.lon_dim).values
            t = t.transpose("time", cfg.data.lat_dim, cfg.data.lon_dim).values
            rh = compute_relative_humidity(q, t, pressure_hpa=spec.level)
            channel_arrays.append(rh)
        else:
            da = ds[spec.name]
            if spec.level is not None:
                da = da.sel(level=spec.level)
            # Transpose by NAME, same reasoning as data/gcs_dataset.py —
            # correct regardless of this store's actual underlying axis
            # order.
            da = da.transpose("time", cfg.data.lat_dim, cfg.data.lon_dim)
            channel_arrays.append(da.values)

    arr = np.stack(channel_arrays, axis=1)  # (T, C, H, W)
    arr = flip_axes_inference(arr, target.flip_lat, target.flip_lon)
    return arr


def rollout(model, x0: torch.Tensor, n_steps: int, device) -> np.ndarray:
    """
    Autoregressive rollout: feed the model's own output back in as the next
    input, n_steps times. Shared between run_inference (below) and
    inference/evaluate.py so the exact same stepping logic is used whether
    or not the result is being scored against ground truth.

    Args:
        x0: normalised initial condition, shape (1, C, H, W).
        n_steps: number of 6-hour steps to roll forward.

    Returns:
        Normalised predictions, shape (n_steps, C, H, W), still on CPU as
        a numpy array (not denormalised — caller's responsibility, since
        run_inference and evaluate.py denormalise at slightly different
        points).
    """
    x = x0.to(device)
    predictions_norm = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = model(x)
            predictions_norm.append(x.cpu().numpy())
    return np.concatenate(predictions_norm, axis=0)


def load_trained_model(cfg: Config, device):
    """Build the model and load cfg.inference.checkpoint_path into it —
    shared between run_inference and evaluate.py so both load the exact
    same way."""
    model = build_model(cfg.model)
    load_checkpoint(
        str(Path(cfg.inference.checkpoint_path).parent),
        Path(cfg.inference.checkpoint_path).name,
        model,
        device=device,
    )
    return model.eval().to(device)


def run_inference(cfg: Config, train_stats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Runs one autoregressive rollout per target in cfg.inference.targets,
    each starting from that target's own store's first available timestep,
    for cfg.inference.forecast_lead_steps steps (28 steps x 6h = 7 days, at
    the default config value). The same trained model/checkpoint is reused
    across all targets — only the input data and its preprocessing differ.

    TODO: currently only forecasts from a single starting point per target
    (that store's first timestep). Loop over multiple start indices if you
    want several independent forecasts for evaluation.

    Returns a dict of {target.name: predictions} — predictions shaped
    (n_steps, C, H, W) in physical (denormalised) units.
    """
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, device)

    Path(cfg.inference.output_dir).mkdir(parents=True, exist_ok=True)

    all_predictions: Dict[str, np.ndarray] = {}
    for target in cfg.inference.targets:
        arr = load_inference_data(cfg, target)
        arr_norm = normalise_for_inference(arr, train_stats)

        # Initial condition: first available timestep, kept batched (1, C, H, W).
        x0 = torch.from_numpy(arr_norm[0:1]).float()
        predictions_norm = rollout(model, x0, cfg.inference.forecast_lead_steps, device)
        predictions = denormalise(predictions_norm, train_stats)

        out_path = Path(cfg.inference.output_dir) / f"{target.name}_forecast.npy"
        np.save(out_path, predictions)

        lead_hours = cfg.inference.forecast_lead_steps * 6
        print(f"[{target.name}] saved {cfg.inference.forecast_lead_steps}-step forecast "
              f"({lead_hours}h / {lead_hours / 24:.1f} days lead time) to {out_path}")

        all_predictions[target.name] = predictions

    return all_predictions
