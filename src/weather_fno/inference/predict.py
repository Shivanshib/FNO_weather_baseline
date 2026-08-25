"""
Shared building blocks for running a trained model at inference time:
fetching a store's data, the autoregressive rollout, and loading a trained
checkpoint. Used by both scripts/evaluate.py and
scripts/predict_single_variable.py -- no CLI of its own.

The model is trained on single 6-hour steps, so a multi-day forecast means
running it forecast_lead_steps times, feeding each output back in as the
next input (autoregressive rollout), not one big forward pass.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from weather_fno.config import Config, InferenceTarget
from weather_fno.data.io import open_dataset
from weather_fno.inference.preprocessing import compute_relative_humidity, flip_axes_inference
from weather_fno.models.fno_baseline import build_model
from weather_fno.utils.checkpoint import load_checkpoint


def load_inference_data(
    cfg: Config,
    target: InferenceTarget,
    n_timesteps: int = 1,
    start_date: Optional[str] = None,
    return_time: bool = False,
):
    """
    Open one inference-time GCS store and build a (T, C, H, W) array
    matching cfg.data.channels, in training channel order.

    Args:
        n_timesteps: how many timesteps to pull, sliced BEFORE reading any
            values out of zarr (default 1 = just the initial condition).
            Pass forecast_lead_steps + 1 to also pull ground truth for
            scoring, as inference/evaluate.py does.
        start_date: start from the first timestep at/after this date
            instead of the store's first available one. Every configured
            store shares the same real time index, so using the same
            start_date for every target keeps them comparable. Left as
            None, the default lands on 1959 -- decades before the training
            window, which mixes up "does this generalise to a new
            resolution" with "does this generalise to a different era".
        return_time: also return the real timestamps for each row (needed
            to look up climatology per lead time -- see
            inference/evaluate.py). False by default so every EXISTING
            caller keeps working unchanged.

    Relative humidity isn't available directly in every store --
    target.derive_relative_humidity switches on deriving it from specific
    humidity + temperature instead (see inference/preprocessing.py).

    Returns:
        arr (T, C, H, W), or (arr, time_values) if return_time=True --
        time_values is read from the store directly (not assumed/computed
        from start_date, which can be None) so it's correct regardless of
        which timestep the rollout actually started from.
    """
    # 1. Open the store and slice down to just the timesteps needed.
    ds = open_dataset(target.gcs_bucket_path)
    if start_date is not None:
        ds = ds.sel(time=slice(start_date, None))
    ds = ds.isel(time=slice(0, n_timesteps))
    time_values = ds["time"].values

    # 2. Pull each configured channel, deriving relative humidity where needed.
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
            da = da.transpose("time", cfg.data.lat_dim, cfg.data.lon_dim)
            channel_arrays.append(da.values)

    # 3. Stack into (T, C, H, W) and apply this target's own orientation fix.
    arr = np.stack(channel_arrays, axis=1)
    arr = flip_axes_inference(arr, target.flip_lat, target.flip_lon)
    return (arr, time_values) if return_time else arr


def rollout(model, x0: torch.Tensor, n_steps: int, device, target_mode: str = "direct") -> np.ndarray:
    """
    Autoregressive rollout: feed the model's own output back in as the
    next input, n_steps times.

    Args:
        x0: normalised initial condition, shape (1, C, H, W).
        n_steps: number of 6-hour steps to roll forward.
        target_mode: "direct" -- the model's output IS the next state.
            "residual" -- the model's output is a DELTA, added onto the
            current state (x + model(x)) to get the next one. Must match
            what the loaded checkpoint was trained with (load_trained_model
            checks this).

    Returns:
        Normalised, FULLY RECONSTRUCTED states, shape (n_steps, C, H, W),
        as a numpy array on CPU. Every element is a full state regardless
        of target_mode -- callers never need to know which mode was used.
        Not denormalised; that's the caller's job.
    """
    x = x0.to(device)
    predictions_norm = []
    with torch.no_grad():
        for _ in range(n_steps):
            out = model(x)
            x = x + out if target_mode == "residual" else out
            predictions_norm.append(x.cpu().numpy())
    return np.concatenate(predictions_norm, axis=0)


def load_trained_model(cfg: Config, device):
    """
    Build the model and load cfg.inference.checkpoint_path into it.

    Raises instead of silently falling back to a random-initialised model
    if the checkpoint is missing, or if it was trained under a different
    target_mode than this config uses -- either would otherwise produce a
    plausible-looking but meaningless forecast with no error at all.
    """
    model = build_model(cfg.model)
    ckpt = load_checkpoint(
        str(Path(cfg.inference.checkpoint_path).parent),
        Path(cfg.inference.checkpoint_path).name,
        model,
        device=device,
    )
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found at '{cfg.inference.checkpoint_path}' "
            f"(resolved relative to cwd={os.getcwd()}). Check the path is correct "
            f"relative to where this is being run from."
        )

    ckpt_target_mode = ckpt.get("target_mode", "direct")  # older checkpoints had no target_mode
    if ckpt_target_mode != cfg.model.target_mode:
        raise ValueError(
            f"Checkpoint at '{cfg.inference.checkpoint_path}' was trained with "
            f"target_mode='{ckpt_target_mode}', but this config is set to "
            f"target_mode='{cfg.model.target_mode}'. Use the matching --config/"
            f"--experiment pair for this checkpoint's target_mode."
        )
    return model.eval().to(device)
