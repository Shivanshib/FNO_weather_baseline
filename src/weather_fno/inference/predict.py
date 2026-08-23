"""
Shared building blocks for running a trained model on inference-time data:
fetching+preprocessing a store's data, the autoregressive rollout itself,
and loading a trained checkpoint. Used by both scripts/evaluate.py (scores
a rollout against ground truth) and scripts/predict_single_variable.py
(saves a full week of one variable's maps) -- this module has no CLI or
`main()` of its own, it's the common logic both of those build on.

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
) -> np.ndarray:
    """
    Open one inference-time GCS store and build a (T, C, H, W) array
    matching cfg.data.channels, in the same channel order used for
    training.

    Only `n_timesteps` are ever pulled (default 1 — just the initial
    condition for an autoregressive rollout), sliced BEFORE pulling any
    values out of zarr — reading every timestep first (as an earlier
    version of this function did) is infeasible for a multi-decade,
    full-resolution store. Pass a larger n_timesteps (e.g.
    forecast_lead_steps + 1) to also pull ground-truth timesteps for
    scoring a forecast against, as inference/evaluate.py does — still
    bounded and deliberate, never "the whole store".

    start_date: if given, starts from the first available timestep AT OR
    AFTER this date (e.g. "2016-06-01") instead of the store's first
    available timestep overall. Every configured inference target shares
    the same underlying real time index (confirmed directly against the
    stores — all start 1959-01-01), so passing the same start_date to every
    target guarantees they're compared on identical real timestamps.
    Without this, the default (None) uses index 0, which is 1959-01-01 for
    these stores — decades before the training window (2000-2014), which
    mixes "does this generalise to unseen resolution" with "does this
    generalise to a completely different era" in a way that's hard to
    disentangle. Set to a date within/near the training or validation
    period for a cleaner resolution-only comparison.

    Relative humidity (r500, r850) isn't available directly in every
    inference store — target.derive_relative_humidity switches on deriving
    it per pressure level from specific humidity and temperature at that
    same level instead. Both derived channels (500 and 850 hPa) are
    handled here, since each has a different ChannelSpec with a different
    .level. Stores that provide relative_humidity directly (like the
    coarse training store) should leave derive_relative_humidity=False.
    """
    # 1. Open the store and slice down to just the timesteps actually
    # needed, before touching any variable's values.
    ds = open_dataset(target.gcs_bucket_path)
    if start_date is not None:
        ds = ds.sel(time=slice(start_date, None))
    ds = ds.isel(time=slice(0, n_timesteps))

    # 2. Pull each configured channel out, in order -- deriving relative
    # humidity where needed, reading directly otherwise.
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

    # 3. Stack into (T, C, H, W) and apply this target's own orientation
    # correction (independent of the training store's flip settings).
    arr = np.stack(channel_arrays, axis=1)
    arr = flip_axes_inference(arr, target.flip_lat, target.flip_lon)
    return arr


def rollout(model, x0: torch.Tensor, n_steps: int, device, target_mode: str = "direct") -> np.ndarray:
    """
    Autoregressive rollout: feed the model's own output back in as the next
    input, n_steps times. Shared by every caller that needs a multi-step
    forecast (inference/evaluate.py, scripts/predict_single_variable.py) so
    the exact same stepping logic is used everywhere, whether or not the
    result is being scored against ground truth.

    Args:
        x0: normalised initial condition, shape (1, C, H, W).
        n_steps: number of 6-hour steps to roll forward.
        target_mode: "direct" (default) — the model's raw output IS the
            next state, used as-is. "residual" — the model's raw output is
            a DELTA, added onto the current state to reconstruct the next
            one (x + model(x)) before it's fed back in or returned. Pass
            cfg.model.target_mode; must match whatever the loaded
            checkpoint was trained with (load_trained_model enforces this).

    Returns:
        Normalised, FULLY RECONSTRUCTED next-state predictions, shape
        (n_steps, C, H, W), still on CPU as a numpy array — regardless of
        target_mode, every element here is a full state, never a raw
        delta, so every caller (denormalise, scoring, plotting) needs no
        target_mode awareness of its own. NOT denormalised, that's the
        caller's responsibility (callers denormalise at slightly different
        points depending on what else they need to do with the array first).
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
    """Build the model and load cfg.inference.checkpoint_path into it —
    shared by every inference entrypoint so they all load the exact same
    way.

    load_checkpoint() returns None when nothing exists at the given path
    -- correct for Trainer's auto-resume (no checkpoint yet just means
    "start fresh"), but for inference a missing checkpoint must never be
    silent: falling through would leave `model` at its random
    initialisation from build_model() and produce a plausible-looking but
    meaningless forecast with no error at all. Explicitly check and raise.
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
            f"(resolved relative to cwd={os.getcwd()}). The model would otherwise "
            f"silently run with its random initial weights instead of the trained "
            f"ones -- check the path is correct relative to where this is being run from."
        )
    # The checkpoint's weights were trained under WHATEVER target_mode
    # produced them (see ModelConfig.target_mode) -- rollout()'s
    # reconstruction math (x + model(x) vs model(x) alone) depends on
    # getting this right, and a mismatch would silently produce a
    # plausible-LOOKING but meaningless forecast rather than an error, the
    # same failure mode the missing-checkpoint check above guards against.
    # ckpt.get(..., "direct") covers checkpoints saved before target_mode
    # existed at all.
    ckpt_target_mode = ckpt.get("target_mode", "direct")
    if ckpt_target_mode != cfg.model.target_mode:
        raise ValueError(
            f"Checkpoint at '{cfg.inference.checkpoint_path}' was trained with "
            f"target_mode='{ckpt_target_mode}', but this config is set to "
            f"target_mode='{cfg.model.target_mode}'. Use the matching --config/"
            f"--experiment pair for this checkpoint's target_mode."
        )
    return model.eval().to(device)
