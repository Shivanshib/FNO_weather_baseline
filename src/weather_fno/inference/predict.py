"""
Run inference on higher-resolution data with a model trained at 64x32.

FNOs are discretization-invariant by construction (the spectral convolution
operates on Fourier modes, not raw grid points), so the same trained model
can in principle be evaluated directly on a higher-resolution grid without
retraining — accuracy at the new resolution should still be validated
against something, though, since this is only a baseline model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import xarray as xr

from weather_fno.config import Config
from weather_fno.data.preprocessing import denormalise
from weather_fno.inference.preprocessing import (
    compute_specific_humidity,
    flip_axes_inference,
    normalise_for_inference,
)
from weather_fno.models.fno_baseline import build_model
from weather_fno.utils.checkpoint import load_checkpoint


def load_inference_data(cfg: Config) -> np.ndarray:
    """
    Open the higher-resolution GCS store, derive specific humidity, and
    apply the same axis corrections used at training time.

    TODO: fill in the actual variable names available at the higher
    resolution once confirmed.
    """
    ds = xr.open_zarr(cfg.inference.gcs_bucket_path, chunks={"time": 1}, storage_options={"token":"anon"})

    channel_arrays = []
    for spec in cfg.data.channels:
        if spec.short_name == "specific_humidity" and cfg.inference.derive_specific_humidity:
            q = compute_specific_humidity(
                relative_humidity=ds["TODO_relative_humidity_var"].values,
                temperature=ds["TODO_temperature_var"].values,
                pressure=ds["TODO_pressure_var"].values,
            )
            channel_arrays.append(q)
        else:
            da = ds[spec.name]
            if spec.level is not None:
                da = da.sel(level=spec.level)
            channel_arrays.append(da.values)

    arr = np.stack(channel_arrays, axis=1)  # (T, C, H, W)
    arr = flip_axes_inference(arr, cfg.data.flip_lat, cfg.data.flip_lon)
    return arr


def run_inference(cfg: Config, train_stats: Dict[str, np.ndarray]) -> np.ndarray:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model)
    load_checkpoint(
        str(Path(cfg.inference.checkpoint_path).parent),
        Path(cfg.inference.checkpoint_path).name,
        model,
        device=device,
    )
    model.eval().to(device)

    arr = load_inference_data(cfg)
    arr_norm = normalise_for_inference(arr, train_stats)
    x = torch.from_numpy(arr_norm).float().to(device)

    with torch.no_grad():
        pred_norm = model(x)

    pred = denormalise(pred_norm.cpu().numpy(), train_stats)

    Path(cfg.inference.output_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(cfg.inference.output_dir) / "predictions.npy", pred)
    return pred
