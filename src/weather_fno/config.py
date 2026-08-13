"""
Central configuration schema.

Loads the YAML config into a plain, typed structure and resolves anything
machine-specific (device) here — this is the ONLY place that should ever
need editing when moving between machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import yaml


@dataclass
class ChannelSpec:
    short_name: str          # e.g. "t850" — used for logging/plots/lookups
                              #  (e.g. selecting a channel by name in a
                              #  notebook); NOT used to decide derivation --
                              #  that matches on `name` instead (see
                              #  inference/predict.py::load_inference_data,
                              #  which derives relative_humidity by name).
    name: str                 # actual variable name in the GCS/zarr store
    level: Optional[int] = None   # pressure level in hPa; None = surface/integrated field


@dataclass
class DataConfig:
    gcs_bucket_path: str
    channels: List[ChannelSpec]
    resolution: List[int]
    flip_lat: bool
    flip_lon: bool
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    stats_cache_path: str
    # Actual dimension NAMES in the zarr store — used to transpose by name
    # rather than assume positional axis order (see project notes on why).
    # Confirmed against all three WeatherBench2 stores this project uses
    # (coarse, native_highres, 1p5deg): all use "latitude"/"longitude".
    lat_dim: str = "latitude"
    lon_dim: str = "longitude"


@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    n_modes: List[int]
    hidden_channels: int
    n_layers: int
    factorization: Optional[str] = None
    rank: Optional[float] = None


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    num_workers: int
    device: str
    early_stopping_patience: int
    output_dir: str
    checkpoint_dir: str
    plot_dir: str
    log_dir: str


@dataclass
class InferenceTarget:
    """One inference-time data source. Different stores need different
    preprocessing — e.g. a native high-resolution store may only provide
    specific humidity (relative humidity must be derived), while a
    resampled store closer to the coarse training data provides relative
    humidity directly but needs the same latitude flip as training."""
    name: str                 # short label, used in output filenames/logs
    gcs_bucket_path: str
    resolution: List[int]
    flip_lat: bool
    flip_lon: bool
    derive_relative_humidity: bool = False


@dataclass
class InferenceConfig:
    checkpoint_path: str
    output_dir: str
    targets: List[InferenceTarget]
    forecast_lead_steps: int = 28  # 28 x 6h = 7 days, at the training cadence
    # Shared starting timestep for every target's rollout -- a single field
    # (not per-target) so all targets are guaranteed to start from the same
    # real timestamp when compared against each other, e.g. "2016-06-01".
    # None (the default) means "the first available timestep in the
    # store", which is 1959-01-01 for these WeatherBench2 stores -- outside
    # the training window entirely. Set an explicit date within/near the
    # training or validation period for a cleaner resolution-generalisation
    # test that isn't also testing generalisation across 40+ years of time.
    start_date: Optional[str] = None


@dataclass
class Config:
    run_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU automatically if CUDA isn't available on this machine."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_config(path: "str | os.PathLike") -> Config:
    # 1. Parse the YAML into a plain nested dict.
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # 2. Build the two list-of-dataclass fields by hand -- `**kwargs`
    # unpacking only works one level deep, so `data.channels` (a list of
    # dicts) and `inference.targets` (same) need converting to lists of
    # ChannelSpec/InferenceTarget BEFORE the parent dataclass is built.
    data_raw = dict(raw["data"])
    data_raw["channels"] = [ChannelSpec(**c) for c in data_raw["channels"]]

    inference_raw = dict(raw["inference"])
    inference_raw["targets"] = [InferenceTarget(**t) for t in inference_raw["targets"]]

    # 3. Build every section's dataclass. Any YAML key that doesn't match a
    # dataclass field name raises TypeError here -- that's deliberate,
    # it's how a typo'd or stale config key gets caught immediately
    # instead of silently being ignored.
    cfg = Config(
        run_name=raw["run_name"],
        data=DataConfig(**data_raw),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        inference=InferenceConfig(**inference_raw),
    )

    # 4. Make sure output directories exist wherever this runs.
    for d in [cfg.training.output_dir, cfg.training.checkpoint_dir,
              cfg.training.plot_dir, cfg.training.log_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return cfg
