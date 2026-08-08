"""
Central configuration schema.

Loads the YAML config into a plain, typed structure and resolves anything
machine-specific (device) here — this is the ONLY place that should ever
need editing when moving between machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import yaml


@dataclass
class ChannelSpec:
    short_name: str          # e.g. "t850" — used for logging/plots and for
                              #  matching special-case channels like a
                              #  derived specific-humidity field
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
class MetricsConfig:
    primary: str
    additional: List[str] = field(default_factory=list)


@dataclass
class InferenceConfig:
    gcs_bucket_path: str
    resolution: List[int]
    checkpoint_path: str
    output_dir: str
    derive_specific_humidity: bool = False


@dataclass
class Config:
    run_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    metrics: MetricsConfig
    inference: InferenceConfig


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU automatically if CUDA isn't available on this machine."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_config(path: "str | os.PathLike") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    data_raw = dict(raw["data"])
    data_raw["channels"] = [ChannelSpec(**c) for c in data_raw["channels"]]

    cfg = Config(
        run_name=raw["run_name"],
        data=DataConfig(**data_raw),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        metrics=MetricsConfig(**raw["metrics"]),
        inference=InferenceConfig(**raw["inference"]),
    )

    # Make sure output directories exist wherever this runs.
    for d in [cfg.training.output_dir, cfg.training.checkpoint_dir,
              cfg.training.plot_dir, cfg.training.log_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return cfg
