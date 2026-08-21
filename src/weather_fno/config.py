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
    # Always auto-derived from run_name by derive_run_paths() -- see its
    # docstring. Never set this in a YAML config; load_config() ignores
    # (and warns about) it if you do.
    stats_cache_path: str = ""
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
    # The one shared, UN-namespaced root across every run (e.g. "outputs")
    # -- machine-specific like `device` (e.g. pointed at a quota-safe disk
    # on a particular GPU node), but never run-specific. checkpoint_dir/
    # plot_dir/log_dir below are namespaced under this by run_name via
    # derive_run_paths() -- never set them directly in a YAML config;
    # load_config() ignores (and warns about) them if you do.
    output_dir: str
    checkpoint_dir: str = ""
    plot_dir: str = ""
    log_dir: str = ""
    # CosineAnnealingLR: smoothly decays the learning rate along a cosine
    # curve from learning_rate down to min_lr over lr_scheduler_t_max
    # epochs (None = default to `epochs` above, so the decay spans the
    # whole training budget). Unlike ReduceLROnPlateau this is scheduled,
    # not reactive to val loss -- it steps every epoch regardless of
    # whether val loss actually improved that epoch. Defaulted so older
    # configs that don't set these still load.
    lr_scheduler_t_max: Optional[int] = None
    min_lr: float = 1.0e-6


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
    # Both always auto-derived from run_name by derive_run_paths() -- see
    # its docstring. Never set these in a YAML config; load_config()
    # ignores (and warns about) them if you do. checkpoint_path defaults
    # to THIS run's own best.pt; point somewhere else by overriding the
    # attribute in code afterward (every notebook's RUN FOLDER cell does
    # exactly this for a downloaded run).
    checkpoint_path: str = ""
    output_dir: str = ""


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


def derive_run_paths(cfg: Config) -> None:
    """
    Namespace every output path this run writes to under
    outputs/{run_name}/, and make sure they all exist. Sets, on `cfg` in
    place:
        training.checkpoint_dir  -> {output_dir}/{run_name}/checkpoints
        training.plot_dir        -> {output_dir}/{run_name}/plots
        training.log_dir         -> {output_dir}/{run_name}/logs
        data.stats_cache_path    -> {output_dir}/{run_name}/stats/normalisation_stats.npz
        inference.output_dir     -> {output_dir}/{run_name}/predictions
        inference.checkpoint_path -> {training.checkpoint_dir}/best.pt

    training.output_dir itself is NOT touched here -- it's the one
    shared, un-namespaced root across every run (see TrainingConfig's
    comment), set directly from the YAML.

    Called once by load_config() below. Call it again yourself after
    changing cfg.run_name post-load (e.g. scripts/sweep.py, before
    building each sweep entry's dataset/trainer) so every derived path
    stays consistent with whichever run_name is currently set -- these
    were previously kept in sync by hand path-by-path (a real bug once:
    sweep.py used to only re-derive checkpoint_dir, leaving
    stats_cache_path shared across every sweep entry and silently
    overwritten by whichever entry trained last).
    """
    run_dir = f"{cfg.training.output_dir}/{cfg.run_name}"
    cfg.training.checkpoint_dir = f"{run_dir}/checkpoints"
    cfg.training.plot_dir = f"{run_dir}/plots"
    cfg.training.log_dir = f"{run_dir}/logs"
    cfg.data.stats_cache_path = f"{run_dir}/stats/normalisation_stats.npz"
    cfg.inference.output_dir = f"{run_dir}/predictions"
    cfg.inference.checkpoint_path = f"{cfg.training.checkpoint_dir}/best.pt"

    for d in [cfg.training.output_dir, cfg.training.checkpoint_dir, cfg.training.plot_dir,
              cfg.training.log_dir, cfg.inference.output_dir,
              str(Path(cfg.data.stats_cache_path).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _warn_if_overridden(raw_section: dict, keys: List[str], section_name: str) -> None:
    """Every key here is now computed by derive_run_paths() -- setting it
    in the YAML has no effect (it gets overwritten immediately after this
    runs), so say so rather than silently ignoring it."""
    for key in keys:
        if key in raw_section:
            print(f"[config] {section_name}.{key} is set in the YAML but is always "
                  f"auto-derived from run_name now (see config.py::derive_run_paths) "
                  f"-- this value is ignored.")


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

    _warn_if_overridden(raw["training"], ["checkpoint_dir", "plot_dir", "log_dir"], "training")
    _warn_if_overridden(data_raw, ["stats_cache_path"], "data")
    _warn_if_overridden(inference_raw, ["checkpoint_path", "output_dir"], "inference")

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

    # 4. Namespace every output path by run_name, and make sure they all
    # exist wherever this runs.
    derive_run_paths(cfg)

    return cfg
