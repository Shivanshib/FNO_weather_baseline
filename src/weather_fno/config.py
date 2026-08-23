"""
Central configuration schema.

Loads the YAML config into a plain, typed structure and resolves anything
machine-specific (device) here — this is the ONLY place that should ever
need editing when moving between machines.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, fields
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
    # "direct": the model predicts u(t+dt) outright (the original/default
    # behaviour). "residual": the model predicts the DELTA
    # du = u(t+dt) - u(t) instead -- the loss compares the model's raw
    # output to the true delta (Trainer._run_epoch), and every
    # autoregressive step reconstructs the full state as x + model(x)
    # before feeding it back in (inference/predict.py::rollout) or scoring
    # it. This is a property of what the CHECKPOINT'S WEIGHTS represent,
    # not just a loss-function choice -- load_trained_model() refuses to
    # load a checkpoint whose recorded target_mode doesn't match this
    # config's, since the reconstruction math differs and a silent
    # mismatch would produce a plausible-looking but meaningless forecast.
    target_mode: str = "direct"


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


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """
    Apply a small dict of overrides on top of an already-loaded Config,
    returning a NEW Config (the original `cfg` is untouched). `overrides`
    mirrors the base YAML's own nested shape -- a fragment of what
    configs/baseline_fno.yaml itself looks like, e.g.:
        {"run_name": "tfno_v1", "model": {"factorization": "tucker", "rank": 0.5}}
    `run_name` may be set at the top level (Config's only top-level scalar
    field); every other key must be nested one level under a section name
    (data/model/training/inference) matching that section's own field
    names -- exactly like slicing a fragment out of the base YAML.

    An unknown section or field name raises ValueError immediately, same
    spirit as load_config's own "a typo'd YAML key is a TypeError, not a
    silently-ignored no-op" -- setattr on a dataclass instance would
    otherwise happily create a phantom attribute nothing ever reads,
    leaving the real field silently at its base-config value.

    Shared by scripts/train.py/evaluate.py/predict_single_variable.py (via
    load_config's own override_path -- the common case) and
    scripts/sweep.py (applied directly, once per SWEEP_GRID entry).
    """
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key == "run_name":
            cfg.run_name = value
            continue
        if not hasattr(cfg, key):
            raise ValueError(f"unknown config section '{key}' in override -- expected one "
                              f"of: run_name, data, model, training, inference")
        section = getattr(cfg, key)
        valid_fields = {f.name for f in fields(section)}
        for field_name, field_value in value.items():
            if field_name not in valid_fields:
                raise ValueError(f"unknown field '{key}.{field_name}' in override -- "
                                  f"'{key}' has: {sorted(valid_fields)}")
            setattr(section, field_name, field_value)
    return cfg


def _validate_n_modes(cfg: Config) -> None:
    """model.n_modes is capped by the training grid's Nyquist limit
    (grid_size // 2 per axis) -- exceeding it either errors deep inside
    neuralop or silently produces a model that doesn't do what the config
    implies. Worth catching here, at config-load time, now that n_modes is
    something experiment override files routinely change."""
    lon_size, lat_size = cfg.data.resolution  # resolution is [lon, lat] = [W, H]
    lat_modes, lon_modes = cfg.model.n_modes  # n_modes is [lat, lon] = [H, W] -- reversed
    for axis, actual, grid_size in [("lat", lat_modes, lat_size), ("lon", lon_modes, lon_size)]:
        ceiling = grid_size // 2
        if actual > ceiling:
            raise ValueError(
                f"model.n_modes {axis}={actual} exceeds the Nyquist ceiling {ceiling} "
                f"(grid_size // 2) for this {lon_size}x{lat_size} (lon x lat) training "
                f"grid -- lower it, or increase data.resolution to match."
            )


def _validate_target_mode(cfg: Config) -> None:
    """model.target_mode selects a genuinely different loss target and
    inference reconstruction (see ModelConfig.target_mode's comment) --
    catch a typo'd value here, at config-load time, rather than having it
    silently fall through as neither "direct" nor "residual" deep inside
    Trainer/rollout."""
    valid = ("direct", "residual")
    if cfg.model.target_mode not in valid:
        raise ValueError(
            f"model.target_mode '{cfg.model.target_mode}' is not one of {valid}."
        )


def save_config_snapshot(cfg: Config) -> None:
    """
    Write the fully-resolved config -- every hyperparameter, every derived
    path, the whole channel/target list -- to {run_dir}/config_used.yaml.
    Makes a downloaded run folder self-documenting on its own: what
    architecture, what data range, what hyperparameters actually produced
    this checkpoint, even if the experiment override file (or the base
    config itself) that built it has since changed or been deleted.

    Called once by scripts/train.py right after load_config(), not from
    load_config() itself -- this is a training-time record of what a run
    actually used, not something every read-only load (a notebook just
    inspecting an existing run) should silently write into that run's
    folder.
    """
    run_dir = Path(cfg.training.checkpoint_dir).parent
    snapshot_path = run_dir / "config_used.yaml"
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, default_flow_style=False)


def load_config(path: "str | os.PathLike", override_path: "str | os.PathLike | None" = None) -> Config:
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

    # 3.5. Apply a small experiment override file on top, if given -- see
    # apply_overrides()'s docstring for the exact merge semantics. Always
    # requires its own run_name: forgetting to set one is the single
    # likeliest mistake here, and without this check it would silently
    # collide with (or auto-resume!) the base config's own run instead of
    # failing loudly.
    if override_path is not None:
        with open(override_path, "r") as f:
            overrides = yaml.safe_load(f)
        if "run_name" not in overrides:
            raise ValueError(
                f"{override_path} must set its own run_name -- every experiment needs a "
                f"distinct one so its outputs never collide with (or silently resume) the "
                f"base config's own run. See configs/experiments/example.yaml."
            )
        cfg = apply_overrides(cfg, overrides)

    # 4. Namespace every output path by run_name (possibly just changed by
    # the override above), and make sure they all exist wherever this
    # runs.
    derive_run_paths(cfg)

    # 5. Catch a real, likely mistake immediately (at config-load time)
    # rather than deep inside neuralop or a multi-hour training run.
    _validate_n_modes(cfg)
    _validate_target_mode(cfg)

    return cfg
