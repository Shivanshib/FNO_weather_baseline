"""
Central configuration schema. Loads the YAML config into a typed
structure and resolves anything machine-specific (device, output paths)
here -- this should be the only place that needs editing when moving to a
different machine.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml


@dataclass
class ChannelSpec:
    """One training/inference channel. `name` is the real variable name in
    the zarr store; `short_name` is a friendlier label used for plots and
    for looking a channel up by name (e.g. in a notebook)."""
    short_name: str
    name: str
    level: Optional[int] = None  # pressure level in hPa; None = surface/integrated field


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
    # Auto-derived from run_name by derive_run_paths() -- never set this by
    # hand in a YAML config.
    stats_cache_path: str = ""
    # Actual dimension names in the zarr store, used to select axes by
    # name instead of assuming a fixed position.
    lat_dim: str = "latitude"
    lon_dim: str = "longitude"
    # True for a training store that doesn't provide relative_humidity
    # directly (e.g. native-resolution ERA5, which only has specific
    # humidity) -- routes every relative_humidity channel through
    # data/preprocessing.py::compute_relative_humidity instead (see
    # data/gcs_dataset.py::_select_channels). False (default) for the
    # coarse baseline store and the 1.5deg store, both of which already
    # provide it directly, same as InferenceTarget's own field of the
    # same name/purpose for inference-time stores.
    derive_relative_humidity: bool = False


@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    n_modes: List[int]
    hidden_channels: int
    n_layers: int
    factorization: Optional[str] = None
    rank: Optional[float] = None
    # "direct": the model predicts u(t+dt) outright. "residual": the model
    # predicts the delta du = u(t+dt) - u(t) instead -- see
    # Trainer._run_epoch (loss target) and inference/predict.py::rollout
    # (state reconstruction). This is a property of the checkpoint's own
    # weights, not just a training-time setting -- load_trained_model()
    # refuses to load a checkpoint whose recorded target_mode doesn't
    # match this config's.
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
    # The one shared, un-namespaced root across every run (machine-
    # specific, e.g. a particular disk). checkpoint_dir/plot_dir/log_dir
    # below are namespaced under this by run_name via derive_run_paths()
    # -- never set them directly in a YAML config.
    output_dir: str
    checkpoint_dir: str = ""
    plot_dir: str = ""
    log_dir: str = ""
    # Pretrain's CosineAnnealingLR: decays learning_rate down to min_lr
    # over lr_scheduler_t_max epochs (None = default to `epochs`, so the
    # decay spans the whole pretrain budget). Scheduled, not reactive to
    # val loss like ReduceLROnPlateau.
    lr_scheduler_t_max: Optional[int] = None
    min_lr: float = 1.0e-6
    # Seeds model weight init and DataLoader shuffle order (see
    # set_seed()) -- the two biggest sources of run-to-run randomness in
    # this pipeline. Same default for every config, so two experiments
    # (e.g. two target_mode variants) are comparable out of the box unless
    # one deliberately overrides it.
    seed: int = 42
    # FourCastNet-style fine-tuning: 0 (default) = plain single-step
    # training for all `epochs`, unchanged from before this existed. > 0
    # = after the `epochs` single-step epochs, train for this many
    # additional epochs on a 2-step autoregressive rollout instead (model
    # predicts x(k+1) from x(k), then x(k+2) from its OWN x(k+1)
    # prediction, loss = sum of both steps' errors against their real
    # ground truth) -- see Trainer.fit()/_run_epoch().
    #
    # Fine-tuning gets its OWN fresh CosineAnnealingLR (Trainer's
    # _start_finetune_phase), NOT a continuation of the pretrain curve --
    # matching the FourCastNet paper's own recipe, where fine-tuning
    # restarts at a lower peak LR rather than picking up wherever pretrain
    # left off. finetune_learning_rate is REQUIRED once finetune_epochs >
    # 0 (validated at config-load time, config.py::_validate_finetune_lr)
    # -- deliberately not defaulted, since what to restart fine-tuning at
    # is a real methodological choice, not something to leave implicit.
    # finetune_lr_scheduler_t_max works exactly like lr_scheduler_t_max,
    # just for this second schedule (None = default to finetune_epochs).
    finetune_epochs: int = 0
    finetune_learning_rate: Optional[float] = None
    finetune_lr_scheduler_t_max: Optional[int] = None


@dataclass
class InferenceTarget:
    """One inference-time data source. Different stores need different
    preprocessing -- e.g. a native high-resolution store may only provide
    specific humidity (relative humidity must be derived), while a
    resampled store closer to the training data provides it directly."""
    name: str  # short label, used in output filenames/logs
    gcs_bucket_path: str
    resolution: List[int]
    flip_lat: bool
    flip_lon: bool
    derive_relative_humidity: bool = False


@dataclass
class InferenceConfig:
    targets: List[InferenceTarget]
    forecast_lead_steps: int = 28  # 28 x 6h = 7 days, at the training cadence
    # Shared starting timestep for every target's rollout, so they're all
    # compared on the same real dates. None (default) = each store's first
    # available timestep, which for these stores is 1959 -- decades before
    # the training window, so set an explicit date for a fairer comparison.
    start_date: Optional[str] = None
    # Both auto-derived from run_name by derive_run_paths() -- never set
    # these by hand. checkpoint_path defaults to this run's own best.pt.
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
    """Fall back to CPU automatically if CUDA isn't available."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    """
    Seed every RNG this project touches: model weight init (torch) and
    DataLoader shuffle order (torch's global RNG, which the default
    RandomSampler draws from). Call once, right after load_config, before
    building the model or any DataLoader.

    Doesn't force CUDA into fully deterministic mode (torch.backends.cudnn
    .deterministic / torch.use_deterministic_algorithms) -- FNO leans on
    FFT-based ops that either lack a deterministic GPU implementation or
    would run noticeably slower under one, for a source of noise that's
    tiny compared to weight init/shuffle order. This gets the two sources
    that actually matter for comparing configs, not bit-for-bit reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_run_paths(cfg: Config) -> None:
    """
    Namespace every output path this run writes to under
    outputs/{run_name}/, and create them all. Sets, on `cfg` in place:
        training.checkpoint_dir   -> {output_dir}/{run_name}/checkpoints
        training.plot_dir         -> {output_dir}/{run_name}/plots
        training.log_dir          -> {output_dir}/{run_name}/logs
        data.stats_cache_path     -> {output_dir}/{run_name}/stats/normalisation_stats.npz
        inference.output_dir      -> {output_dir}/{run_name}/predictions
        inference.checkpoint_path -> {training.checkpoint_dir}/best.pt

    training.output_dir itself is untouched -- it's the one shared root
    every run's folder sits under, set directly from the YAML.

    Called once by load_config(). Call it again yourself if you change
    cfg.run_name afterwards (scripts/sweep.py does this per sweep entry)
    so every derived path stays in sync with the new run_name.
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
    """These keys are always computed by derive_run_paths() -- setting one
    in the YAML has no effect, so warn instead of silently ignoring it."""
    for key in keys:
        if key in raw_section:
            print(f"[config] {section_name}.{key} is set in the YAML but is always "
                  f"auto-derived from run_name now (see config.py::derive_run_paths) "
                  f"-- this value is ignored.")


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """
    Apply a small dict of overrides on top of an already-loaded Config,
    returning a NEW Config (the original is untouched). `overrides`
    mirrors the base YAML's nested shape, e.g.:
        {"run_name": "tfno_v1", "model": {"factorization": "tucker", "rank": 0.5}}
    `run_name` may be set at the top level; every other key must be
    nested one level under a section name (data/model/training/inference).

    Raises ValueError on an unknown section or field name (a typo) rather
    than silently creating a phantom attribute nothing reads.

    Used by load_config()'s override_path (one experiment file) and by
    scripts/sweep.py (once per SWEEP_GRID entry).
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
    (grid_size // 2 per axis). Catches this at config-load time instead of
    erroring deep inside neuralop or silently building the wrong model."""
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
    """model.target_mode must be "direct" or "residual" -- catch a typo
    here rather than have it silently fall through as neither."""
    valid = ("direct", "residual")
    if cfg.model.target_mode not in valid:
        raise ValueError(
            f"model.target_mode '{cfg.model.target_mode}' is not one of {valid}."
        )


def _validate_finetune_lr(cfg: Config) -> None:
    """training.finetune_learning_rate must be set once finetune_epochs >
    0 -- fine-tuning restarts with its OWN fresh cosine schedule (see
    Trainer._start_finetune_phase), so what to restart it at is a real
    methodological choice, not something to leave to a silent default."""
    if cfg.training.finetune_epochs > 0 and cfg.training.finetune_learning_rate is None:
        raise ValueError(
            "training.finetune_epochs > 0 requires training.finetune_learning_rate to "
            "also be set -- fine-tuning uses its own fresh cosine schedule, starting at "
            "this LR (FourCastNet's own recipe uses a lower peak LR for fine-tuning than "
            "pretraining, e.g. 1/5th), not a value chosen for you."
        )


def save_config_snapshot(cfg: Config) -> None:
    """
    Write the fully-resolved config (every hyperparameter, every derived
    path, the whole channel/target list) to {run_dir}/config_used.yaml, so
    a downloaded run folder is self-documenting even after the experiment
    override file (or base config) that built it has changed or gone.

    Called by scripts/train.py right after load_config(), not from
    load_config() itself -- this is a training-time record of what a run
    actually used, not something a read-only load should write.
    """
    run_dir = Path(cfg.training.checkpoint_dir).parent
    snapshot_path = run_dir / "config_used.yaml"
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, default_flow_style=False)


def load_config(path: "str | os.PathLike", override_path: "str | os.PathLike | None" = None) -> Config:
    """Load configs/*.yaml into a Config, optionally merging a small
    experiment override file on top (see apply_overrides). Also derives
    every run-namespaced output path and validates model.n_modes/
    target_mode before returning."""
    # 1. Parse the YAML into a plain nested dict.
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # 2. `**kwargs` unpacking only works one level deep, so the two
    # list-of-dict fields (data.channels, inference.targets) need
    # converting to lists of ChannelSpec/InferenceTarget by hand first.
    data_raw = dict(raw["data"])
    data_raw["channels"] = [ChannelSpec(**c) for c in data_raw["channels"]]

    inference_raw = dict(raw["inference"])
    inference_raw["targets"] = [InferenceTarget(**t) for t in inference_raw["targets"]]

    _warn_if_overridden(raw["training"], ["checkpoint_dir", "plot_dir", "log_dir"], "training")
    _warn_if_overridden(data_raw, ["stats_cache_path"], "data")
    _warn_if_overridden(inference_raw, ["checkpoint_path", "output_dir"], "inference")

    # 3. Build every section's dataclass. A YAML key that doesn't match a
    # dataclass field raises TypeError here -- deliberate, so a typo'd
    # config key is caught immediately instead of silently ignored.
    cfg = Config(
        run_name=raw["run_name"],
        data=DataConfig(**data_raw),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        inference=InferenceConfig(**inference_raw),
    )

    # 3.5. Merge in a small experiment override file, if given. It must
    # set its own run_name -- forgetting to would otherwise silently
    # collide with (or auto-resume) the base config's own run.
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

    # 4. Namespace every output path by run_name.
    derive_run_paths(cfg)

    # 5. Catch likely config mistakes now, not deep inside a training run.
    _validate_n_modes(cfg)
    _validate_target_mode(cfg)
    _validate_finetune_lr(cfg)

    return cfg
