"""
The full scripts/train.py training pipeline as an importable function --
scripts/ stays a thin CLI wrapper around this (see README's Structure
section), and scripts/run_seed_ensemble.py calls it once per seed too,
without duplicating the finetune-loader/climatology logic.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from weather_fno.config import Config
from weather_fno.data.climatology import climatology_cache_is_valid, compute_and_save_climatology
from weather_fno.data.gcs_dataset import MultiStepDataset
from weather_fno.data.preprocessing import denormalise
from weather_fno.data.split import build_train_val_datasets
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer
from weather_fno.utils.plotting import plot_history


def run_training(cfg: Config, device) -> dict:
    """
    Build the data pipeline (+ climatology + 2-step fine-tune loaders, if
    needed) and model, train via Trainer.fit(), save the loss plots, and
    return the resulting history dict.

    Assumes the caller has already called set_seed(cfg.training.seed) and
    save_config_snapshot(cfg) -- both are genuinely CLI-entrypoint-level
    concerns (exactly what to seed before, and when to snapshot the
    config) rather than part of "run one training loop", so scripts/
    train.py's main() still does them itself before calling this.
    """
    train_ds, val_ds = build_train_val_datasets(cfg.data)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                               shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                             shuffle=False, num_workers=cfg.training.num_workers)

 
    climatology_path = cfg.data.stats_cache_path.replace("normalisation_stats", "climatology")
    if climatology_cache_is_valid(climatology_path):
        print(f"[climatology] already cached for this training split at {climatology_path} "
              f"-- skipping recomputation.")
    else:
 
        train_physical = denormalise(train_ds.data.numpy(), train_ds.stats)
        compute_and_save_climatology(train_physical, train_ds.time_values, climatology_path)

    # 2-step fine-tune loaders, only built if actually needed -- reuse
    # train_ds/val_ds's already-fetched data (MultiStepDataset doesn't
    # hit GCS itself), just paired 2 steps ahead instead of 1.
    train_loader_2step = val_loader_2step = None
    if cfg.training.finetune_epochs > 0:
        train_loader_2step = DataLoader(MultiStepDataset(train_ds.data, n_future_steps=2),
                                         batch_size=cfg.training.batch_size,
                                         shuffle=True, num_workers=cfg.training.num_workers)
        val_loader_2step = DataLoader(MultiStepDataset(val_ds.data, n_future_steps=2),
                                       batch_size=cfg.training.batch_size,
                                       shuffle=False, num_workers=cfg.training.num_workers)

    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)

    # Latitude weights for the loss -- real values from the store, not a
    # linspace guess, since equiangular grids don't space evenly pole-to-pole.
    weights = lat_weights(train_ds.lat_values)

    trainer = Trainer(model, optimizer, cfg.training, weights, device, target_mode=cfg.model.target_mode)
    history = trainer.fit(train_loader, val_loader, train_loader_2step, val_loader_2step)

    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss.png", cfg.run_name,
                 pretrain_epochs=cfg.training.epochs)
    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss_log.png",
                 cfg.run_name, log_scale=True, pretrain_epochs=cfg.training.epochs)

    return history
