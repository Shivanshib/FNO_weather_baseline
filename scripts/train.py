"""
CLI entrypoint. Same command works on cream, vanilla, or external compute —
only configs/baseline_fno.yaml (or an env-specified alternative) changes.

Usage:
    python scripts/train.py --config configs/baseline_fno.yaml
    # Sweeping a few parameters without touching the base config or the
    # code -- a small override file, see configs/experiments/example.yaml:
    python scripts/train.py --config configs/baseline_fno.yaml --experiment configs/experiments/example.yaml
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from weather_fno.config import load_config, resolve_device, save_config_snapshot
from weather_fno.data.split import build_train_val_datasets
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer
from weather_fno.utils.plotting import plot_history


def main():
    # 1. Parse CLI args and load config.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # Small override file layered on top of --config, so a whole
    # experiment (architecture, epochs, ...) lives in one small file
    # instead of editing the base config -- see configs/experiments/example.yaml.
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.experiment)
    device = resolve_device(cfg.training.device)

    # Record exactly what this run used, before training starts, so a
    # downloaded run folder stays self-documenting even if the experiment
    # file or base config later changes.
    save_config_snapshot(cfg)

    # 2. Data pipeline.
    train_ds, val_ds = build_train_val_datasets(cfg.data)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                               shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                             shuffle=False, num_workers=cfg.training.num_workers)

    # 3. Model and optimizer.
    model = build_model(cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)

    # 4. Latitude weights for the loss -- real values from the store, not
    # a linspace guess, since equiangular grids don't space evenly pole-to-pole.
    weights = lat_weights(train_ds.lat_values)

    # 5. Train, then plot the loss curve (linear and log scale -- log
    # scale stays readable once loss has dropped enough to flatten the
    # linear plot).
    trainer = Trainer(model, optimizer, cfg.training, weights, device, target_mode=cfg.model.target_mode)
    history = trainer.fit(train_loader, val_loader)

    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss.png", cfg.run_name)
    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss_log.png",
                 cfg.run_name, log_scale=True)


if __name__ == "__main__":
    main()
