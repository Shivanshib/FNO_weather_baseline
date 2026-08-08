"""
CLI entrypoint. Same command works on cream, vanilla, or external compute —
only configs/baseline_fno.yaml (or an env-specified alternative) changes.

Usage:
    python scripts/train.py --config configs/baseline_fno.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from weather_fno.config import load_config, resolve_device
from weather_fno.data.split import build_train_val_datasets
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer
from weather_fno.utils.plotting import plot_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.training.device)

    train_ds, val_ds = build_train_val_datasets(cfg.data)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                               shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                             shuffle=False, num_workers=cfg.training.num_workers)

    model = build_model(cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)

    # TODO: fill in the actual latitude values for the training grid, e.g.
    # read them from the dataset itself rather than hardcoding.
    lat_degrees = np.linspace(-90, 90, cfg.data.resolution[1])
    weights = lat_weights(lat_degrees)

    trainer = Trainer(model, optimizer, cfg.training, weights, device)
    history = trainer.fit(train_loader, val_loader)

    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss.png", cfg.run_name)


if __name__ == "__main__":
    main()
