"""
CLI entrypoint. Same command works on cream, vanilla, or external compute —
only configs/baseline_fno.yaml (or an env-specified alternative) changes.

Usage:
    python scripts/train.py --config configs/baseline_fno.yaml
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from weather_fno.config import load_config, resolve_device
from weather_fno.data.split import build_train_val_datasets
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer
from weather_fno.utils.plotting import plot_history


def main():
    # 1. create argument parser
    parser = argparse.ArgumentParser()
    
    # will pass in the .yaml file that we set up when running the code from the terminal
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # stores all hyper parameters into a python object
    cfg = load_config(args.config)
    device = resolve_device(cfg.training.device)


    # 2. Data Pipeline
    # initialise training and validation datasets
    train_ds, val_ds = build_train_val_datasets(cfg.data)

    # wraps datasets into pytorch dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                               shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                             shuffle=False, num_workers=cfg.training.num_workers)

    # 3. Model and Optimisation Init
    # builds fno model
    model = build_model(cfg.model)
    # builds optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)

    # 4. Set up physics informed Adjustments
    # Real latitude values pulled from the store itself (see
    # GCSWeatherDataset.lat_values) — not a guessed/linspace approximation,
    # since most equiangular grids don't actually space evenly pole-to-pole.
    weights = lat_weights(train_ds.lat_values)

    # 5. Set up training loop and visualisation
    trainer = Trainer(model, optimizer, cfg.training, weights, device)
    history = trainer.fit(train_loader, val_loader)

    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss.png", cfg.run_name)


if __name__ == "__main__":
    main()
