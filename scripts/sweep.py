"""
Lightweight hyperparameter sweep for baseline FNO tuning.

Deliberately simple: loop over a few config overrides, train each, log
final val loss to CSV. Swap in a proper sweep tool (Optuna, W&B sweeps)
later if this stops being enough — the config-driven design means nothing
else has to change.

Usage:
    python scripts/sweep.py --config configs/baseline_fno.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from weather_fno.config import load_config, resolve_device
from weather_fno.data.split import build_train_val_datasets
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer

# TODO: fill in the hyperparameter combinations you actually want to sweep.
SWEEP_GRID = [
    {"model.n_modes": [8, 4], "model.hidden_channels": 32},
    {"model.n_modes": [16, 8], "model.hidden_channels": 64},
    {"model.n_modes": [24, 12], "model.hidden_channels": 64},
]


def apply_overrides(cfg, overrides: dict):
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        section, field = dotted_key.split(".")
        setattr(getattr(cfg, section), field, value)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    results_path = Path(base_cfg.training.output_dir) / "sweep_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["overrides", "best_val_loss"])

        for overrides in SWEEP_GRID:
            cfg = apply_overrides(base_cfg, overrides)
            cfg.run_name = f"{base_cfg.run_name}_{overrides}"
            device = resolve_device(cfg.training.device)

            train_ds, val_ds = build_train_val_datasets(cfg.data)
            train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                       shuffle=True, num_workers=cfg.training.num_workers)
            val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                                     shuffle=False, num_workers=cfg.training.num_workers)

            model = build_model(cfg.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                          weight_decay=cfg.training.weight_decay)

            lat_degrees = np.linspace(-90, 90, cfg.data.resolution[1])
            weights = lat_weights(lat_degrees)

            trainer = Trainer(model, optimizer, cfg.training, weights, device)
            trainer.fit(train_loader, val_loader)

            writer.writerow([overrides, trainer.best_val_loss])
            f.flush()


if __name__ == "__main__":
    main()
