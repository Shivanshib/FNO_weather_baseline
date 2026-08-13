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

import torch
from torch.utils.data import DataLoader

from weather_fno.config import load_config, resolve_device
from weather_fno.data.split import build_train_val_datasets
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer

# TODO: fill in the hyperparameter combinations you actually want to sweep.
# These three are just valid placeholders (respecting the 64x32 training
# grid's Nyquist ceiling of [16, 32] -- see configs/baseline_fno.yaml's
# model.n_modes comment), not a considered choice of what to sweep.
SWEEP_GRID = [
    {"model.n_modes": [8, 16], "model.hidden_channels": 32},
    {"model.n_modes": [12, 24], "model.hidden_channels": 64},
    {"model.n_modes": [16, 32], "model.hidden_channels": 64},
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
        writer.writerow(["run_name", "overrides", "best_val_loss"])

        for i, overrides in enumerate(SWEEP_GRID):
            # 1. Build this entry's config: apply overrides, and give it
            # its own run_name AND its own checkpoint_dir. The checkpoint
            # dir specifically matters -- without a distinct one per
            # entry, every entry after the first would silently
            # auto-resume from the PREVIOUS entry's checkpoint (Trainer's
            # auto-resume just looks for checkpoint_dir/latest.pt, it has
            # no idea a different architecture was requested) instead of
            # training fresh.
            cfg = apply_overrides(base_cfg, overrides)
            cfg.run_name = f"{base_cfg.run_name}_sweep{i}"
            cfg.training.checkpoint_dir = str(Path(base_cfg.training.checkpoint_dir) / f"sweep_{i}")
            device = resolve_device(cfg.training.device)

            # 2. Data pipeline -- same as train.py, rebuilt fresh per
            # entry since some overrides could in principle affect data
            # config too (not just model config).
            train_ds, val_ds = build_train_val_datasets(cfg.data)
            train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                       shuffle=True, num_workers=cfg.training.num_workers)
            val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                                     shuffle=False, num_workers=cfg.training.num_workers)

            # 3. Model + optimizer, built from THIS entry's (possibly
            # overridden) model config.
            model = build_model(cfg.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                          weight_decay=cfg.training.weight_decay)

            weights = lat_weights(train_ds.lat_values)

            # 4. Train this entry to completion (or early stopping), then
            # log its result immediately -- flushed after every row so
            # partial sweep progress survives a crash partway through.
            trainer = Trainer(model, optimizer, cfg.training, weights, device)
            trainer.fit(train_loader, val_loader)

            writer.writerow([cfg.run_name, overrides, trainer.best_val_loss])
            f.flush()


if __name__ == "__main__":
    main()
