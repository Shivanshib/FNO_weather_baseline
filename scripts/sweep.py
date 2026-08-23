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
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from weather_fno.config import apply_overrides, derive_run_paths, load_config, resolve_device, save_config_snapshot
from weather_fno.data.split import build_train_val_datasets
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer

# TODO: fill in the hyperparameter combinations you actually want to
# sweep -- these three are just valid placeholders. Same nested shape as
# configs/experiments/*.yaml (both go through config.py::apply_overrides).
# No need to set run_name per entry -- the loop below gives each one its own.
SWEEP_GRID = [
    {"model": {"n_modes": [8, 16], "hidden_channels": 32}},
    {"model": {"n_modes": [12, 24], "hidden_channels": 64}},
    {"model": {"n_modes": [16, 32], "hidden_channels": 64}},
]


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
            # 1. Build this entry's config: apply overrides, give it its
            # own run_name, and re-derive every namespaced path (not just
            # checkpoint_dir -- otherwise entries could silently share
            # cached stats, or auto-resume from a previous entry's
            # checkpoint instead of training fresh).
            cfg = apply_overrides(base_cfg, overrides)
            cfg.run_name = f"{base_cfg.run_name}_sweep{i}"
            derive_run_paths(cfg)
            save_config_snapshot(cfg)
            device = resolve_device(cfg.training.device)

            # 2. Data pipeline, rebuilt fresh per entry.
            train_ds, val_ds = build_train_val_datasets(cfg.data)
            train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                       shuffle=True, num_workers=cfg.training.num_workers)
            val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                                     shuffle=False, num_workers=cfg.training.num_workers)

            # 3. Model + optimizer from this entry's config.
            model = build_model(cfg.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                          weight_decay=cfg.training.weight_decay)

            weights = lat_weights(train_ds.lat_values)

            # 4. Train, then log the result immediately -- flushed after
            # every row so progress survives a crash partway through.
            trainer = Trainer(model, optimizer, cfg.training, weights, device, target_mode=cfg.model.target_mode)
            trainer.fit(train_loader, val_loader)

            writer.writerow([cfg.run_name, overrides, trainer.best_val_loss])
            f.flush()


if __name__ == "__main__":
    main()
