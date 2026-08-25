"""
CLI entrypoint. Same command works on cream, vanilla, or external compute —
only configs/baseline_fno.yaml (or an env-specified alternative) changes.

Usage:
    python scripts/train.py --config configs/baseline_fno.yaml
    # Sweeping a few parameters without touching the base config or the
    # code -- a small override file, see configs/experiments/example.yaml:
    python scripts/train.py --config configs/baseline_fno.yaml --experiment configs/experiments/example.yaml
    # FourCastNet-style 2-step fine-tuning after pretraining -- see
    # configs/experiments/twostep_finetune.yaml:
    python scripts/train.py --config configs/baseline_fno.yaml --experiment configs/experiments/twostep_finetune.yaml
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from weather_fno.config import load_config, resolve_device, save_config_snapshot, set_seed
from weather_fno.data.climatology import compute_and_save_climatology
from weather_fno.data.gcs_dataset import MultiStepDataset
from weather_fno.data.preprocessing import denormalise
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

    # Seed before anything that consumes randomness -- model weight init
    # and DataLoader shuffle order -- so two runs with the same seed are
    # actually comparable (see set_seed's docstring).
    set_seed(cfg.training.seed)

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

    # Climatology baseline for evaluation (inference/evaluate.py) and ACC
    # (training/metrics.py::lat_weighted_acc) -- computed once here, from
    # the training split's own data (already fetched above, no extra GCS
    # cost), and cached so evaluate.py never needs to recompute it. Skips
    # (with a warning) rather than failing if train_start/train_end don't
    # span a full year -- see compute_and_save_climatology's docstring.
    climatology_path = cfg.data.stats_cache_path.replace("normalisation_stats", "climatology")
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

    # 3. Model and optimizer.
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)

    # 4. Latitude weights for the loss -- real values from the store, not
    # a linspace guess, since equiangular grids don't space evenly pole-to-pole.
    weights = lat_weights(train_ds.lat_values)

    # 5. Train, then plot the loss curve (linear and log scale -- log
    # scale stays readable once loss has dropped enough to flatten the
    # linear plot).
    trainer = Trainer(model, optimizer, cfg.training, weights, device, target_mode=cfg.model.target_mode)
    history = trainer.fit(train_loader, val_loader, train_loader_2step, val_loader_2step)

    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss.png", cfg.run_name,
                 pretrain_epochs=cfg.training.epochs)
    plot_history(history, f"{cfg.training.plot_dir}/{cfg.run_name}_loss_log.png",
                 cfg.run_name, log_scale=True, pretrain_epochs=cfg.training.epochs)


if __name__ == "__main__":
    main()
