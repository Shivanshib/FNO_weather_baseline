"""
Fast end-to-end smoke test: data fetch, model build, training, checkpoint
save + resume, plotting, and inference -- all in well under a minute.

Runs the SAME config as a real run, but with tiny date ranges/epoch count,
and every output path redirected to outputs/smoketest/ so it never touches
a real run's checkpoints or cache.

Usage:
    python scripts/smoke_test.py --config configs/baseline_fno.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from weather_fno.config import derive_run_paths, load_config, resolve_device, set_seed
from weather_fno.data.split import build_train_val_datasets
from weather_fno.inference.predict import load_inference_data
from weather_fno.inference.preprocessing import normalise_for_inference
from weather_fno.models.fno_baseline import build_model
from weather_fno.training.metrics import lat_weights
from weather_fno.training.trainer import Trainer
from weather_fno.utils.plotting import plot_history

SMOKETEST_DIR = Path("outputs/smoketest")


@contextmanager
def step(name: str):
    print(f"-> {name}...", flush=True)
    t0 = time.time()
    try:
        yield
    except Exception:
        print(f"[FAIL] {name} ({time.time() - t0:.1f}s)")
        raise
    else:
        print(f"[OK]   {name} ({time.time() - t0:.1f}s)")


def build_smoketest_config(base_config_path: str):
    """Same config as a real run, but with a tiny date range/epoch count
    and its own run_name ("smoketest") so every output path lands under
    outputs/smoketest/, fully isolated from a real run's checkpoints."""
    cfg = load_config(base_config_path)

    cfg.run_name = "smoketest"
    cfg.data.train_start = "2000-01-01"
    cfg.data.train_end = "2000-01-03"
    cfg.data.val_start = "2000-02-01"
    cfg.data.val_end = "2000-02-02"
    cfg.training.epochs = 2
    # This script builds/calls Trainer directly and never constructs the
    # 2-step fine-tune loaders (see scripts/train.py) -- force this off
    # regardless of what --config sets, or fit() would demand loaders
    # this script never builds.
    cfg.training.finetune_epochs = 0
    # NOT overriding device/num_workers -- validating those on THIS
    # machine is the whole point of this script.

    derive_run_paths(cfg)  # re-derive now that run_name changed
    assert cfg.training.checkpoint_dir == str(SMOKETEST_DIR / "checkpoints")

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_fno.yaml")
    args = parser.parse_args()

    t_start = time.time()
    print(f"Smoke test against {args.config}\n")

    # Fresh start every run -- a stale smoketest checkpoint left over from
    # a previous run would make the resume check below meaningless.
    shutil.rmtree(SMOKETEST_DIR, ignore_errors=True)

    with step("load config"):
        cfg = build_smoketest_config(args.config)
        set_seed(cfg.training.seed)

    with step(f"resolve device (configured: {cfg.training.device})"):
        device = resolve_device(cfg.training.device)
        print(f"     using: {device}")

    with step("build train/val datasets (tests GCS reachability from this machine)"):
        train_ds, val_ds = build_train_val_datasets(cfg.data)
        print(f"     train: {tuple(train_ds.data.shape)}, val: {tuple(val_ds.data.shape)}")
        assert train_ds.data.shape[1] == cfg.model.in_channels, "channel count mismatch"
        assert not torch.isnan(train_ds.data).any(), "NaNs in train data"

    with step(f"build DataLoaders (num_workers={cfg.training.num_workers})"):
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                   shuffle=True, num_workers=cfg.training.num_workers)
        val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                                 shuffle=False, num_workers=cfg.training.num_workers)

    with step("build model + optimizer"):
        model = build_model(cfg.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate,
                                      weight_decay=cfg.training.weight_decay)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"     {n_params:,} parameters")

    with step("train for 2 epochs + checkpoint"):
        weights = lat_weights(train_ds.lat_values)
        trainer = Trainer(model, optimizer, cfg.training, weights, device, target_mode=cfg.model.target_mode)
        history = trainer.fit(train_loader, val_loader)
        assert len(history["train_loss"]) == 2
        assert (Path(cfg.training.checkpoint_dir) / "latest.pt").exists()
        assert (Path(cfg.training.checkpoint_dir) / "best.pt").exists()

    with step("resume from checkpoint"):
        cfg.training.epochs = 4
        model2 = build_model(cfg.model)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=cfg.training.learning_rate,
                                       weight_decay=cfg.training.weight_decay)
        trainer2 = Trainer(model2, optimizer2, cfg.training, weights, device, target_mode=cfg.model.target_mode)
        assert trainer2.start_epoch == 2, f"expected to resume at epoch 2, got {trainer2.start_epoch}"
        history2 = trainer2.fit(train_loader, val_loader)
        assert len(history2["train_loss"]) == 4

    with step("plot training history"):
        plot_path = f"{cfg.training.plot_dir}/{cfg.run_name}_loss.png"
        plot_history(history2, plot_path, cfg.run_name)
        assert Path(plot_path).exists()

        log_plot_path = f"{cfg.training.plot_dir}/{cfg.run_name}_loss_log.png"
        plot_history(history2, log_plot_path, cfg.run_name, log_scale=True)
        assert Path(log_plot_path).exists()

    with step("inference: derive/read channels + one forward pass per target"):
        model2.eval()
        train_stats = train_ds.stats
        for target in cfg.inference.targets:
            arr = load_inference_data(cfg, target)
            arr_norm = normalise_for_inference(arr, train_stats)
            x = torch.from_numpy(arr_norm).float().to(device)
            with torch.no_grad():
                y = model2(x)
            assert y.shape == x.shape, f"{target.name}: shape mismatch {tuple(y.shape)} vs {tuple(x.shape)}"
            assert not torch.isnan(y).any(), f"{target.name}: NaNs in model output"
            print(f"     {target.name}: input {tuple(x.shape)} -> output {tuple(y.shape)}, OK")

    print(f"\nALL CHECKS PASSED in {time.time() - t_start:.1f}s -- safe to start a real training run.")
    print(f"(smoke test artifacts left in {SMOKETEST_DIR}/ for inspection -- "
          f"harmless, unrelated to real checkpoints/cache, safe to delete any time)")


if __name__ == "__main__":
    main()
