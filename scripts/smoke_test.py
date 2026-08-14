"""
Fast end-to-end smoke test for the whole pipeline: data fetch, model
build, training, checkpoint save + resume, plotting, and inference
(including relative-humidity derivation) -- all in well under a minute.

Runs against the SAME config as a real run (channels, flip settings, GCS
paths, model architecture) but overrides the date ranges and epoch count
to a tiny slice, and redirects every output path to outputs/smoketest/ so
it can never touch or interfere with a real run's checkpoints, cache, or
auto-resume state.

Meant to be run right after SSHing into a new machine (e.g. a university
GPU node) and before starting a real overnight run -- confirms GCS is
reachable from THIS machine (some cluster compute nodes have no outbound
internet access, only the login node does -- worth knowing before, not
during, an overnight job), the GPU is visible, the configured
num_workers setting actually works here, and the full pipeline -- including
the checkpoint RESUME path specifically -- runs correctly, without waiting
minutes/hours to find out something's wrong.

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

from weather_fno.config import load_config, resolve_device
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
    cfg = load_config(base_config_path)

    # Tiny time windows -- just enough timesteps for a handful of (x, y)
    # pairs on each side of the train/val split.
    cfg.run_name = "smoketest"
    cfg.data.train_start = "2000-01-01"
    cfg.data.train_end = "2000-01-03"
    cfg.data.val_start = "2000-02-01"
    cfg.data.val_end = "2000-02-02"
    cfg.data.stats_cache_path = str(SMOKETEST_DIR / "stats" / "normalisation_stats.npz")

    cfg.training.epochs = 2
    cfg.training.output_dir = str(SMOKETEST_DIR)
    cfg.training.checkpoint_dir = str(SMOKETEST_DIR / "checkpoints")
    cfg.training.plot_dir = str(SMOKETEST_DIR / "plots")
    cfg.training.log_dir = str(SMOKETEST_DIR / "logs")
    # Deliberately NOT overriding device/num_workers -- those are exactly
    # what we want this smoke test to validate on THIS machine.

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
        trainer = Trainer(model, optimizer, cfg.training, weights, device)
        history = trainer.fit(train_loader, val_loader)
        assert len(history["train_loss"]) == 2
        assert (Path(cfg.training.checkpoint_dir) / "latest.pt").exists()
        assert (Path(cfg.training.checkpoint_dir) / "best.pt").exists()

    with step("resume from checkpoint (this exact path broke on PyTorch >=2.6 -- see CODE_REFERENCE.md)"):
        cfg.training.epochs = 4
        model2 = build_model(cfg.model)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=cfg.training.learning_rate,
                                       weight_decay=cfg.training.weight_decay)
        trainer2 = Trainer(model2, optimizer2, cfg.training, weights, device)
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
