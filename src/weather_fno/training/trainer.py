"""
Training loop: train/val epochs, checkpointing, early stopping, history
logging for plotting.

Main process, per call to fit():
  for each epoch:
    1. one gradient-descent pass over the training set (_run_epoch,
       train=True) -- lat-weighted MSE, backprop, optimizer step per batch
    2. one no-grad pass over the validation set (_run_epoch, train=False)
       -- same loss, no weight updates
    3. save latest.pt unconditionally (atomic write, see utils/checkpoint)
    4. if val loss improved, also save best.pt and reset the early-stopping
       counter; otherwise increment it and stop once it hits
       early_stopping_patience
Auto-resume (in __init__, before any of the above starts) means "rerun the
identical command" is always safe -- it just continues from latest.pt if
one already exists, no flag needed.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from weather_fno.training.metrics import lat_weighted_mse
from weather_fno.utils.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, cfg, lat_weight_tensor, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.cfg = cfg
        self.lat_weight_tensor = lat_weight_tensor
        self.device = device

        self.history = {"train_loss": [], "val_loss": []}
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # Auto-resume: if a checkpoint already exists, pick up from there
        # with no flag required — makes "rerun the same command" safe.
        ckpt = load_checkpoint(cfg.checkpoint_dir, "latest.pt", self.model, self.optimizer, device)
        if ckpt is not None:
            self.start_epoch = ckpt["epoch"] + 1
            self.best_val_loss = ckpt["best_val_loss"]
            self.history = ckpt["history"]
            print(f"Resumed from checkpoint at epoch {ckpt['epoch']}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(mode=train)
        total_loss, n_batches = 0.0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(train):
                pred = self.model(x)
                loss = lat_weighted_mse(pred, y, self.lat_weight_tensor)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        patience_counter = 0

        for epoch in range(self.start_epoch, self.cfg.epochs):
            # 1-2. Train then validate this epoch.
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s")

            # 3. Always checkpoint latest.pt, regardless of whether this
            # was the best epoch -- this is what auto-resume picks up from.
            state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                "config": self.cfg,
            }
            save_checkpoint(state, self.cfg.checkpoint_dir, "latest.pt")

            # 4. best.pt + early stopping, driven by validation loss only.
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                state["best_val_loss"] = self.best_val_loss
                save_checkpoint(state, self.cfg.checkpoint_dir, "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for "
                          f"{self.cfg.early_stopping_patience} epochs)")
                    break

        return self.history
