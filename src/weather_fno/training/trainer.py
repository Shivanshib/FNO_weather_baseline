"""
Training loop: train/val epochs, LR scheduling, checkpointing, early
stopping, history logging for plotting.

Main process, per call to fit():
  for each epoch:
    1. one gradient-descent pass over the training set (_run_epoch,
       train=True) -- lat-weighted MSE, backprop, optimizer step per batch
    2. one no-grad pass over the validation set (_run_epoch, train=False)
       -- same loss, no weight updates
    3. step the CosineAnnealingLR scheduler -- smoothly decays the learning
       rate along a cosine curve from learning_rate down to min_lr over
       lr_scheduler_t_max epochs. Unlike ReduceLROnPlateau this is
       SCHEDULED, not reactive: it steps every epoch on a fixed curve
       regardless of whether val loss actually improved that epoch.
    4. save latest.pt unconditionally (atomic write, see utils/checkpoint)
    5. if val loss improved, also save best.pt and reset the early-stopping
       counter; otherwise increment it and stop once it hits
       early_stopping_patience -- a plain safety net independent of the LR
       schedule, since the cosine curve itself doesn't respond to
       plateaus the way ReduceLROnPlateau did
Auto-resume (in __init__, before any of the above starts) means "rerun the
identical command" is always safe -- it just continues from latest.pt if
one already exists, no flag needed. The scheduler resumes to the correct
point on its curve too, replayed from the checkpointed epoch count rather
than restored wholesale -- see the resume comment in __init__ for why.
"""

from __future__ import annotations

import time
import warnings

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

        # T_max defaults to the full epoch budget, so the cosine curve
        # reaches min_lr right at the end of training (if early stopping
        # cuts a run short, the curve just doesn't finish -- fine).
        t_max = cfg.lr_scheduler_t_max or cfg.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=cfg.min_lr,
        )

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
            # Replay the scheduler to the epoch already reached, rather
            # than restoring a saved scheduler state_dict wholesale. A
            # state_dict carries T_max/eta_min/base_lrs too, not just the
            # epoch position -- if `epochs` (hence T_max) was deliberately
            # changed since this checkpoint was saved (e.g. extending a
            # run that undershot), a wholesale restore would silently
            # bring back the OLD T_max, and since cosine is periodic,
            # stepping past a stale T_max sends the LR back UP instead of
            # continuing to decay.
            #
            # Must use the closed-form step(epoch=...) here, NOT a loop of
            # bare step() calls -- verified directly (see CODE_REFERENCE.md):
            # bare step()'s "chainable" form computes the next LR
            # recursively from the OPTIMIZER'S CURRENT lr, and
            # optimizer.load_state_dict() just above (inside
            # load_checkpoint) already overwrote that to whatever value was
            # saved -- already fully decayed, in the common case of
            # resuming near the end of a run. A loop of bare steps from
            # that corrupted starting point stays corrupted forever (empty
            # -- degenerates to a flat line at min_lr); the closed form
            # recomputes purely from epoch count and THIS run's own
            # base_lrs (captured at scheduler construction, above, before
            # load_checkpoint touched anything), so it recovers correctly
            # regardless of what the optimizer's lr currently says. The
            # `epoch` parameter is soft-deprecated by PyTorch in favour of
            # bare step() -- that advice doesn't apply to this one-off
            # resume-time replay, so the warning is expected and silenced.
            # +1: at save time, `epoch` epochs (0..ckpt["epoch"] inclusive)
            # had each already called .step() once, on top of the implicit
            # last_epoch=0 set at construction -- so the scheduler's TRUE
            # position was ckpt["epoch"] + 1 steps in, not ckpt["epoch"].
            # Verified directly against an uninterrupted reference run
            # (see CODE_REFERENCE.md) -- without +1 every resumed run
            # quietly repeats one epoch's LR value twice.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
                warnings.filterwarnings("ignore", message=".*epoch parameter.*")
                self.scheduler.step(ckpt["epoch"] + 1)
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

            # 3. Step the cosine LR schedule -- no val_loss argument, unlike
            # ReduceLROnPlateau: this moves the LR along its curve every
            # epoch on a fixed schedule, not in response to plateaus.
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"lr {current_lr:.2e} | {elapsed:.1f}s")

            # 4. Always checkpoint latest.pt, regardless of whether this
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

            # 5. best.pt + early stopping, driven by validation loss only --
            # a plain safety net independent of the LR schedule now, since
            # the cosine curve doesn't react to plateaus itself.
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
