"""
Training loop: train/val epochs, LR scheduling, checkpointing, early
stopping, and (optionally) FourCastNet-style 2-step fine-tuning.

fit() does this every epoch:
  1. one gradient-descent pass over train (lat-weighted MSE)
  2. one no-grad pass over val (same loss, no weight updates)
  3. step the CosineAnnealingLR scheduler (decays LR every epoch on a
     fixed curve, unlike ReduceLROnPlateau which reacts to val loss)
  4. save latest.pt unconditionally
  5. if val loss improved, also save best.pt and reset the early-stopping
     counter; otherwise count towards early_stopping_patience

The first cfg.epochs epochs are plain single-step training. If
cfg.finetune_epochs > 0, cfg.finetune_epochs MORE epochs follow after
that, training on a 2-step autoregressive rollout instead (see
_run_epoch's n_future_steps branch) -- both phases share one continuous
epoch counter and one LR schedule, so auto-resume works the same way
across the whole run, pretrain or fine-tune, no separate state needed.

Auto-resume happens in __init__: if latest.pt already exists, training
picks up from there automatically, so rerunning the same command is
always safe.
"""

from __future__ import annotations

import time
import warnings

import torch
from torch.utils.data import DataLoader

from weather_fno.training.metrics import lat_weighted_mse
from weather_fno.utils.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, cfg, lat_weight_tensor, device, target_mode: str = "direct"):
        """
        Args:
            cfg: a TrainingConfig (not the full Config).
            target_mode: "direct" -- loss target is y (the full next
                state). "residual" -- loss target is (y - x) instead, so
                the model learns to predict the delta. Pass
                cfg.model.target_mode from the caller.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.cfg = cfg
        self.lat_weight_tensor = lat_weight_tensor
        self.device = device
        self.target_mode = target_mode

        # T_max defaults to the full epoch budget -- pretrain PLUS
        # fine-tune, so the cosine curve reaches min_lr right at the end
        # of the whole run (fine-tuning continues the same decaying
        # curve, not a separate schedule).
        t_max = cfg.lr_scheduler_t_max or (cfg.epochs + cfg.finetune_epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=cfg.min_lr,
        )

        self.history = {"train_loss": [], "val_loss": []}
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # Auto-resume from latest.pt if it exists.
        ckpt = load_checkpoint(cfg.checkpoint_dir, "latest.pt", self.model, self.optimizer, device)
        if ckpt is not None:
            # A checkpoint's weights were trained under whatever
            # target_mode produced them -- resuming under a different one
            # would train against the wrong loss target. ckpt.get(...)
            # default covers checkpoints saved before target_mode existed.
            ckpt_target_mode = ckpt.get("target_mode", "direct")
            if ckpt_target_mode != self.target_mode:
                raise ValueError(
                    f"Checkpoint at '{cfg.checkpoint_dir}/latest.pt' was trained with "
                    f"target_mode='{ckpt_target_mode}', but this run is configured with "
                    f"target_mode='{self.target_mode}'. Use a distinct run_name for a "
                    f"fresh run instead."
                )
            self.start_epoch = ckpt["epoch"] + 1
            self.best_val_loss = ckpt["best_val_loss"]
            self.history = ckpt["history"]

            # Replaying the scheduler correctly on resume is trickier than
            # it looks -- two things matter here:
            #   1. Use the CLOSED-FORM scheduler.step(epoch=...), not a
            #      loop of bare step() calls. Bare step() computes the
            #      next LR from the OPTIMIZER'S CURRENT lr, which
            #      load_checkpoint just overwrote to its already-decayed,
            #      end-of-run value -- looping bare step() from there
            #      never recovers. The closed form recomputes purely from
            #      epoch count and this run's own base_lrs instead, so
            #      it's unaffected by whatever lr got loaded.
            #   2. Use ckpt["epoch"] + 1, not ckpt["epoch"]. At save time,
            #      epoch 0..ckpt["epoch"] had each already called .step()
            #      once, so the scheduler's true position is one step
            #      further than the saved epoch number.
            # (Both verified against an uninterrupted reference run --
            # see CODE_REFERENCE.md. The scheduler.step(epoch=...) form is
            # soft-deprecated by PyTorch, hence the warning suppression.)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
                warnings.filterwarnings("ignore", message=".*epoch parameter.*")
                self.scheduler.step(ckpt["epoch"] + 1)
            print(f"Resumed from checkpoint at epoch {ckpt['epoch']}")

    def _run_epoch(self, loader: DataLoader, train: bool, n_future_steps: int = 1) -> float:
        """
        One pass over `loader`. Updates weights only if train=True.

        n_future_steps=1 (pretrain): loader yields (x, y) pairs, y shape
        (B, C, H, W) -- one forward pass, one loss term, same as always.

        n_future_steps>1 (fine-tune): loader yields (x, y) from a
        MultiStepDataset, y shape (B, n_future_steps, C, H, W). Rolls the
        model forward n_future_steps times, feeding its own reconstructed
        output back in as the next input each time (exactly like
        inference/predict.py::rollout, not like being handed the true
        intermediate states) -- and sums the loss from every step before
        one single backward pass, matching FourCastNet's fine-tuning
        procedure.
        """
        self.model.train(mode=train)
        total_loss, n_batches = 0.0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(train):
                if n_future_steps == 1:
                    pred = self.model(x)
                    target = (y - x) if self.target_mode == "residual" else y
                    loss = lat_weighted_mse(pred, target, self.lat_weight_tensor)
                else:
                    state = x
                    loss = 0.0
                    for step in range(n_future_steps):
                        y_step = y[:, step]
                        raw = self.model(state)
                        target = (y_step - state) if self.target_mode == "residual" else y_step
                        loss = loss + lat_weighted_mse(raw, target, self.lat_weight_tensor)
                        state = (state + raw) if self.target_mode == "residual" else raw

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            train_loader_2step: DataLoader = None, val_loader_2step: DataLoader = None):
        """
        Train until cfg.epochs + cfg.finetune_epochs, or early stopping,
        whichever comes first. Returns the history dict (train_loss,
        val_loss, epoch_time_seconds -- one entry per epoch, all three
        the same length, and all persisted across resumes).

        train_loader_2step/val_loader_2step (built from a MultiStepDataset
        with n_future_steps=2, see scripts/train.py) are required if
        cfg.finetune_epochs > 0 -- that's the fine-tune phase's data.
        """
        if self.cfg.finetune_epochs > 0 and (train_loader_2step is None or val_loader_2step is None):
            raise ValueError(
                "cfg.finetune_epochs > 0 but train_loader_2step/val_loader_2step weren't given."
            )

        total_epochs = self.cfg.epochs + self.cfg.finetune_epochs
        patience_counter = 0

        for epoch in range(self.start_epoch, total_epochs):
            finetune = epoch >= self.cfg.epochs
            if finetune and epoch == self.cfg.epochs:
                # Fresh phase, different loss scale (fine-tune sums TWO
                # steps' MSE) -- a pretrain-phase best_val_loss would
                # otherwise block best.pt from ever updating again.
                self.best_val_loss = float("inf")
                patience_counter = 0

            loader, vloader, n_future_steps = (
                (train_loader_2step, val_loader_2step, 2) if finetune
                else (train_loader, val_loader, 1)
            )

            t0 = time.time()
            train_loss = self._run_epoch(loader, train=True, n_future_steps=n_future_steps)
            val_loss = self._run_epoch(vloader, train=False, n_future_steps=n_future_steps)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            # Pure compute time (excludes data-fetch/setup) -- persisted
            # so total training cost survives resumes and is comparable
            # across runs without re-timing anything. setdefault covers
            # checkpoints saved before this field existed.
            self.history.setdefault("epoch_time_seconds", []).append(elapsed)

            # Cosine schedule moves every epoch on its fixed curve --
            # unlike ReduceLROnPlateau, it takes no val_loss argument.
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            phase = "finetune" if finetune else "pretrain"
            print(f"epoch {epoch:03d} [{phase}] | train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"lr {current_lr:.2e} | {elapsed:.1f}s")

            # Always save latest.pt -- this is what auto-resume picks up.
            state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                "config": self.cfg,
                "target_mode": self.target_mode,
            }
            save_checkpoint(state, self.cfg.checkpoint_dir, "latest.pt")

            # best.pt + early stopping, driven by val loss only.
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

        total_seconds = sum(self.history.get("epoch_time_seconds", []))
        print(f"Total training compute time (sum of every epoch this run has ever done, "
              f"across all resumes -- excludes data-fetch/setup time): "
              f"{total_seconds / 3600:.2f}h ({total_seconds:.0f}s)")
        return self.history
