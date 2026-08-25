"""
Trains several experiments in sequence, unattended -- e.g. leave this
running in a tmux/screen session overnight to cover every experiment file
you want trained, and come back later to evaluate them all
(scripts/evaluate_experiments.py).

Each experiment runs as its own SEPARATE subprocess (a fresh
`python scripts/train.py --experiment ...` call), not an in-process loop
like scripts/sweep.py -- deliberately, so a crash in one experiment (e.g.
a CUDA "unspecified launch failure" on a shared GPU) can't taint the CUDA
context for the ones after it. A crashed experiment is logged and
skipped; the rest still run. Since Trainer auto-resumes from its own
checkpoint, rerunning this same command later retries any
failed/interrupted experiment from wherever it left off, and quickly
no-ops on any experiment that's already fully trained.

The summary also reports, per experiment (read back from its own
checkpoint, not recomputed): wall-clock time for the whole subprocess
(includes data fetch/setup), pure training compute time (sum of every
epoch's own time, from Trainer.fit's history -- excludes fetch/setup, so
it's the fairer number for comparing different architectures' actual
training cost), and parameter count -- everything you need to compare the
"cost" of different models side by side without re-timing anything.

Usage:
    python scripts/run_experiments.py --config configs/baseline_fno.yaml \\
        --experiments configs/experiments/target_mode_direct.yaml \\
                       configs/experiments/target_mode_residual.yaml \\
                       configs/experiments/twostep_finetune.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from weather_fno.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_fno.yaml")
    parser.add_argument("--experiments", type=str, nargs="+", required=True,
                         help="One or more --experiment file paths to train in sequence.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    summary_path = Path(base_cfg.training.output_dir) / "batch_train_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    try:
        for experiment_path in args.experiments:
            label = Path(experiment_path).stem
            print(f"\n{'=' * 70}\n[{label}] starting...\n{'=' * 70}", flush=True)

            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, "scripts/train.py", "--config", args.config, "--experiment", experiment_path]
            )
            wall_time = time.time() - t0

            status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"

            # Read back what this experiment actually cost, straight from
            # its own checkpoint -- not recomputed, just collected. None
            # for a failed experiment (there may be no checkpoint at all).
            train_time_seconds = n_params = None
            if proc.returncode == 0:
                cfg = load_config(args.config, override_path=experiment_path)
                ckpt = torch.load(Path(cfg.training.checkpoint_dir) / "latest.pt",
                                   map_location="cpu", weights_only=False)
                train_time_seconds = sum(ckpt["history"].get("epoch_time_seconds", []))
                # neuralop's state_dict() includes a non-tensor "_metadata"
                # entry (construction hyperparameters) alongside the real
                # weights -- filter to actual tensors, confirmed directly
                # rather than assumed (every value being a tensor is NOT
                # a safe assumption for this model's state_dict()).
                n_params = sum(t.numel() for t in ckpt["model_state"].values() if torch.is_tensor(t))

            print(f"[{label}] {status} (wall clock: {wall_time / 60:.1f} min)", flush=True)
            results.append((label, status, wall_time, train_time_seconds, n_params))
    finally:
        # Written even on a KeyboardInterrupt/early exit, so a partial
        # batch still leaves a record of what ran and what didn't.
        with open(summary_path, "w") as f:
            for label, status, wall_time, train_time_seconds, n_params in results:
                train_str = f"{train_time_seconds / 3600:.2f}h train-compute" if train_time_seconds else "n/a"
                params_str = f"{n_params:,} params" if n_params else "n/a"
                f.write(f"{label:30s} {status:20s} {wall_time / 60:6.1f} min wall-clock, "
                        f"{train_str:22s} {params_str}\n")

    print(f"\n{'=' * 70}\nSummary (also written to {summary_path})\n{'=' * 70}")
    for label, status, wall_time, train_time_seconds, n_params in results:
        train_str = f"{train_time_seconds / 3600:.2f}h train-compute" if train_time_seconds else "n/a"
        params_str = f"{n_params:,} params" if n_params else "n/a"
        print(f"  {label:30s} {status:20s} {wall_time / 60:6.1f} min wall-clock, "
              f"{train_str:22s} {params_str}")

    failed = [r for r in results if r[1] != "OK"]
    if failed:
        print(f"\n{len(failed)} experiment(s) failed -- rerun this same command later to "
              f"retry them (auto-resume picks each one up from its own last checkpoint). "
              f"Already-finished experiments still re-fetch their data on a rerun (a few "
              f"minutes) but harmlessly no-op on training itself, so it's always safe to "
              f"pass the SAME full --experiments list again rather than figuring out which "
              f"ones actually still need it.")


if __name__ == "__main__":
    main()
