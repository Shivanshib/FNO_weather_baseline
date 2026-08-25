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
            elapsed = time.time() - t0

            status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"[{label}] {status} ({elapsed / 60:.1f} min)", flush=True)
            results.append((label, status, elapsed))
    finally:
        # Written even on a KeyboardInterrupt/early exit, so a partial
        # batch still leaves a record of what ran and what didn't.
        with open(summary_path, "w") as f:
            for label, status, elapsed in results:
                f.write(f"{label:30s} {status:20s} {elapsed / 60:.1f} min\n")

    print(f"\n{'=' * 70}\nSummary (also written to {summary_path})\n{'=' * 70}")
    for label, status, elapsed in results:
        print(f"  {label:30s} {status:20s} {elapsed / 60:.1f} min")

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
