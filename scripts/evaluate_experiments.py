"""
Evaluates several trained experiments in sequence -- the companion to
run_experiments.py's batch training. Each experiment runs as its own
subprocess (same crash-isolation reasoning as run_experiments.py), then a
combined summary table (mean RMSE across channels at the final lead time,
model vs persistence, per target, per experiment) gets printed and saved
to CSV, so you can see every experiment's results side by side without
hunting through each run's own outputs/{run_name}/predictions/ folder
individually.

Doesn't recompute anything -- reads back the {target}_eval_metrics.npz
each scripts/evaluate.py call already saves, same as any other consumer
of those files (e.g. notebooks/inspect_predictions.ipynb).

Usage:
    python scripts/evaluate_experiments.py --config configs/baseline_fno.yaml \\
        --experiments configs/experiments/target_mode_direct.yaml \\
                       configs/experiments/target_mode_residual.yaml \\
        --targets coarse
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from weather_fno.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_fno.yaml")
    parser.add_argument("--experiments", type=str, nargs="+", required=True,
                         help="One or more --experiment file paths to evaluate in sequence.")
    parser.add_argument("--targets", type=str, nargs="+", default=None,
                         help="Passed straight through to each evaluate.py call, e.g. --targets coarse.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    summary_path = Path(base_cfg.training.output_dir) / "batch_eval_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []  # (run_label, target_name, model_rmse, persistence_rmse)

    for experiment_path in args.experiments:
        label = Path(experiment_path).stem
        print(f"\n{'=' * 70}\n[{label}] evaluating...\n{'=' * 70}", flush=True)

        cmd = [sys.executable, "scripts/evaluate.py", "--config", args.config, "--experiment", experiment_path]
        if args.targets:
            cmd += ["--targets", *args.targets]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[{label}] FAILED (exit {proc.returncode}) -- excluded from the summary")
            continue

        # Read back whatever this run's own evaluate.py call just saved --
        # same targets it was just run against (args.targets if given,
        # otherwise every target configured for this run).
        cfg = load_config(args.config, override_path=experiment_path)
        targets = [t for t in cfg.inference.targets if not args.targets or t.name in args.targets]
        for target in targets:
            metrics_path = Path(cfg.inference.output_dir) / f"{target.name}_eval_metrics.npz"
            if not metrics_path.exists():
                continue
            m = np.load(metrics_path)
            rows.append((label, target.name,
                         float(m["model_rmse"][-1].mean()),
                         float(m["persistence_rmse"][-1].mean())))

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "target", "final_model_rmse", "final_persistence_rmse"])
        writer.writerows(rows)

    print(f"\n{'=' * 70}\nSummary -- mean RMSE across channels at the final lead time "
          f"(also written to {summary_path})\n{'=' * 70}")
    print(f"{'experiment':30s} {'target':15s} {'model':>10s} {'persistence':>12s}")
    for label, target_name, model_rmse, persistence_rmse in rows:
        print(f"{label:30s} {target_name:15s} {model_rmse:10.4g} {persistence_rmse:12.4g}")


if __name__ == "__main__":
    main()
