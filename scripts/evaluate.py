"""
CLI entrypoint: evaluate a trained model's autoregressive forecast against
ground truth on every configured inference target -- lat-weighted RMSE per
lead time vs a persistence baseline (and a climatology baseline, on
coarse-grid targets -- see data/climatology.py), plus the model's anomaly
correlation coefficient (ACC) vs lead time, plus forecast-vs-actual map
plots.

Saves, per target, into cfg.inference.output_dir:
  {target.name}_eval_metrics.npz       lead_hours, model_rmse, model_rmse_banded
                                        (per latitude band, see
                                        training/metrics.py::lat_banded_rmse_per_channel),
                                        lat_band_labels, persistence_rmse,
                                        + climatology_rmse/model_acc if available
  {target.name}_rmse_vs_lead_time.png  scorecard for a few headline channels
  {target.name}_acc_vs_lead_time.png   ACC scorecard (only for targets with climatology)
  {target.name}_{MAP_CHANNEL}_maps.png ground truth / forecast / error maps

climatology_rmse/model_acc need outputs/stats_{train_start}_{train_end}/
climatology.npz (shared across every run using that same training split,
see config.py::derive_run_paths), computed automatically by train.py --
for a run trained before this feature existed (or before this sharing
existed), run scripts/compute_climatology.py once to backfill it (doesn't
need the model/checkpoint, so it's safe to run any time). A run without
it just gets model_rmse/persistence_rmse as before, with a printed
warning, not an error.

Usage:
    python scripts/evaluate.py --config configs/baseline_fno.yaml
    # Evaluating a specific experiment: pass the SAME --experiment file
    # used to train it, so run_name (and hence checkpoint_path) resolves
    # to that run, not the base config's own -- see
    # configs/experiments/example.yaml.
    python scripts/evaluate.py --config configs/baseline_fno.yaml --experiment configs/experiments/example.yaml
    # Restrict to specific inference targets (by name, from --config's
    # inference.targets), e.g. only the coarse (in-distribution) grid:
    python scripts/evaluate.py --config configs/baseline_fno.yaml --targets coarse
"""

from __future__ import annotations

import argparse

from weather_fno.config import load_config
from weather_fno.inference.evaluate import run_full_evaluation

# Small, standard set of variables for the RMSE scorecard -- surface temp,
# mid-tropospheric height, sea-level pressure, near-surface wind. Edit this
# list (using any short_name from configs/baseline_fno.yaml's channel list)
# to change what gets plotted.
HEADLINE_CHANNELS = ["t2m", "z500", "mslp", "u10"]

# Which single channel to show full ground-truth/forecast/error maps for.
MAP_CHANNEL = "t2m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_fno.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    # Restrict evaluation to a subset of inference.targets by name (e.g.
    # just "coarse"), instead of every target configured in --config.
    parser.add_argument("--targets", type=str, nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.experiment)

    if args.targets is not None:
        available = {t.name for t in cfg.inference.targets}
        unknown = set(args.targets) - available
        if unknown:
            raise ValueError(f"--targets {sorted(unknown)} not found. Available: {sorted(available)}")
        cfg.inference.targets = [t for t in cfg.inference.targets if t.name in args.targets]

    # The actual work (load stats/climatology, score every target, save
    # metrics + plots) lives in inference/evaluate.py::run_full_evaluation
    # -- shared with scripts/run_seed_ensemble.py, which calls it once per
    # seed -- this file is just CLI arg parsing on top.
    run_full_evaluation(cfg, headline_channels=HEADLINE_CHANNELS, map_channel=MAP_CHANNEL)


if __name__ == "__main__":
    main()
