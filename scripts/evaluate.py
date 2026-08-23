"""
CLI entrypoint: evaluate a trained model's autoregressive forecast against
ground truth on every configured inference target -- lat-weighted RMSE per
lead time vs a persistence baseline, plus forecast-vs-actual map plots.

Saves, per target, into cfg.inference.output_dir:
  {target.name}_eval_metrics.npz       lead_hours, model_rmse, persistence_rmse
  {target.name}_rmse_vs_lead_time.png  scorecard for a few headline channels
  {target.name}_{MAP_CHANNEL}_maps.png ground truth / forecast / error maps

Usage:
    python scripts/evaluate.py --config configs/baseline_fno.yaml
    # Evaluating a specific experiment: pass the SAME --experiment file
    # used to train it, so run_name (and hence checkpoint_path) resolves
    # to that run, not the base config's own -- see
    # configs/experiments/example.yaml.
    python scripts/evaluate.py --config configs/baseline_fno.yaml --experiment configs/experiments/example.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from weather_fno.config import load_config
from weather_fno.inference.evaluate import get_target_lat_lon, run_evaluation
from weather_fno.utils.plotting import plot_forecast_maps, plot_rmse_vs_lead_time

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
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.experiment)

    # 1. Load the exact normalisation stats the training run fit -- never
    # re-fit stats on inference/eval data.
    train_cache_path = cfg.data.stats_cache_path.replace("normalisation_stats", "train_cache")
    cached = np.load(train_cache_path)
    train_stats = {"mean": cached["mean"], "std": cached["std"]}

    # 2. Run the full scored rollout for every configured target (all the
    # actual compute happens here -- everything below is just plotting
    # already-computed results).
    results = run_evaluation(cfg, train_stats)

    out_dir = Path(cfg.inference.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Per target: save the raw metrics, then two plots from them.
    for target in cfg.inference.targets:
        result = results[target.name]

        metrics_path = out_dir / f"{target.name}_eval_metrics.npz"
        np.savez(metrics_path, lead_hours=result["lead_hours"],
                 model_rmse=result["model_rmse"], persistence_rmse=result["persistence_rmse"])
        print(f"[{target.name}] saved metrics to {metrics_path}")

        scorecard_path = out_dir / f"{target.name}_rmse_vs_lead_time.png"
        plot_rmse_vs_lead_time(result, cfg.data.channels, HEADLINE_CHANNELS,
                                str(scorecard_path), title=f"{target.name} -- RMSE vs lead time")
        print(f"[{target.name}] saved scorecard plot to {scorecard_path}")

        # A handful of representative lead times: ~15% in, ~halfway, and
        # the final step -- not hardcoded step numbers, so this adapts if
        # forecast_lead_steps changes.
        n_steps = cfg.inference.forecast_lead_steps
        lead_step_indices = sorted(set([
            max(0, round(n_steps * 0.15) - 1),
            max(0, round(n_steps * 0.5) - 1),
            n_steps - 1,
        ]))

        lat, lon = get_target_lat_lon(cfg, target)
        maps_path = out_dir / f"{target.name}_{MAP_CHANNEL}_maps.png"
        plot_forecast_maps(result, cfg.data.channels, MAP_CHANNEL, lat, lon,
                            lead_step_indices, str(maps_path), title=target.name)
        print(f"[{target.name}] saved forecast maps to {maps_path}")


if __name__ == "__main__":
    main()
