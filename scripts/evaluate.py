"""
CLI entrypoint: evaluate a trained model's autoregressive forecast against
ground truth on every configured inference target -- lat-weighted RMSE per
lead time vs a persistence baseline (and a climatology baseline, on
coarse-grid targets -- see data/climatology.py), plus the model's anomaly
correlation coefficient (ACC) vs lead time, plus forecast-vs-actual map
plots.

Saves, per target, into cfg.inference.output_dir:
  {target.name}_eval_metrics.npz       lead_hours, model_rmse, persistence_rmse,
                                        + climatology_rmse/model_acc if available
  {target.name}_rmse_vs_lead_time.png  scorecard for a few headline channels
  {target.name}_acc_vs_lead_time.png   ACC scorecard (only for targets with climatology)
  {target.name}_{MAP_CHANNEL}_maps.png ground truth / forecast / error maps

climatology_rmse/model_acc need outputs/{run_name}/stats/climatology.npz,
computed automatically by train.py -- for a run trained before this
feature existed, run scripts/compute_climatology.py once to backfill it
(doesn't need the model/checkpoint, so it's safe to run any time). A run
without it just gets model_rmse/persistence_rmse as before, with a
printed warning, not an error.

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
from pathlib import Path

import numpy as np

from weather_fno.config import load_config
from weather_fno.inference.evaluate import get_target_lat_lon, run_evaluation
from weather_fno.utils.plotting import plot_acc_vs_lead_time, plot_forecast_maps, plot_rmse_vs_lead_time

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

    # 1. Load the exact normalisation stats the training run fit -- never
    # re-fit stats on inference/eval data.
    train_cache_path = cfg.data.stats_cache_path.replace("normalisation_stats", "train_cache")
    cached = np.load(train_cache_path)
    train_stats = {"mean": cached["mean"], "std": cached["std"]}

    # Climatology is optional -- a run trained before this feature existed
    # won't have it cached yet. Degrade gracefully (warn, skip
    # climatology/ACC) rather than failing the whole evaluation.
    climatology_path = cfg.data.stats_cache_path.replace("normalisation_stats", "climatology")
    if Path(climatology_path).exists():
        clim_cached = np.load(climatology_path)
        climatology = {"climatology": clim_cached["climatology"], "hours_of_day": clim_cached["hours_of_day"]}
    else:
        climatology = None
        print(f"[evaluate] no climatology cached at {climatology_path} -- skipping the "
              f"climatology baseline and ACC (run scripts/compute_climatology.py to backfill it).")

    # 2. Run the full scored rollout for every configured target (all the
    # actual compute happens here -- everything below is just plotting
    # already-computed results).
    results = run_evaluation(cfg, train_stats, climatology=climatology)

    out_dir = Path(cfg.inference.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Per target: save the raw metrics, then the plots from them.
    for target in cfg.inference.targets:
        result = results[target.name]
        has_climatology = "model_acc" in result

        metrics_path = out_dir / f"{target.name}_eval_metrics.npz"
        metrics_to_save = {"lead_hours": result["lead_hours"], "model_rmse": result["model_rmse"],
                            "persistence_rmse": result["persistence_rmse"],
                            "rollout_time_seconds": result["rollout_time_seconds"]}
        if has_climatology:
            metrics_to_save["climatology_rmse"] = result["climatology_rmse"]
            metrics_to_save["model_acc"] = result["model_acc"]
        np.savez(metrics_path, **metrics_to_save)
        print(f"[{target.name}] saved metrics to {metrics_path}")

        scorecard_path = out_dir / f"{target.name}_rmse_vs_lead_time.png"
        plot_rmse_vs_lead_time(result, cfg.data.channels, HEADLINE_CHANNELS,
                                str(scorecard_path), title=f"{target.name} -- RMSE vs lead time")
        print(f"[{target.name}] saved scorecard plot to {scorecard_path}")

        if has_climatology:
            acc_path = out_dir / f"{target.name}_acc_vs_lead_time.png"
            plot_acc_vs_lead_time(result, cfg.data.channels, HEADLINE_CHANNELS,
                                   str(acc_path), title=f"{target.name} -- ACC vs lead time")
            print(f"[{target.name}] saved ACC plot to {acc_path}")

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
