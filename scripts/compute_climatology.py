"""
Backfills outputs/{run_name}/stats/climatology.npz for a run that was
trained BEFORE climatology existed (train.py now computes this
automatically for every new run -- this script is only for runs started
earlier). Needs only the training split's data, not the model/checkpoint
at all, so this never touches or retrains anything.

Usage:
    python scripts/compute_climatology.py --config configs/baseline_fno.yaml --experiment configs/experiments/target_mode_direct.yaml
"""

from __future__ import annotations

import argparse

from weather_fno.config import load_config
from weather_fno.data.climatology import compute_and_save_climatology
from weather_fno.data.gcs_dataset import GCSWeatherDataset
from weather_fno.data.preprocessing import denormalise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_fno.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.experiment)

    # Train split only -- climatology doesn't need val data, so this
    # builds GCSWeatherDataset directly rather than going through
    # build_train_val_datasets (which would also fetch val for nothing).
    train_ds = GCSWeatherDataset(
        gcs_bucket_path=cfg.data.gcs_bucket_path,
        channels=cfg.data.channels,
        start=cfg.data.train_start,
        end=cfg.data.train_end,
        flip_lat=cfg.data.flip_lat,
        flip_lon=cfg.data.flip_lon,
        lat_dim=cfg.data.lat_dim,
        lon_dim=cfg.data.lon_dim,
    )

    climatology_path = cfg.data.stats_cache_path.replace("normalisation_stats", "climatology")
    train_physical = denormalise(train_ds.data.numpy(), train_ds.stats)
    compute_and_save_climatology(train_physical, train_ds.time_values, climatology_path)


if __name__ == "__main__":
    main()
