"""
Time-based train/val split helpers.

Weather data is a time series — a random split leaks information between
adjacent, highly correlated timesteps. Always split by date instead.
"""

from __future__ import annotations

from weather_fno.config import DataConfig
from weather_fno.data.gcs_dataset import GCSWeatherDataset


def build_train_val_datasets(data_cfg: DataConfig):
    # Builds train FIRST (fits fresh normalisation stats), then val reusing
    # those same stats -- val must never fit its own, or it's no longer a
    # fair held-out measure of how well train-derived normalisation
    # generalises.
    train_cache = data_cfg.stats_cache_path.replace("normalisation_stats", "train_cache")
    val_cache = data_cfg.stats_cache_path.replace("normalisation_stats", "val_cache")

    train_ds = GCSWeatherDataset(
        gcs_bucket_path=data_cfg.gcs_bucket_path,
        channels=data_cfg.channels,
        start=data_cfg.train_start,
        end=data_cfg.train_end,
        flip_lat=data_cfg.flip_lat,
        flip_lon=data_cfg.flip_lon,
        lat_dim=data_cfg.lat_dim,
        lon_dim=data_cfg.lon_dim,
        stats=None,  # computed fresh from the training split
        cache_path=train_cache,
    )

    val_ds = GCSWeatherDataset(
        gcs_bucket_path=data_cfg.gcs_bucket_path,
        channels=data_cfg.channels,
        start=data_cfg.val_start,
        end=data_cfg.val_end,
        flip_lat=data_cfg.flip_lat,
        flip_lon=data_cfg.flip_lon,
        lat_dim=data_cfg.lat_dim,
        lon_dim=data_cfg.lon_dim,
        stats=train_ds.stats,  # reuse TRAIN stats — never fit stats on val
        cache_path=val_cache,
    )

    return train_ds, val_ds
