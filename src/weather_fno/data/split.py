"""
Builds the train and validation datasets, split by date (not randomly --
weather timesteps are highly correlated, so a random split would leak
information between train and val).
"""

from __future__ import annotations

from weather_fno.config import DataConfig
from weather_fno.data.gcs_dataset import GCSWeatherDataset


def build_train_val_datasets(data_cfg: DataConfig):
    """Build train first (fits fresh normalisation stats from its own
    data), then val reusing those same stats -- val must never fit its
    own stats, or it stops being a fair measure of generalisation."""
    train_cache = data_cfg.stats_cache_path.replace("normalisation_stats", "train_cache")

    train_ds = GCSWeatherDataset(
        gcs_bucket_path=data_cfg.gcs_bucket_path,
        channels=data_cfg.channels,
        start=data_cfg.train_start,
        end=data_cfg.train_end,
        flip_lat=data_cfg.flip_lat,
        flip_lon=data_cfg.flip_lon,
        lat_dim=data_cfg.lat_dim,
        lon_dim=data_cfg.lon_dim,
        stats=None,  # None = fit fresh stats from this (training) data
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
        stats=train_ds.stats,  # reuse TRAIN stats -- never fit stats on val
    )

    return train_ds, val_ds
