"""
Streaming Dataset for coarse-resolution ERA5-style data stored as zarr on GCS.

The 64x32, 20-channel training data is small enough to fit comfortably in
memory, so this class opens the store lazily via xarray + gcsfs, applies
preprocessing ONCE, and caches the result (in memory + optionally to disk)
rather than re-reading/reprocessing from GCS every epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from weather_fno.config import ChannelSpec
from weather_fno.data.preprocessing import flip_axes, normalise


def _select_channel(ds: xr.Dataset, spec: ChannelSpec) -> np.ndarray:
    """Pull a single channel out of the store, selecting a pressure level
    when the field isn't surface/integrated (level is None)."""
    da = ds[spec.name]
    if spec.level is not None:
        da = da.sel(level=spec.level)
    return da.values


class GCSWeatherDataset(Dataset):
    def __init__(
        self,
        gcs_bucket_path: str,
        channels: List[ChannelSpec],
        start: str,
        end: str,
        flip_lat: bool,
        flip_lon: bool,
        stats: Optional[Dict[str, np.ndarray]] = None,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            gcs_bucket_path: e.g. "gs://TODO-bucket/TODO-path.zarr"
            channels: ordered list of variable names to stack into channels.
            start, end: inclusive date strings bounding this split.
            flip_lat, flip_lon: axis corrections for this data source.
            stats: normalisation stats dict {"mean": ..., "std": ...}. Pass
                the stats computed on the TRAIN split when building the val
                dataset, so val is normalised identically to train.
            cache_path: optional path to cache the preprocessed array to disk
                so repeat runs skip GCS entirely.
        """
        self.channels = channels
        self.flip_lat = flip_lat
        self.flip_lon = flip_lon

        if cache_path and Path(cache_path).exists():
            cached = np.load(cache_path)
            arr = cached["data"]
            self.stats = {"mean": cached["mean"], "std": cached["std"]}
        else:
            ds = xr.open_zarr(gcs_bucket_path, chunks={"time": 1})
            ds = ds.sel(time=slice(start, end))

            # TODO: confirm variable naming (spec.name) matches your GCS
            # store's schema exactly.
            arr = np.stack([_select_channel(ds, c) for c in channels], axis=1)  # (T, C, H, W)

            arr = flip_axes(arr, flip_lat=flip_lat, flip_lon=flip_lon)

            arr, self.stats = normalise(arr, stats=stats)

            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, data=arr,
                         mean=self.stats["mean"], std=self.stats["std"])

        self.data = torch.from_numpy(arr).float()

    def __len__(self) -> int:
        # TODO: adjust once you decide the input/target pairing (e.g. single
        # timestep -> next timestep for a baseline autoregressive setup).
        return self.data.shape[0] - 1

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y
