"""
PyTorch Dataset for the coarse (64x32, 20-channel) ERA5-style training
data, streamed from a GCS zarr store.

The whole split fits in memory, so __init__ fetches and normalises it all
up front. Only the small normalisation stats (mean/std/lat_values) ever
get cached to disk -- not the full array, which can be several GB and
would risk filling up a shared/quota-limited disk. So every run re-fetches
from GCS on startup instead (a few minutes, but bounded).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from weather_fno.config import ChannelSpec
from weather_fno.data.io import open_dataset
from weather_fno.data.preprocessing import flip_axes, normalise


def _select_channels(
    ds: xr.Dataset, channels: List[ChannelSpec], lat_dim: str, lon_dim: str
) -> List[np.ndarray]:
    """
    Pull every configured channel out of the store, in configured order.

    Several channels share the same variable at different pressure levels
    (e.g. z1000/z850/z500 are all `geopotential`). Levels of the same
    variable usually live in the same zarr chunk, so we group by variable
    name and fetch all needed levels in one `.sel(level=[...])` call --
    fetching each level separately would re-download the same chunk once
    per level.

    Transposes to (time, [level,] lat, lon) by dimension NAME, not
    position, so this is correct regardless of how the store lays out its
    axes internally.
    """
    by_name: Dict[str, List[ChannelSpec]] = {}
    for spec in channels:
        by_name.setdefault(spec.name, []).append(spec)

    values_by_id: Dict[int, np.ndarray] = {}
    for name, specs in by_name.items():
        t0 = time.time()
        da = ds[name]
        if "level" in da.dims:
            levels = [s.level for s in specs]
            da = da.sel(level=levels).transpose("time", "level", lat_dim, lon_dim)
            values = da.values  # one fetch, covering every level of this variable
            for i, spec in enumerate(specs):
                values_by_id[id(spec)] = values[:, i]
            level_desc = f"{len(levels)} level(s): {levels}"
        else:
            da = da.transpose("time", lat_dim, lon_dim)
            values = da.values
            for spec in specs:
                values_by_id[id(spec)] = values
            level_desc = "surface/integrated"
        print(f"  fetched {name} ({level_desc}) in {time.time() - t0:.1f}s")

    return [values_by_id[id(spec)] for spec in channels]


class GCSWeatherDataset(Dataset):
    """One (train or val) split of the training data. __getitem__ returns
    (x, y) pairs of adjacent 6-hourly timesteps for next-step prediction."""

    def __init__(
        self,
        gcs_bucket_path: str,
        channels: List[ChannelSpec],
        start: str,
        end: str,
        flip_lat: bool,
        flip_lon: bool,
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
        stats: Optional[Dict[str, np.ndarray]] = None,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            gcs_bucket_path: zarr path of the training store.
            channels: ordered list of variables to stack into channels.
            start, end: inclusive date strings bounding this split.
            flip_lat, flip_lon: orientation fixes for this store (e.g. it
                might store latitude north-to-south instead of the other
                way round) -- separate from axis ORDER, which is always
                handled correctly above via the named transpose.
            lat_dim, lon_dim: the store's actual dimension names.
            stats: {"mean": ..., "std": ...} to normalise with. Pass None
                to fit fresh stats from this split's own data (the TRAIN
                split only) -- pass the train split's stats back in here
                when building the val split, so both are normalised the
                same way.
            cache_path: where to save the fitted stats, so a later process
                (evaluate.py, predict_single_variable.py) can reuse them
                without needing this run's data in memory. Only saved when
                stats=None -- i.e. only when this call is the one actually
                fitting fresh stats.
        """
        self.channels = channels
        self.flip_lat = flip_lat
        self.flip_lon = flip_lon

        # 1. Open the store (lazy, nothing downloaded yet) and narrow to
        # this split's date range.
        ds = open_dataset(gcs_bucket_path)
        ds = ds.sel(time=slice(start, end))

        # 2. Real latitude values, flipped the same way as the data below
        # so weights[i] always lines up with row i of self.data.
        lat_values = ds[lat_dim].values
        if flip_lat:
            lat_values = lat_values[::-1]
        self.lat_values = lat_values

        # 3. Fetch every configured channel and stack into (T, C, H, W).
        print(f"Fetching {len(channels)} channels ({start} to {end}) from {gcs_bucket_path}...")
        t0 = time.time()
        arr = np.stack(_select_channels(ds, channels, lat_dim, lon_dim), axis=1)
        print(f"Done fetching in {time.time() - t0:.1f}s")

        # 4. Orientation fix, then per-channel standardisation.
        arr = flip_axes(arr, flip_lat=flip_lat, flip_lon=flip_lon)
        arr, self.stats = normalise(arr, stats=stats)

        # 5. Cache the (tiny) stats only -- see class docstring for why we
        # never cache the full array.
        if cache_path and stats is None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, mean=self.stats["mean"], std=self.stats["std"],
                     lat_values=self.lat_values)

        self.data = torch.from_numpy(arr).float()

    def __len__(self) -> int:
        # One less than the timestep count -- __getitem__ pairs index i
        # with i+1.
        return self.data.shape[0] - 1

    def __getitem__(self, idx: int):
        # x = current timestep, y = next timestep. Every store here is
        # uniformly 6-hourly, so this pairing is exactly "predict 6h
        # ahead" -- there's no separate horizon setting anywhere else.
        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y
