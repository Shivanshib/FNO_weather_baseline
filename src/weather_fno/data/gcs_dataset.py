"""
Streaming Dataset for coarse-resolution ERA5-style data stored as zarr on GCS.

The 64x32, 20-channel training data is small enough to fit comfortably in
memory, so this class opens the store lazily via xarray + gcsfs and applies
preprocessing ONCE per process. Only the small normalisation stats
(mean/std/lat_values, a few KB) are ever persisted to disk -- NOT the full
preprocessed array, which can be several GB for the full training date
range and risks blowing a disk quota on space-constrained filesystems
(university HPC home directories are commonly quota-limited). Every
dataset build therefore always re-fetches from GCS -- a bounded, one-time
network cost per process start (a few minutes for the full training
range) rather than an unbounded disk-space cost.
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
    """Pull every configured channel out of the store, in configured order.

    Several channels share the same underlying variable at different
    pressure levels (e.g. z1000/z850/z500/z50 are all `geopotential`).
    Zarr reads are chunk-granular — if a variable's levels sit in the same
    chunk (common; there are usually only a handful of pressure levels),
    selecting each level separately re-downloads and re-decompresses that
    same chunk once per level requested. Grouping by variable NAME and
    pulling every needed level in a single `.sel(level=[...])` call fetches
    each distinct variable exactly once, regardless of how many channels
    it feeds.

    Transposes to a guaranteed (time, [level,] lat_dim, lon_dim) axis order
    BY NAME rather than assuming positional order — correct regardless of
    how the store physically lays the array out.
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
            gcs_bucket_path: e.g. "gs://TODO-bucket/TODO-path.zarr"
            channels: ordered list of variable names to stack into channels.
            start, end: inclusive date strings bounding this split.
            flip_lat, flip_lon: orientation corrections (e.g. store runs
                north-to-south but the model expects south-to-north) — this
                is DIFFERENT from axis order, which is now always handled
                correctly via the named transpose above.
            lat_dim, lon_dim: actual dimension names in the store.
            stats: normalisation stats dict {"mean": ..., "std": ...}. Pass
                the stats computed on the TRAIN split when building the val
                dataset, so val is normalised identically to train.
            cache_path: optional path to persist the fitted normalisation
                stats (mean/std/lat_values only — NOT the full data array,
                see module docstring) so a later, separate process (e.g.
                scripts/infer.py) can reuse the exact stats a training run
                fit without needing that run's data still in memory. Only
                written when stats=None (i.e. this call is FITTING fresh
                stats — the train split), never when reusing stats passed
                in from elsewhere (val/inference).
        """
        self.channels = channels
        self.flip_lat = flip_lat
        self.flip_lon = flip_lon

        ds = open_dataset(gcs_bucket_path)
        ds = ds.sel(time=slice(start, end))

        # Real latitude values from the store — flipped to match the
        # data array below if flip_lat is set, so weights[i] always
        # corresponds to row i of self.data regardless of orientation.
        lat_values = ds[lat_dim].values
        if flip_lat:
            lat_values = lat_values[::-1]
        self.lat_values = lat_values

        # TODO: confirm variable naming (spec.name) matches your GCS
        # store's schema exactly.
        print(f"Fetching {len(channels)} channels ({start} to {end}) from {gcs_bucket_path}...")
        t0 = time.time()
        arr = np.stack(
            _select_channels(ds, channels, lat_dim, lon_dim), axis=1
        )  # (T, C, H, W) — H=lat_dim, W=lon_dim, guaranteed by the transpose above
        print(f"Done fetching in {time.time() - t0:.1f}s")

        arr = flip_axes(arr, flip_lat=flip_lat, flip_lon=flip_lon)

        arr, self.stats = normalise(arr, stats=stats)

        if cache_path and stats is None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, mean=self.stats["mean"], std=self.stats["std"],
                     lat_values=self.lat_values)

        self.data = torch.from_numpy(arr).float()

    def __len__(self) -> int:
        # TODO: adjust once you decide the input/target pairing (e.g. single
        # timestep -> next timestep for a baseline autoregressive setup).
        return self.data.shape[0] - 1

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y
