"""
GCS zarr store access.

Kept as a single-function module so both the training dataset
(data/gcs_dataset.py) and the inference loader (inference/predict.py) open
stores the exact same way — no risk of one path picking up different
storage options than the other.
"""

from __future__ import annotations

import xarray as xr


def open_dataset(gcs_bucket_path: str) -> xr.Dataset:
    """
    Open a zarr store on GCS.

    WeatherBench2's public buckets are readable anonymously, so anonymous
    access is tried first, with consolidated metadata (a single small
    `.zmetadata` read instead of listing the whole store). Prints which
    path actually succeeded — this only ever opens the store lazily (no
    array data is downloaded here), but with three fallback tiers it's
    otherwise impossible to tell which one silently fired.
    """
    try:
        ds = xr.open_zarr(gcs_bucket_path, storage_options={"token": "anon"}, consolidated=True)
        print(f"[open_dataset] {gcs_bucket_path}: opened anonymously (consolidated metadata)")
        return ds
    except Exception as e:
        print(f"[open_dataset] {gcs_bucket_path}: anonymous+consolidated open failed "
              f"({e!r}); retrying anonymous without consolidated metadata "
              f"(slower — lists the store's full structure)...")

    # Still anonymous here deliberately — a public bucket almost never needs
    # real credentials, and jumping straight to default-credential
    # resolution on a machine with none configured can hang for a while
    # probing for them before it even gets to opening the store.
    try:
        ds = xr.open_zarr(gcs_bucket_path, storage_options={"token": "anon"})
        print(f"[open_dataset] {gcs_bucket_path}: opened anonymously (non-consolidated)")
        return ds
    except Exception as e:
        print(f"[open_dataset] {gcs_bucket_path}: anonymous open failed too "
              f"({e!r}); falling back to default GCS credentials...")

    return xr.open_zarr(gcs_bucket_path)
