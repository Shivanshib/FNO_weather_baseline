"""
Opens a zarr store on GCS. One function, used by both the training
dataset and the inference loader, so every store gets opened the same way.
"""

from __future__ import annotations

import xarray as xr


def open_dataset(gcs_bucket_path: str) -> xr.Dataset:
    """
    Lazily open a zarr store (no array data downloaded yet).

    WeatherBench2's buckets are public, so we try anonymous access first
    (fast path: consolidated metadata = one small file read instead of
    listing the whole store), then fall back step by step if that fails.
    Still anonymous on the second try too — jumping straight to real
    credential lookup can hang for a while on a machine that has none
    configured, and public buckets basically never need credentials anyway.
    """
    try:
        ds = xr.open_zarr(gcs_bucket_path, storage_options={"token": "anon"}, consolidated=True)
        print(f"[open_dataset] {gcs_bucket_path}: opened anonymously (consolidated metadata)")
        return ds
    except Exception as e:
        print(f"[open_dataset] {gcs_bucket_path}: anonymous+consolidated open failed "
              f"({e!r}); retrying without consolidated metadata...")

    try:
        ds = xr.open_zarr(gcs_bucket_path, storage_options={"token": "anon"})
        print(f"[open_dataset] {gcs_bucket_path}: opened anonymously (non-consolidated)")
        return ds
    except Exception as e:
        print(f"[open_dataset] {gcs_bucket_path}: anonymous open failed too "
              f"({e!r}); falling back to default GCS credentials...")

    return xr.open_zarr(gcs_bucket_path)
