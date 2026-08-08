"""
One-off inspection script — run this BEFORE trusting any of the axis-order,
dimension-name, or channel-name assumptions currently baked into
configs/baseline_fno.yaml.

Opens the configured GCS stores (training + inference) and prints back the
ground truth: which variables actually exist, what their dimensions are
called and what order they're in, which way latitude runs, what longitude
convention is used, and what time range is really available. Fix
configs/baseline_fno.yaml to match whatever this disagrees with — don't
edit this script to make the mismatch go away.

Usage:
    python scripts/inspect_store.py --config configs/baseline_fno.yaml
"""

from __future__ import annotations

import argparse

import xarray as xr

from weather_fno.config import Config, load_config


def inspect(gcs_bucket_path: str, cfg: Config, label: str) -> None:
    print(f"\n{'=' * 70}\nInspecting: {label}\n{gcs_bucket_path}\n{'=' * 70}")

    ds = xr.open_zarr(gcs_bucket_path, chunks={"time": 1})

    print(f"\nAvailable data_vars ({len(ds.data_vars)}):")
    for v in sorted(ds.data_vars):
        print(f"  - {v}  dims={ds[v].dims}")

    print("\nChecking configured channels against this store:")
    for spec in cfg.data.channels:
        if spec.name not in ds.data_vars:
            print(f"  MISSING  {spec.short_name:6s} -> '{spec.name}' not found in store")
            continue
        da = ds[spec.name]
        level_info = ""
        if spec.level is not None:
            if "level" in da.dims:
                available = da["level"].values
                ok = spec.level in available
                status = "OK" if ok else f"NOT IN {list(available)}"
                level_info = f" | level={spec.level} [{status}]"
            else:
                level_info = " | WARNING: spec has a level but this variable has no 'level' dim"
        print(f"  {spec.short_name:6s} -> {spec.name:32s} dims={da.dims}{level_info}")

    print("\nDimension name check:")
    lat_dim, lon_dim = cfg.data.lat_dim, cfg.data.lon_dim
    print(f"  configured lat_dim='{lat_dim}' -> "
          f"{'present' if lat_dim in ds.dims else 'NOT FOUND'} (store dims: {list(ds.dims)})")
    print(f"  configured lon_dim='{lon_dim}' -> "
          f"{'present' if lon_dim in ds.dims else 'NOT FOUND'} (store dims: {list(ds.dims)})")

    if lat_dim in ds.coords:
        lat_vals = ds[lat_dim].values
        descending = lat_vals[0] > lat_vals[-1]
        print("\nLatitude orientation:")
        print(f"  first={lat_vals[0]}, last={lat_vals[-1]}  "
              f"-> runs {'DESCENDING (north to south)' if descending else 'ASCENDING (south to north)'}")
        print(f"  current config has flip_lat={cfg.data.flip_lat}")

    if lon_dim in ds.coords:
        lon_vals = ds[lon_dim].values
        convention = "[0, 360)" if lon_vals.max() > 180 else "[-180, 180)"
        print("\nLongitude convention:")
        print(f"  range=[{lon_vals.min()}, {lon_vals.max()}] -> looks like {convention}")

    if "time" in ds.coords:
        t = ds["time"].values
        print(f"\nTime range available: {t.min()} to {t.max()}  ({len(t)} steps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    inspect(cfg.data.gcs_bucket_path, cfg, label="training store (coarse, 64x32)")
    inspect(cfg.inference.gcs_bucket_path, cfg, label="inference store (higher-resolution)")


if __name__ == "__main__":
    main()