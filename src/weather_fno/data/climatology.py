"""
Climatological mean per (day-of-year, hour-of-day), computed once from the
training split and cached alongside the normalisation stats.

Coarse-grid only: computing this for the native-resolution store would
mean fetching ~15 years of native data just for a baseline, which this
project deliberately avoids (see gcs_dataset.py's own docstring on why
data never gets fetched more than the minimum needed).

Binned by (day-of-year, hour-of-day) rather than a smooth function fit --
with only ~15 years of training data that's a noisy mean per exact day,
so a circular moving average over day-of-year smooths it out (wraps
Dec 31 into Jan 1), instead of fitting harmonics.

Cache size: for the 20-channel, 64x32 baseline grid, the cached
climatology.npz is ~200MB (366 days x 4 hours x 20 channels x 32x64
grid, float32)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

DEFAULT_SMOOTHING_DAYS = 15


def _day_of_year_and_hour(time_values: np.ndarray):
    """1-indexed day-of-year (1-366, leap-year aware) and hour-of-day
    (0-23) for each timestamp, via plain datetime64 arithmetic (no pandas
    dependency needed for this)."""
    t = np.asarray(time_values, dtype="datetime64[ns]")
    year_start = t.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (t.astype("datetime64[D]") - year_start).astype(int) + 1
    hour_of_day = (t.astype("datetime64[h]") - t.astype("datetime64[D]")).astype(int)
    return day_of_year, hour_of_day


def _circular_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving average over axis 0 (366 day-of-year bins), wrapping
    circularly so day 366 is treated as adjacent to day 1."""
    if window <= 1:
        return arr
    half = window // 2
    padded = np.concatenate([arr[-half:], arr, arr[:half]], axis=0)
    kernel_len = 2 * half + 1
    cumsum = np.cumsum(padded, axis=0)
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)
    return (cumsum[kernel_len:] - cumsum[:-kernel_len]) / kernel_len


def compute_climatology(
    data: np.ndarray,
    time_values: np.ndarray,
    smoothing_window_days: int = DEFAULT_SMOOTHING_DAYS,
) -> Dict[str, np.ndarray]:
    """
    Args:
        data: (T, C, H, W), PHYSICAL units (denormalise before calling --
            climatology needs to be directly comparable to ground truth).
        time_values: (T,) real timestamps matching data's rows (e.g.
            GCSWeatherDataset.time_values).
        smoothing_window_days: circular moving-average window over
            day-of-year.

    Returns:
        {"climatology": (366, n_hours, C, H, W) float32,
         "hours_of_day": (n_hours,) int array, e.g. [0, 6, 12, 18]}
        -- ready to np.savez_compressed directly.
    """
    doy, hour = _day_of_year_and_hour(time_values)
    hours_of_day = sorted(set(hour.tolist()))
    hour_index = np.array([hours_of_day.index(h) for h in hour])

    T, C, H, W = data.shape
    raw = np.zeros((366, len(hours_of_day), C, H, W), dtype=np.float64)
    counts = np.zeros((366, len(hours_of_day)), dtype=np.int64)
    np.add.at(raw, (doy - 1, hour_index), data)
    np.add.at(counts, (doy - 1, hour_index), 1)

    # Fail loudly rather than silently dividing by zero -- would only
    # happen if train_start/train_end don't span at least one full year,
    # in which case climatology isn't a meaningful baseline anyway.
    if (counts == 0).any():
        n_missing = int((counts == 0).sum())
        raise ValueError(
            f"climatology has {n_missing} (day_of_year, hour) bin(s) with zero training "
            f"samples -- train_start/train_end must span at least one full year for every "
            f"day-of-year to be represented."
        )
    raw /= counts[:, :, None, None, None]

    smoothed = _circular_smooth(raw, window=smoothing_window_days)
    return {"climatology": smoothed.astype(np.float32), "hours_of_day": np.array(hours_of_day)}


def climatology_cache_is_valid(path: str) -> bool:
    """
    True if `path` exists AND loads as a real, complete climatology.npz
    (has both expected keys) -- False for a missing file OR a corrupted/
    truncated one (e.g. an interrupted np.savez_compressed from a killed
    process -- confirmed to actually happen, not a hypothetical: two
    tucker-sweep seed-43 runs' shared climatology.npz were caught exactly
    this way, ~50-100MB short of the real ~200MB and unreadable).

    Used to decide whether it's safe to SKIP recomputing a shared
    climatology cache (config.py::derive_run_paths -- data.stats_cache_path
    is shared across every run using the same training split) -- an
    existence-only check would happily "reuse" a corrupted file forever,
    since nothing else ever rewrites it once skipped.
    """
    if not Path(path).exists():
        return False
    try:
        with np.load(path) as d:
            return "climatology" in d and "hours_of_day" in d
    except Exception:
        return False


def compute_and_save_climatology(data: np.ndarray, time_values: np.ndarray, out_path: str) -> bool:
    """
    Compute climatology from (already-denormalised, physical-units)
    training data and save it to out_path -- UNLESS the training range
    spans less than a full year, in which case this just prints a warning
    and returns False rather than raising. A full climatology genuinely
    can't be built from less than a year of data (every day-of-year needs
    at least one sample); that's expected and fine for a short test run
    (e.g. scripts/smoke_test.py's tiny date windows), not a bug worth
    crashing training over.

    Shared by scripts/train.py (called automatically every run) and
    scripts/compute_climatology.py (backfills it for a run trained before
    this existed), so both apply the exact same span check.

    Returns:
        True if climatology was computed and saved, False if skipped.
    """
    span_days = int((time_values[-1] - time_values[0]) / np.timedelta64(1, "D")) + 1
    if span_days < 366:
        print(f"[climatology] training data spans only {span_days} day(s) -- need at least "
              f"366 to cover every day-of-year, so skipping climatology (expected for a "
              f"short test run; a real training run should still get it as long as "
              f"train_start/train_end span at least a year).")
        return False

    climatology = compute_climatology(data, time_values)
    np.savez_compressed(out_path, climatology=climatology["climatology"],
                         hours_of_day=climatology["hours_of_day"])
    print(f"Saved climatology to {out_path}")
    return True


def query_climatology(climatology: Dict[str, np.ndarray], time_values: np.ndarray) -> np.ndarray:
    """
    Look up the climatological (C, H, W) field for each of `time_values`.

    Args:
        climatology: as returned by compute_climatology (or loaded back
            from its .npz cache).
        time_values: (N,) real timestamps to query.

    Returns:
        (N, C, H, W) climatological fields, physical units.
    """
    doy, hour = _day_of_year_and_hour(time_values)
    hours_of_day: List[int] = list(climatology["hours_of_day"])
    missing = sorted(set(hour.tolist()) - set(hours_of_day))
    if missing:
        raise ValueError(
            f"climatology has no data for hour-of-day {missing} -- it was built from a "
            f"store with hours {hours_of_day}, but this query needs {sorted(set(hour.tolist()))}."
        )
    hour_index = np.array([hours_of_day.index(h) for h in hour])
    return climatology["climatology"][doy - 1, hour_index]
