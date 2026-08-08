# Baseline FNO — coarse-resolution weather forecasting

A baseline Fourier Neural Operator trained on 64x32, 20-channel coarse
ERA5-style data streamed from a GCS zarr store, with inference support on
higher-resolution data.

## Setup

```bash
source /opt/Python/Python-3.11.5_Setup.csh   # or whichever version you're using
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # if you add a pyproject/setup.py so `weather_fno` is importable
```

Everything machine-specific (device, worker count, output paths) lives in
`configs/baseline_fno.yaml` — moving to a different machine (blaze/cream/
vanilla/external compute) should only ever mean editing that file.

## Fill in before running

- `configs/baseline_fno.yaml`: GCS bucket paths, channel names, date ranges
  for train/val, axis-flip flags, inference resolution.
- `src/weather_fno/inference/preprocessing.py`: `compute_specific_humidity`
  — needs the actual raw variable names/units from the higher-resolution
  store.
- `src/weather_fno/training/metrics.py`: additional metrics beyond
  lat-weighted MSE (RMSE, ACC, etc.) once finalised.

## Usage

```bash
# Train (auto-resumes from outputs/checkpoints/latest.pt if present)
python scripts/train.py --config configs/baseline_fno.yaml

# Sweep a few hyperparameter combinations
python scripts/sweep.py --config configs/baseline_fno.yaml

# Run inference on higher-resolution data with the best checkpoint
python scripts/infer.py --config configs/baseline_fno.yaml
```

## Structure

```
configs/            All hyperparameters, paths, machine settings
src/weather_fno/
  data/              GCS streaming, normalisation, axis flips, time-based split
  models/            Thin FNO builder wrapping neuralop
  training/          Trainer (checkpointing, early stopping), metrics
  utils/             Atomic checkpoint save/load, loss-curve plotting
  inference/         Higher-resolution preprocessing (incl. specific
                     humidity derivation) and prediction
scripts/             Thin CLI entrypoints: train.py, sweep.py, infer.py
outputs/             checkpoints/, plots/, logs/, stats/, predictions/ (gitignored)
```
