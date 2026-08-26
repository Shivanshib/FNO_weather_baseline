"""
CLI entrypoint. Same command works on cream, vanilla, or external compute —
only configs/baseline_fno.yaml (or an env-specified alternative) changes.

Usage:
    python scripts/train.py --config configs/baseline_fno.yaml
    # Sweeping a few parameters without touching the base config or the
    # code -- a small override file, see configs/experiments/example.yaml:
    python scripts/train.py --config configs/baseline_fno.yaml --experiment configs/experiments/example.yaml
    # FourCastNet-style 2-step fine-tuning after pretraining -- see
    # configs/experiments/twostep_finetune.yaml:
    python scripts/train.py --config configs/baseline_fno.yaml --experiment configs/experiments/twostep_finetune.yaml
"""

from __future__ import annotations

import argparse

from weather_fno.config import load_config, resolve_device, save_config_snapshot, set_seed
from weather_fno.training.run import run_training


def main():
    # Parse CLI args and load config.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # Small override file layered on top of --config, so a whole
    # experiment (architecture, epochs, ...) lives in one small file
    # instead of editing the base config -- see configs/experiments/example.yaml.
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.experiment)
    device = resolve_device(cfg.training.device)

    # Seed before anything that consumes randomness -- model weight init
    # and DataLoader shuffle order -- so two runs with the same seed are
    # actually comparable (see set_seed's docstring).
    set_seed(cfg.training.seed)

    # Record exactly what this run used, before training starts, so a
    # downloaded run folder stays self-documenting even if the experiment
    # file or base config later changes.
    save_config_snapshot(cfg)

    # The actual pipeline (data, climatology, model, Trainer.fit, loss
    # plots) lives in training/run.py::run_training -- shared with
    # scripts/run_seed_ensemble.py, which calls it once per seed -- this
    # file is just CLI arg parsing + the two seed/snapshot calls above,
    # which are genuinely single-run-entrypoint concerns.
    run_training(cfg, device)


if __name__ == "__main__":
    main()
