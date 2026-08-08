"""
CLI entrypoint for inference on higher-resolution data.

Usage:
    python scripts/infer.py --config configs/baseline_fno.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from weather_fno.config import load_config
from weather_fno.inference.predict import run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Reuse the exact normalisation stats computed on the training split —
    # never re-fit stats on inference data.
    train_cache_path = cfg.data.stats_cache_path.replace("normalisation_stats", "train_cache")
    cached = np.load(train_cache_path)
    train_stats = {"mean": cached["mean"], "std": cached["std"]}

    run_inference(cfg, train_stats)


if __name__ == "__main__":
    main()
