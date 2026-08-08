"""
Baseline FNO model builder.

A thin wrapper around neuralop's FNO class — deliberately not subclassed or
forked, so upstream fixes/updates just work. Every architectural choice
comes from the config, which is what makes hyperparameter sweeps trivial:
sweep.py just builds different Config objects and calls this same function.
"""

from __future__ import annotations

from neuralop.models import FNO

from weather_fno.config import ModelConfig


def build_model(cfg: ModelConfig) -> FNO:
    return FNO(
        n_modes=tuple(cfg.n_modes),
        hidden_channels=cfg.hidden_channels,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        n_layers=cfg.n_layers,
        # Leave as dense (factorization=None) for the baseline. Set
        # factorization="tucker" + a rank in (0, 1] in the config to switch
        # to a Tensor FNO later — see project notes on when that's worth it.
        factorization=cfg.factorization,
        rank=cfg.rank,
    )
