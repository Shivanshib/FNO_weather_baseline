"""
Builds the FNO model from config. Thin wrapper around neuralop's FNO class
(not subclassed, so upstream updates just work) -- every architectural
choice comes from ModelConfig, so a hyperparameter sweep is just calling
this with a different config.
"""

from __future__ import annotations

from neuralop.models import FNO

from weather_fno.config import ModelConfig


def build_model(cfg: ModelConfig) -> FNO:
    """Build an FNO from a ModelConfig. factorization/rank stay None for a
    plain dense FNO; set factorization="tucker" + a rank in (0, 1] for a
    compressed Tensor FNO (TFNO) instead."""
    return FNO(
        n_modes=tuple(cfg.n_modes),
        hidden_channels=cfg.hidden_channels,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        n_layers=cfg.n_layers,
        factorization=cfg.factorization,
        rank=cfg.rank,
    )
