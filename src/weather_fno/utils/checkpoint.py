"""
Atomic checkpoint save/load.

Saves plain state dicts (never the model object itself, which pickles class
definitions and breaks if the code changes later). Writes to a temp file and
renames atomically so a killed/interrupted session never leaves a corrupt
checkpoint file behind.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(state: dict, ckpt_dir: str, filename: str = "latest.pt") -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    final_path = os.path.join(ckpt_dir, filename)
    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir)
    os.close(fd)
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)  # atomic on the same filesystem


def load_checkpoint(
    ckpt_dir: str,
    filename: str,
    model,
    optimizer=None,
    device: str = "cpu",
) -> Optional[dict]:
    path = os.path.join(ckpt_dir, filename)
    if not os.path.exists(path):
        return None  # nothing to resume from

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt  # caller reads epoch, best_val_loss, history, etc.
