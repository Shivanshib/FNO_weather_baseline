"""
Save/load model checkpoints as plain dicts (never the model object itself,
which would pickle class definitions and break if the code later changes).
Saving is atomic (write to a temp file, then rename) so a crash never
leaves a half-written, corrupt checkpoint behind.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(state: dict, ckpt_dir: str, filename: str = "latest.pt") -> None:
    """Write `state` to ckpt_dir/filename, atomically."""
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
    """Load ckpt_dir/filename into `model` (and `optimizer`, if given).
    Returns the full checkpoint dict (epoch, history, etc.) so the caller
    can pick up training/inference state from it, or None if the file
    doesn't exist yet."""
    path = os.path.join(ckpt_dir, filename)
    if not os.path.exists(path):
        return None  # nothing to resume from

    # weights_only=False because the checkpoint also carries a
    # TrainingConfig object, not just tensors -- PyTorch >=2.6 defaults to
    # weights_only=True, which refuses to unpickle that and breaks
    # resuming. Safe here since we only ever load checkpoints this project
    # itself produced.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
