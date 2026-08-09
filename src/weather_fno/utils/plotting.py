"""Plot training history (loss curves)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display available on a headless SSH session
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_history(history: dict, out_path: str, run_name: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train", color=cm.viridis(0.2))
    ax.plot(history["val_loss"], label="val", color=cm.viridis(0.7))
    ax.set_xlabel("epoch")
    ax.set_ylabel("lat-weighted MSE")
    ax.set_title(f"Training history {run_name}".strip())
    ax.legend()
    ax.grid(alpha=0.3)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
