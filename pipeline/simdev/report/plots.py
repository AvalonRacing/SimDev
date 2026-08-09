from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_force_history(
    df: pd.DataFrame, out_path: Path, window: tuple[int, int]
) -> Path:
    """Cd and Cl against iteration, with the averaging window shaded."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    for axis, column in zip(axes, ("Cd", "Cl")):
        if column not in df:
            continue
        axis.plot(df["Time"], df[column], linewidth=1.0)
        axis.axvspan(window[0], window[1], alpha=0.15, label="averaging window")
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_residuals(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9, 4))

    for column in df.columns:
        if column == "Time" or not column.endswith("_initial"):
            continue
        axis.semilogy(df["Time"], df[column], linewidth=1.0, label=column)

    axis.set_xlabel("iteration")
    axis.set_ylabel("initial residual")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
