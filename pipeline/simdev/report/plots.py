from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


# Fixed y-limits for the coefficient panels, so two runs can be read against
# each other by eye. Autoscaling gives every run an axis fitted to its own
# wobble, which is what makes a 5% shift between meshes look identical to a
# 0.5% one - the comparison an A/B is actually for.
#
# THESE ARE THE CAR'S RANGES. The Ahmed validation case runs Cd ~ 0.3 and a
# small POSITIVE Cl, which would sit squashed against the floor of the Cd panel
# and entirely off the Cl panel. Give Ahmed its own entry, or empty this dict,
# before reading a force plot from it - a trace that has left the axis looks
# the same as one that was never plotted.
FORCE_AXIS_LIMITS: dict[str, tuple[float, float]] = {
    "Cd": (0.0, 2.0),
    "Cl": (-3.0, 0.0),
}


def settled_from(df: pd.DataFrame, window: tuple[int, int]) -> float:
    """First iteration to scale the y-axis from, skipping the startup transient.

    THE PLOT IS USELESS WITHOUT THIS AND THAT IS NOT A COSMETIC POINT. A
    cornering case starts from still air in a rotating frame, so iteration 1
    is the whole domain being spun up: on the 20.88M run Cd came in at 36.4
    against a settled 0.99 and Cl at -16.5 against -1.62. Autoscaled over the
    full history, the converged region - the only part anybody reads a force
    plot for - is a flat line one pixel thick, and a 5% drift in it is
    invisible. The transient is still drawn; it just runs off the top.

    Anchored on the averaging window rather than on a fixed fraction of the
    run, because the window is already the pipeline's own statement of which
    iterations it trusts. Three window widths of run-up before it shows the
    approach without letting the transient back in - at the default 50-
    iteration window on a 250-iteration run that is iteration 50, which is
    where this case has in fact settled. Clamped into the run so a short
    smoke case, where three widths reaches back past iteration 0, still gets
    a sensible axis instead of the full transient.
    """
    time = df["Time"]
    start, end = float(time.iloc[0]), float(time.iloc[-1])
    width = max(window[1] - window[0], 1)
    return min(max(window[0] - 3 * width, start + 0.10 * (end - start)), window[0])


def plot_force_history(
    df: pd.DataFrame, out_path: Path, window: tuple[int, int]
) -> Path:
    """Cd and Cl against iteration, on fixed axes so runs compare by eye.

    Each panel carries the mean and the +/- 1 sigma band over the averaging
    window as well as the trace, because "is it converged" is a question
    about the size of the wobble against the size of the band, and reading
    that off an unannotated line is guesswork.
    """
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    x_from = settled_from(df, window)

    for axis, column in zip(axes, ("Cd", "Cl")):
        if column not in df:
            continue
        axis.plot(df["Time"], df[column], linewidth=1.0, color="C0")
        axis.axvspan(window[0], window[1], alpha=0.15, label="averaging window")

        settled = df[df["Time"] >= x_from][column]
        band = df[(df["Time"] >= window[0]) & (df["Time"] <= window[1])][column]
        if len(band):
            mean, std = float(band.mean()), float(band.std())
            axis.axhline(mean, color="C3", linewidth=0.9, linestyle="--",
                         label=f"{column} = {mean:.4f} +/- {std:.4f}")
            axis.axhspan(mean - std, mean + std, color="C3", alpha=0.12)

        limits = FORCE_AXIS_LIMITS.get(column)
        if limits is not None:
            axis.set_ylim(*limits)
        elif len(settled):
            lo, hi = float(settled.min()), float(settled.max())
            pad = max((hi - lo) * 0.25, abs(hi) * 0.02, 1e-6)
            axis.set_ylim(lo - pad, hi + pad)

        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize="small")

    # x KEEPS THE WHOLE HISTORY even though y does not. The transient runs off
    # the top of the axis, and a trace entering from off-scale and settling is
    # exactly the picture that says it decayed and roughly where - which is
    # information a plot cropped to the settled region throws away for the
    # sake of a fifth of the width.
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
