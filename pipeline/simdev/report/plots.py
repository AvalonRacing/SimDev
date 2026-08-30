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


def _settled(series: pd.Series, time: pd.Series, window: tuple[int, int]) -> pd.Series:
    """The part of a trace worth scaling to: the run-up plus the window."""
    frame = pd.DataFrame({"Time": time, "v": series}).dropna()
    if frame.empty:
        return frame["v"]
    x_from = settled_from(frame.rename(columns={"v": "x"}), window)
    return frame[frame["Time"] >= x_from]["v"]


def panel_limits(
    series: pd.Series, window: tuple[int, int], time: pd.Series | None = None
) -> tuple[float, float]:
    """y-limits sized to THIS component's oscillation, not to the vehicle's.

    A per-patch Cl runs from about -1 on the body to a few hundredths on a
    suspension arm, and the oscillation on the small ones is smaller again.
    Drawn on the shared FORCE_AXIS_LIMITS the quiet components are flat lines
    and the swing - the only thing an attribution plot is for - is invisible.
    So every panel gets its own axis, fitted to the settled region of its own
    trace.

    ANCHORED ON THE AVERAGING WINDOW, NOT ON THE SETTLED REGION, and the
    approach is allowed to run off the panel. Anchoring on the settled region
    is not robust: a transient that has decayed to a few percent of its
    starting value is still tens of times the oscillation amplitude, so one
    slow-decaying component drags its own axis out until the swing is a flat
    line again - the exact failure this helper exists to prevent, reappearing
    through the scaling rule. The window is the pipeline's own statement of
    which iterations it trusts, so the axis is sized to that and padded by
    1.5x its height on each side to show the approach coming in. A trace
    entering from off-scale and settling is the same contract
    plot_force_history already documents for the transient.
    """
    if time is None:
        time = pd.Series(range(1, len(series) + 1), index=series.index)
    frame = pd.DataFrame({"Time": time, "v": series}).dropna()
    if frame.empty:
        return (-1.0, 1.0)

    band = frame[(frame["Time"] >= window[0]) & (frame["Time"] <= window[1])]["v"]
    if len(band) < 2:
        # No window to anchor on - a smoke run, or a component that stopped
        # early. Fall back to the settled region rather than returning
        # something arbitrary.
        band = _settled(series, time, window)
    if band.empty:
        return (-1.0, 1.0)

    lo, hi = float(band.min()), float(band.max())
    pad = max((hi - lo) * 1.5, abs(hi) * 0.01, 1e-9)
    return (lo - pad, hi + pad)


def oscillation(df: pd.DataFrame, window: tuple[int, int], column: str) -> float:
    """Standard deviation of one component over the averaging window.

    The window, not the whole history: a component that thrashed during the
    startup transient and has been quiet since is not what is driving the
    limit cycle, and ranking on the full trace would put it first.
    """
    if column not in df or df.empty:
        return 0.0
    band = df[(df["Time"] >= window[0]) & (df["Time"] <= window[1])][column]
    if len(band) < 2:
        band = df[column]
    return float(band.std()) if len(band) > 1 else 0.0


def rank_by_oscillation(
    components: dict[str, pd.DataFrame], window: tuple[int, int], column: str
) -> list[str]:
    """Component names, loudest first.

    The plot answers "which part of the car is oscillating", so the ordering
    is the answer and the panels are just the evidence.
    """
    return sorted(
        components, key=lambda n: oscillation(components[n], window, column), reverse=True
    )


def plot_component_forces(
    components: dict[str, pd.DataFrame],
    out_path: Path,
    window: tuple[int, int],
    column: str = "Cl",
) -> Path:
    """One panel per force patch, each on its own scale, loudest first.

    WHAT THIS IS FOR. The aggregate trace says the car's Cl oscillates; it
    cannot say which part of the car is doing it, and a limit cycle fed by the
    rear wing stalling and one fed by a front tyre wake look identical in the
    total while wanting opposite fixes. These panels split the total by patch
    so the swing can be attributed before anyone re-tunes numerics against it.

    Each patch's coefficient is its SHARE of the vehicle's, because every
    per-patch function object carries the aggregate's Aref/lRef/CofR - so the
    panels sum to the total and the sigma printed on each is directly
    comparable with the others and with the whole-car number.

    Every panel is autoscaled to itself (see panel_limits) and the title
    carries the mean, the sigma and that sigma as a share of the summed
    per-patch sigma - which is the number that actually answers "where is it
    coming from".
    """
    usable = {n: d for n, d in components.items() if column in d and not d.empty}
    if not usable:
        usable = {}

    order = rank_by_oscillation(usable, window, column)
    sigmas = {n: oscillation(usable[n], window, column) for n in order}
    total_sigma = sum(sigmas.values()) or 1.0

    n = max(len(order), 1)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.0 * cols, 1.9 * rows), squeeze=False, sharex=True
    )
    flat = [a for row in axes for a in row]

    for axis, name in zip(flat, order):
        df = usable[name]
        axis.plot(df["Time"], df[column], linewidth=0.9, color="C0")
        axis.axvspan(window[0], window[1], alpha=0.12, color="C1")

        band = df[(df["Time"] >= window[0]) & (df["Time"] <= window[1])][column]
        mean = float(band.mean()) if len(band) else float("nan")
        if len(band):
            axis.axhline(mean, color="C3", linewidth=0.8, linestyle="--")
            axis.axhspan(mean - sigmas[name], mean + sigmas[name],
                         color="C3", alpha=0.12)

        axis.set_ylim(*panel_limits(df[column], window, df["Time"]))
        axis.set_title(
            f"{name}   {column}={mean:.4f}   sigma={sigmas[name]:.4f}"
            f"   ({sigmas[name] / total_sigma:.0%} of summed sigma)",
            fontsize="small", loc="left",
        )
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize="small")

    for axis in flat[len(order):]:
        axis.set_visible(False)
    for axis in flat[max(len(order) - cols, 0):len(order)]:
        axis.set_xlabel("iteration")

    fig.suptitle(
        f"per-component {column} - panels are individually scaled, "
        "ordered by oscillation over the averaging window",
        fontsize="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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
