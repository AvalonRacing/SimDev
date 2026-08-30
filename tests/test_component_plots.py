from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simdev.report.plots import (
    panel_limits,
    plot_component_forces,
    rank_by_oscillation,
)
from simdev.run.parsers import read_component_coeffs

WINDOW = (200, 250)


def _trace(mean: float, swing: float, n: int = 250, transient: float = 0.0):
    """A component trace: a big startup transient decaying into a sine of
    amplitude `swing` about `mean`."""
    t = np.arange(1, n + 1)
    osc = swing * np.sin(2 * np.pi * t / 40.0)
    decay = transient * np.exp(-t / 20.0)
    return pd.DataFrame({"Time": t, "Cd": np.full(n, 0.1), "Cl": mean + osc + decay})


# --- the scale is the point -------------------------------------------------


def test_panel_limits_are_sized_to_the_oscillation_not_the_value() -> None:
    """A component swinging +/-0.002 about -0.05 must get an axis a couple of
    hundredths tall, not one that spans the vehicle's whole Cl range. On a
    shared -3..0 axis that trace is a flat line and the oscillation - the only
    thing the plot is for - is invisible.
    """
    lo, hi = panel_limits(_trace(-0.05, 0.002)["Cl"], WINDOW)
    assert hi - lo < 0.02
    assert lo < -0.05 < hi


def test_panel_limits_scale_with_the_component() -> None:
    """A dominant component gets a taller axis than a quiet one, so each is
    read at its own magnitude."""
    big = panel_limits(_trace(-1.0, 0.20)["Cl"], WINDOW)
    small = panel_limits(_trace(-0.05, 0.002)["Cl"], WINDOW)
    assert (big[1] - big[0]) > 10 * (small[1] - small[0])


def test_panel_limits_ignore_the_startup_transient() -> None:
    """Iteration 1 of a cornering case is the domain spinning up. Scaling to
    it makes the settled region a one-pixel line, which is the failure this
    whole helper exists to avoid."""
    lo, hi = panel_limits(_trace(-1.0, 0.05, transient=-40.0)["Cl"], WINDOW)
    assert hi - lo < 1.0
    assert lo > -5.0


# --- attribution ------------------------------------------------------------


def test_components_are_ranked_by_how_much_they_oscillate() -> None:
    """The question is 'where does the oscillation come from', so the loudest
    component is the one to put first."""
    components = {
        "Body": _trace(-1.0, 0.01),
        "Wing": _trace(-0.5, 0.30),
        "Tire_FL": _trace(-0.1, 0.05),
    }
    assert rank_by_oscillation(components, WINDOW, "Cl") == ["Wing", "Tire_FL", "Body"]


def test_ranking_uses_the_averaging_window_not_the_whole_history() -> None:
    """A component that was violent during the transient and is quiet now is
    not the one driving the limit cycle."""
    components = {
        "settled": _trace(-1.0, 0.01, transient=-50.0),
        "noisy": _trace(-1.0, 0.10),
    }
    assert rank_by_oscillation(components, WINDOW, "Cl")[0] == "noisy"


# --- the plot ---------------------------------------------------------------


def test_plot_writes_one_panel_per_component(tmp_path: Path) -> None:
    components = {n: _trace(-0.3, 0.02) for n in ("Body", "Wing", "Tire_FL")}
    out = plot_component_forces(components, tmp_path / "c.png", WINDOW, "Cl")
    assert out.exists() and out.stat().st_size > 0


def test_plot_survives_a_component_with_no_data(tmp_path: Path) -> None:
    """A patch whose function object wrote a header and no rows must not take
    the whole report down with it."""
    components = {"Body": _trace(-0.3, 0.02), "Empty": pd.DataFrame(columns=["Time", "Cl"])}
    out = plot_component_forces(components, tmp_path / "c.png", WINDOW, "Cl")
    assert out.exists()


# --- reading them off disk --------------------------------------------------


def test_read_component_coeffs_keys_by_patch(tmp_path: Path) -> None:
    for patch in ("Body", "Tire_FL"):
        d = tmp_path / "postProcessing" / f"forceCoeffs_{patch}" / "0"
        d.mkdir(parents=True)
        (d / "coefficient.dat").write_text(
            "# Time\tCd\tCl\n1\t0.5\t-1.0\n2\t0.5\t-1.1\n", encoding="utf-8"
        )
    got = read_component_coeffs(tmp_path)
    assert set(got) == {"Body", "Tire_FL"}
    assert got["Body"]["Cl"].iloc[-1] == pytest.approx(-1.1)


def test_read_component_coeffs_does_not_pick_up_the_aggregate(tmp_path: Path) -> None:
    """postProcessing/forceCoeffs is the whole-vehicle object. Swept in as a
    'component' it would double the total and outrank every real patch."""
    for name in ("forceCoeffs", "forceCoeffs_Body"):
        d = tmp_path / "postProcessing" / name / "0"
        d.mkdir(parents=True)
        (d / "coefficient.dat").write_text(
            "# Time\tCd\tCl\n1\t0.5\t-1.0\n", encoding="utf-8"
        )
    assert set(read_component_coeffs(tmp_path)) == {"Body"}
