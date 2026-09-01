"""Dimensional forces, moments, centre of pressure and per-group coefficients.

Everything here is a window mean over the iterations `check_convergence`
already chose, which is also `fieldAverage`'s timeStart - so a number in
report.tsv and a picture in results/images describe the same flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

# Below this, in coefficient terms, the denominator of a COP ratio is noise
# and the ratio it produces is a large number with no meaning. Report nothing
# and say why.
MIN_COEFFICIENT = 0.02


def window_mean(frame: pd.DataFrame, column: str, window: tuple[int, int]) -> float:
    """Mean of one column over the averaging window. NaN if the window is empty."""
    rows = frame[(frame["Time"] >= window[0]) & (frame["Time"] <= window[1])]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows[column].mean())


@dataclass(frozen=True)
class CentreOfPressure:
    """Three diagnostics, NOT the coordinates of one point.

    A net force plus a net moment defines a line of action. `x` and `z` are
    two different readings of the same pitching moment - `x` attributes all
    of M_y to downforce, `z` attributes all of it to drag - so they do not
    describe a single position and were never meant to. The record carries
    cop_convention: "ratio" so a later reader cannot mistake them for one.
    """

    x: float | None
    y: float | None
    z: float | None
    balance_front_pct: float | None
    reasons: list[str] = field(default_factory=list)


def centre_of_pressure(
    force: tuple[float, float, float],
    moment: tuple[float, float, float],
    c_of_r: tuple[float, float, float],
    force_scale: float,
    axles: tuple[float, float] | None = None,
) -> CentreOfPressure:
    """Motorsport-convention COP and front aero balance.

    With r the position of the resultant relative to the centre of rotation,
    M = r x F expands to

        M_x = r_y F_z - r_z F_y
        M_y = r_z F_x - r_x F_z
        M_z = r_x F_y - r_y F_x

    and each reported coordinate drops the second term of its own equation:

        COP_x = x_ref - M_y / F_z
        COP_y = y_ref + M_x / F_z
        COP_z = z_ref + M_y / F_x

    M_z is not used and cannot be: it contains no r_z at all, so it
    constrains nothing this convention reports.

    `force_scale` is 0.5 rho U_inf^2 A_ref in newtons - the force a unit
    coefficient would produce - and only sets the guard threshold.
    """
    fx, fy, fz = force
    mx, my, _mz = moment
    x_ref, y_ref, z_ref = c_of_r
    floor = MIN_COEFFICIENT * abs(force_scale)
    reasons: list[str] = []

    if abs(fz) >= floor:
        cop_x: float | None = x_ref - my / fz
        cop_y: float | None = y_ref + mx / fz
    else:
        cop_x = cop_y = None
        reasons.append(
            f"COP_x and COP_y undefined: |F_z| = {abs(fz):.4g} N is under "
            f"{MIN_COEFFICIENT} of the reference force {abs(force_scale):.4g} N"
        )

    if abs(fx) >= floor:
        cop_z: float | None = z_ref + my / fx
    else:
        cop_z = None
        reasons.append(
            f"COP_z undefined: |F_x| = {abs(fx):.4g} N is under "
            f"{MIN_COEFFICIENT} of the reference force {abs(force_scale):.4g} N"
        )

    balance: float | None = None
    if cop_x is not None and axles is not None:
        x_front, x_rear = axles
        wheelbase = x_front - x_rear
        if abs(wheelbase) > 1e-9:
            balance = (cop_x - x_rear) / wheelbase * 100.0
        else:
            reasons.append(
                "aero balance undefined: the front and rear axles share an x"
            )

    return CentreOfPressure(cop_x, cop_y, cop_z, balance, reasons)
