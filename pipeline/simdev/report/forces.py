"""Dimensional forces, moments, centre of pressure and per-group coefficients.

Everything here is a window mean over the iterations `check_convergence`
already chose, which is also `fieldAverage`'s timeStart - so a number in
report.tsv and a picture in results/images describe the same flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd

if TYPE_CHECKING:
    from simdev.config.schema import CaseSpec

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


COEFFICIENTS = ("Cd", "Cl")


def group_coefficients(
    components: Mapping[str, pd.DataFrame],
    groups: Mapping[str, Sequence[str]],
    window: tuple[int, int],
) -> dict[str, dict[str, float]]:
    """Window-mean Cd and Cl per group, summed from the per-patch objects.

    controlDict gives every forceCoeffs_<patch> the vehicle's own Aref, lRef
    and CofR, so each patch's Cd and Cl is its SHARE of the vehicle
    coefficient and groups are plain sums. That is what makes
    cd_body + cd_wing + cd_other == cd hold to floating point, and it is why
    per-patch reference areas would have been useless here: eleven numbers on
    eleven scales that add up to nothing.

    A patch with no data makes its whole group NaN rather than summing what
    happens to be present. A group that silently omits a member disagrees
    with the total by an amount nobody can see.
    """
    out: dict[str, dict[str, float]] = {}

    for group, patches in groups.items():
        totals = {c: 0.0 for c in COEFFICIENTS}
        for patch in patches:
            frame = components.get(patch)
            if frame is None:
                totals = {c: float("nan") for c in COEFFICIENTS}
                break
            for coefficient in COEFFICIENTS:
                totals[coefficient] += window_mean(frame, coefficient, window)
        out[group] = totals

    return out


@dataclass(frozen=True)
class ForceReport:
    """Everything report.tsv needs beyond what post already computes.

    `force`, `moment` and `cop` are None together: they all come from the
    same two files, and a run made before the forces object existed has
    neither. That is a gap in the record, not a failed run.
    """

    force: tuple[float, float, float] | None
    moment: tuple[float, float, float] | None
    cop: CentreOfPressure | None
    groups: dict[str, dict[str, float]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    # The full per-iteration frames this report was built from - the same
    # force.dat and moment.dat that gave `force` and `moment` their window
    # means. Carried here so a caller building the COP history plot (which
    # needs every iteration, not the mean) can reuse them instead of
    # re-resolving and re-parsing the identical files with a second
    # find_latest call site. None exactly when `force`/`moment`/`cop` are.
    force_frame: pd.DataFrame | None = None
    moment_frame: pd.DataFrame | None = None


def axles_from_prepare(run_dir: Path) -> tuple[float, float] | None:
    """Mean front and rear wheel-centre x, as prepare measured them.

    Steering rotates a wheel about its own steering axis and moves the centre
    by millimetres in x, so this is stable enough to divide by - which is why
    the balance denominator comes from here rather than from the geometry
    bounding box, which moves with the bodywork.

    None when the case has no wheels. The Ahmed body must not be given an
    invented wheelbase so that a balance column can be filled in.
    """
    from simdev.run.status import read_status

    status = read_status(run_dir, "prepare")
    if status is None:
        return None
    wheels = status.detail.get("wheels") or {}
    front = [w["origin"][0] for name, w in wheels.items() if name.startswith("F")]
    rear = [w["origin"][0] for name, w in wheels.items() if name.startswith("R")]
    if not front or not rear:
        return None
    return (sum(front) / len(front), sum(rear) / len(rear))


def build_force_report(
    run_dir: Path, spec: "CaseSpec", window: tuple[int, int]
) -> ForceReport:
    """Window means of F and M, the COP triple, and the group coefficients."""
    from simdev.run.parsers import read_component_coeffs, read_force_vectors
    from simdev.stages.common import find_latest

    reasons: list[str] = []

    groups = group_coefficients(
        read_component_coeffs(run_dir), spec.post.groups, window
    )

    try:
        force_frame = read_force_vectors(
            find_latest(run_dir, "forces/*/force.dat")
        )
        moment_frame = read_force_vectors(
            find_latest(run_dir, "forces/*/moment.dat")
        )
    except (FileNotFoundError, ValueError) as error:
        reasons.append(
            "no dimensional forces: could not read the 'forces' function "
            f"object ({error}). Runs made before it was added to controlDict "
            "have coefficients but no newtons"
        )
        return ForceReport(None, None, None, groups, reasons)

    force = tuple(window_mean(force_frame, axis, window) for axis in "xyz")
    moment = tuple(window_mean(moment_frame, axis, window) for axis in "xyz")

    # The force a unit coefficient would produce. Only sets the COP guard.
    force_scale = (
        0.5 * spec.flow.rho * spec.flow.u_inf**2 * spec.a_ref_effective
    )
    cop = centre_of_pressure(
        force, moment, tuple(spec.forces.c_of_r), force_scale,
        axles=axles_from_prepare(run_dir),
    )
    reasons.extend(cop.reasons)

    return ForceReport(
        force, moment, cop, groups, reasons,
        force_frame=force_frame, moment_frame=moment_frame,
    )


def cop_history(
    forces: pd.DataFrame,
    moments: pd.DataFrame,
    c_of_r: tuple[float, float, float],
    force_scale: float,
    axles: tuple[float, float] | None,
) -> pd.DataFrame:
    """COP_x and front balance per iteration, for the plot.

    Balance is the number that actually gets tuned, and whether it swings
    half a percent or five across the limit cycle is invisible in a windowed
    mean. Guarded iterations come back NaN so the trace breaks rather than
    spiking - matplotlib draws a gap, which is the honest picture of an
    iteration where the denominator was noise.
    """
    merged = forces.merge(moments, on="Time", suffixes=("_f", "_m"))
    rows = []
    for _, row in merged.iterrows():
        cop = centre_of_pressure(
            (row["x_f"], row["y_f"], row["z_f"]),
            (row["x_m"], row["y_m"], row["z_m"]),
            c_of_r,
            force_scale,
            axles=axles,
        )
        rows.append(
            {
                "Time": row["Time"],
                "COP_x": float("nan") if cop.x is None else cop.x,
                "balance_front_pct": (
                    float("nan")
                    if cop.balance_front_pct is None
                    else cop.balance_front_pct
                ),
            }
        )
    return pd.DataFrame(rows)
