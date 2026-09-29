from __future__ import annotations

import pandas as pd
import pytest

from simdev.report.forces import (
    MIN_COEFFICIENT,
    centre_of_pressure,
    window_mean,
)

# 0.5 * 1.225 * 15^2 * 0.081 -> a plausible reference force for the car.
FORCE_SCALE = 11.16
ORIGIN = (0.0, 0.0, 0.0)


def test_window_mean_uses_only_the_window() -> None:
    frame = pd.DataFrame({"Time": [1, 2, 3, 4], "Cd": [9.0, 9.0, 1.0, 3.0]})
    assert window_mean(frame, "Cd", (3, 4)) == pytest.approx(2.0)


def test_window_mean_of_an_empty_window_is_nan() -> None:
    frame = pd.DataFrame({"Time": [1, 2], "Cd": [1.0, 2.0]})
    assert pd.isna(window_mean(frame, "Cd", (50, 60)))


def test_cop_x_is_where_downforce_acts() -> None:
    """100 N of downforce 0.1 m ahead of the CofR.

    M = r x F = (0.1, 0, 0) x (0, 0, -100) = (0, 10, 0), so COP_x must come
    back at +0.1 m.
    """
    cop = centre_of_pressure((0.0, 0.0, -100.0), (0.0, 10.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.x == pytest.approx(0.1)


def test_cop_y_is_where_downforce_acts_laterally() -> None:
    # Same force 0.05 m to the car's left: M = (0, 0.05, 0) x (0, 0, -100).
    cop = centre_of_pressure((0.0, 0.0, -100.0), (-5.0, 0.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.y == pytest.approx(0.05)


def test_cop_z_is_the_height_drag_acts_at() -> None:
    # 50 N of drag at 0.08 m: M = (0, 0, 0.08) x (50, 0, 0) = (0, 4, 0).
    cop = centre_of_pressure((50.0, 0.0, 0.0), (0.0, 4.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.z == pytest.approx(0.08)


def test_cop_x_and_cop_z_are_not_one_point() -> None:
    """Both read the same pitching moment and they disagree by construction.

    COP_x attributes all of M_y to downforce; COP_z attributes all of it to
    drag. Pinned as a test so nobody later "fixes" the disagreement and
    silently changes what both columns mean.
    """
    cop = centre_of_pressure((50.0, 0.0, -100.0), (0.0, 10.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.x == pytest.approx(0.1)
    assert cop.z == pytest.approx(0.2)


def test_balance_is_100_percent_at_the_front_axle() -> None:
    cop = centre_of_pressure(
        (0.0, 0.0, -100.0), (0.0, 20.0, 0.0), ORIGIN, FORCE_SCALE, axles=(0.2, -0.2)
    )
    assert cop.x == pytest.approx(0.2)
    assert cop.balance_front_pct == pytest.approx(100.0)


def test_balance_is_50_percent_at_the_wheelbase_centre() -> None:
    cop = centre_of_pressure(
        (0.0, 0.0, -100.0), (0.0, 0.0, 0.0), ORIGIN, FORCE_SCALE, axles=(0.2, -0.2)
    )
    assert cop.balance_front_pct == pytest.approx(50.0)


def test_a_vanishing_denominator_gives_no_number_and_a_reason() -> None:
    """Not a large meaningless number. Empty, with the reason recorded."""
    tiny = MIN_COEFFICIENT * FORCE_SCALE * 0.5
    cop = centre_of_pressure((0.0, 0.0, tiny), (0.0, 10.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.x is None and cop.y is None and cop.balance_front_pct is None
    assert any("F_z" in reason for reason in cop.reasons)


def test_cop_is_relative_to_the_centre_of_rotation() -> None:
    cop = centre_of_pressure(
        (0.0, 0.0, -100.0), (0.0, 10.0, 0.0), (0.5, 0.0, 0.0), FORCE_SCALE
    )
    assert cop.x == pytest.approx(0.6)
