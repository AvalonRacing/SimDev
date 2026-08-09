from __future__ import annotations

import pytest

from scripts.mesh_independence import monotonic_convergence, within_tolerance


def test_monotonic_increasing_series_is_monotonic() -> None:
    assert monotonic_convergence([0.30, 0.33, 0.345]) is True


def test_monotonic_decreasing_series_is_monotonic() -> None:
    assert monotonic_convergence([0.40, 0.36, 0.348]) is True


def test_oscillating_series_is_not_monotonic() -> None:
    assert monotonic_convergence([0.30, 0.40, 0.33]) is False


def test_two_points_are_insufficient() -> None:
    assert monotonic_convergence([0.30, 0.35]) is False


def test_within_tolerance_accepts_a_ten_percent_error() -> None:
    assert within_tolerance(0.32, target=0.30, tolerance=0.10) is True


def test_within_tolerance_rejects_a_twenty_percent_error() -> None:
    assert within_tolerance(0.36, target=0.30, tolerance=0.10) is False


@pytest.mark.parametrize("value", [0.30, 0.33, 0.27])
def test_tolerance_band_is_symmetric(value: float) -> None:
    assert within_tolerance(value, target=0.30, tolerance=0.10) is True
