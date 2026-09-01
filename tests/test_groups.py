from __future__ import annotations

import pandas as pd
import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.config.validate import ValidationError, validate
from simdev.report.forces import group_coefficients

WINDOW = (2, 3)


def _component(cd: float, cl: float) -> pd.DataFrame:
    # Constant over the window, and deliberately wrong before it, so a test
    # that ignores the window fails rather than coincidentally passing.
    return pd.DataFrame(
        {"Time": [1, 2, 3], "Cd": [99.0, cd, cd], "Cl": [99.0, cl, cl]}
    )


def test_a_group_is_the_sum_of_its_patches() -> None:
    components = {
        "Body": _component(0.30, -0.80),
        "Wing": _component(0.20, -0.60),
        "Chassis": _component(0.10, -0.10),
    }
    groups = {"body": ("Body",), "wing": ("Wing",), "other": ("Chassis",)}
    result = group_coefficients(components, groups, WINDOW)
    assert result["body"]["Cd"] == pytest.approx(0.30)
    assert result["other"]["Cl"] == pytest.approx(-0.10)


def test_the_groups_sum_to_the_vehicle_total() -> None:
    """The identity that makes a pasted row self-checking.

    controlDict gives every forceCoeffs_<patch> the vehicle's own Aref, lRef
    and CofR, so each patch's Cd is its SHARE of the vehicle coefficient.
    Groups are therefore plain sums.
    """
    components = {
        "Body": _component(0.30, -0.80),
        "Wing": _component(0.20, -0.60),
        "Chassis": _component(0.10, -0.10),
        "Tire_FL": _component(0.05, -0.02),
    }
    groups = {"body": ("Body",), "wing": ("Wing",), "other": ("Chassis", "Tire_FL")}
    result = group_coefficients(components, groups, WINDOW)
    total = sum(g["Cd"] for g in result.values())
    assert total == pytest.approx(0.65)


def test_a_missing_patch_makes_its_group_nan_not_zero() -> None:
    """A patch whose function object never ran is unknown, not zero.

    Silently summing what happens to be present would make the group columns
    disagree with the total by an amount nobody can see.
    """
    groups = {"body": ("Body", "Chassis")}
    result = group_coefficients({"Body": _component(0.3, -0.8)}, groups, WINDOW)
    assert pd.isna(result["body"]["Cd"])
