"""Absolute cell sizes, in metres, pinned per patch and per profile.

Refinement levels are *relative* to `base_cell_size`, which makes re-basing
the background a coordinated edit across three profiles and every per-patch
level in the case file. Get one patch wrong and nothing complains: the case
resolves, the mesh builds, and the near-wall resolution on that one part is
silently half or double what it was. The wing would still be called level 7.

So the levels are not what this file checks. It checks the metres they come
out as, which is the quantity the physics actually depends on, and it is the
reason a future re-basing can be done by arithmetic rather than by care.

If you re-base and a test here fails, do not update the number to match: work
out which patch was missed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simdev.config.resolve import load_case

CASE = Path(__file__).resolve().parents[1] / "cases" / "car" / "config.yaml"

# What each surface must resolve to, in metres, whatever the background is.
#
#   0.75 mm - bodywork, tyres, the wing and the suspension links. The coarsest
#             that still puts ~56 cells on a 42 mm wing chord and 5 across a
#             4 mm control arm.
#   1.5  mm - the MRF sleeves, which are cell zones rather than walls and only
#             have to enclose the rim cleanly.
PRODUCTION_CELL_SIZE = {
    "Body": 0.00075,
    "Chassis": 0.00075,
    "Wing": 0.00075,
    "Tire_FL": 0.00075,
    "Tire_FR": 0.00075,
    "Tire_RL": 0.00075,
    "Tire_RR": 0.00075,
    "SUS_FL": 0.00075,
    "SUS_FR": 0.00075,
    "SUS_RL": 0.00075,
    "SUS_RR": 0.00075,
    "MRF_FL": 0.0015,
    "MRF_FR": 0.0015,
    "MRF_RL": 0.0015,
    "MRF_RR": 0.0015,
}

# The case-wide surface cell each profile resolves to. Every profile is one
# clean factor of two from the next, which is what makes the grid ladder in
# scripts/mesh_independence.py meaningful.
PROFILE_CELL_SIZE = {
    "car": 0.00075,
    "car_dev": 0.0015,
    "car_smoke": 0.0125,
}


def _spec(profile: str):
    return load_case(CASE, profile, None, None)


@pytest.mark.parametrize("name,expected", sorted(PRODUCTION_CELL_SIZE.items()))
def test_each_patch_resolves_to_its_designed_cell_size(name, expected) -> None:
    spec = _spec("car")
    patch = next(p for p in spec.geometry.patches if p.name == name)
    assert spec.surface_cell_size_for(patch) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("profile,expected", sorted(PROFILE_CELL_SIZE.items()))
def test_each_profile_resolves_to_its_designed_cell_size(profile, expected) -> None:
    assert _spec(profile).surface_cell_size == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("designed", sorted(set(PRODUCTION_CELL_SIZE.values())))
def test_every_designed_cell_size_is_reachable_from_the_background(designed) -> None:
    """Re-basing must stay exact, not merely close.

    Every refinement level is a halving, so a size the background cannot reach
    by repeated halving is a size no level can express. 96 / 0.75 = 128 and
    96 / 1.5 = 64; a re-basing to 72 mm would make it 96 and 48, and 96 is not
    a power of two - the wing would land on 0.5625 mm and every number in this
    file would have to move with it.

    Deliberately measured against the design table above rather than against
    surface_cell_size_for(). Comparing the background to a size *derived* from
    the background is circular - it is 2^level by construction and cannot ever
    fail - which is exactly the shape of vacuous test this file exists to
    avoid.
    """
    ratio = _spec("car").mesh.base_cell_size / designed
    assert ratio == pytest.approx(round(ratio), rel=1e-9), (
        f"a {designed * 1e3:.4f} mm cell is {ratio:.4f} background cells; "
        "no refinement level can express it"
    )
    assert round(ratio) & (round(ratio) - 1) == 0, (
        f"the background is {round(ratio)}x a {designed * 1e3:.4f} mm cell, "
        "which is not a power of two"
    )


def test_the_layer_stack_still_fits_the_cell_it_is_carved_from() -> None:
    """The stack is sized in metres and the cell is sized in levels.

    They are set in different files, so a re-basing that moved the cell
    without moving the levels would show up here first: six layers at 40 um
    and ER 1.25 want 450 um, and 0.65 of a 0.75 mm cell is 487 um.
    """
    spec = _spec("car")
    assert spec.n_layers_effective == spec.mesh.n_layers == 6
    assert spec.layer_stack_thickness(6) < spec.mesh.max_layer_cell_ratio * 0.00075
