from __future__ import annotations

import pytest

from simdev.geometry.roles import ROLE_TRAITS, PatchRole, force_roles, traits


def test_every_role_has_traits() -> None:
    assert set(ROLE_TRAITS) == set(PatchRole)


def test_only_body_and_tyre_contribute_to_forces() -> None:
    assert force_roles() == frozenset({PatchRole.BODY, PatchRole.TYRE})


def test_mrf_zone_is_excluded_from_forces() -> None:
    # Regression guard: the benchmark pipeline needed a name filter for this.
    assert traits(PatchRole.MRF_ZONE).in_forces is False


def test_ground_is_not_in_forces() -> None:
    assert traits(PatchRole.GROUND).in_forces is False


@pytest.mark.parametrize("role", [PatchRole.BODY, PatchRole.TYRE, PatchRole.GROUND])
def test_wall_roles_are_walls(role: PatchRole) -> None:
    assert traits(role).is_wall is True


@pytest.mark.parametrize("role", [PatchRole.SYMMETRY, PatchRole.INLET, PatchRole.OUTLET])
def test_non_wall_roles_are_not_walls(role: PatchRole) -> None:
    assert traits(role).is_wall is False


def test_role_values_match_openfoam_naming() -> None:
    assert PatchRole.MRF_ZONE.value == "mrfZone"
    assert PatchRole.BODY.value == "body"
