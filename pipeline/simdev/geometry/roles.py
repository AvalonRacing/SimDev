from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PatchRole(str, Enum):
    """Role of a surface. Drives BCs, refinement, force inclusion and MRF together."""

    BODY = "body"
    TYRE = "tyre"
    GROUND = "ground"
    SYMMETRY = "symmetry"
    FARFIELD = "farfield"
    INLET = "inlet"
    OUTLET = "outlet"
    MRF_ZONE = "mrfZone"


@dataclass(frozen=True)
class RoleTraits:
    in_forces: bool
    is_wall: bool
    refinement: str
    # Whether this role's surface arrives as an STL. The alternative is a face
    # of the background mesh that blockMesh names, and the two are mutually
    # exclusive: a role that expects an STL and does not find one is a typo,
    # not an empty patch, and used to be skipped in silence.
    from_stl: bool


ROLE_TRAITS: dict[PatchRole, RoleTraits] = {
    PatchRole.BODY: RoleTraits(
        in_forces=True, is_wall=True, refinement="high", from_stl=True
    ),
    PatchRole.TYRE: RoleTraits(
        in_forces=True, is_wall=True, refinement="high", from_stl=True
    ),
    PatchRole.GROUND: RoleTraits(
        in_forces=False, is_wall=True, refinement="medium", from_stl=False
    ),
    PatchRole.SYMMETRY: RoleTraits(
        in_forces=False, is_wall=False, refinement="none", from_stl=False
    ),
    PatchRole.FARFIELD: RoleTraits(
        in_forces=False, is_wall=False, refinement="none", from_stl=False
    ),
    PatchRole.INLET: RoleTraits(
        in_forces=False, is_wall=False, refinement="none", from_stl=False
    ),
    PatchRole.OUTLET: RoleTraits(
        in_forces=False, is_wall=False, refinement="none", from_stl=False
    ),
    # An MRF cell zone is a closed volume, not a boundary. It carries no
    # boundary condition, must never enter force integration, and its faces
    # stay internal so the flow passes through them.
    PatchRole.MRF_ZONE: RoleTraits(
        in_forces=False, is_wall=False, refinement="none", from_stl=True
    ),
}


def traits(role: PatchRole) -> RoleTraits:
    return ROLE_TRAITS[role]


def force_roles() -> frozenset[PatchRole]:
    """Roles whose patches are integrated for forces."""
    return frozenset(r for r, t in ROLE_TRAITS.items() if t.in_forces)
