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


ROLE_TRAITS: dict[PatchRole, RoleTraits] = {
    PatchRole.BODY: RoleTraits(in_forces=True, is_wall=True, refinement="high"),
    PatchRole.TYRE: RoleTraits(in_forces=True, is_wall=True, refinement="high"),
    PatchRole.GROUND: RoleTraits(in_forces=False, is_wall=True, refinement="medium"),
    PatchRole.SYMMETRY: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    PatchRole.FARFIELD: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    PatchRole.INLET: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    PatchRole.OUTLET: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    # An MRF cell zone is not a patch and must never enter force integration.
    PatchRole.MRF_ZONE: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
}


def traits(role: PatchRole) -> RoleTraits:
    return ROLE_TRAITS[role]


def force_roles() -> frozenset[PatchRole]:
    """Roles whose patches are integrated for forces."""
    return frozenset(r for r, t in ROLE_TRAITS.items() if t.in_forces)
