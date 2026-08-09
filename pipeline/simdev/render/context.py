from __future__ import annotations

from pathlib import Path
from typing import Any

from simdev.config.schema import CaseSpec, WallTreatment
from simdev.domain.base import DomainBox
from simdev.geometry.roles import PatchRole, traits

C_MU = 0.09

WALL_FUNCTIONS: dict[WallTreatment, dict[str, str]] = {
    WallTreatment.HIGH_Y_PLUS: {
        "nut": "nutkWallFunction",
        "k": "kqRWallFunction",
        "omega": "omegaWallFunction",
    },
    WallTreatment.LOW_Y_PLUS: {
        "nut": "nutLowReWallFunction",
        "k": "kLowReWallFunction",
        "omega": "omegaWallFunction",
    },
    WallTreatment.SPALDING: {
        "nut": "nutUSpaldingWallFunction",
        "k": "kqRWallFunction",
        "omega": "omegaWallFunction",
    },
}


def inlet_turbulence(spec: CaseSpec) -> tuple[float, float]:
    """Freestream k and omega from turbulence intensity and length scale."""
    k = 1.5 * (spec.flow.turbulence_intensity * spec.flow.u_inf) ** 2
    omega = k**0.5 / (C_MU**0.25 * spec.flow.turbulence_length_scale)
    return k, omega


def location_in_mesh(domain: DomainBox) -> tuple[float, float, float]:
    """A point in the fluid, upstream of the body and off the symmetry plane."""
    dx = domain.x_max - domain.x_min
    return (
        domain.x_min + 0.10 * dx,
        domain.y_min + 0.25 * (domain.y_max - domain.y_min),
        domain.z_min + 0.50 * (domain.z_max - domain.z_min),
    )


def build_context(
    spec: CaseSpec, domain: DomainBox, geometry_files: dict[str, Path]
) -> dict[str, Any]:
    k, omega = inlet_turbulence(spec)
    wall_fns = WALL_FUNCTIONS[spec.physics.wall_treatment]

    wall_patches = [
        p.name for p in spec.geometry.patches if traits(p.role).is_wall
    ]
    refined_patches = [
        {
            "name": p.name,
            "file": Path(geometry_files[p.name]).name,
            "level_min": spec.mesh.surface_refinement_min,
            "level_max": spec.mesh.surface_refinement_max,
            "n_layers": spec.mesh.n_layers if traits(p.role).refinement == "high" else 0,
        }
        for p in spec.geometry.patches
        if p.name in geometry_files
    ]

    return {
        "spec": spec,
        "domain": domain,
        "geometry_files": {n: Path(p).name for n, p in geometry_files.items()},
        "refined_patches": refined_patches,
        "wall_patches": wall_patches,
        "force_patches": [
            p.name
            for p in spec.geometry.patches
            if traits(p.role).in_forces
        ],
        "wall_fns": wall_fns,
        "k_inlet": k,
        "omega_inlet": omega,
        "nut_inlet": k / omega,
        "a_ref": spec.a_ref_effective,
        "location_in_mesh": location_in_mesh(domain),
        "ground_is_moving": spec.ground.motion.value == "moving",
        "symmetry_patch": domain.symmetry,
    }
