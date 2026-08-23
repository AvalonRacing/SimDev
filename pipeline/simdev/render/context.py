from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from simdev.config.schema import (
    CaseSpec,
    GroundMotion,
    Mode,
    WallTreatment,
    WheelRotation,
)
from simdev.domain.annulus import SectorMesh
from simdev.domain.base import Domain, DomainSector
from simdev.geometry.roles import PatchRole, traits
from simdev.geometry.wheels import Wheel

C_MU = 0.09

# Cell zone covering every cell, which is what the cornering frame rotates.
#
# It is an ordinary named zone, not a keyword: MRFZone::read looks the name
# up in the mesh and aborts with "cannot find MRF cellZone all" if nothing
# created it. blockMesh names it on the block itself - the name sits between
# the vertex list and the cell counts - so it exists before snappyHexMesh
# runs and is inherited by every cell snappy splits out of the background.
CORNER_ZONE = "all"

# The combined vehicle surface that refinement shells measure distance from.
#
# It is deliberately NOT one of the patches. It appears in snappy's `geometry`
# so distances can be taken against it, and never in `refinementSurfaces`, so
# it creates no boundary patch, carries no boundary condition, is never
# snapped to and never enters force integration. It exists only as a distance
# field, and one combined surface means snappy computes one rather than
# fifteen.
VEHICLE_SURFACE = "vehicle"

# A synthetic surface swept along the domain's own cornering arc, that the
# wake shells measure distance from - same role as VEHICLE_SURFACE, same
# exclusion from refinementSurfaces, just following the path instead of the
# car. Written by prepare._write_wake_surface only when spec.domain.wake is
# set and the domain is an annulus; the name exists here too so there is one
# spelling of it, matching VEHICLE_SURFACE.
WAKE_SURFACE = "wake"

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


def location_in_mesh(domain: Domain) -> tuple[float, float, float]:
    """A point in the fluid, upstream of the body and off the symmetry plane.

    For a sector the domain's bounding box is mostly *outside* the annulus -
    the box corners sit beyond both radial walls - so a fraction of the
    bounding box would land snappy's seed point in empty space and it would
    keep the wrong side of the mesh. The curved case is therefore placed in
    the sector's own polar coordinates instead.
    """
    if isinstance(domain, DomainSector):
        theta = domain.theta_inlet + 0.10 * (domain.theta_outlet - domain.theta_inlet)
        return domain.point(
            theta,
            (domain.r_min + domain.radius) / 2.0,
            domain.z_min + 0.50 * (domain.z_max - domain.z_min),
        )

    # A tenth of the way in from the *inlet* end, which is not always the -x
    # end: with the tunnel reversed this side is the long downstream tail,
    # and seeding there is still in the fluid but a long way from the car.
    dx = domain.x_max - domain.x_min
    return (
        domain.inlet_x + domain.flow_sign * 0.10 * dx,
        domain.y_min + 0.25 * (domain.y_max - domain.y_min),
        domain.z_min + 0.50 * (domain.z_max - domain.z_min),
    )


def refinement_boxes(spec: CaseSpec, domain: DomainBox) -> list[dict[str, Any]]:
    """Resolve each refinement region to an absolute, domain-clipped box.

    Regions are declared in body lengths off the geometry bounding box; snappy
    wants metres. Clipping matters because a region that pokes outside the
    domain still refines every background cell it touches on the way out,
    which is a silent way to double a cell count.
    """
    length = domain.geom_length
    boxes: list[dict[str, Any]] = []

    for region in spec.domain.refinement_regions:
        half_width = region.half_width * length
        box_min = (
            max(domain.x_min, domain.geom_min[0] + region.x_start * length),
            max(domain.y_min, -half_width),
            domain.z_min,
        )
        box_max = (
            min(domain.x_max, domain.geom_max[0] + region.x_end * length),
            min(domain.y_max, half_width),
            min(domain.z_max, domain.z_min + region.height * length),
        )
        boxes.append(
            {
                "name": region.name,
                "level": region.level,
                "min": box_min,
                "max": box_max,
            }
        )

    return boxes


def refinement_shells(spec: CaseSpec, domain: DomainBox) -> list[dict[str, Any]]:
    """Resolve each shell to an absolute distance in metres, finest first.

    snappy reads `levels` as a list of (distance, level) pairs and applies the
    first one whose distance contains the cell, so the order is not cosmetic:
    listed coarse-first, the outermost shell would swallow everything inside
    it and the fine shells would never apply. Sorting here rather than trusting
    the case file means a shell list written in either order behaves the same.
    """
    length = domain.geom_length
    shells = [
        {"distance": shell.distance * length, "level": shell.level}
        for shell in spec.domain.refinement_shells
    ]
    return sorted(shells, key=lambda s: s["distance"])


def wake_shells(spec: CaseSpec, domain: DomainBox) -> list[dict[str, Any]]:
    """Resolve wake shells to an absolute distance in metres, finest first.

    Same resolution and ordering as refinement_shells - see its docstring -
    just measured from WAKE_SURFACE instead of VEHICLE_SURFACE. Empty
    whenever refinement_shells() would need to check, so callers can test
    this list directly rather than re-deriving domain.wake's applicability.
    """
    if spec.domain.wake is None or not isinstance(domain, DomainSector):
        return []
    length = domain.geom_length
    shells = [
        {"distance": shell.distance * length, "level": shell.level}
        for shell in spec.domain.wake.shells
    ]
    return sorted(shells, key=lambda s: s["distance"])


def ground_cell_size(spec: CaseSpec, domain: DomainBox) -> float:
    """Background cell size at the ground plane, under the car.

    The ground is a blockMesh patch, so it never appears in refinementSurfaces
    and snappy never surface-refines it. Its cell is the background cell,
    divided down only where volume refinement reaches the floor. Sizing the
    ground's prism stack against the coarser unrefined cell would ask for more
    layers than fit under the car.

    Two things reach the floor, and both have to count:

    - a refinement *region*, when its box starts at z_min - which the wake box
      does;
    - a refinement *shell*, whenever it is thicker than the ride height, which
      any useful near-field shell is. The car sits ~10 mm off the road and the
      innermost shell is tens of millimetres, so the ground under the car is
      refined to that shell's level whether or not anybody drew a box there.

    Missing the second is how the ground ends up asking for a stack sized
    against a 96 mm cell that does not exist anywhere near the vehicle.
    """
    level = 0
    for region, box in zip(
        spec.domain.refinement_regions, refinement_boxes(spec, domain)
    ):
        if box["min"][2] <= domain.z_min + 1e-12:
            level = max(level, region.level)

    ride_height = max(0.0, domain.geom_min[2] - domain.z_min)
    for shell in refinement_shells(spec, domain):
        if shell["distance"] >= ride_height:
            level = max(level, shell["level"])

    return spec.mesh.base_cell_size / 2**level


def layer_patches(spec: CaseSpec, domain: DomainBox) -> list[dict[str, Any]]:
    """Every wall patch and the layers it can carry.

    Layers follow is_wall, not refinement == 'high'. A wall patch gets a wall
    function, and a wall function assumes its first cell centre sits in the
    log layer - so a wall without layers is a wall whose turbulence model is
    being evaluated somewhere it is not valid. The ground is the case in
    point: it is a blockMesh patch, so it was skipped by the STL-driven layer
    logic and ran with its first cell 60 mm off the floor.
    """
    patches: list[dict[str, Any]] = []
    for patch in spec.geometry.patches:
        if not traits(patch.role).is_wall:
            continue
        if traits(patch.role).refinement == "high":
            n_layers = spec.n_layers_for(patch)
        else:
            n_layers = spec.n_layers_in_cell(ground_cell_size(spec, domain))
        if patch.n_layers is not None:
            n_layers = min(n_layers, patch.n_layers)
        patches.append({"name": patch.name, "n_layers": n_layers})
    return patches


@dataclass(frozen=True)
class Bc:
    patch: str
    entries: dict[str, str]


def _vec(x: float, y: float, z: float) -> str:
    return f"uniform ({x} {y} {z})"


def _foam_vec(v) -> str:
    """A vector as an OpenFOAM *dictionary* entry: space separated."""
    return f"({float(v[0]):.10g} {float(v[1]):.10g} {float(v[2]):.10g})"


def _cpp_vec(v) -> str:
    """A vector as a C++ constructor argument list: comma separated.

    Not the same as _foam_vec, and the difference is invisible until
    OpenFOAM tries to compile a coded boundary condition and reports
    "expected ')' before numeric constant" from a generated file.
    """
    return f"({float(v[0]):.10g}, {float(v[1]):.10g}, {float(v[2]):.10g})"


@dataclass(frozen=True)
class CornerFrame:
    """The rotating reference frame a cornering case is solved in.

    Everything the case does differently from a straight-line run follows
    from these three numbers, so they are computed once and shared: the
    domain is swept about this axis, MRFProperties rotates about it, and the
    wheels are driven by the road speed it implies at their own radius.
    """

    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    omega: float

    @property
    def omega_vector(self) -> np.ndarray:
        return self.omega * np.asarray(self.axis, dtype=float)

    def velocity_at(self, point) -> np.ndarray:
        """Absolute velocity of a point that rotates with the frame."""
        offset = np.asarray(point, dtype=float) - np.asarray(self.origin, dtype=float)
        return np.cross(self.omega_vector, offset)

    def road_velocity_at(self, point) -> np.ndarray:
        """Road velocity as seen in the rotating frame.

        The road is at rest in the ground frame, so in the rotating frame it
        runs backwards at the local tangential speed. This is what a wheel
        has to match, and it differs between the inside and outside of the
        corner - which is the whole reason wheel speeds cannot be one number.
        """
        return -self.velocity_at(point)


def corner_frame(spec: CaseSpec, domain: Domain) -> CornerFrame | None:
    """The cornering frame, or None for a straight-line case."""
    omega = spec.omega_signed
    if spec.physics.mode is not Mode.CORNERING or omega is None:
        return None
    if not isinstance(domain, DomainSector):
        raise ValueError(
            "a cornering case needs the annulus domain; the rotating frame "
            "has no centre to turn about without one"
        )
    return CornerFrame(
        origin=(domain.centre[0], domain.centre[1], 0.0),
        axis=(0.0, 0.0, 1.0),
        omega=omega,
    )


def wheel_speeds(
    spec: CaseSpec,
    wheels: dict[str, Wheel],
    frame: CornerFrame | None,
) -> dict[str, dict[str, Any]]:
    """Rolling speed for each wheel, solved against the road it stands on.

    Straight-line: every wheel sees the same road speed, so they all turn at
    U/r - but even then r is measured, not declared, so a change of tyre or
    ride height carries through.

    Cornering: each wheel sits at its own radius from the corner centre and
    therefore on road moving at its own speed. The outer wheels turn faster
    than the inner ones by the ratio of their radii, and at a 3 m corner with
    a 0.19 m track that is a 6% spread. Hard-coding one wheel speed would put
    that error on every corner of the car.
    """
    # In the car's frame the road runs with the flow, so it follows the
    # tunnel direction rather than +x.
    road_reference = (
        np.asarray(spec.freestream, dtype=float)
        if spec.ground.motion is GroundMotion.MOVING
        else np.zeros(3)
    )

    speeds: dict[str, dict[str, Any]] = {}
    for name, wheel in wheels.items():
        contact = wheel.contact_point()
        if frame is None:
            road = road_reference
        else:
            road = frame.road_velocity_at(contact)

        omega, slip = wheel.spin_omega(road)
        speeds[name] = {
            "wheel": wheel,
            "omega": omega,
            "slip": slip,
            "road_speed": float(np.linalg.norm(road)),
            "surface_speed": abs(omega) * wheel.radius,
        }
    return speeds


def _tyre_bc(
    wheel: Wheel, omega: float, frame: CornerFrame | None
) -> dict[str, str]:
    """Velocity boundary condition for a surface that turns with a wheel.

    Straight-line is a single rotation and `rotatingWallVelocity` expresses
    it exactly.

    Cornering is not. The tyre is carried around the corner *and* spins about
    its own axis, and those are rotations about two different, non-
    intersecting axes - a screw motion, which no single OpenFOAM rotating-wall
    condition can describe. OpenFOAM wants the absolute velocity on a patch
    excluded from the MRF frame, so the two rotations are summed explicitly
    in a coded condition. Getting this wrong is not subtle: leaving the
    corner term out parks a spinning wheel in space while the car drives away
    from it.
    """
    if frame is None:
        return {
            "type": "rotatingWallVelocity",
            "origin": _foam_vec(wheel.origin),
            "axis": _foam_vec(wheel.axis),
            "omega": f"{omega:.10g}",
        }

    lines = [
        "#{",
        "            const vectorField& c = patch().Cf();",
        f"            const vector wheelOmega{_cpp_vec(omega * wheel.direction)};",
        f"            const vector wheelOrigin{_cpp_vec(wheel.origin)};",
        f"            const vector frameOmega{_cpp_vec(frame.omega_vector)};",
        f"            const vector frameOrigin{_cpp_vec(frame.origin)};",
        "            operator==",
        "            (",
        "                (wheelOmega ^ (c - wheelOrigin))",
        "              + (frameOmega ^ (c - frameOrigin))",
        "            );",
        "        #}",
    ]
    return {
        "type": "codedFixedValue",
        "value": "uniform (0 0 0)",
        "name": f"corneringWheel{wheel.wheel}",
        "code": "\n".join(lines),
    }


def build_bcs(
    spec: CaseSpec,
    k: float,
    omega: float,
    nut: float,
    wall_fns: dict[str, str],
    speeds: dict[str, dict[str, Any]] | None = None,
    frame: CornerFrame | None = None,
) -> dict[str, list[Bc]]:
    """Boundary conditions per field per patch. All branching happens here."""
    freestream = _vec(*spec.freestream)
    moving_ground = spec.ground.motion is GroundMotion.MOVING
    speeds = speeds or {}

    # In a rotating frame OpenFOAM solves the absolute velocity, and the
    # patches excluded from the frame carry absolute values. Far from the car
    # the air really is at rest over the track, so the oncoming flow is not
    # prescribed at all - it emerges from the frame rotation. Writing a
    # freestream velocity here as well would drive the case twice.
    cornering = frame is not None
    still_air = _vec(0.0, 0.0, 0.0)
    onset = still_air if cornering else freestream

    fields = ("U", "p", "k", "omega", "nut")
    bcs: dict[str, list[Bc]] = {f: [] for f in fields}

    for patch in spec.geometry.patches:
        role = patch.role
        name = patch.name

        # An MRF zone is a closed volume that becomes a cell zone and a set of
        # internal faces. It is never a boundary patch, so writing a boundary
        # condition for it would name a patch that does not exist and abort
        # the solver on startup.
        if role is PatchRole.MRF_ZONE:
            continue

        if role is PatchRole.INLET:
            entries = {
                "U": {"type": "fixedValue", "value": onset},
                "p": {"type": "zeroGradient"},
                "k": {"type": "fixedValue", "value": f"uniform {k}"},
                "omega": {"type": "fixedValue", "value": f"uniform {omega}"},
                "nut": {"type": "calculated", "value": f"uniform {nut}"},
            }
        elif role is PatchRole.OUTLET:
            entries = {
                "U": {
                    "type": "inletOutlet",
                    "inletValue": _vec(0.0, 0.0, 0.0),
                    "value": onset,
                },
                "p": {"type": "fixedValue", "value": "uniform 0"},
                "k": {
                    "type": "inletOutlet",
                    "inletValue": f"uniform {k}",
                    "value": f"uniform {k}",
                },
                "omega": {
                    "type": "inletOutlet",
                    "inletValue": f"uniform {omega}",
                    "value": f"uniform {omega}",
                },
                "nut": {"type": "calculated", "value": f"uniform {nut}"},
            }
        elif role is PatchRole.SYMMETRY:
            entries = {f: {"type": "symmetry"} for f in fields}
        elif role is PatchRole.FARFIELD:
            entries = {f: {"type": "slip"} for f in fields}
        elif role is PatchRole.GROUND:
            if cornering:
                # The road is genuinely stationary over the ground, and the
                # ground patch is excluded from the rotating frame, so its
                # absolute velocity is zero. The apparent sweep of the road
                # under the car comes out of the frame transform rather than
                # out of a velocity written here - which is also why it
                # correctly varies across the track width.
                u_entry = {"type": "noSlip"}
            elif moving_ground:
                u_entry = {"type": "fixedValue", "value": freestream}
            else:
                u_entry = {"type": "noSlip"}
            entries = {
                "U": u_entry,
                "p": {"type": "zeroGradient"},
                "k": {"type": wall_fns["k"], "value": f"uniform {k}"},
                "omega": {"type": wall_fns["omega"], "value": f"uniform {omega}"},
                "nut": {"type": wall_fns["nut"], "value": "uniform 0"},
            }
        else:  # BODY, TYRE
            spinning = (
                patch.wheel is not None
                and patch.wheel in speeds
                and spec.physics.wheel_rotation is WheelRotation.SPINNING
            )
            if spinning:
                entry = speeds[patch.wheel]
                u_entry = _tyre_bc(entry["wheel"], entry["omega"], frame)
            else:
                u_entry = {"type": "noSlip"}
            entries = {
                "U": u_entry,
                "p": {"type": "zeroGradient"},
                "k": {"type": wall_fns["k"], "value": f"uniform {k}"},
                "omega": {"type": wall_fns["omega"], "value": f"uniform {omega}"},
                "nut": {"type": wall_fns["nut"], "value": "uniform 0"},
            }

        for field in fields:
            bcs[field].append(Bc(patch=name, entries=entries[field]))

    return bcs


def sleeves_are_cell_zones(frame: CornerFrame | None) -> bool:
    """Whether the wheel sleeves become cellZones, and so MRF zones, at all.

    True only for a straight-line case. It is asked in two places that have
    to agree - mrf_zones() below, which writes constant/MRFProperties, and
    build_context(), which decides whether snappy makes the sleeve a cellZone
    surface or a plain refinement region - and disagreeing is silent in both
    directions. A zone in the dictionary that the mesh does not have aborts
    the solver on startup; a zone in the mesh that the dictionary does not
    name leaves those cells inertial in a rotating domain, and nothing says
    so. See mrf_zones() for why cornering has none.
    """
    return frame is None


def mrf_zones(
    spec: CaseSpec,
    domain: Domain,
    speeds: dict[str, dict[str, Any]],
    frame: CornerFrame | None,
) -> list[dict[str, Any]]:
    """The rotating cell zones, which differ between the two modes.

    Straight-line: one zone per wheel, covering the rim and spokes, turning
    about that wheel's own measured axis. This is what makes a wheel pump air
    rather than merely present a moving skin. The four sleeves are metres
    apart in a domain with no other cellZone in it, so every face on a sleeve
    boundary is shared with unzoned fluid and belongs to exactly one frame.

    Cornering: ONE zone, `all`, over the whole domain - and no wheel zones,
    which is why the sleeves are not cellZone surfaces at all in this mode
    (see sleeves_are_cell_zones and build_context).

    ADJACENT MRF ZONES DOUBLE-COUNT THE FRAME FLUX ON THE FACES THEY SHARE,
    AND IN A CORNERING CASE EVERY SLEEVE FACE IS SUCH A FACE. `all` covers
    the background and snappy moves the sleeve cells out of it into their own
    zone - polyTopoChange carries one cellZone label per cell
    (polyTopoChange.H:211), so the assignment is a move, not a copy - which
    leaves the two zones sharing the whole sleeve boundary. MRFZone then
    marks a face as belonging to a zone when EITHER of its cells is in that
    zone (MRFZone.C:80-87), so those faces are internal faces of both, and
    MRFZoneList::makeRelative simply loops the zones subtracting each frame's
    Omega ^ (Cf - origin) in turn (MRFZoneList.C:249-254). The flux entering
    the sleeve is therefore not the flux leaving the fluid around it.

    MEASURED, BECAUSE THE OBVIOUS OBJECTION IS THAT TWO ZONES DESCRIBING THE
    SAME MOTION SHOULD BEHAVE LIKE ONE. A 4 x 4 x 1 duct at 1 m/s in a frame
    turning at 1 rad/s, meshed and run twice with everything identical except
    whether the cells are declared as one cellZone or as two adjacent halves
    carrying the SAME origin, axis and omega:

        max |dU| = 0.437 m/s on a 1 m/s inlet, mean 0.072 m/s

    Same mesh, same frame, same boundary conditions; the entire difference is
    the interface. So this is not a compound-motion approximation that could
    be traded off against the ~1% Coriolis argument - it is a discretisation
    error at the sleeve boundary, and it is there whether the wheel zone
    carries the wheel's spin rate or the corner's own.

    WHAT CORNERING GIVES UP IS RIM PUMPING, WHICH IS THE LIMITATION THE
    HANDBOOK ALREADY NAMED. A cell carries one frame rotation and cannot be
    going round the corner and round the wheel at once, so the sleeve was
    never going to express the real motion; the tyre and rim walls still
    carry the full spin-plus-carry velocity through _tyre_bc's coded
    condition, so the wheels turn, they just do not drive the air inside the
    rim. Doing it properly needs a sliding mesh, not a second MRF zone.

    The tyres stay on the corner frame's non-rotating patch list either way.
    MRFZone::correctBoundaryVelocity overwrites every *included* patch face
    with Omega ^ (Cf - origin) on each SIMPLE iteration, discarding whatever
    the boundary condition computed, and the rate comparison is the wrong
    one for that: the corner acts on a 4 m lever and the spin on a 0.033 m
    one, so the two linear speeds are 15.0 and 12.6 m/s - the same order,
    both of them essentially the road speed, as they must be for a rolling
    wheel. Excluding the tyres leaves _tyre_bc's value alone.
    """
    if not sleeves_are_cell_zones(frame):
        excluded = [
            p.name
            for p in spec.geometry.patches
            if p.role in (PatchRole.GROUND, PatchRole.INLET, PatchRole.OUTLET, PatchRole.FARFIELD)
            or (p.wheel is not None and traits(p.role).is_wall)
        ]

        return [
            {
                "name": "cornerFrame",
                "cell_zone": CORNER_ZONE,
                "origin": frame.origin,
                "axis": frame.axis,
                "omega": frame.omega,
                "non_rotating_patches": excluded,
            }
        ]

    zones: list[dict[str, Any]] = []
    for patch in spec.geometry.patches:
        if patch.role is not PatchRole.MRF_ZONE or patch.wheel not in speeds:
            continue
        entry = speeds[patch.wheel]
        zones.append(
            {
                "name": patch.name,
                "cell_zone": patch.name,
                "origin": entry["wheel"].origin,
                "axis": entry["wheel"].axis,
                "omega": entry["omega"],
                "non_rotating_patches": [],
            }
        )
    return zones


def build_context(
    spec: CaseSpec,
    domain: Domain,
    geometry_files: dict[str, Path],
    wheels: dict[str, Wheel] | None = None,
) -> dict[str, Any]:
    k, omega = inlet_turbulence(spec)
    wall_fns = WALL_FUNCTIONS[spec.physics.wall_treatment]
    frame = corner_frame(spec, domain)
    speeds = wheel_speeds(spec, wheels or {}, frame)
    zones = mrf_zones(spec, domain, speeds, frame)

    wall_patches = [
        p.name for p in spec.geometry.patches if traits(p.role).is_wall
    ]
    # A sleeve that is not a cellZone must not be a refinementSurface either.
    # A refinementSurface without a cellZone is a WALL: snappy would snap to
    # it, give it a boundary patch nothing writes a condition for, and put a
    # closed shell of rubber-thin wall inside each wheel. So in cornering the
    # sleeves leave refinementSurfaces entirely and come back below as
    # refinement regions, which refine the same cells and create no surface.
    cell_zones = sleeves_are_cell_zones(frame)
    meshed_as_surface = [
        p
        for p in spec.geometry.patches
        if p.name in geometry_files
        and (cell_zones or p.role is not PatchRole.MRF_ZONE)
    ]

    refined_patches = [
        {
            "name": p.name,
            "file": Path(geometry_files[p.name]).name,
            "level_min": spec.patch_refinement(p)[0],
            "level_max": spec.patch_refinement(p)[1],
            # Per-patch, and never the raw request: asking snappy for more
            # layers than fit its cell makes it truncate and drop them on
            # exactly the small parts. See CaseSpec.n_layers_for().
            "n_layers": (
                spec.n_layers_for(p) if traits(p.role).refinement == "high" else 0
            ),
            "is_cell_zone": p.role is PatchRole.MRF_ZONE,
        }
        for p in meshed_as_surface
    ]

    # Closed volumes that refine the cells they contain and nothing else -
    # no patch, no faceZone, no cellZone. The wheel sleeves in cornering,
    # where they cannot be MRF zones (see mrf_zones) but the air inside the
    # rim still wants the resolution their old surface refinement gave it.
    refinement_volumes = [
        {
            "name": p.name,
            "file": Path(geometry_files[p.name]).name,
            "level": spec.patch_refinement(p)[1],
        }
        for p in spec.geometry.patches
        if p.role is PatchRole.MRF_ZONE
        and not cell_zones
        and p.name in geometry_files
    ]

    # The initial internal field. Prescribing a freestream everywhere in a
    # cornering case would seed the domain with a uniform velocity that no
    # boundary supports, and the first iterations would be spent unwinding
    # it; still air is both the correct absolute state and the quiet start.
    internal_u = (
        "uniform (0 0 0)" if frame else f"uniform {_foam_vec(spec.freestream)}"
    )

    is_sector = isinstance(domain, DomainSector)
    sector_mesh = SectorMesh(domain) if is_sector else None

    return {
        "spec": spec,
        "domain": domain,
        "geometry_files": {n: Path(p).name for n, p in geometry_files.items()},
        "refined_patches": refined_patches,
        "refinement_volumes": refinement_volumes,
        "refinement_boxes": refinement_boxes(spec, domain),
        "refinement_shells": refinement_shells(spec, domain),
        # The combined vehicle surface the shells measure distance from. Named
        # here rather than in the template so there is one spelling of it, and
        # empty when no shells are declared so no unused surface is loaded.
        "shell_surface": (
            VEHICLE_SURFACE if spec.domain.refinement_shells else ""
        ),
        # Same pattern, for the curved wake region - see WAKE_SURFACE and
        # WakeRefinement. Empty (and so absent from snappy's geometry) unless
        # the case declares domain.wake AND resolved to an annulus domain;
        # a box domain has no arc to sweep the surface along.
        "wake_surface": (
            WAKE_SURFACE
            if spec.domain.wake is not None and isinstance(domain, DomainSector)
            else ""
        ),
        "wake_shells": wake_shells(spec, domain),
        "layer_patches": layer_patches(spec, domain),
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
        "drag_dir": _foam_vec(spec.drag_dir),
        "pitch_axis": _foam_vec(spec.pitch_axis),
        "location_in_mesh": location_in_mesh(domain),
        "ground_is_moving": spec.ground.motion.value == "moving",
        "symmetry_patch": domain.symmetry,
        "mrf_zones": zones,
        "corner_frame": frame,
        # blockMesh has to create the whole-domain zone; MRFProperties only
        # refers to it. Empty for a straight-line case, which has no frame
        # rotation and wants no zone over the background mesh.
        "background_cell_zone": CORNER_ZONE if frame else "",
        "wheel_speeds": speeds,
        "sector": sector_mesh,
        "sector_boundary": sector_mesh.boundary() if sector_mesh else [],
        "bcs": build_bcs(spec, k, omega, k / omega, wall_fns, speeds, frame),
        "field_meta": {
            "U": ("volVectorField", "[0 1 -1 0 0 0 0]", internal_u),
            "p": ("volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0"),
            "k": ("volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k}"),
            "omega": ("volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {omega}"),
            "nut": ("volScalarField", "[0 2 -1 0 0 0 0]", f"uniform {k / omega}"),
        },
    }
