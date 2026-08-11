"""The cornering domain: a virtual wind tunnel bent around the corner.

Steady cornering is only steady in a frame that turns with the car, so this
domain and the rotating frame in `render/context.py` are two halves of one
model and neither is meaningful alone. The sector supplies the curved path;
the frame supplies the Coriolis and centrifugal terms that make travelling
that path a physical thing rather than a duct.

Why a sector rather than the box:

In the rotating frame the air far from the car is at rest in the *ground*
frame, which means it is in solid-body rotation about the corner centre. Its
streamlines are circles. A box domain cuts those circles at an angle, so
every outer face is simultaneously an inlet and an outlet and the pressure
field has nowhere clean to anchor. A sector's end faces are perpendicular to
the flow and its radial faces are parallel to it, which is the same
relationship the box has to straight-line flow.
"""

from __future__ import annotations

import math
from typing import Sequence

from simdev.config.schema import CaseSpec, Mode
from simdev.domain.base import ArcBlock, DomainSector
from simdev.geometry.roles import PatchRole

# Beyond this the sector's two end faces start to face one another, and wake
# leaving the outlet is close enough to the inlet to contaminate it.
MAX_USEFUL_SWEEP = math.pi


class AnnulusDomainBuilder:
    """Curved tunnel about a vertical axis, ground plane at z = 0.

    Reads exactly the same domain numbers as the box builder and reinterprets
    them for a curved path: upstream and downstream lengths become arc length
    along the vehicle's own radius, half width becomes radial half-extent,
    height is unchanged. A case therefore keeps its domain proportions when
    it is switched from straight to cornering, and a difference between the
    two runs is a difference in physics rather than in tunnel size.
    """

    def build(
        self,
        spec: CaseSpec,
        geom_bounds: tuple[Sequence[float], Sequence[float]],
    ) -> DomainSector:
        if spec.physics.mode is not Mode.CORNERING:
            raise ValueError(
                "the annulus domain describes a curved path and only applies "
                f"to mode 'cornering', not '{spec.physics.mode.value}'"
            )
        radius = spec.physics.corner_radius
        if not radius:
            raise ValueError("cornering requires a positive corner_radius")

        lo, hi = geom_bounds
        length = hi[0] - lo[0]
        d = spec.domain

        # The corner centre sits abeam the vehicle, on the side it turns
        # toward, at the corner radius. Abeam rather than at the nose so the
        # vehicle's own path radius is exactly corner_radius.
        car_x = (lo[0] + hi[0]) / 2.0
        car_y = (lo[1] + hi[1]) / 2.0
        centre = (car_x, car_y + spec.corner_side * radius)

        # Angle from the corner centre out to the car. The offset above is
        # purely in y, so this is one of the two vertical directions.
        theta_car = math.atan2(-spec.corner_side * radius, 0.0)

        # Which way round the sector the air arrives from.
        #
        # The tangent at the vehicle points along corner_side in x, and the
        # air moves along flow_sign in x, so the two agree when their product
        # is positive - and then the inlet lies at *decreasing* theta.
        # Derived rather than stated, for the same reason omega_signed is: a
        # flipped sector mirrors the whole case and still converges.
        inlet_direction = -spec.flow_sign * spec.corner_side

        theta_inlet = theta_car + inlet_direction * d.upstream_lengths * length / radius
        theta_outlet = (
            theta_car - inlet_direction * d.downstream_lengths * length / radius
        )

        half_width = d.half_width_lengths * length
        r_min = max(radius - half_width, 1.0e-6)
        r_max = radius + half_width
        z_min = 0.0
        z_max = d.height_lengths * length

        base = spec.mesh.base_cell_size
        n_radial = max(1, round((r_max - r_min) / base))
        n_vertical = max(1, round((z_max - z_min) / base))

        blocks = self._blocks(
            min(theta_inlet, theta_outlet),
            max(theta_inlet, theta_outlet),
            radius,
            base,
            d.arc_segments_per_quadrant,
        )

        symmetry = (
            _only(spec, PatchRole.SYMMETRY) if spec.half_model else None
        )

        return DomainSector(
            centre=centre,
            radius=radius,
            r_min=r_min,
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
            theta_car=theta_car,
            theta_inlet=theta_inlet,
            theta_outlet=theta_outlet,
            blocks=blocks,
            inlet_at_theta_min=theta_inlet < theta_outlet,
            n_radial=n_radial,
            n_vertical=n_vertical,
            inlet=_only(spec, PatchRole.INLET),
            outlet=_only(spec, PatchRole.OUTLET),
            ground=_only(spec, PatchRole.GROUND),
            farfield=_only(spec, PatchRole.FARFIELD),
            symmetry=symmetry,
            geom_min=(lo[0], lo[1], lo[2]),
            geom_max=(hi[0], hi[1], hi[2]),
        )

    @staticmethod
    def _blocks(
        theta_min: float,
        theta_max: float,
        radius: float,
        base_cell: float,
        segments_per_quadrant: int,
    ) -> tuple[ArcBlock, ...]:
        """Split the sweep into blocks short enough to stay circular."""
        sweep = theta_max - theta_min
        max_segment = (math.pi / 2.0) / segments_per_quadrant
        count = max(1, math.ceil(sweep / max_segment))

        # Circumferential cells are sized on the arc at the vehicle's radius,
        # so a cell here is the same size as a streamwise cell in the box
        # domain at the same resolution profile.
        total_cells = max(count, round(sweep * radius / base_cell))
        per_block = max(1, round(total_cells / count))

        step = sweep / count
        return tuple(
            ArcBlock(
                theta_start=theta_min + i * step,
                theta_end=theta_min + (i + 1) * step,
                n_circumferential=per_block,
            )
            for i in range(count)
        )


# Local axes of every block: x radial, y circumferential, z vertical. Chosen
# so that x cross y points along +z for an increasing theta step, which is
# what makes the hex right-handed and its volume positive. Reversing either
# the radial or the angular order inverts every cell in the domain, and
# blockMesh reports that as "negative volume" a long way from the cause.
_RING_R_MIN_Z_MIN = 0
_RING_R_MAX_Z_MIN = 1
_RING_R_MAX_Z_MAX = 2
_RING_R_MIN_Z_MAX = 3

# blockMesh's outward-facing orderings for the six faces of a hex.
_FACE_X_MIN = (0, 4, 7, 3)
_FACE_X_MAX = (1, 2, 6, 5)
_FACE_Y_MIN = (0, 1, 5, 4)
_FACE_Y_MAX = (3, 7, 6, 2)
_FACE_Z_MIN = (0, 3, 2, 1)
_FACE_Z_MAX = (4, 5, 6, 7)


class SectorMesh:
    """blockMesh topology for a sector: vertices, hexes, arcs, boundary faces.

    Built here rather than in the template because a curved block mesh is
    real geometry - vertex indices, handedness, arc interpolation points -
    and `render/templates/` is required to contain no logic. The template
    loops over what this produces and nothing else.
    """

    def __init__(self, domain: DomainSector) -> None:
        self.domain = domain
        self.vertices: list[tuple[float, float, float]] = []
        self.hexes: list[tuple[tuple[int, ...], tuple[int, int, int]]] = []
        self.arcs: list[tuple[int, int, tuple[float, float, float]]] = []
        self._build()

    def _ring(self, theta: float) -> int:
        """Append the four vertices of one radial/vertical corner ring."""
        base = len(self.vertices)
        d = self.domain
        for r, z in (
            (d.r_min, d.z_min),
            (d.r_max, d.z_min),
            (d.r_max, d.z_max),
            (d.r_min, d.z_max),
        ):
            self.vertices.append(d.point(theta, r, z))
        return base

    def _build(self) -> None:
        d = self.domain
        thetas = [b.theta_start for b in d.blocks] + [d.blocks[-1].theta_end]
        rings = [self._ring(theta) for theta in thetas]

        for index, block in enumerate(d.blocks):
            a, b = rings[index], rings[index + 1]
            # x radial (a: r_min -> r_max), y circumferential (a -> b),
            # z vertical. See the note above on handedness.
            corners = (
                a + _RING_R_MIN_Z_MIN,
                a + _RING_R_MAX_Z_MIN,
                b + _RING_R_MAX_Z_MIN,
                b + _RING_R_MIN_Z_MIN,
                a + _RING_R_MIN_Z_MAX,
                a + _RING_R_MAX_Z_MAX,
                b + _RING_R_MAX_Z_MAX,
                b + _RING_R_MIN_Z_MAX,
            )
            self.hexes.append(
                (corners, (d.n_radial, block.n_circumferential, d.n_vertical))
            )

            mid = (block.theta_start + block.theta_end) / 2.0
            for offset, r, z in (
                (_RING_R_MIN_Z_MIN, d.r_min, d.z_min),
                (_RING_R_MAX_Z_MIN, d.r_max, d.z_min),
                (_RING_R_MAX_Z_MAX, d.r_max, d.z_max),
                (_RING_R_MIN_Z_MAX, d.r_min, d.z_max),
            ):
                self.arcs.append((a + offset, b + offset, d.point(mid, r, z)))

    def _face(self, block_index: int, face: tuple[int, ...]) -> tuple[int, ...]:
        corners = self.hexes[block_index][0]
        return tuple(corners[i] for i in face)

    def boundary(self) -> list[tuple[str, str, list[tuple[int, ...]]]]:
        """(patch name, blockMesh patch type, faces), in dictionary order."""
        d = self.domain
        last = len(self.hexes) - 1

        end_min = self._face(0, _FACE_Y_MIN)
        end_max = self._face(last, _FACE_Y_MAX)
        inlet_face, outlet_face = (
            (end_min, end_max) if d.inlet_at_theta_min else (end_max, end_min)
        )

        # Radial walls and the roof are all far field. In a rotating frame
        # the still air outside is in solid-body rotation, so its streamlines
        # run parallel to both radial faces - the same relationship a box
        # domain's side walls have to straight-line flow.
        farfield: list[tuple[int, ...]] = []
        ground: list[tuple[int, ...]] = []
        for index in range(len(self.hexes)):
            farfield.append(self._face(index, _FACE_X_MIN))
            farfield.append(self._face(index, _FACE_X_MAX))
            farfield.append(self._face(index, _FACE_Z_MAX))
            ground.append(self._face(index, _FACE_Z_MIN))

        return [
            (d.inlet, "patch", [inlet_face]),
            (d.outlet, "patch", [outlet_face]),
            (d.ground, "wall", ground),
            (d.farfield, "patch", farfield),
        ]


def check_sweep(domain: DomainSector) -> list[str]:
    """Warn when the tunnel has curved far enough to bite its own tail."""
    if domain.sweep > MAX_USEFUL_SWEEP:
        return [
            f"the cornering sector sweeps {math.degrees(domain.sweep):.0f} "
            f"degrees at a {domain.radius:.2f} m radius, so its inlet and "
            "outlet face each other and wake leaving the domain re-enters it. "
            "Shorten domain.downstream_lengths, or accept that this corner is "
            "too tight for this much tunnel"
        ]
    return []


def _only(spec: CaseSpec, role: PatchRole) -> str:
    names = spec.patches_with_role(role)
    if len(names) != 1:
        raise ValueError(
            f"annulus domain needs exactly one '{role.value}' patch, found {names}"
        )
    return names[0]
