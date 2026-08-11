"""Wheel axes measured from the geometry, never assumed.

The CAD arrives in driving states. A ride-height change, a steering input or
a roll attitude moves every wheel, tilts its axis and alters the loaded
radius, and any of those written down as a constant in a config file is a
number that will be wrong the first time the state changes and silent about
it. So nothing here is declared: the axis, the centre, the rolling radius and
the width are all measured from the surfaces of the driving state actually
being run.

What *is* declared is which surfaces belong to which wheel
(``PatchSpec.wheel``), because that is a modelling decision rather than a
measurement.

The measurement is the axis of revolution of a solid of revolution. For the
area-weighted second-moment tensor of such a surface, the two axes transverse
to the symmetry axis carry equal moments and the symmetry axis carries a
different one - whether larger or smaller depends on whether the part is a
disc or a rod, which is exactly why the axis is picked as *the odd one out*
rather than as the largest or the smallest. A wheel is a disc, an MRF sleeve
is nearly a cube in this ratio, and a long driveshaft is a rod; one rule
covers all three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import trimesh

from simdev.geometry.roles import PatchRole


class WheelPatch(Protocol):
    """The three fields of a PatchSpec this module needs.

    Structural rather than an import of CaseSpec, so the measurement can be
    unit-tested against two cylinders and a name without constructing a whole
    resolved case around them.
    """

    name: str
    role: PatchRole
    wheel: str | None

# Above this, the two transverse moments differ enough that the surface is
# not a solid of revolution and no axis can be trusted from it.
MAX_AXISYMMETRY_DEFECT = 0.05

# A wheel whose axis is within this of vertical has no meaningful contact
# patch, and the rolling-speed solve below is singular.
MIN_AXIS_HORIZONTALITY = 0.1


class WheelGeometryError(Exception):
    pass


@dataclass(frozen=True)
class Wheel:
    """One wheel, as measured from its own surfaces.

    `origin` is a point on the rotation axis - the wheel centre - and `axis`
    is a unit vector along it. Both are in the pipeline frame, after the
    import scale and translate, so they can be written straight into a
    boundary condition.
    """

    wheel: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    width: float
    # Outer radius of the surface the axis was taken from. When that is an
    # MRF sleeve this is the reach of the rotating cell zone, and it has to
    # stay inside the tyre: a zone that pokes out through the tread would
    # spin the free stream.
    axis_surface_radius: float
    axisymmetry_defect: float
    axis_source: str
    radius_source: str

    @property
    def centre(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=float)

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self.axis, dtype=float)

    def contact_offset(self) -> np.ndarray:
        """Vector from the wheel centre to the contact patch.

        The lowest point of the wheel, which is the component of -z
        perpendicular to the axis, scaled to the rolling radius. Derived from
        the axis rather than assumed to be straight down, so a cambered or
        steered wheel still resolves its own contact point.
        """
        axis = self.direction
        down = np.array([0.0, 0.0, -1.0])
        perpendicular = down - float(np.dot(down, axis)) * axis
        norm = float(np.linalg.norm(perpendicular))
        if norm < MIN_AXIS_HORIZONTALITY:
            raise WheelGeometryError(
                f"wheel '{self.wheel}' has an axis {self.axis} within "
                f"{math.degrees(math.asin(max(norm, 0.0))):.1f} degrees of "
                "vertical; it has no contact patch and no rolling speed"
            )
        return self.radius * perpendicular / norm

    def contact_point(self) -> np.ndarray:
        return self.centre + self.contact_offset()

    def spin_omega(self, road_velocity: np.ndarray) -> tuple[float, float]:
        """Angular speed about `axis` that matches the road, and the slip left over.

        Rolling without slip means the contact patch moves with the road.
        Solving that as a least-squares problem rather than reading a sign
        off a diagram keeps the result correct for a wheel that is steered,
        cambered, or on the inside of a corner - all of which change the
        answer and none of which change the code.

        The second return value is the magnitude of the road velocity that no
        rotation about this axis can reproduce. For a straight-running wheel
        it is zero; for a steered wheel it is the lateral slip, and it is
        returned rather than hidden because a large value means the axis and
        the direction of travel disagree.
        """
        offset = self.contact_offset()
        tangent = np.cross(self.direction, offset)
        tangent_sq = float(np.dot(tangent, tangent))
        if tangent_sq <= 0.0:
            raise WheelGeometryError(f"wheel '{self.wheel}' has zero rolling radius")

        omega = float(np.dot(road_velocity, tangent)) / tangent_sq
        residual = road_velocity - omega * tangent
        return omega, float(np.linalg.norm(residual))

    def surface_velocity(self, points: np.ndarray, omega: float) -> np.ndarray:
        """Velocity of the spinning surface at `points`, in the wheel's frame."""
        return np.cross(omega * self.direction, np.asarray(points) - self.centre)


def _surface_moments(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid and second-moment eigen-decomposition of a surface.

    The second moment is integrated *exactly* over each triangle rather than
    approximated by its centroid, using

        integral over T of x x^T dA = (A/12) [ sum_k v_k v_k^T
                                             + (sum_k v_k)(sum_k v_k)^T ]

    which matters more than it sounds. Tessellators fan a flat disc from a
    single rim vertex, so the triangle *centroids* of a cylinder's end caps
    bunch to one side and are not axisymmetric at all, even though the disc
    they cover is. On the real MRF sleeve that bias put the two transverse
    moments 24% apart and made a machined cylinder look like it had no axis.
    Exact integration depends only on the surface covered, not on how it was
    cut up, so the answer no longer moves when the exporter changes.
    """
    triangles = mesh.triangles
    areas = mesh.area_faces
    total = float(areas.sum())
    if total <= 0.0:
        raise WheelGeometryError("surface has zero area")

    centroids = triangles.mean(axis=1)
    centre = (centroids * areas[:, None]).sum(axis=0) / total

    vertex_sum = triangles.sum(axis=1)
    outer_of_sum = vertex_sum[:, :, None] * vertex_sum[:, None, :]
    sum_of_outer = (triangles[:, :, :, None] * triangles[:, :, None, :]).sum(axis=1)

    raw = ((areas[:, None, None] / 12.0) * (sum_of_outer + outer_of_sum)).sum(axis=0)
    tensor = raw / total - np.outer(centre, centre)

    values, vectors = np.linalg.eigh(tensor)
    return centre, values, vectors


def axis_of_revolution(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, float]:
    """(point on axis, unit axis, axisymmetry defect) for a solid of revolution.

    The defect is the relative difference between the two transverse moments.
    Zero for a true solid of revolution; large for anything that is not one,
    which is the check that a control arm has not been labelled as a wheel.
    """
    centre, values, vectors = _surface_moments(mesh)

    # The symmetry axis is the eigenvalue that differs from the other two.
    # Comparing each against the mean of its two companions identifies it
    # without assuming a disc (axis smallest) or a rod (axis largest).
    distinctness = [
        abs(values[i] - (values[(i + 1) % 3] + values[(i + 2) % 3]) / 2.0)
        for i in range(3)
    ]
    index = int(np.argmax(distinctness))
    others = [values[(index + 1) % 3], values[(index + 2) % 3]]

    scale = max(abs(others[0]), abs(others[1]), 1e-30)
    defect = abs(others[0] - others[1]) / scale

    axis = vectors[:, index]
    axis = axis / np.linalg.norm(axis)
    # Eigenvector signs are arbitrary. Fix one deterministically so the same
    # geometry always yields the same axis; every rotation speed downstream
    # carries its own sign, solved against the road, so this choice is free.
    largest = int(np.argmax(np.abs(axis)))
    if axis[largest] < 0.0:
        axis = -axis

    return centre, axis, float(defect)


def _radial_extent(
    mesh: trimesh.Trimesh, origin: np.ndarray, axis: np.ndarray
) -> tuple[float, float]:
    """(outer radius, axial width) of a surface about a given axis.

    Deliberately not an inner radius. Whether a bore shows up as a minimum
    radius depends on whether the tessellator put a vertex at the centre of
    the end cap, which is a property of the exporter rather than of the part.
    """
    offsets = mesh.vertices - origin
    axial = offsets @ axis
    radial = np.linalg.norm(offsets - axial[:, None] * axis, axis=1)
    return float(radial.max()), float(axial.max() - axial.min())


def measure_wheel(
    wheel: str,
    axis_mesh: trimesh.Trimesh,
    axis_source: str,
    radius_mesh: trimesh.Trimesh,
    radius_source: str,
) -> Wheel:
    """Measure one wheel from the surface that defines its axis and its tread.

    Two surfaces because they answer different questions. The MRF sleeve is a
    machined cylinder and gives the cleanest possible axis; the tyre is what
    actually touches the road and is the only thing that can give a rolling
    radius. Using the sleeve's radius would under-report the rolling radius by
    the tread depth and drive every wheel too slowly.
    """
    origin, axis, defect = axis_of_revolution(axis_mesh)

    if defect > MAX_AXISYMMETRY_DEFECT:
        raise WheelGeometryError(
            f"surface '{axis_source}' for wheel '{wheel}' is not a solid of "
            f"revolution: its two transverse moments differ by {defect:.1%} "
            f"(limit {MAX_AXISYMMETRY_DEFECT:.0%}). No rotation axis can be "
            "measured from it. Check that the right part carries "
            f"'wheel: {wheel}'"
        )

    # The axis point comes from the axis surface, but the wheel centre should
    # sit at the middle of the tread: recentre along the axis using the tyre,
    # keeping the axis line itself.
    tread_offsets = radius_mesh.vertices - origin
    axial = tread_offsets @ axis
    origin = origin + axis * float(axial.min() + axial.max()) / 2.0

    radius, width = _radial_extent(radius_mesh, origin, axis)
    axis_surface_radius, _ = _radial_extent(axis_mesh, origin, axis)

    return Wheel(
        wheel=wheel,
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        radius=radius,
        width=width,
        axis_surface_radius=axis_surface_radius,
        axisymmetry_defect=defect,
        axis_source=axis_source,
        radius_source=radius_source,
    )


def derive_wheels(
    patches: Sequence[WheelPatch],
    meshes: dict[str, trimesh.Trimesh],
) -> dict[str, Wheel]:
    """Measure every wheel declared in the patch set.

    Which surface defines the axis, and which defines the radius, is chosen
    by role rather than by name:

    - the axis comes from the wheel's `mrfZone` surface when there is one,
      because that sleeve is a turned cylinder and gives an exact axis, and
      from the `tyre` otherwise;
    - the radius always comes from the `tyre`, because that is the only
      surface that touches the road.

    A wheel with no tyre surface is an error rather than a default: a rolling
    speed guessed from a rim would be quietly too slow everywhere.
    """
    by_wheel: dict[str, list] = {}
    for patch in patches:
        if patch.wheel is not None:
            by_wheel.setdefault(patch.wheel, []).append(patch)

    wheels: dict[str, Wheel] = {}
    for wheel, group in by_wheel.items():
        available = [p for p in group if p.name in meshes]
        tyres = [p for p in available if p.role is PatchRole.TYRE]
        sleeves = [p for p in available if p.role is PatchRole.MRF_ZONE]

        if not tyres:
            raise WheelGeometryError(
                f"wheel '{wheel}' has no patch with role 'tyre' whose surface "
                f"was found; its rolling radius cannot be measured. Patches "
                f"carrying this wheel id: {[p.name for p in group]}"
            )
        if len(tyres) > 1:
            raise WheelGeometryError(
                f"wheel '{wheel}' has {len(tyres)} 'tyre' patches "
                f"({[p.name for p in tyres]}); exactly one defines the "
                "rolling radius"
            )

        tyre = tyres[0]
        axis_patch = sleeves[0] if sleeves else tyre
        wheels[wheel] = measure_wheel(
            wheel=wheel,
            axis_mesh=meshes[axis_patch.name],
            axis_source=axis_patch.name,
            radius_mesh=meshes[tyre.name],
            radius_source=tyre.name,
        )

    return wheels
