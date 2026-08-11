from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from simdev.config.schema import CaseSpec


@runtime_checkable
class Domain(Protocol):
    """What every domain kind must answer, whatever shape it is.

    Deliberately an axis-aligned bounding box plus patch names rather than a
    shape description. Everything downstream that touches the domain wants
    one of two things: somewhere to put a point (locationInMesh, refinement
    region clipping) or a patch name to hang a boundary condition on. Neither
    needs to know whether the tunnel is straight or curved, so neither has to
    branch on it - only blockMeshDict does.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    inlet: str
    outlet: str
    ground: str
    farfield: str
    symmetry: str | None
    geom_min: tuple[float, float, float]
    geom_max: tuple[float, float, float]

    @property
    def cell_count(self) -> int: ...

    @property
    def cross_section_area(self) -> float: ...

    @property
    def block_mesh_template(self) -> str: ...


@dataclass(frozen=True)
class DomainBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    n_cells: tuple[int, int, int]
    inlet: str
    outlet: str
    ground: str
    farfield: str
    symmetry: str | None
    # +1 if the freestream blows along +x, -1 if along -x. Decides which end
    # of the box the air enters through.
    flow_sign: float = 1.0
    # Bounding box of the geometry the domain was built around. Refinement
    # regions are anchored to the model, not the tunnel, so they need this.
    geom_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geom_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def geom_length(self) -> float:
        return self.geom_max[0] - self.geom_min[0]

    @property
    def x_min_patch(self) -> str:
        """Patch on the -x face. The inlet only when the flow runs +x."""
        return self.inlet if self.flow_sign > 0 else self.outlet

    @property
    def x_max_patch(self) -> str:
        return self.outlet if self.flow_sign > 0 else self.inlet

    @property
    def inlet_x(self) -> float:
        """The x coordinate the air enters at."""
        return self.x_min if self.flow_sign > 0 else self.x_max

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    @property
    def cross_section_area(self) -> float:
        _, dy, dz = self.size
        return dy * dz

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.n_cells
        return nx * ny * nz

    @property
    def block_mesh_template(self) -> str:
        return "blockMeshDict.jinja"


@dataclass(frozen=True)
class ArcBlock:
    """One circumferential block of the sector, in absolute coordinates.

    blockMesh draws a curved edge as a circular arc through a single
    interpolation point, which only traces a circle faithfully over a modest
    angle. A cornering domain can span well past 90 degrees - at a 3 m radius
    a fifteen-body-length tunnel sweeps about 126 - so the sector is built
    from several blocks in series rather than one, and each keeps its arc
    short enough to stay round.
    """

    theta_start: float
    theta_end: float
    n_circumferential: int


@dataclass(frozen=True)
class DomainSector:
    """An annular sector: the tunnel bent around the corner the car is taking.

    The car sits at `radius` from `centre`, on a vertical axis. Radial extent
    is the tunnel's width, the swept angle is its length, and height is
    unchanged from the straight case - so a case keeps the same domain
    numbers whether it is cornering or not.

    Angles are measured in the usual sense about +z from the +x axis, and
    `theta_car` is where the vehicle sits. The inlet is placed on whichever
    side of the car the air arrives from, which depends on the direction of
    the turn; see AnnulusDomainBuilder.
    """

    centre: tuple[float, float]
    radius: float
    r_min: float
    r_max: float
    z_min: float
    z_max: float
    theta_car: float
    theta_inlet: float
    theta_outlet: float
    # Always ordered by increasing theta, whichever way the air flows. Block
    # handedness depends on the sign of the angular step, so generating them
    # in flow order would silently invert every cell on a right-hand corner;
    # which end is the inlet is recorded separately instead.
    blocks: tuple[ArcBlock, ...]
    inlet_at_theta_min: bool
    n_radial: int
    n_vertical: int
    inlet: str
    outlet: str
    ground: str
    farfield: str
    symmetry: str | None
    geom_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geom_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def geom_length(self) -> float:
        return self.geom_max[0] - self.geom_min[0]

    @property
    def sweep(self) -> float:
        """Total swept angle, always positive."""
        return abs(self.theta_inlet - self.theta_outlet)

    @property
    def arc_length(self) -> float:
        """Tunnel length along the vehicle's own path."""
        return self.sweep * self.radius

    @property
    def cross_section_area(self) -> float:
        return (self.r_max - self.r_min) * (self.z_max - self.z_min)

    @property
    def cell_count(self) -> int:
        circumferential = sum(b.n_circumferential for b in self.blocks)
        return circumferential * self.n_radial * self.n_vertical

    @property
    def block_mesh_template(self) -> str:
        return "blockMeshDict_annulus.jinja"

    def point(self, theta: float, r: float, z: float) -> tuple[float, float, float]:
        return (
            self.centre[0] + r * math.cos(theta),
            self.centre[1] + r * math.sin(theta),
            z,
        )

    def _corner_points(self) -> list[tuple[float, float, float]]:
        thetas = [b.theta_start for b in self.blocks] + [self.blocks[-1].theta_end]
        return [
            self.point(theta, r, z)
            for theta in thetas
            for r in (self.r_min, self.r_max)
            for z in (self.z_min, self.z_max)
        ]

    # The sector's own axis-aligned bounding box. Used for placing
    # locationInMesh and for clipping refinement regions, both of which only
    # need "somewhere inside" rather than the true curved boundary.
    @property
    def x_min(self) -> float:
        return min(p[0] for p in self._corner_points())

    @property
    def x_max(self) -> float:
        return max(p[0] for p in self._corner_points())

    @property
    def y_min(self) -> float:
        return min(p[1] for p in self._corner_points())

    @property
    def y_max(self) -> float:
        return max(p[1] for p in self._corner_points())

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )


class DomainBuilder(Protocol):
    def build(
        self,
        spec: CaseSpec,
        geom_bounds: tuple[Sequence[float], Sequence[float]],
    ) -> Domain: ...
