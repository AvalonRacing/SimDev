from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from simdev.config.schema import CaseSpec


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


class DomainBuilder(Protocol):
    def build(
        self,
        spec: CaseSpec,
        geom_bounds: tuple[Sequence[float], Sequence[float]],
    ) -> DomainBox: ...
