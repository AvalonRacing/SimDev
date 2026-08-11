from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh

# A mesh whose longest edge is this many times the expected length is
# almost certainly authored in millimetres.
UNIT_SUSPICION_FACTOR = 100.0


@dataclass(frozen=True)
class StlInfo:
    n_triangles: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    is_watertight: bool
    surface_area: float

    @property
    def extent(self) -> tuple[float, float, float]:
        return (
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        )


def stl_info(mesh: trimesh.Trimesh) -> StlInfo:
    lo, hi = mesh.bounds
    return StlInfo(
        n_triangles=len(mesh.faces),
        bounds_min=(float(lo[0]), float(lo[1]), float(lo[2])),
        bounds_max=(float(hi[0]), float(hi[1]), float(hi[2])),
        is_watertight=bool(mesh.is_watertight),
        surface_area=float(mesh.area),
    )


def load_surface(path: Path) -> trimesh.Trimesh:
    """Load an STL with coincident vertices merged.

    STL stores every facet with its own copy of each vertex, so a perfectly
    sound solid arrives as a cloud of unconnected triangles. Without the
    merge, watertightness is False for every mesh ever exported and the
    warning means nothing.
    """
    mesh = trimesh.load_mesh(Path(path), process=True)
    mesh.merge_vertices()
    return mesh


def read_stl_info(path: Path) -> StlInfo:
    return stl_info(load_surface(path))


def import_surface(
    source: Path,
    target: Path,
    scale: float = 1.0,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """Place a CAD surface into the pipeline frame and write it out.

    The transform is applied once, here, and the *transformed* surface is
    what lands in constant/triSurface. snappyHexMesh, the bounding box, the
    frontal area and the wheel axes then all read the same metres in the same
    frame, and no consumer has to know the CAD was authored in millimetres.

    Writing the placed copy rather than scaling at mesh time also means the
    file OpenFOAM reads is the file the pipeline measured, so a geometry
    question can be answered by opening the run directory.
    """
    mesh = load_surface(source)

    if scale != 1.0:
        mesh.apply_scale(scale)
    if any(translate):
        mesh.apply_translation(translate)

    target.parent.mkdir(parents=True, exist_ok=True)
    # Binary: the car's body alone is ~936k triangles, and ASCII STL of that
    # is a few hundred MB per part.
    target.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    return mesh


def projected_frontal_area(mesh: trimesh.Trimesh, axis: int = 0) -> float:
    """Exact projected area along `axis`, via a shapely union of the triangles.

    `precise` unions the triangles through shapely instead of trimesh's fast
    boundary-edge path, and `rpad=0.0` disables the pad/de-pad step trimesh
    applies to bridge gaps between regions. That padding costs ~5e-7 relative
    area, which would make the blockage assertion depend on mesh scale rather
    than on the geometry.
    """
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    projection = mesh.projected(normal, precise=True, rpad=0.0)
    return float(sum(polygon.area for polygon in projection.polygons_full))


def check_geometry(info: StlInfo, expected_length: float | None = None) -> list[str]:
    warnings: list[str] = []

    if not info.is_watertight:
        warnings.append(
            "geometry is not watertight; snappyHexMesh will leak into the "
            "interior and the mesh will be unusable"
        )

    if expected_length is not None:
        longest = max(info.extent)
        if longest > expected_length * UNIT_SUSPICION_FACTOR:
            warnings.append(
                f"largest extent is {longest:.1f} against an expected "
                f"{expected_length:.3f}; unit mismatch, geometry is probably "
                "in millimetres"
            )

    return warnings
