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


def place_surface(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh:
    """Convert a CAD surface's units. Nothing else.

    Scale only, and deliberately so. Attitude - yaw, pitch, roll, steering,
    camber - and ride height are set in CAD and arrive baked into the part
    positions, so the pipeline never rotates, translates or snaps the
    geometry. Every transform it is allowed to apply is a place where the
    simulated car can differ from the drawn one, invisibly.

    When the CAD frame and the tunnel disagree about which way the car
    faces, the *tunnel* is reversed (flow.direction), not the car.
    """
    if scale != 1.0:
        mesh.apply_scale(scale)
    return mesh


def write_surface(mesh: trimesh.Trimesh, target: Path) -> Path:
    """Write the placed surface where OpenFOAM will read it.

    Binary, because the body alone tessellates to ~356k triangles and ASCII
    STL of that is hundreds of megabytes.

    The file written is the file the pipeline measured, so any question about
    what was actually simulated is answered by opening the run directory
    rather than by re-deriving the transform.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    return target


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
