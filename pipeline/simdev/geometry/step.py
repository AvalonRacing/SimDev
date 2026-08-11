"""STEP to STL conversion, cached on what actually affects the result.

The CAD arrives as STEP (one file per patch, NURBS, millimetres) and
snappyHexMesh wants triangles. gmsh does the tessellation through its bundled
OpenCASCADE kernel, so there is no FreeCAD or CAD-vendor dependency.

Two things about this are worth stating up front because both have already
cost time on this project:

**Tessellation quality is a physics setting, not a file format detail.** Too
coarse and a curved surface becomes a faceted one that separates in the wrong
place; too fine and the surface mesh is larger than the volume mesh built from
it. It is therefore config, in metres like every other length, and it goes
into the cache key.

**Never measure a STEP file's bounding box without tessellating it.** OCC's
bounding boxes are built from NURBS control hulls and run large - the earlier
session recorded one about 6x too big. Every dimension this pipeline reports
comes from the triangles, not the B-rep.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR_NAME = ".simdev-cache"


class StepConversionError(Exception):
    pass


@dataclass(frozen=True)
class Tessellation:
    """Surface mesh sizing, in metres.

    `curvature_segments` is elements per full circle, which is the control
    that actually matters on a vehicle: it is what decides whether a 6 mm
    suspension link is a hexagon or a cylinder, independently of how large
    the part is.
    """

    max_edge: float
    min_edge: float
    curvature_segments: int

    def key(self) -> str:
        return json.dumps(
            {
                "max_edge": self.max_edge,
                "min_edge": self.min_edge,
                "curvature_segments": self.curvature_segments,
            },
            sort_keys=True,
        )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(source: Path, tessellation: Tessellation, scale: float) -> str:
    """Everything that changes the triangles, and nothing that does not.

    Scale is in here because the mesh sizes are given in metres and have to
    be converted into the CAD's own units before gmsh sees them - so the same
    file at a different scale is a different tessellation.

    Deliberately *not* in here: the rotation and translation. Those are rigid
    transforms applied to the triangles afterwards, so a ride-height change
    or a new driving-state placement reuses the cached tessellation instead
    of re-tessellating a 5 MB body for nothing.
    """
    payload = f"{_file_digest(source)}|{tessellation.key()}|{scale!r}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def tessellate(
    source: Path,
    target: Path,
    tessellation: Tessellation,
    scale: float,
) -> Path:
    """Tessellate one STEP file to a binary STL, in the CAD's own units."""
    try:
        import gmsh
    except Exception as error:  # pragma: no cover - depends on the install
        raise StepConversionError(
            "geometry.kind is 'step' but gmsh could not be imported "
            f"({error}). It is a declared dependency; on Ubuntu it also needs "
            "the system library libGLU: sudo apt-get install -y libglu1-mesa. "
            "See docs/environment-setup.md"
        ) from error

    # Mesh sizes are configured in metres; gmsh works in the file's own units.
    to_source_units = 1.0 / scale if scale else 1.0

    target.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.occ.importShapes(str(source))
        gmsh.model.occ.synchronize()

        if not gmsh.model.getEntities(3):
            raise StepConversionError(
                f"{source.name} contains no solids. A surface-only STEP cannot "
                "be meshed against: snappyHexMesh needs closed volumes"
            )

        gmsh.option.setNumber("Mesh.MeshSizeMax", tessellation.max_edge * to_source_units)
        gmsh.option.setNumber("Mesh.MeshSizeMin", tessellation.min_edge * to_source_units)
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", tessellation.curvature_segments
        )
        # Sizes come from the settings above, not from whatever the exporter
        # happened to leave in the file.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Binary", 1)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(target))
    finally:
        gmsh.finalize()

    if not target.exists():
        raise StepConversionError(f"gmsh produced no output for {source.name}")
    return target


def convert(
    source: Path,
    cache_dir: Path,
    tessellation: Tessellation,
    scale: float,
) -> tuple[Path, bool]:
    """Tessellate `source`, reusing the cache when nothing relevant changed.

    Returns the cached STL and whether it had to be built. Conversion is the
    slow part of prepare on this geometry - the body alone is 5 MB of NURBS -
    and a case is re-prepared far more often than the CAD changes.
    """
    key = cache_key(source, tessellation, scale)
    cached = Path(cache_dir) / f"{source.stem}-{key}.stl"

    if cached.exists() and cached.stat().st_size > 0:
        return cached, False

    tessellate(source, cached, tessellation, scale)
    return cached, True


def default_cache_dir(source_dir: Path) -> Path:
    """Beside the CAD, so clearing it is obvious and it follows the geometry.

    Falls back to the user cache when the CAD lives somewhere unwritable,
    which is the normal case for a shared or read-only geometry drop.
    """
    candidate = Path(source_dir) / CACHE_DIR_NAME
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".writable"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        fallback = Path(base) / "simdev" / "step"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
