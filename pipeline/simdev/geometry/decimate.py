"""Simplify a surface that is measured from rather than meshed against.

snappyHexMesh answers a distance-mode refinement region by asking an octree
over the surface's triangles for the nearest point, and it rebuilds that octree
on every rank each time the mesh is redistributed. The cost is therefore set by
the triangle count of the surface, not by how finely the mesh near it is
resolved - so a distance field built from the full-resolution CAD tessellation
is paying tessellation prices for an answer that only has to be good to within
a cell or two.

WHY CLUSTERING AND NOT QUADRIC DECIMATION, which is the usual tool. A quadric
error metric measures how far the surface moves, and collapsing a 4 mm
suspension link flat moves almost no surface at all - so it is precisely the
operation the metric likes best. Losing that link from the distance field
would switch off the refinement around it, and the refinement between body,
chassis and wishbones is what the shells were added for.

Rossignac-Borrel clustering has the property the quadric metric lacks: the
error is bounded by the grid cell, so no part of the surface can move further
than the tolerance however aggressively the triangle count falls. That makes
the tolerance a statement about how far the refinement boundary may shift,
which is the decision that actually has to be made here.

WHAT IT IS NOT is a guarantee about which features survive. Whether a feature
thinner than the tolerance collapses depends on where the grid boundary
happens to fall across it: a 4 mm link sitting inside one 20 mm cell loses
both walls, and the same link straddling a cell boundary keeps them. So the
tolerance cannot be sized by arithmetic alone, and prepare measures the
surface area each part actually retained instead - a collapsed feature takes
its area with it, while honest simplification of a curved panel does not.

It also needs nothing that is not already a dependency. trimesh's own
`simplify_quadric_decimation` defers to `fast_simplification`, which is not
installed and would have to be installed in WSL too.
"""

from __future__ import annotations

import numpy as np
import trimesh


def cluster_vertices(mesh: trimesh.Trimesh, tolerance: float) -> trimesh.Trimesh:
    """Merge every vertex sharing a `tolerance`-sized grid cell.

    Returns a new surface; `mesh` is not modified. A tolerance of zero (or
    less) returns the surface unchanged, which is what makes the feature
    default to off.

    The representative of a cluster is the mean of its members rather than one
    of them, so a curved panel keeps its position instead of stepping onto the
    grid. No vertex moves further than the cell diagonal either way.
    """
    if tolerance <= 0.0:
        return mesh.copy()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    cell = np.floor(vertices / tolerance).astype(np.int64)
    _, index = np.unique(cell, axis=0, return_inverse=True)
    # numpy 2 returns this shaped like the input along `axis`; older numpy
    # returns it flat. Ravel so the face remap below is right on both.
    index = np.asarray(index).ravel()

    counts = np.bincount(index)
    merged = np.column_stack(
        [
            np.bincount(index, weights=vertices[:, axis], minlength=len(counts))
            / counts
            for axis in range(3)
        ]
    )

    faces = index[np.asarray(mesh.faces)]
    # A triangle whose corners landed in the same cell has collapsed to a line
    # or a point. Left in, it is a zero-area sliver that the octree still has
    # to carry - which is the cost this whole module exists to avoid.
    kept = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )

    out = trimesh.Trimesh(vertices=merged, faces=faces[kept], process=False)
    out.remove_unreferenced_vertices()
    return out
