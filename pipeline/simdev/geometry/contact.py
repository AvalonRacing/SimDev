"""Tyre contact patches: cut the tyre near the road, extrude the cut to it.

A loaded tyre is drawn deflected into the road, so the tyre surface meets the
ground plane tangentially. That intersection is the worst thing you can hand
snappyHexMesh: the wedge between tread and road closes to zero angle, and the
cells that have to fill it are slivers. They fail checkMesh on skewness, they
refuse layers, and when they do mesh they put a spurious separation line right
where the wake of the wheel is born.

The fix is the standard one. Cut the tyre on a horizontal plane a fraction of
a millimetre above the road, throw away everything below it, and extrude the
resulting cross-section straight down through the ground plane. What was a
tangential wedge becomes a vertical wall meeting the road at ninety degrees,
and the footprint snappy resolves is the real contact patch rather than
whatever the clipping happened to leave.

Two details are worth stating because both are easy to get wrong:

**The extrusion runs *past* the road, not to it.** Ending it exactly on z = 0
would leave a face coplanar with the ground patch, which is its own class of
snapping failure. Running it below and letting snappy clip it keeps the
intersection a clean edge.

**The bottom is capped only so the STL stays closed.** That cap lives below
z = 0, where the background mesh has no cells, so snappy never intersects it.
It exists because an open surface makes every watertightness check downstream
meaningless, not because anything simulates it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh import grouping

# A vertex is "on the cut plane" within this, in metres. The slice puts them
# there exactly; the tolerance only absorbs the round trip through float32
# when a surface has already been through an STL.
ON_PLANE_TOL = 1.0e-7


class ContactPatchError(Exception):
    pass


@dataclass(frozen=True)
class ContactPatch:
    """What the cut turned out to be, for the run record.

    `area` is the footprint the tyre presents to the road - the area of the
    cross-section at the cut plane - which is the number to sanity-check
    against the load the tyre is carrying. `depth` is how far the tyre was
    drawn into the road, i.e. how much geometry the cut removed.
    """

    area: float
    n_loops: int
    cut_height: float
    floor_z: float
    depth: float


def _boundary_edges_on_plane(
    mesh: trimesh.Trimesh, cut_height: float
) -> np.ndarray:
    """Directed edges bounding a hole that lies in the cut plane.

    Directed as their single owning face uses them, so the winding of
    everything built off them follows without a signed-area heuristic - which
    matters, because a treaded tyre can cut into more than one loop and an
    interior loop runs the opposite way round.

    Restricted to the cut plane on purpose. A tyre that arrived with a hole in
    it somewhere else is a defect `check_geometry` already reports, and
    extruding that hole to the road would turn a reported problem into a
    silently invented one.
    """
    once = grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(once) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    edges = mesh.edges[once]
    z = mesh.vertices[edges][:, :, 2]
    on_plane = np.all(np.abs(z - cut_height) <= ON_PLANE_TOL, axis=1)
    return edges[on_plane]


def _loops(boundary: np.ndarray) -> list[list[int]]:
    """Walk the hole boundary into closed loops.

    Every vertex on a manifold hole boundary has exactly one outgoing and one
    incoming edge, so the walk is a lookup rather than a search. A vertex with
    two outgoing edges means two loops meet at a point, which no cut of a
    sound tyre produces and which no cap could be built for.
    """
    successor: dict[int, int] = {}
    for a, b in boundary:
        # The hole runs opposite to the face that owns the edge.
        if int(b) in successor:
            raise ContactPatchError(
                "the cut cross-section pinches to a point: vertex "
                f"{int(b)} starts two boundary edges. The tyre surface is "
                "self-touching at the cut height"
            )
        successor[int(b)] = int(a)

    loops: list[list[int]] = []
    unvisited = set(successor)
    while unvisited:
        start = unvisited.pop()
        loop = [start]
        node = successor[start]
        while node != start:
            if node not in unvisited:
                raise ContactPatchError(
                    "the cut cross-section does not close into a loop"
                )
            unvisited.discard(node)
            loop.append(node)
            node = successor[node]
        loops.append(loop)
    return loops


def _signed_area(points: np.ndarray) -> float:
    """Shoelace of a loop projected onto the road."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def extrude_contact_patch(
    mesh: trimesh.Trimesh,
    cut_height: float,
    floor_z: float,
) -> tuple[trimesh.Trimesh, ContactPatch | None]:
    """Cut `mesh` at `cut_height` and extrude the cross-section to `floor_z`.

    Returns the new surface and what the cut turned out to be, or the mesh
    unchanged and None when nothing reached below the cut plane - a tyre that
    is not touching the road has no contact patch to build, and inventing one
    would hide that.
    """
    if floor_z >= cut_height:
        raise ContactPatchError(
            f"contact patch floor {floor_z} is not below the cut height "
            f"{cut_height}; there is nothing to extrude"
        )

    lowest = float(mesh.bounds[0][2])
    if lowest >= cut_height:
        return mesh, None

    upper = mesh.slice_plane(
        plane_origin=(0.0, 0.0, cut_height),
        plane_normal=(0.0, 0.0, 1.0),
        cap=False,
    )
    if upper is None or len(upper.faces) == 0:
        raise ContactPatchError(
            f"the whole surface lies below the cut plane at z = {cut_height}; "
            "it is buried in the road, not resting on it"
        )
    upper.merge_vertices()

    boundary = _boundary_edges_on_plane(upper, cut_height)
    if len(boundary) == 0:
        return mesh, None

    loops = _loops(boundary)

    vertices = [np.asarray(upper.vertices, dtype=np.float64)]
    faces = [np.asarray(upper.faces, dtype=np.int64)]
    next_index = len(upper.vertices)

    # One lowered copy of each boundary vertex, so the walls are exactly
    # vertical: same x and y, floor z.
    boundary_vertices = np.unique(boundary)
    lowered = upper.vertices[boundary_vertices].copy()
    lowered[:, 2] = floor_z
    vertices.append(lowered)
    below = {int(v): next_index + i for i, v in enumerate(boundary_vertices)}
    next_index += len(boundary_vertices)

    # Walls. The hole's boundary runs b -> a, so the quad hung from it is
    # wound to continue the surface rather than to face back into it.
    walls = []
    for a, b in boundary:
        a, b = int(a), int(b)
        walls.append([b, a, below[a]])
        walls.append([b, below[a], below[b]])
    faces.append(np.asarray(walls, dtype=np.int64))

    # Floor. A fan from each loop's centroid: it is below the road, where the
    # background mesh has no cells, so it is never intersected and only has to
    # close the surface.
    area = 0.0
    fan = []
    for loop in loops:
        points = upper.vertices[loop]
        area += _signed_area(points)

        centre = np.array(
            [points[:, 0].mean(), points[:, 1].mean(), floor_z], dtype=np.float64
        )
        vertices.append(centre[None, :])
        hub = next_index
        next_index += 1

        for i, node in enumerate(loop):
            nxt = loop[(i + 1) % len(loop)]
            # The walls left this loop running below[nxt] -> below[node];
            # the cap closes it the other way.
            fan.append([hub, below[node], below[nxt]])
    faces.append(np.asarray(fan, dtype=np.int64))

    result = trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        process=False,
    )
    result.merge_vertices()

    if not result.is_watertight:
        raise ContactPatchError(
            "the extruded contact patch did not close. The tyre surface is "
            "probably not watertight above the cut plane either"
        )

    return result, ContactPatch(
        area=abs(area),
        n_loops=len(loops),
        cut_height=cut_height,
        floor_z=floor_z,
        depth=cut_height - lowest,
    )
