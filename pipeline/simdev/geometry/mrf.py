"""MRF sleeves that interfere with the tyre instead of coinciding with it.

An MRF sleeve is a closed volume whose faces become a cellZone and a faceZone
rather than a wall. It has to enclose the air in the wheel, which means it
lives just inside the tyre - and "just inside" is where CAD naturally puts it
*exactly on* the tyre's bore, because both were drawn from the same nominal
diameter.

That is the one place it must not be. A faceZone surface lying on a wall
surface is degenerate for snappyHexMesh: it has to snap the wall and insert a
zone boundary in the same location, and what it produces instead is baffles,
faceZones that come back "multiply connected (shared edge)", non-manifold
points, and intermittently a face with reversed orientation. It was measured
on this car: the sleeve wall at r = 27.00 mm against a tyre bore at
r = 26.95 mm, with the outer cap coplanar with the sidewall, so about three
quarters of the sleeve was welded to the tyre. checkMesh reported all four
zones multiply connected with 381 non-manifold points, and max
non-orthogonality sat above the gate until the sleeves were taken out.

The fix is not to shrink the sleeve away from the tyre - that leaves the zone
short of the air it is supposed to rotate, and a near miss is as fragile as a
hit. It is to push it deliberately *into* the tyre, so the two surfaces
plainly intersect rather than plainly touch. Where the sleeve is buried inside
tyre material there are no fluid cells at all, so no zone boundary is created
there, and the cellZone comes out bounded by the tyre's own wall - which is
what it should have been bounded by all along.

Two directions, and they are not the same:

- **Radially it grows**, straight into the carcass between bore and tread.
- **Axially it shrinks**, pulling each end cap off the sidewall plane it was
  flush with and back into the wheel's own air. Growing it axially instead
  would push the cap out past the sidewall into the free stream, and a
  rotating zone in the free stream spins air that should be still.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# Points used per surface when measuring the gap. The clouds are compared with
# a KD-tree rather than by exact surface projection, because exact projection
# in trimesh needs rtree, and that is a system library this pipeline has been
# bitten by before (see docs/environment-setup.md on libGLU).
#
# 20k is deliberately modest, and enough. The question is only ever whether
# two surfaces are lying on each other or clear of each other - roughly zero
# against roughly a millimetre - not what the gap is to four figures. A
# coincident pair is found at any density, and sparse sampling can only ever
# *over*-estimate a gap, so the failure direction is "did not notice a
# coincidence", never "moved a sleeve that was fine".
#
# It is also called for every wheel of every prepare. At 200k it added seconds
# per case and minutes to the test suite for precision nothing reads.
CLOUD_POINTS = 20_000


class MrfGeometryError(Exception):
    pass


@dataclass(frozen=True)
class SleeveFit:
    """What one sleeve was, and what was done about it."""

    wheel: str
    sleeve: str
    tyre: str
    clearance: float
    interference: float
    radius_before: float
    radius_after: float
    tyre_radius: float

    @property
    def moved(self) -> bool:
        return self.interference > 0.0


def surface_gap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Smallest distance between two surfaces, in metres.

    Zero means they touch or intersect. This does not distinguish the two, and
    does not need to: both are the degenerate case.
    """
    pa = a.sample(CLOUD_POINTS)
    pb = b.sample(CLOUD_POINTS)
    d_ab, _ = cKDTree(pb).query(pa)
    d_ba, _ = cKDTree(pa).query(pb)
    return float(min(d_ab.min(), d_ba.min()))


def push_into_tyre(
    sleeve: trimesh.Trimesh,
    origin: np.ndarray,
    axis: np.ndarray,
    interference: float,
) -> trimesh.Trimesh:
    """Grow the sleeve radially and inset it axially, both by `interference`.

    Worked in the wheel's own cylindrical frame rather than along vertex
    normals. A normal offset would move the end caps outward too, and the cap
    that matters is the one already flush with the outboard sidewall - pushing
    that one out puts the rotating zone in the free stream, which is a worse
    fault than the one being repaired.
    """
    origin = np.asarray(origin, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    offsets = sleeve.vertices - origin
    axial = offsets @ axis
    radial = offsets - axial[:, None] * axis
    r = np.linalg.norm(radial, axis=1)

    # Radially outward, into the carcass.
    scale = np.where(r > 1e-12, (r + interference) / np.maximum(r, 1e-12), 1.0)
    grown = radial * scale[:, None]

    # Axially inward, off the sidewall planes.
    lo, hi = float(axial.min()), float(axial.max())
    half = 0.5 * (hi - lo)
    if half <= interference:
        raise MrfGeometryError(
            f"sleeve is {half * 2e3:.2f} mm long and the interference is "
            f"{interference * 1e3:.2f} mm; insetting both caps would collapse it"
        )
    mid = 0.5 * (hi + lo)
    pulled = mid + (axial - mid) * ((half - interference) / half)

    moved = sleeve.copy()
    moved.vertices = origin + grown + pulled[:, None] * axis
    return moved


def outer_radius(mesh: trimesh.Trimesh, origin, axis) -> float:
    origin = np.asarray(origin, dtype=float)
    axis = np.asarray(axis, dtype=float)
    offsets = mesh.vertices - origin
    axial = offsets @ axis
    return float(np.linalg.norm(offsets - axial[:, None] * axis, axis=1).max())
