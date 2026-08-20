"""The tyre/road junction, which is the hardest thing on the car to mesh.

A tyre drawn deflected into the road meets it tangentially, and the wedge
between tread and ground closes to zero angle. The cut-and-extrude turns that
into a vertical wall at ninety degrees. What has to be true afterwards is
narrow and checkable: the surface is still closed, the wall is actually
vertical, and nothing was invented for a tyre that was not touching the road.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from simdev.geometry.contact import ContactPatchError, extrude_contact_patch

CUT = 0.0005
FLOOR = -0.002
RADIUS = 0.033
WIDTH = 0.027


def tyre(
    sink: float = 0.0015, r_min: float = 0.012, sections: int = 128
) -> trimesh.Trimesh:
    """A tyre on its axis along y, sunk `sink` into the road at z = 0.

    `sections` is left fine by default because the footprint is a chord of the
    tread and a coarse polygon is measurably flatter at the bottom than the
    circle it stands for - at trimesh's default 32 it reads 3% wide.
    """
    transform = trimesh.transformations.translation_matrix(
        [0.0, 0.0, RADIUS - sink]
    ) @ trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0])
    return trimesh.creation.annulus(
        r_min=r_min,
        r_max=RADIUS,
        height=WIDTH,
        sections=sections,
        transform=transform,
    )


def _section_area(mesh: trimesh.Trimesh, z: float) -> float:
    section = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    planar, _ = section.to_2D()
    return sum(abs(polygon.area) for polygon in planar.polygons_full)


# --- the surface stays sound ---------------------------------------------


def test_the_cut_surface_is_still_closed() -> None:
    """An open tyre makes every watertightness check downstream meaningless."""
    cut, _ = extrude_contact_patch(tyre(), CUT, FLOOR)

    assert cut.is_watertight
    assert cut.is_winding_consistent
    assert cut.volume > 0.0


def test_the_extrusion_is_vertical() -> None:
    """The whole point: a wall at ninety degrees, not a ramp.

    A constant cross-section from the cut plane down to the floor is what
    "vertical" means in a form that does not depend on how the tyre was
    tessellated.
    """
    cut, patch = extrude_contact_patch(tyre(), CUT, FLOOR)

    # Tight, but not to the last bit: the tolerance has to clear the noise in
    # slicing a triangle a hair below its own top edge, while a ramp would
    # miss by tens of percent.
    at_cut = _section_area(cut, CUT - 1e-5)
    for z in (0.0, -0.001, FLOOR + 1e-5):
        assert _section_area(cut, z) == pytest.approx(at_cut, rel=1e-4)
    assert patch.area == pytest.approx(at_cut, rel=1e-4)


def test_the_wall_runs_past_the_road_rather_than_stopping_on_it() -> None:
    """A face coplanar with the ground patch is its own snapping failure."""
    cut, patch = extrude_contact_patch(tyre(), CUT, FLOOR)

    assert float(cut.bounds[0][2]) == pytest.approx(FLOOR)
    assert patch.floor_z == FLOOR


def test_nothing_of_the_tyre_survives_below_the_cut_except_the_wall() -> None:
    original = tyre()
    cut, _ = extrude_contact_patch(original, CUT, FLOOR)

    # Every vertex below the cut plane is one of the lowered copies.
    below = cut.vertices[cut.vertices[:, 2] < CUT - 1e-9]
    assert len(below) > 0
    assert np.allclose(below[:, 2], FLOOR)


def test_the_footprint_is_the_size_the_deflection_implies() -> None:
    """A 1.5 mm deflection on a 33 mm tyre gives a footprint of known length.

    The cut plane sits `depth` below the top of the tread's swept circle, so
    the footprint is the chord at that depth: sqrt(r^2 - (r - depth)^2) each
    side of the axis, over the width of the tread. That pins the area to the
    geometry rather than to whatever the code happened to produce.
    """
    sink = 0.0015
    cut, patch = extrude_contact_patch(tyre(sink=sink), CUT, FLOOR)

    assert patch.depth == pytest.approx(CUT + sink, rel=1e-6)

    half_length = math.sqrt(RADIUS**2 - (RADIUS - patch.depth) ** 2)
    assert patch.area == pytest.approx(2 * half_length * WIDTH, rel=0.02)


# --- what it refuses to do -----------------------------------------------


def test_a_tyre_that_is_not_touching_the_road_is_left_alone() -> None:
    """No contact, no contact patch. Inventing one would hide the fault."""
    floating = tyre(sink=-0.002)
    cut, patch = extrude_contact_patch(floating, CUT, FLOOR)

    assert cut is floating
    assert patch is None


def test_a_tyre_resting_exactly_on_the_road_still_gets_a_footprint() -> None:
    """Zero deflection is the awkward case: tangential contact, no depth."""
    cut, patch = extrude_contact_patch(tyre(sink=0.0), CUT, FLOOR)

    assert cut.is_watertight
    assert patch is not None
    assert patch.area > 0.0


def test_a_surface_buried_in_the_road_is_an_error() -> None:
    buried = tyre(sink=0.10)
    with pytest.raises(ContactPatchError, match="buried in the road"):
        extrude_contact_patch(buried, CUT, FLOOR)


def test_a_floor_above_the_cut_is_an_error() -> None:
    with pytest.raises(ContactPatchError, match="nothing to extrude"):
        extrude_contact_patch(tyre(), CUT, CUT + 0.001)


# --- shapes that cut into more than one loop -----------------------------


def test_a_treaded_tyre_cuts_into_several_footprints() -> None:
    """Two blocks of tread give two separate contact patches, each closed.

    The winding of an extruded loop follows from the face that owns its
    boundary edge rather than from a signed area, which is what makes this
    work without knowing which loop is which.
    """
    blocks = [
        trimesh.creation.box(
            extents=(0.010, 0.008, 0.010),
            transform=trimesh.transformations.translation_matrix(
                [0.0, y, 0.0035]
            ),
        )
        for y in (-0.008, 0.008)
    ]
    hub = trimesh.creation.box(
        extents=(0.030, 0.027, 0.020),
        transform=trimesh.transformations.translation_matrix([0.0, 0.0, 0.018]),
    )
    treaded = trimesh.util.concatenate(blocks + [hub])

    cut, patch = extrude_contact_patch(treaded, CUT, FLOOR)

    assert cut.is_watertight
    assert cut.is_winding_consistent
    assert patch.n_loops == 2
    assert patch.area == pytest.approx(2 * 0.010 * 0.008, rel=1e-6)
