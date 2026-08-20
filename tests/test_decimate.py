"""Vertex clustering, for the one surface that is measured from and not meshed.

Quadric decimation is the usual choice and it is the wrong one here: its error
metric is happy to flatten a 4 mm suspension link into a sliver, because doing
so barely moves any surface. That is exactly the feature this case cannot
afford to lose - the shells exist to refine the air *between* body, chassis and
wishbones.

Clustering has the property the quadric metric lacks: the error is bounded by
the grid, so nothing thinner than the tolerance survives and nothing thicker
can be removed. The tolerance is therefore a statement about which features
are kept, which is the decision that actually has to be made.
"""

from __future__ import annotations

import numpy as np
import trimesh

from simdev.geometry.decimate import cluster_vertices


def test_vertices_closer_than_the_tolerance_are_merged() -> None:
    fine = trimesh.creation.icosphere(subdivisions=4, radius=0.05)

    coarse = cluster_vertices(fine, 0.01)

    assert len(coarse.faces) < len(fine.faces)


def test_a_surface_already_coarser_than_the_tolerance_is_untouched() -> None:
    """Clustering can only ever merge. A box has nothing to merge."""
    box = trimesh.creation.box(extents=(0.40, 0.18, 0.10))

    assert len(cluster_vertices(box, 0.01).faces) == len(box.faces)


def test_no_vertex_moves_further_than_the_grid_diagonal() -> None:
    """The bound that makes the tolerance mean something.

    A cluster's representative is the mean of vertices sharing one grid cell,
    so no vertex can be displaced further than the cell's diagonal.
    """
    fine = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    tolerance = 0.01

    coarse = cluster_vertices(fine, tolerance)

    moved = np.abs(coarse.bounds - fine.bounds).max()
    assert moved <= tolerance * np.sqrt(3.0)


def test_a_tolerance_of_zero_returns_the_surface_unchanged() -> None:
    fine = trimesh.creation.icosphere(subdivisions=3, radius=0.05)

    unchanged = cluster_vertices(fine, 0.0)

    assert len(unchanged.faces) == len(fine.faces)


def test_degenerate_faces_are_dropped_rather_than_written_out() -> None:
    """Collapsing a triangle's vertices into one cell leaves a zero-area face.

    Written to STL those become slivers that snappy's octree still has to
    carry, which defeats the point.
    """
    fine = trimesh.creation.icosphere(subdivisions=4, radius=0.05)

    coarse = cluster_vertices(fine, 0.02)

    assert coarse.area > 0.0
    assert np.all(coarse.area_faces > 0.0)


def test_a_feature_thinner_than_the_tolerance_can_collapse() -> None:
    """Stated as a test because it is the risk the whole guard exists for.

    A 4 mm link whose two walls fall in one 20 mm cell loses both of them and
    takes its surface area with it. In the distance field that link has simply
    ceased to exist, and stops pulling any refinement around itself.

    Note "can", not "does": whether it happens depends on where the grid
    boundary falls relative to the feature, so this is not something the
    tolerance alone can rule out - which is why prepare measures the area each
    part actually retained instead of trusting the arithmetic.
    """
    link = trimesh.creation.box(extents=(0.004, 0.004, 0.12))
    # Placed inside one cell rather than straddling a boundary. Centred on the
    # origin the same link survives, which is the phase-dependence itself.
    link.apply_translation([0.01, 0.01, 0.0])

    collapsed = cluster_vertices(link, 0.02)

    assert collapsed.area < 0.5 * link.area


def test_the_same_feature_survives_when_it_straddles_a_cell_boundary() -> None:
    """The other half of the phase-dependence, so nobody reads the test above
    as a guarantee and sizes a tolerance against it."""
    link = trimesh.creation.box(extents=(0.004, 0.004, 0.12))

    assert cluster_vertices(link, 0.02).area == link.area
