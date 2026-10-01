"""Everything the renderer needs, as plain data.

The renderer runs under the system Python (see viz/pv_render.py) and cannot
import simdev, so this module is the interface between them. It carries
numbers and file paths and nothing else: no expression strings, because a
formula shipped as text is an eval waiting to be added and it would put the
cornering-frame arithmetic in two places at once.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from simdev.viz.views import SLICE_FIELDS, SURFACE_FIELDS, Views

# Car axes: nose +x, up +z, therefore car-left +y. Confirmed against the CAD.
# Each entry is (direction the camera looks, up vector); the camera is placed
# opposite its direction, so `front` sits at +x and looks back at the nose.
SURFACE_VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front":  ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "rear":   (( 1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left":   (( 0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "right":  (( 0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    # Top and bottom put the car's LONG axis across the long side of the
    # frame: up = +y, so the car lies lengthwise in a landscape image
    # instead of standing on end in it and wasting both margins.
    "top":    (( 0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "bottom": (( 0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "iso":    ((-1.0 / math.sqrt(3), -1.0 / math.sqrt(3), -1.0 / math.sqrt(3)),
               (0.0, 0.0, 1.0)),
}

# Parallel projection, so this only has to be outside the geometry.
CAMERA_DISTANCE = 2.0

# The direction each slice family is viewed along, and its up vector.
#   x: from downstream looking upstream, so car-left (+y) is on the right -
#      the conventional way to read streamwise vortices.
#   y: from the car's right, so the nose points right.
#   z: from above, nose up. A plan view.
SLICE_VIEW = {
    "x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "z": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# The direction the index counts in, per axis.
#
# x counts from the FRONT of the car to behind it. The car travels +x and
# the air arrives from +x, so the front-most plane is the largest offset and
# the index descends from it - flipping through cp_x/ in filename order then
# walks the car nose to tail. y and z count upward in offset, which is
# right-to-left and ground-up respectively.
COUNT_DESCENDING = {"x": True, "y": False, "z": False}


def slice_index(axis: str, offset: float, offsets: Sequence[float]) -> int:
    """1-based position of this plane in viewing order along its axis."""
    ordered = sorted(offsets, reverse=COUNT_DESCENDING.get(axis, False))
    return ordered.index(offset) + 1


def slice_name(axis: str, offset: float) -> str:
    """Signed, fixed-width, three decimals: `x_+0.120`, `x_-0.300`.

    Two runs with identical offsets always produce identical names, enabling
    file-by-file alignment across runs. That alignment is the whole reason
    a later side-by-side tool is a small job.

    IMPORTANT: Lexicographic order is NOT geometric order. The format uses
    '+' (ASCII 43) for positive and '-' (ASCII 45) for negative, so all
    positive values sort before all negative ones: ['x_+0.000', 'x_+0.300',
    'x_-0.020', 'x_-0.300']. Furthermore, within negatives, 'x_-0.020' sorts
    before 'x_-0.300' (larger magnitudes sort later). Any task that needs
    geometric order (smallest to largest offset) must sort on the numeric
    offset, not on these names. A contact sheet builder sorting by these names
    would put the nose at frame top, skip ahead to +x, then jump to -x and
    walk backward.
    """
    return f"{axis}_{offset:+.3f}"


def _camera(
    focal: tuple[float, float, float],
    direction: Sequence[float],
    up: Sequence[float],
    parallel_scale: float,
) -> dict[str, Any]:
    return {
        "focal": list(focal),
        "position": [focal[i] - CAMERA_DISTANCE * direction[i] for i in range(3)],
        "up": list(up),
        # NEVER fitted to the data. A camera that rescales itself draws two
        # runs at two magnifications with nothing in the image saying so.
        "parallel_scale": parallel_scale,
    }


def _rotate(vector: Sequence[float], axes: Sequence[Sequence[float]]) -> list[float]:
    """Express a car-frame vector in mesh coordinates.

    `axes` is (x_car, y_car, z_car), each a unit vector in mesh coordinates,
    so this is just v[0]*x_car + v[1]*y_car + v[2]*z_car. With axes = the
    identity it returns the vector unchanged, which is what a case with no
    measured yaw gets - one code path, no special casing.
    """
    return [
        sum(vector[k] * axes[k][i] for k in range(3))
        for i in range(3)
    ]


CAR_AXES_IDENTITY: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def build_render_plan(
    u_inf: float,
    views: Views,
    datum: tuple[float, float, float],
    frame: Mapping[str, Any] | None,
    images_dir: Path,
    stamp: Mapping[str, Any],
    axes: Sequence[Sequence[float]] = CAR_AXES_IDENTITY,
) -> dict[str, Any]:
    """The whole picture suite, as one JSON-serialisable dict.

    `axes` are the car's own axes in mesh coordinates, from
    viz.datum.car_axes. Every camera direction, every up vector, every slice
    normal and every offset is expressed in them, so a driving state posed
    at a different body-slip angle still yields the same view of the car
    rather than the same view of the mesh. Defaults to the mesh axes.
    """
    plan: dict[str, Any] = {
        "views_digest": views.digest,
        "resolution": list(views.resolution),
        "streamlines": views.streamlines,
        "datum": list(datum),
        "car_axes": [list(a) for a in axes],
        "stamp": dict(stamp),
        # The renderer builds U_rel, cp and cpt from these. Straight-line and
        # cornering differ only in these numbers, so there is one code path.
        "frame": {
            "mode": "cornering" if frame else "straight",
            "omega": float(frame["omega"]) if frame else 0.0,
            "origin": list(frame["origin"]) if frame else [0.0, 0.0, 0.0],
            "u_inf": float(u_inf),
        },
        "fields": {
            name: {"limits": list(style.limits), "colormap": style.colormap}
            for name, style in views.fields.items()
        },
        "slices": [],
        "surfaces": [],
    }

    for axis, spec_axis in views.axes.items():
        direction, up = SLICE_VIEW[axis]
        index = AXIS_INDEX[axis]
        all_offsets = spec_axis.offsets()
        index_of = {
            o: slice_index(axis, o, all_offsets) for o in all_offsets
        }
        for offset in all_offsets:
            name = slice_name(axis, offset)

            # The plane's normal is the CAR's axis, not the mesh's, so an
            # "x slice" cuts across the car rather than across the domain.
            normal = _rotate(
                [1.0 if i == index else 0.0 for i in range(3)], axes
            )
            # ...and the offset walks along that same rotated axis from the
            # datum, so plane k of two differently-posed states cuts the
            # same place on the car.
            point = [datum[i] + offset * normal[i] for i in range(3)]

            # Focal point tracks the plane along its own normal and stays
            # pinned to the datum in the other two axes, so the car sits in
            # the same pixels in every run and every state.
            if axis == "z":
                focal = [datum[0], datum[1], datum[2] + offset]
            else:
                focal = [
                    datum[0] + offset * normal[0],
                    datum[1] + offset * normal[1],
                    datum[2] + views.focus_height,
                ]

            plan["slices"].append({
                "name": name,
                "axis": axis,
                "offset": offset,
                "point": point,
                "normal": normal,
                # "sample" is not seeded here: images.py fills it in for
                # every slice and every surface once it has actually found
                # the file on disk (find_sample), and overwrites whatever
                # this function wrote before the plan is ever read. Seeding
                # it here would carry a results/samples/... path nothing
                # produced, and a caller that skips that discovery step -
                # a future `simdev compare`, say - would silently ship it.
                "camera": _camera(
                    (focal[0], focal[1], focal[2]),
                    _rotate(direction, axes),
                    _rotate(up, axes),
                    views.parallel_scale[axis],
                ),
                "images": [
                    {
                        "field": field,
                        # vort on a plane is the component NORMAL to it - the
                        # one that shows streamwise vortices punching through.
                        "component": index if field == "vort" else None,
                        # FLAT LAYOUT: one directory per field-and-axis,
                        # e.g. cp_x/ holding cp_x_+0.120.png. Every
                        # directory under results/images therefore holds
                        # pictures and nothing else - no intermediate
                        # levels to click through to reach one.
                        # ZERO-PADDED INDEX FIRST, so the directory lists
                        # in viewing order. The signed offset alone does
                        # not sort: "+" is ASCII 43 and "-" is 45, so
                        # every positive station sorted before every
                        # negative one and the sequence jumped about.
                        "out": str(
                            images_dir / f"{field}_{axis}"
                            / f"{field}_{axis}_{index_of[offset]:02d}_{offset:+.3f}.png"
                        ),
                    }
                    for field in SLICE_FIELDS
                ],
            })

    # SURFACE VIEWS STAY IN THE GLOBAL AXES, UNLIKE THE SLICES.
    #
    # The slices are rotated into the car's frame so plane k cuts the same
    # station on the car whatever attitude it is posed at. These seven are
    # the overall views, and holding them in the domain's own axes means a
    # yawed car LOOKS yawed - the attitude stays visible in the picture
    # rather than being rotated out of it.
    focal = (datum[0], datum[1], datum[2] + views.focus_height)
    for name, (direction, up) in SURFACE_VIEWS.items():
        plan["surfaces"].append({
            "name": name,
            "camera": _camera(
                focal,
                direction,
                up,
                # The iso view sees the car along its diagonal, which is
                # longer than any single axis - at the slice scale the nose
                # and wing ran off the frame. Its own, wider setting.
                views.surface_scale[name],
            ),
            "images": [
                {
                    "field": field,
                    "component": None,
                    "out": str(images_dir / f"surface_{field}" / f"{name}.png"),
                }
                for field in SURFACE_FIELDS
            ],
        })

    return plan
