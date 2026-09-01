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
    "top":    (( 0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    "bottom": (( 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
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


def slice_name(axis: str, offset: float) -> str:
    """Signed, fixed-width, three decimals: `x_+0.120`, `x_-0.300`.

    Signed and zero-padded so the names sort in geometric order and two runs'
    image directories line up file for file. That alignment is the whole
    reason a later side-by-side tool is a small job.
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


def build_render_plan(
    u_inf: float,
    views: Views,
    datum: tuple[float, float, float],
    frame: Mapping[str, Any] | None,
    samples_dir: Path,
    images_dir: Path,
    stamp: Mapping[str, Any],
) -> dict[str, Any]:
    """The whole picture suite, as one JSON-serialisable dict."""
    plan: dict[str, Any] = {
        "views_digest": views.digest,
        "resolution": list(views.resolution),
        "streamlines": views.streamlines,
        "datum": list(datum),
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
        for offset in spec_axis.offsets():
            name = slice_name(axis, offset)

            point = list(datum)
            point[index] = datum[index] + offset

            # Focal point tracks the plane along its own normal and stays
            # pinned to the datum in the other two axes, so the car sits in
            # the same pixels in every run and every state.
            focal = [datum[0], datum[1], datum[2] + views.focus_height]
            if axis == "z":
                focal = [datum[0], datum[1], datum[2] + offset]
            else:
                focal[index] = datum[index] + offset

            plan["slices"].append({
                "name": name,
                "axis": axis,
                "offset": offset,
                "point": point,
                "normal": [1.0 if i == index else 0.0 for i in range(3)],
                "sample": str(samples_dir / f"{name}"),
                "camera": _camera(
                    (focal[0], focal[1], focal[2]),
                    direction,
                    up,
                    views.parallel_scale[axis],
                ),
                "images": [
                    {
                        "field": field,
                        # vort on a plane is the component NORMAL to it - the
                        # one that shows streamwise vortices punching through.
                        "component": index if field == "vort" else None,
                        "out": str(images_dir / "slices" / axis / field / f"{field}_{name}.png"),
                    }
                    for field in SLICE_FIELDS
                ],
            })

    focal = (datum[0], datum[1], datum[2] + views.focus_height)
    for name, (direction, up) in SURFACE_VIEWS.items():
        plan["surfaces"].append({
            "name": name,
            "camera": _camera(focal, direction, up, max(views.parallel_scale.values())),
            "images": [
                {
                    "field": field,
                    "component": None,
                    "out": str(images_dir / "surface" / field / f"{name}.png"),
                }
                for field in SURFACE_FIELDS
            ],
        })

    return plan
