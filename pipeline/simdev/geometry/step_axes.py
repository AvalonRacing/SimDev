"""Read a part's own coordinate system out of a STEP file.

WHY READ IT RATHER THAN MEASURE IT. The car's attitude has to be known to
frame a picture in the car's own axes, and viz/datum.py can fit it from the
geometry - but a fit is an inference. The CAD already states the answer: the
designer placed the part on a coordinate system, and that placement is in
the file. Reading it is exact, independent of how the surface happens to be
tessellated, and identical for every driving state exported from the same
model. The fit stays as the fallback.

The two agree closely in practice - 0.8 degrees apart on CAD/Testcase/Body -
which is the check that neither is wrong, not a reason to prefer either.

WHAT IS PARSED. STEP carries the placement as

    ITEM_DEFINED_TRANSFORMATION($, $, #from, #to)
    #to = AXIS2_PLACEMENT_3D('', #origin, #axis, #refdir)
    #axis   = DIRECTION('axis',   (x, y, z))   -> the part's local z
    #refdir = DIRECTION('refdir', (x, y, z))   -> the part's local x

This is deliberately a narrow, literal reader rather than a STEP library:
it resolves four entity references and reads two direction triples. Anything
it does not recognise raises, and the caller falls back to the measured fit
rather than guessing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_TRANSFORM = re.compile(
    r"ITEM_DEFINED_TRANSFORMATION\s*\(\s*\$?\s*,\s*\$?\s*,\s*#(\d+)\s*,\s*#(\d+)\s*\)"
)
_PLACEMENT = re.compile(
    r"^#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\([^,]*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*\)",
    re.MULTILINE,
)
_DIRECTION = re.compile(
    r"^#(\d+)\s*=\s*DIRECTION\s*\([^,]*,\s*\(([^)]*)\)\s*\)", re.MULTILINE
)


class StepAxesError(Exception):
    """The file carries no placement this reader understands."""


def _directions(text: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for match in _DIRECTION.finditer(text):
        try:
            values = [float(v) for v in match.group(2).split(",")]
        except ValueError:
            continue
        if len(values) == 3:
            out[int(match.group(1))] = np.array(values, dtype=float)
    return out


def read_step_axes(path: Path) -> tuple[tuple[float, float, float], ...]:
    """The part's axes as (x_car, y_car, z_car) unit vectors.

    Oriented to this pipeline's convention - nose along +x, up along +z -
    rather than to whatever sign the CAD happened to store. The Body part's
    own x axis points REARWARD in CAD/Testcase, so this flip is load-bearing
    and not a formality: without it every view renders back to front.

    The frame is re-orthogonalised on the way out. STEP stores axis and
    refdir as exact perpendiculars, but they arrive as decimal text and a
    camera basis that is not exactly orthonormal shears the picture.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    transform = _TRANSFORM.search(text)
    if transform is None:
        raise StepAxesError(f"{path}: no ITEM_DEFINED_TRANSFORMATION")
    target = int(transform.group(2))

    placements = {
        int(m.group(1)): (int(m.group(3)), int(m.group(4)))
        for m in _PLACEMENT.finditer(text)
    }
    if target not in placements:
        raise StepAxesError(f"{path}: #{target} is not an AXIS2_PLACEMENT_3D")

    axis_ref, refdir_ref = placements[target]
    directions = _directions(text)
    if axis_ref not in directions or refdir_ref not in directions:
        raise StepAxesError(f"{path}: placement #{target} has no direction triples")

    vertical = directions[axis_ref]
    longitudinal = directions[refdir_ref]

    for vector in (vertical, longitudinal):
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0.0:
            raise StepAxesError(f"{path}: placement #{target} has a degenerate axis")

    vertical = vertical / np.linalg.norm(vertical)
    longitudinal = longitudinal / np.linalg.norm(longitudinal)

    # This pipeline's convention, not the CAD's: nose along +x, up along +z.
    if longitudinal[0] < 0.0:
        longitudinal = -longitudinal
    if vertical[2] < 0.0:
        vertical = -vertical

    lateral = np.cross(vertical, longitudinal)
    if np.linalg.norm(lateral) < 1e-9:
        raise StepAxesError(f"{path}: placement #{target} axes are parallel")
    lateral /= np.linalg.norm(lateral)
    longitudinal = np.cross(lateral, vertical)
    longitudinal /= np.linalg.norm(longitudinal)

    return (
        tuple(float(v) for v in longitudinal),
        tuple(float(v) for v in lateral),
        tuple(float(v) for v in vertical),
    )
