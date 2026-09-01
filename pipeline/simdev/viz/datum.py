"""The car-frame origin every picture is centred and measured from."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def car_datum(
    meshes: Mapping[str, object], patches: Sequence[str]
) -> tuple[tuple[float, float, float], list[str]]:
    """Bounding-box centre of the datum patches in x and y; ground in z.

    z is the ground plane, which is fixed at 0 in every case this pipeline
    builds - so a picture's vertical framing is anchored to the road rather
    than to whatever the bodywork happens to reach.

    WHY THE CHASSIS. Wheels move with steering angle. Body and Wing are the
    surfaces under development, and an anchor that shifts when the thing
    being measured is redesigned anchors nothing - two runs would be framed
    differently *because* the wing changed, which is precisely the difference
    the pictures are meant to show. docs/handbook.md calls the Chassis an
    aero dummy that nobody iterates. That is the property wanted here.

    Falls back to every available surface, with a reason, when none of the
    named patches exist - the Ahmed validation case has no Chassis and still
    needs to be framed.
    """
    reasons: list[str] = []
    selected = [meshes[p] for p in patches if p in meshes]

    if not selected:
        if not meshes:
            raise ValueError("no geometry to take a datum from")
        reasons.append(
            f"no datum patch among {', '.join(patches)}: the datum is the "
            "bounding box of every surface instead, so it moves when any of "
            "them is redesigned"
        )
        selected = list(meshes.values())

    lo = np.min([m.bounds[0] for m in selected], axis=0)  # type: ignore[attr-defined]
    hi = np.max([m.bounds[1] for m in selected], axis=0)  # type: ignore[attr-defined]

    return (
        (float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0), 0.0),
        reasons,
    )
