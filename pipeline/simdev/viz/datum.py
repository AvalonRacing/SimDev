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


# Below this ratio between consecutive principal extents, two of the car's
# axes are too close in length for the fit to tell them apart, and the frame
# that comes out is a guess. A car chassis separates cleanly - about 3.4
# between long and lateral, 2.1 between lateral and vertical.
MIN_SEPARATION = 1.3


def car_axes(
    meshes: Mapping[str, object], patches: Sequence[str]
) -> tuple[tuple[tuple[float, float, float], ...], dict[str, float], list[str]]:
    """The car's own axes - yaw, pitch AND roll - in mesh coordinates.

    WHY THIS EXISTS. The attitude is baked into the CAD: the testcase pose
    carries about 9.9 degrees of body slip, 1.6 of pitch and 2.7 of roll,
    measured off the Chassis. Frame a picture on the mesh axes and the car
    sits skewed by all three - and a driving state exported under braking or
    at a different steer angle sits skewed differently, so the same nominal
    view of two states is two different views, with nothing in either
    picture saying so. Anchoring the cameras and the slice normals to the
    car is what makes two states comparable.

    ALL THREE ANGLES ARE TAKEN, so a slice keeps the same orientation
    relative to the bodywork whatever the car is doing. The cost is that the
    road no longer sits exactly level in frame - it tilts by the pitch and
    roll of the pose - which is the right trade when the subject of the
    picture is the car rather than the track.

    The frame is an AREA-WEIGHTED principal-axis fit of the datum patches.
    Area-weighted rather than per-vertex because tessellation density varies
    across a surface and vertex counts would let a finely-triangulated
    region pull the fit toward itself.

    Signs are pinned to the CAD's own build convention - nose along +x, up
    along +z - and the frame is re-orthogonalised afterwards so it is exactly
    orthonormal and right-handed rather than merely nearly so.

    Falls back to the mesh axes, with a reason, when the fit is too poorly
    conditioned to trust.

    Returns (axes, angles, reasons) where axes is (x_car, y_car, z_car) in
    mesh coordinates and angles holds yaw/pitch/roll in degrees.
    """
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    flat = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    reasons: list[str] = []

    selected = [meshes[p] for p in patches if p in meshes]
    if not selected:
        if not meshes:
            raise ValueError("no geometry to take car axes from")
        reasons.append(
            f"no datum patch among {', '.join(patches)}: car axes fall back "
            "to the mesh axes, so a posed car will sit skewed in frame"
        )
        return identity, flat, reasons

    centres = np.concatenate(
        [np.asarray(m.triangles_center) for m in selected]  # type: ignore[attr-defined]
    )
    weights = np.concatenate(
        [np.asarray(m.area_faces) for m in selected]  # type: ignore[attr-defined]
    )
    total = weights.sum()
    if total <= 0.0:
        reasons.append("datum surface has no area; car axes fall back to the mesh axes")
        return identity, flat, reasons

    mean = (centres * weights[:, None]).sum(axis=0) / total
    deviation = centres - mean
    covariance = (deviation * weights[:, None]).T @ deviation / total

    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    if values[2] <= 0.0:
        reasons.append("datum surface is degenerate; car axes fall back to the mesh axes")
        return identity, flat, reasons

    separations = (
        float(np.sqrt(values[0] / values[1])),
        float(np.sqrt(values[1] / values[2])),
    )
    if min(separations) < MIN_SEPARATION:
        reasons.append(
            f"datum surface's principal extents are too close to separate "
            f"(ratios {separations[0]:.2f}, {separations[1]:.2f}; need "
            f"{MIN_SEPARATION}): car axes fall back to the mesh axes"
        )
        return identity, flat, reasons

    longitudinal, vertical = vectors[:, 0], vectors[:, 2]
    # The eigenvector signs are arbitrary. The CAD is built nose along +x and
    # up along +z, so the halves agreeing with those are the right ones.
    # Getting either backwards renders every view mirrored or upside down.
    if longitudinal[0] < 0.0:
        longitudinal = -longitudinal
    if vertical[2] < 0.0:
        vertical = -vertical

    # Re-orthogonalise. eigh returns orthogonal vectors, but the sign fixes
    # above and floating-point drift leave the triple only nearly so, and a
    # camera basis that is not exactly orthonormal shears the picture.
    lateral = np.cross(vertical, longitudinal)
    lateral /= np.linalg.norm(lateral)
    longitudinal = np.cross(lateral, vertical)
    longitudinal /= np.linalg.norm(longitudinal)
    vertical = vertical / np.linalg.norm(vertical)

    angles = {
        "yaw_deg": float(np.degrees(np.arctan2(longitudinal[1], longitudinal[0]))),
        "pitch_deg": float(np.degrees(np.arcsin(-longitudinal[2]))),
        "roll_deg": float(np.degrees(np.arctan2(lateral[2], vertical[2]))),
    }
    axes = (
        tuple(float(v) for v in longitudinal),
        tuple(float(v) for v in lateral),
        tuple(float(v) for v in vertical),
    )
    return axes, angles, reasons
