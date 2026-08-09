from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from simdev.config.schema import AhmedParams

N_NOSE_SECTIONS = 24


def _nose_inset(x: float, radius: float) -> float:
    """Circular fillet profile: d(x) = R - sqrt(R^2 - (R-x)^2) for x < R."""
    if x >= radius:
        return 0.0
    return radius - math.sqrt(max(radius * radius - (radius - x) ** 2, 0.0))


def _sections(params: AhmedParams) -> list[tuple[float, float, float, float]]:
    """Return (x, half_width, z_bottom, z_top) for each cross-section."""
    theta = math.radians(params.slant_angle_deg)
    slant_dx = params.slant_length * math.cos(theta)
    slant_dz = params.slant_length * math.sin(theta)
    slant_start = params.length - slant_dx

    xs: list[float] = [
        params.nose_radius * i / N_NOSE_SECTIONS for i in range(N_NOSE_SECTIONS)
    ]
    xs += [params.nose_radius, slant_start, params.length]
    xs = sorted(set(round(x, 12) for x in xs))

    out: list[tuple[float, float, float, float]] = []
    for x in xs:
        inset = _nose_inset(x, params.nose_radius)
        half_width = params.width / 2.0 - inset
        z_bottom = params.ground_clearance + inset
        z_top = params.ground_clearance + params.height - inset
        if x > slant_start:
            z_top -= (x - slant_start) * math.tan(theta)
        out.append((x, half_width, z_bottom, z_top))

    # Guard the arithmetic that produces the tail height. The tail is reached
    # via (x - slant_start) * tan(theta) but specified as slant_length *
    # sin(theta); those agree algebraically and differ in the last bit, so this
    # is a tolerance check rather than an equality.
    assert math.isclose(
        out[-1][3],
        params.ground_clearance + params.height - slant_dz,
        abs_tol=1e-12,
    )
    return out


def _ring(half_width: float, z_bottom: float, z_top: float, x: float) -> np.ndarray:
    """Four corners of a cross-section, counter-clockwise seen from -x."""
    return np.array(
        [
            [x, -half_width, z_bottom],
            [x, half_width, z_bottom],
            [x, half_width, z_top],
            [x, -half_width, z_top],
        ]
    )


def build_body(params: AhmedParams) -> trimesh.Trimesh:
    sections = _sections(params)
    rings = [_ring(hw, zb, zt, x) for x, hw, zb, zt in sections]

    vertices = np.vstack(rings)
    faces: list[list[int]] = []

    for i in range(len(rings) - 1):
        a = i * 4
        b = (i + 1) * 4
        for j in range(4):
            k = (j + 1) % 4
            faces.append([a + j, b + j, b + k])
            faces.append([a + j, b + k, a + k])

    # Caps. Front ring wound to face -x, rear ring to face +x.
    faces.append([0, 2, 1])
    faces.append([0, 3, 2])
    last = (len(rings) - 1) * 4
    faces.append([last, last + 1, last + 2])
    faces.append([last, last + 2, last + 3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def build_stilts(params: AhmedParams) -> trimesh.Trimesh:
    """Four cylinders spanning the ground clearance gap."""
    radius = params.stilt_diameter / 2.0
    height = params.ground_clearance
    x_front = 0.25 * params.length
    x_rear = 0.75 * params.length
    y_off = params.width / 4.0

    parts: list[trimesh.Trimesh] = []
    for x in (x_front, x_rear):
        for y in (-y_off, y_off):
            cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
            cyl.apply_translation([x, y, height / 2.0])
            parts.append(cyl)
    return trimesh.util.concatenate(parts)


def frontal_area(params: AhmedParams) -> float:
    """Conventional Ahmed reference area: width x height, stilts excluded."""
    return params.width * params.height


def write_ahmed_stl(params: AhmedParams, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {"body": out_dir / "body.stl"}
    build_body(params).export(paths["body"])

    if params.include_stilts:
        paths["stilts"] = out_dir / "stilts.stl"
        build_stilts(params).export(paths["stilts"])

    return paths
