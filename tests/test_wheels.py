from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
import trimesh

from simdev.geometry.roles import PatchRole
from simdev.geometry.wheels import (
    MAX_AXISYMMETRY_DEFECT,
    Wheel,
    WheelGeometryError,
    axis_of_revolution,
    derive_wheels,
    measure_wheel,
)


@dataclass
class FakePatch:
    name: str
    role: PatchRole
    wheel: str | None = None


def cylinder(radius: float, height: float, transform=None, sections: int = 64):
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def annulus(outer: float, inner: float, height: float, transform=None):
    """A tyre: an outer cylinder with a concentric bore."""
    mesh = trimesh.creation.annulus(r_min=inner, r_max=outer, height=height)
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def rotation(angle_deg: float, axis) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(math.radians(angle_deg), axis)


def translation(vector) -> np.ndarray:
    return trimesh.transformations.translation_matrix(vector)


# --- the axis measurement -------------------------------------------------


def test_axis_of_a_z_aligned_cylinder_is_z() -> None:
    _, axis, defect = axis_of_revolution(cylinder(0.027, 0.022))
    assert abs(abs(axis[2]) - 1.0) < 1e-9
    assert defect < 1e-6


def test_axis_follows_an_arbitrary_rotation() -> None:
    """The axis is measured, so an unusual attitude is not a special case."""
    transform = rotation(37.0, [1.0, 0.0, 0.0]) @ rotation(21.0, [0.0, 1.0, 0.0])
    mesh = cylinder(0.027, 0.022, transform)

    _, axis, defect = axis_of_revolution(mesh)

    expected = (transform[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    assert defect < 1e-6
    assert abs(abs(float(np.dot(axis, expected))) - 1.0) < 1e-6


def test_axis_is_found_for_a_disc_and_for_a_rod() -> None:
    """One rule covers both, which is why the odd moment out is used.

    A wide flat wheel and a long thin driveshaft put the symmetry axis at
    opposite ends of the eigenvalue ordering; picking 'largest' or 'smallest'
    would get one of them backwards.
    """
    for radius, height in ((0.033, 0.005), (0.003, 0.056)):
        _, axis, defect = axis_of_revolution(cylinder(radius, height))
        assert defect < 1e-6, (radius, height)
        assert abs(abs(axis[2]) - 1.0) < 1e-6, (radius, height)


def test_exact_integration_survives_a_fan_triangulated_cap() -> None:
    """The bug this replaced: fanned end caps faked a 24% asymmetry.

    Tessellators cut a flat disc as a fan from one rim vertex, so the triangle
    centroids bunch to one side even though the disc is axisymmetric. A
    centroid-weighted moment sees that bunching; an exact integral does not.
    """
    mesh = cylinder(0.027, 0.022, sections=48)
    fanned = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

    _, axis, defect = axis_of_revolution(fanned)

    assert defect < 1e-4
    assert abs(abs(axis[2]) - 1.0) < 1e-6


def test_a_non_axisymmetric_part_is_rejected() -> None:
    """A control arm labelled as a wheel must fail loudly, not average out."""
    box = trimesh.creation.box(extents=(0.056, 0.004, 0.074))
    _, _, defect = axis_of_revolution(box)
    assert defect > MAX_AXISYMMETRY_DEFECT


# --- the full wheel measurement -------------------------------------------


def make_wheel(transform=None, tyre_radius: float = 0.033) -> Wheel:
    return measure_wheel(
        wheel="FL",
        axis_mesh=cylinder(0.027, 0.022, transform),
        axis_source="MRF_FL",
        radius_mesh=annulus(tyre_radius, 0.005, 0.027, transform),
        radius_source="Tire_FL",
    )


def test_radius_comes_from_the_tyre_not_the_sleeve() -> None:
    """Taking the radius from the MRF sleeve would drive every wheel too slow."""
    wheel = make_wheel()
    assert wheel.radius == pytest.approx(0.033, rel=1e-3)
    assert wheel.axis_source == "MRF_FL"
    assert wheel.radius_source == "Tire_FL"


def test_sleeve_radius_is_reported_and_sits_inside_the_tread() -> None:
    """The MRF cell zone must not reach out through the tyre."""
    wheel = make_wheel()
    assert wheel.axis_surface_radius == pytest.approx(0.027, rel=1e-3)
    assert wheel.axis_surface_radius < wheel.radius


def test_measurement_rejects_a_sleeve_that_is_not_a_solid_of_revolution() -> None:
    with pytest.raises(WheelGeometryError, match="not a solid of revolution"):
        measure_wheel(
            wheel="FL",
            axis_mesh=trimesh.creation.box(extents=(0.056, 0.004, 0.074)),
            axis_source="LCA_FL",
            radius_mesh=annulus(0.033, 0.005, 0.027),
            radius_source="Tire_FL",
        )


# --- rolling speed --------------------------------------------------------


def test_contact_patch_moves_with_the_road() -> None:
    """The property the whole sign convention exists to satisfy."""
    wheel = make_wheel(rotation(90.0, [1.0, 0.0, 0.0]))  # axis along y
    road = np.array([15.0, 0.0, 0.0])

    omega, slip = wheel.spin_omega(road)
    velocity = wheel.surface_velocity(wheel.contact_point()[None, :], omega)[0]

    assert slip == pytest.approx(0.0, abs=1e-9)
    assert velocity == pytest.approx(road, abs=1e-6)


def test_rolling_speed_magnitude_is_u_over_r() -> None:
    wheel = make_wheel(rotation(90.0, [1.0, 0.0, 0.0]))
    omega, _ = wheel.spin_omega(np.array([15.0, 0.0, 0.0]))
    assert abs(omega) == pytest.approx(15.0 / wheel.radius, rel=1e-6)


def test_top_of_the_wheel_moves_opposite_the_contact_patch() -> None:
    """In the car frame the wheel centre is fixed, so the top runs backwards."""
    wheel = make_wheel(rotation(90.0, [1.0, 0.0, 0.0]))
    omega, _ = wheel.spin_omega(np.array([15.0, 0.0, 0.0]))

    top = wheel.centre - wheel.contact_offset()
    velocity = wheel.surface_velocity(top[None, :], omega)[0]

    assert velocity == pytest.approx([-15.0, 0.0, 0.0], abs=1e-6)


def test_reversing_the_road_reverses_the_wheel() -> None:
    wheel = make_wheel(rotation(90.0, [1.0, 0.0, 0.0]))
    forward, _ = wheel.spin_omega(np.array([15.0, 0.0, 0.0]))
    backward, _ = wheel.spin_omega(np.array([-15.0, 0.0, 0.0]))
    assert forward == pytest.approx(-backward)


def test_a_steered_wheel_reports_its_slip() -> None:
    """Lateral road velocity cannot be rolled away, and is surfaced not hidden."""
    steer = 20.0
    wheel = make_wheel(rotation(90.0, [1.0, 0.0, 0.0]) @ rotation(steer, [0.0, 1.0, 0.0]))

    road = np.array([15.0, 0.0, 0.0])
    omega, slip = wheel.spin_omega(road)

    assert slip == pytest.approx(15.0 * math.sin(math.radians(steer)), rel=1e-3)
    assert abs(omega) == pytest.approx(
        15.0 * math.cos(math.radians(steer)) / wheel.radius, rel=1e-3
    )


def test_a_vertical_axis_has_no_contact_patch() -> None:
    wheel = make_wheel()  # axis along z
    with pytest.raises(WheelGeometryError, match="vertical"):
        wheel.contact_offset()


# --- pairing surfaces to wheels -------------------------------------------


def _corner_meshes(offset) -> dict:
    transform = translation(offset) @ rotation(90.0, [1.0, 0.0, 0.0])
    return {
        "MRF_FL": cylinder(0.027, 0.022, transform),
        "Tire_FL": annulus(0.033, 0.005, 0.027, transform),
    }


def test_derive_wheels_pairs_by_declared_id_not_by_name() -> None:
    patches = [
        FakePatch("MRF_FL", PatchRole.MRF_ZONE, "FL"),
        FakePatch("Tire_FL", PatchRole.TYRE, "FL"),
        FakePatch("Body", PatchRole.BODY),
    ]
    wheels = derive_wheels(patches, _corner_meshes([0.15, -0.09, 0.033]))

    assert set(wheels) == {"FL"}
    assert wheels["FL"].origin == pytest.approx((0.15, -0.09, 0.033), abs=1e-6)


def test_derive_wheels_falls_back_to_the_tyre_for_the_axis() -> None:
    patches = [FakePatch("Tire_FL", PatchRole.TYRE, "FL")]
    meshes = {"Tire_FL": _corner_meshes([0.0, 0.0, 0.033])["Tire_FL"]}

    wheels = derive_wheels(patches, meshes)

    assert wheels["FL"].axis_source == "Tire_FL"
    assert wheels["FL"].radius == pytest.approx(0.033, rel=1e-3)


def test_a_wheel_without_a_tyre_is_an_error() -> None:
    patches = [FakePatch("MRF_FL", PatchRole.MRF_ZONE, "FL")]
    meshes = {"MRF_FL": _corner_meshes([0.0, 0.0, 0.033])["MRF_FL"]}

    with pytest.raises(WheelGeometryError, match="no patch with role 'tyre'"):
        derive_wheels(patches, meshes)


def test_two_tyres_on_one_wheel_is_an_error() -> None:
    patches = [
        FakePatch("Tire_FL", PatchRole.TYRE, "FL"),
        FakePatch("Tire_FL_inner", PatchRole.TYRE, "FL"),
    ]
    meshes = _corner_meshes([0.0, 0.0, 0.033])
    meshes["Tire_FL_inner"] = meshes["Tire_FL"]

    with pytest.raises(WheelGeometryError, match="exactly one"):
        derive_wheels(patches, meshes)


def test_each_corner_is_measured_where_it_actually_sits() -> None:
    """The point of measuring: four corners, four different positions."""
    corners = {
        "FL": [0.15, -0.09, 0.033],
        "FR": [0.15, 0.09, 0.033],
        "RL": [-0.15, -0.09, 0.033],
        "RR": [-0.15, 0.09, 0.033],
    }
    patches = []
    meshes = {}
    for wheel, offset in corners.items():
        patches.append(FakePatch(f"Tire_{wheel}", PatchRole.TYRE, wheel))
        transform = translation(offset) @ rotation(90.0, [1.0, 0.0, 0.0])
        meshes[f"Tire_{wheel}"] = annulus(0.033, 0.005, 0.027, transform)

    wheels = derive_wheels(patches, meshes)

    for wheel, offset in corners.items():
        assert wheels[wheel].origin == pytest.approx(offset, abs=1e-6)
