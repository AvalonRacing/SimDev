"""End-to-end prepare against a multi-part vehicle in assembly position.

The real CAD arrives as one STL per part, positioned in the vehicle frame,
and everything the cornering work depends on - wheel axes, rolling radii,
MRF cell zones, ride height - is measured from those positions rather than
configured. That measurement is the thing worth testing, so the fixture here
is a crude but *correctly assembled* car: a hull, four wheels at four
different places, and a sleeve inside each wheel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import trimesh
import yaml

from simdev.stages.prepare import prepare

WHEELBASE = 0.26
TRACK = 0.19
TYRE_RADIUS = 0.033
SLEEVE_RADIUS = 0.027

CORNERS = {
    "FL": (WHEELBASE / 2, -TRACK / 2),
    "FR": (WHEELBASE / 2, TRACK / 2),
    "RL": (-WHEELBASE / 2, -TRACK / 2),
    "RR": (-WHEELBASE / 2, TRACK / 2),
}


def _place(x: float, y: float) -> object:
    """Wheel transform: spin axis along y, centre one tyre radius off the road."""
    return trimesh.transformations.translation_matrix(
        [x, y, TYRE_RADIUS]
    ) @ trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0])


def write_car(directory: Path) -> None:
    """A car sitting on the road at z = 0, in millimetres like the real export."""
    directory.mkdir(parents=True, exist_ok=True)
    parts: dict[str, trimesh.Trimesh] = {}

    body = trimesh.creation.box(extents=(0.40, 0.18, 0.10))
    body.apply_translation([0.0, 0.0, 0.02 + 0.05])
    parts["Body"] = body

    for wheel, (x, y) in CORNERS.items():
        transform = _place(x, y)
        parts[f"Tire_{wheel}"] = trimesh.creation.annulus(
            r_min=0.005, r_max=TYRE_RADIUS, height=0.027, transform=transform
        )
        parts[f"MRF_{wheel}"] = trimesh.creation.cylinder(
            radius=SLEEVE_RADIUS, height=0.022, transform=transform
        )

    for name, mesh in parts.items():
        # Authored in millimetres, as the CAD is, so geometry.scale is
        # exercised rather than bypassed.
        mesh.apply_scale(1000.0)
        (directory / f"{name}.stl").write_bytes(trimesh.exchange.stl.export_stl(mesh))


def car_case(stl_dir: Path, **overrides) -> dict:
    case = {
        "name": "car",
        "flow": {"u_inf": 12.0, "turbulence_length_scale": 0.02},
        "ground": {"motion": "moving"},
        "physics": {"wall_treatment": "high_y_plus", "mode": "straight"},
        "domain": {"kind": "box", "max_blockage": 0.05},
        "mesh": {
            "base_cell_size": 0.024,
            "surface_refinement_min": 2,
            "surface_refinement_max": 3,
            "n_layers": 4,
            "first_layer_thickness": 6.0e-4,
        },
        "forces": {"a_ref_full": 0.0217, "l_ref": 0.40},
        "solve": {"max_iterations": 200, "n_ranks": 4},
        "post": {"yplus_min": 30.0, "yplus_max": 300.0},
        "geometry": {
            "kind": "stl",
            "source_dir": str(stl_dir),
            "scale": 0.001,
            "symmetric": False,
            "patches": [
                {"name": "Body", "role": "body"},
                *[
                    {"name": f"Tire_{w}", "role": "tyre", "wheel": w}
                    for w in CORNERS
                ],
                *[
                    {"name": f"MRF_{w}", "role": "mrfZone", "wheel": w}
                    for w in CORNERS
                ],
                {"name": "ground", "role": "ground"},
                {"name": "inlet", "role": "inlet"},
                {"name": "outlet", "role": "outlet"},
                {"name": "farfield", "role": "farfield"},
            ],
        },
    }
    for path, value in overrides.items():
        target = case
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return case


@pytest.fixture
def car(tmp_path: Path):
    stl_dir = tmp_path / "cad"
    write_car(stl_dir)

    def run(run_dir_name: str = "run", profile: str = "car_dev", **overrides):
        case_path = tmp_path / f"{run_dir_name}.yaml"
        case_path.write_text(
            yaml.safe_dump(car_case(stl_dir, **overrides)), encoding="utf-8"
        )
        return prepare(case_path, tmp_path / run_dir_name, profile=profile)

    return run


# --- import ---------------------------------------------------------------


def test_geometry_is_scaled_into_metres_on_import(car) -> None:
    result = car()
    body = trimesh.load_mesh(
        result.run_dir / "constant" / "triSurface" / "Body.stl", process=True
    )
    assert body.extents[0] == pytest.approx(0.40, rel=1e-3)


def test_every_declared_surface_is_written(car) -> None:
    result = car()
    written = {p.stem for p in (result.run_dir / "constant" / "triSurface").glob("*.stl")}
    assert written == {"Body", *[f"Tire_{w}" for w in CORNERS], *[f"MRF_{w}" for w in CORNERS]}


def test_a_missing_part_is_an_error_not_a_silent_skip(car, tmp_path) -> None:
    (tmp_path / "cad" / "Tire_RR.stl").unlink()
    with pytest.raises(FileNotFoundError, match="Tire_RR"):
        car("missing")


def test_a_body_through_the_road_is_reported_but_not_rejected(car, tmp_path) -> None:
    """Attitude is the modeller's to own, so this reports rather than stops.

    At a big enough roll or dive a splitter really does touch the road, and
    that is a condition to simulate. It stays loud because it is equally a
    symptom of a mis-positioned export.
    """
    sunk = trimesh.creation.box(extents=(0.40, 0.18, 0.10))
    sunk.apply_translation([0.0, 0.0, 0.03])
    sunk.apply_scale(1000.0)
    (tmp_path / "cad" / "Body.stl").write_bytes(trimesh.exchange.stl.export_stl(sunk))

    result = car("sunk")

    assert any("crosses the road plane" in w for w in result.warnings)


# --- measurement ----------------------------------------------------------


def test_each_wheel_is_measured_where_it_sits(car) -> None:
    result = car()
    assert set(result.wheels) == set(CORNERS)
    for wheel, (x, y) in CORNERS.items():
        assert result.wheels[wheel].origin == pytest.approx((x, y, TYRE_RADIUS), abs=1e-4)


def test_rolling_radius_comes_from_the_tyre(car) -> None:
    result = car()
    for wheel in CORNERS:
        assert result.wheels[wheel].radius == pytest.approx(TYRE_RADIUS, rel=1e-2)
        assert result.wheels[wheel].axis_surface_radius == pytest.approx(
            SLEEVE_RADIUS, rel=1e-2
        )


def test_wheel_axes_are_horizontal_and_across_the_car(car) -> None:
    result = car()
    for wheel in CORNERS:
        axis = result.wheels[wheel].direction
        assert abs(axis[1]) == pytest.approx(1.0, abs=1e-6)


def test_measured_wheels_are_recorded_in_the_run_status(car) -> None:
    result = car()
    status = json.loads(
        (result.run_dir / "status" / "prepare.json").read_text(encoding="utf-8")
    )
    recorded = status["detail"]["wheels"]
    assert set(recorded) == set(CORNERS)
    for wheel in CORNERS:
        assert recorded[wheel]["surface_speed"] == pytest.approx(12.0, rel=1e-2)


# --- rendered case --------------------------------------------------------


def test_tyres_get_a_rotating_wall_and_the_body_does_not(car) -> None:
    result = car()
    field = (result.run_dir / "0" / "U").read_text(encoding="utf-8")
    assert "rotatingWallVelocity" in field
    body_block = field.split("Body")[1].split("}")[0]
    assert "noSlip" in body_block


def test_mrf_zones_become_cell_zones_and_carry_no_boundary_condition(car) -> None:
    result = car()
    snappy = (result.run_dir / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    field = (result.run_dir / "0" / "U").read_text(encoding="utf-8")

    for wheel in CORNERS:
        assert f"cellZone        MRF_{wheel};" in snappy
        assert f"\n    MRF_{wheel}\n" not in field


def test_each_wheel_gets_its_own_mrf_entry(car) -> None:
    result = car()
    properties = (result.run_dir / "constant" / "MRFProperties").read_text(
        encoding="utf-8"
    )
    for wheel in CORNERS:
        assert f"cellZone        MRF_{wheel};" in properties


def test_a_full_model_keeps_its_whole_reference_area(car) -> None:
    """The asymmetric car must never be halved. See handbook section 3.3."""
    result = car()
    assert result.spec.half_model is False
    assert result.spec.a_ref_effective == pytest.approx(0.0217)
    control = (result.run_dir / "system" / "controlDict").read_text(encoding="utf-8")
    assert "Aref            0.0217" in control


# --- cornering ------------------------------------------------------------


def test_cornering_renders_a_whole_domain_rotating_frame(car) -> None:
    result = car(
        "corner",
        **{
            "physics.mode": "cornering",
            "physics.corner_radius": 3.0,
            "physics.corner_direction": "left",
            "domain.kind": "annulus",
            "ground.motion": "static",
        },
    )
    properties = (result.run_dir / "constant" / "MRFProperties").read_text(
        encoding="utf-8"
    )
    assert "cellZone        all;" in properties
    assert "nonRotatingPatches" in properties
    assert "ground" in properties.split("nonRotatingPatches")[1].split(";")[0]


def test_cornering_wheels_are_driven_by_a_composed_rotation(car) -> None:
    result = car(
        "corner2",
        **{
            "physics.mode": "cornering",
            "physics.corner_radius": 3.0,
            "domain.kind": "annulus",
            "ground.motion": "static",
        },
    )
    field = (result.run_dir / "0" / "U").read_text(encoding="utf-8")
    assert "codedFixedValue" in field
    assert "frameOmega" in field
    assert "wheelOmega" in field


def test_inner_and_outer_wheels_turn_at_different_speeds(car) -> None:
    result = car(
        "corner3",
        **{
            "physics.mode": "cornering",
            "physics.corner_radius": 3.0,
            "physics.corner_direction": "left",
            "domain.kind": "annulus",
            "ground.motion": "static",
        },
    )
    status = json.loads(
        (result.run_dir / "status" / "prepare.json").read_text(encoding="utf-8")
    )
    speeds = {w: status["detail"]["wheels"][w]["surface_speed"] for w in CORNERS}

    # Left-hand corner: the centre is at -y, so the +y wheels run outside.
    assert speeds["FR"] > speeds["FL"]
    assert speeds["RR"] > speeds["RL"]
    assert speeds["FR"] / speeds["FL"] == pytest.approx(
        (3.0 + TRACK / 2) / (3.0 - TRACK / 2), rel=1e-2
    )


# --- attitude is the modeller's, not the pipeline's -----------------------
#
# An RC car spends most of its cornering life in heavy understeer, so large
# steer and body slip angles are the normal operating point. Nothing about
# them may stop a run.


def _steered_car(directory: Path, steer_deg: float) -> None:
    """The same car with every wheel turned hard out of square."""
    write_car(directory)
    for wheel, (x, y) in CORNERS.items():
        transform = (
            trimesh.transformations.translation_matrix([x, y, TYRE_RADIUS])
            @ trimesh.transformations.rotation_matrix(
                math.radians(steer_deg), [0.0, 0.0, 1.0]
            )
            @ trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0])
        )
        for name, mesh in (
            (f"Tire_{wheel}", trimesh.creation.annulus(
                r_min=0.005, r_max=TYRE_RADIUS, height=0.027, transform=transform)),
            (f"MRF_{wheel}", trimesh.creation.cylinder(
                radius=SLEEVE_RADIUS, height=0.022, transform=transform)),
        ):
            mesh.apply_scale(1000.0)
            (directory / f"{name}.stl").write_bytes(
                trimesh.exchange.stl.export_stl(mesh)
            )


@pytest.mark.parametrize("steer_deg", [10.0, 25.0, 45.0])
def test_a_heavily_yawed_car_prepares_without_failing(tmp_path, steer_deg) -> None:
    cad = tmp_path / "cad"
    _steered_car(cad, steer_deg)

    case_path = tmp_path / "yawed.yaml"
    case_path.write_text(yaml.safe_dump(car_case(cad)), encoding="utf-8")

    result = prepare(case_path, tmp_path / "yawed", profile="car_dev")

    assert set(result.wheels) == set(CORNERS)
    for wheel in CORNERS:
        assert result.wheels[wheel].radius == pytest.approx(TYRE_RADIUS, rel=1e-2)


def test_slip_is_reported_as_an_angle_not_as_a_fault(tmp_path) -> None:
    cad = tmp_path / "cad"
    _steered_car(cad, 30.0)

    case_path = tmp_path / "slip.yaml"
    case_path.write_text(yaml.safe_dump(car_case(cad)), encoding="utf-8")
    result = prepare(case_path, tmp_path / "slip", profile="car_dev")

    reported = [w for w in result.warnings if "slip angle" in w]
    assert len(reported) == 1
    # Measured, not judged: no language suggesting the pose is wrong.
    assert "mistake" not in reported[0]

    status = json.loads(
        (result.run_dir / "status" / "prepare.json").read_text(encoding="utf-8")
    )
    for wheel in CORNERS:
        assert status["detail"]["wheels"][wheel]["slip_deg"] == pytest.approx(
            30.0, abs=0.5
        )


def test_a_wheel_turned_hard_still_rolls_at_the_projected_speed(tmp_path) -> None:
    """Only the component along the wheel's own rolling direction turns it."""
    cad = tmp_path / "cad"
    _steered_car(cad, 30.0)

    case_path = tmp_path / "roll.yaml"
    case_path.write_text(yaml.safe_dump(car_case(cad)), encoding="utf-8")
    result = prepare(case_path, tmp_path / "roll", profile="car_dev")

    status = json.loads(
        (result.run_dir / "status" / "prepare.json").read_text(encoding="utf-8")
    )
    for wheel in CORNERS:
        recorded = status["detail"]["wheels"][wheel]
        assert recorded["surface_speed"] == pytest.approx(
            12.0 * math.cos(math.radians(30.0)), rel=1e-2
        )
