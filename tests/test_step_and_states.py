"""STEP import, the placement transform, and driving states.

The CAD arrives as STEP in one folder per driving state. Three things have to
hold for that to be usable, and each has a way of failing quietly:

- tessellation has to produce closed surfaces, or snappy leaks;
- the placement transform has to put the car in the pipeline frame the right
  way round, without mirroring it;
- switching driving state has to change everything that moves together, in
  one place, and leave a spec in which every value is still explicit.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import trimesh
import yaml

from simdev.config.resolve import UnknownDrivingStateError, resolve
from simdev.config.schema import FlowDirection, Mode
from simdev.geometry.stl import load_surface, place_surface
from simdev.stages.prepare import prepare

gmsh = pytest.importorskip("gmsh", reason="STEP import needs gmsh")

from simdev.geometry.step import (  # noqa: E402
    StepConversionError,
    Tessellation,
    cache_key,
    convert,
)

TESS = Tessellation(max_edge=0.004, min_edge=0.0004, curvature_segments=16)


# --- fixtures -------------------------------------------------------------


def write_step_cylinder(
    path: Path, radius: float, height: float, centre=(0.0, 0.0, 0.0)
) -> Path:
    """A STEP solid in millimetres, as the real export is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.clear()
        gmsh.model.occ.addCylinder(
            centre[0], centre[1], centre[2] - height / 2, 0, 0, height, radius
        )
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def write_step_box(path: Path, extents, centre=(0.0, 0.0, 0.0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.clear()
        gmsh.model.occ.addBox(
            centre[0] - extents[0] / 2,
            centre[1] - extents[1] / 2,
            centre[2] - extents[2] / 2,
            *extents,
        )
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


# --- tessellation ---------------------------------------------------------


def test_a_step_solid_tessellates_watertight(tmp_path: Path) -> None:
    """Not watertight means snappyHexMesh leaks into the interior."""
    source = write_step_cylinder(tmp_path / "part.step", 27.0, 22.0)
    out, built = convert(source, tmp_path / "cache", TESS, 0.001)

    mesh = load_surface(out)

    assert built is True
    assert mesh.is_watertight


def test_tessellation_preserves_dimensions(tmp_path: Path) -> None:
    source = write_step_cylinder(tmp_path / "part.step", 27.0, 22.0)
    out, _ = convert(source, tmp_path / "cache", TESS, 0.001)

    extents = load_surface(out).extents

    # Still in the CAD's own millimetres; scale is applied when it is placed.
    assert extents[2] == pytest.approx(22.0, rel=1e-6)
    assert extents[0] == pytest.approx(54.0, rel=2e-2)


def test_curvature_setting_controls_how_round_a_cylinder_is(tmp_path: Path) -> None:
    """The knob that decides whether a 6 mm link is a hexagon or a cylinder."""
    source = write_step_cylinder(tmp_path / "part.step", 27.0, 22.0)

    coarse, _ = convert(
        source, tmp_path / "c1", Tessellation(0.05, 0.001, 8), 0.001
    )
    fine, _ = convert(source, tmp_path / "c2", Tessellation(0.05, 0.001, 64), 0.001)

    assert len(load_surface(fine).faces) > len(load_surface(coarse).faces)


def test_a_step_file_with_no_solids_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.clear()
        gmsh.model.occ.addRectangle(0, 0, 0, 10, 10)
        gmsh.model.occ.synchronize()
        gmsh.write(str(empty))
    finally:
        gmsh.finalize()

    with pytest.raises(StepConversionError, match="no solids"):
        convert(empty, tmp_path / "cache", TESS, 0.001)


# --- caching --------------------------------------------------------------


def test_conversion_is_cached(tmp_path: Path) -> None:
    source = write_step_cylinder(tmp_path / "part.step", 27.0, 22.0)
    cache = tmp_path / "cache"

    first, built_first = convert(source, cache, TESS, 0.001)
    second, built_second = convert(source, cache, TESS, 0.001)

    assert built_first is True
    assert built_second is False
    assert first == second


def test_tessellation_settings_change_the_cache_key(tmp_path: Path) -> None:
    source = write_step_cylinder(tmp_path / "part.step", 27.0, 22.0)
    assert cache_key(source, TESS, 0.001) != cache_key(
        source, Tessellation(0.002, 0.0004, 16), 0.001
    )


def test_placement_does_not_change_the_cache_key(tmp_path: Path) -> None:
    """Rotation and translation are applied to the triangles afterwards.

    Re-tessellating a 5 MB body because the car moved 2 mm would make every
    ride-height change cost a minute for nothing.
    """
    source = write_step_cylinder(tmp_path / "part.step", 27.0, 22.0)
    assert cache_key(source, TESS, 0.001) == cache_key(source, TESS, 0.001)


def test_editing_the_cad_invalidates_the_cache(tmp_path: Path) -> None:
    source = tmp_path / "part.step"
    write_step_cylinder(source, 27.0, 22.0)
    before = cache_key(source, TESS, 0.001)

    write_step_cylinder(source, 33.0, 22.0)

    assert cache_key(source, TESS, 0.001) != before


# --- the import applies units only -----------------------------------------


def test_import_converts_units_and_nothing_else() -> None:
    """The CAD is the truth. Every transform is a way to differ from it."""
    mesh = trimesh.creation.box(extents=(400.0, 200.0, 100.0))
    mesh.apply_translation([1000.0, 300.0, -50.0])
    before = mesh.bounds.mean(axis=0) / 1000.0

    placed = place_surface(mesh.copy(), scale=0.001)

    assert placed.bounds.mean(axis=0) == pytest.approx(before, abs=1e-12)
    assert placed.extents == pytest.approx([0.4, 0.2, 0.1], rel=1e-12)


def test_place_surface_takes_no_transform_arguments() -> None:
    """A regression guard: rotation and translation were removed on purpose."""
    import inspect

    parameters = set(inspect.signature(place_surface).parameters)
    assert parameters == {"mesh", "scale"}


# --- the road plane -------------------------------------------------------


def _car_case(source_dir: Path, **overrides) -> dict:
    case = {
        "name": "car",
        "flow": {"u_inf": 15.0, "turbulence_length_scale": 0.02},
        "ground": {"motion": "moving"},
        "physics": {"wall_treatment": "high_y_plus", "mode": "straight"},
        "domain": {"kind": "box", "max_blockage": 0.2},
        "mesh": {
            "base_cell_size": 0.024,
            "surface_refinement_min": 2,
            "surface_refinement_max": 3,
            "n_layers": 4,
            "first_layer_thickness": 5.0e-4,
        },
        "forces": {"a_ref_full": 0.0217, "l_ref": 0.40},
        "solve": {"max_iterations": 200, "n_ranks": 4},
        "post": {"yplus_min": 30.0, "yplus_max": 300.0},
        "geometry": {
            "kind": "step",
            "source_dir": str(source_dir),
            "scale": 0.001,
            "symmetric": False,
            "tessellation": {
                "max_edge": 0.006,
                "min_edge": 0.001,
                "curvature_segments": 12,
            },
            "patches": [
                {"name": "Body", "role": "body"},
                {"name": "Tire_FL", "role": "tyre", "wheel": "FL"},
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
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return case


def _write_wheeled_car(cad: Path, body_centre_z: float, tyre_centre_z: float) -> None:
    write_step_box(cad / "Body.step", (400.0, 180.0, 100.0), centre=(0.0, 0.0, body_centre_z))
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.clear()
        # Tyre: axis along y, radius 33.
        gmsh.model.occ.addCylinder(130.0, -13.5, tyre_centre_z, 0, 27.0, 0, 33.0)
        gmsh.model.occ.synchronize()
        gmsh.write(str(cad / "Tire_FL.step"))
    finally:
        gmsh.finalize()


@pytest.fixture
def sunken_car(tmp_path: Path):
    """A car whose tyre dips 1.4 mm below z = 0, as the real export does."""
    cad = tmp_path / "cad"
    _write_wheeled_car(cad, body_centre_z=80.0, tyre_centre_z=31.6)

    def run(name: str = "run", **overrides):
        case_path = tmp_path / f"{name}.yaml"
        case_path.write_text(
            yaml.safe_dump(_car_case(cad, **overrides)), encoding="utf-8"
        )
        return prepare(case_path, tmp_path / name, profile="car_dev")

    return run


def test_tyres_may_cross_the_road_plane(sunken_car) -> None:
    """The part below z = 0 is the contact patch, not an error."""
    result = sunken_car("contact")

    tyre = load_surface(result.run_dir / "constant" / "triSurface" / "Tire_FL.stl")

    assert float(tyre.bounds[0][2]) < 0.0
    assert any("contact patch depth" in w for w in result.warnings)


def test_the_car_is_left_exactly_where_the_cad_put_it(sunken_car) -> None:
    """No rotation, no translation, no ground snapping - on any part.

    Run with the contact patch off, so this measures placement alone. The one
    thing the pipeline is allowed to change about the geometry is the tyre at
    the road, and it gets its own tests below.
    """
    result = sunken_car("asis", **{"geometry.contact_patch.enabled": False})

    surfaces = result.run_dir / "constant" / "triSurface"
    body = load_surface(surfaces / "Body.stl")
    tyre = load_surface(surfaces / "Tire_FL.stl")

    # Authored at 30..130 mm and -1.4..64.6 mm; nothing has moved.
    assert float(body.bounds[0][2]) == pytest.approx(0.030, abs=1e-6)
    assert float(tyre.bounds[0][2]) == pytest.approx(-0.0014, abs=1e-6)


def test_the_contact_patch_squares_the_tyre_off_at_the_road(sunken_car) -> None:
    """The tyre is cut just above z = 0 and extruded down through it."""
    result = sunken_car("squared")

    surfaces = result.run_dir / "constant" / "triSurface"
    tyre = load_surface(surfaces / "Tire_FL.stl")
    settings = result.spec.geometry.contact_patch

    assert float(tyre.bounds[0][2]) == pytest.approx(
        -settings.depth_below_road, abs=1e-6
    )
    # Authored at 64.6 mm; the top of the tyre is untouched.
    assert float(tyre.bounds[1][2]) == pytest.approx(0.0646, abs=1e-4)

    recorded = result.contact_patches["Tire_FL"]
    assert recorded.area > 0.0
    assert recorded.depth == pytest.approx(0.0014 + settings.cut_height, abs=1e-6)


def test_only_the_tyres_are_squared_off(sunken_car) -> None:
    """A splitter on the tarmac is a condition to simulate, not a defect.

    Squaring bodywork off at the road would change the shape being tested;
    snappyHexMesh clips it instead, exactly as before.
    """
    result = sunken_car("bodyonly")

    body = load_surface(result.run_dir / "constant" / "triSurface" / "Body.stl")

    assert float(body.bounds[0][2]) == pytest.approx(0.030, abs=1e-6)
    assert set(result.contact_patches) == {"Tire_FL"}


def test_the_rolling_radius_is_measured_before_the_tyre_is_cut(sunken_car) -> None:
    """The extruded corners sit further from the axis than the tread does.

    Measured off the cut surface the rolling radius reads millimetres high and
    drives the wheel too fast, so the measurement has to happen first. The
    tyre is authored 33 mm in radius, centred 31.6 mm off the road.
    """
    result = sunken_car("radius")

    assert result.wheels["FL"].radius == pytest.approx(0.033, rel=1e-2)


def test_a_non_tyre_part_through_the_road_is_reported(tmp_path: Path) -> None:
    """Reported, not rejected: ride height belongs to the CAD."""
    cad = tmp_path / "cad"
    # Body centred at z = 40 mm spans -10..90: it is through the road.
    _write_wheeled_car(cad, body_centre_z=40.0, tyre_centre_z=31.6)

    case_path = tmp_path / "sunk.yaml"
    case_path.write_text(yaml.safe_dump(_car_case(cad)), encoding="utf-8")

    result = prepare(case_path, tmp_path / "sunk", profile="car_dev")

    assert any("crosses the road plane" in w for w in result.warnings)


def test_a_floating_car_is_reported(tmp_path: Path) -> None:
    cad = tmp_path / "cad"
    _write_wheeled_car(cad, body_centre_z=90.0, tyre_centre_z=40.0)

    case_path = tmp_path / "float.yaml"
    case_path.write_text(yaml.safe_dump(_car_case(cad)), encoding="utf-8")
    result = prepare(case_path, tmp_path / "float", profile="car_dev")

    assert any("no tyre reaches the road" in w for w in result.warnings)


# --- driving states -------------------------------------------------------


BASE_CASE = {
    "name": "car",
    "flow": {"u_inf": 10.0, "turbulence_length_scale": 0.02},
    "ground": {"motion": "moving"},
    "physics": {"wall_treatment": "high_y_plus"},
    "forces": {"a_ref_full": 0.02, "l_ref": 0.44},
    "geometry": {
        "kind": "step",
        "source_dir": "CAD/Base",
        "patches": [{"name": "Body", "role": "body"}],
    },
}


def _with_states(selected: str | None, **extra) -> dict:
    case = {
        **BASE_CASE,
        "driving_states": {
            "testcase": {
                "geometry": {"source_dir": "CAD/Testcase"},
                "flow": {"u_inf": 15.0},
                "physics": {"mode": "cornering", "corner_radius": 4.0},
                "domain": {"kind": "annulus"},
            },
            "braking": {
                "geometry": {"source_dir": "CAD/Braking"},
                "flow": {"u_inf": 20.0},
            },
        },
        **extra,
    }
    if selected is not None:
        case["driving_state"] = selected
    return case


def test_selecting_a_state_applies_everything_it_changes() -> None:
    spec = resolve(_with_states("testcase"), profile="car_dev")

    assert spec.flow.u_inf == 15.0
    assert spec.physics.mode is Mode.CORNERING
    assert spec.physics.corner_radius == 4.0
    assert spec.domain.kind == "annulus"
    assert spec.geometry.source_dir == "CAD/Testcase"


def test_switching_state_is_one_line() -> None:
    testcase = resolve(_with_states("testcase"), profile="car_dev")
    braking = resolve(_with_states("braking"), profile="car_dev")

    assert braking.flow.u_inf == 20.0
    assert braking.geometry.source_dir == "CAD/Braking"
    # Untouched by the braking state, so it falls back to the case file.
    assert braking.physics.mode is Mode.STRAIGHT
    assert testcase.geometry.source_dir != braking.geometry.source_dir


def test_the_selected_state_is_recorded_for_provenance() -> None:
    assert resolve(_with_states("testcase"), profile="car_dev").driving_state == "testcase"


def test_unselected_states_do_not_reach_the_spec() -> None:
    """Otherwise editing the braking state would invalidate cornering runs."""
    spec = resolve(_with_states("testcase"), profile="car_dev")
    dumped = spec.model_dump(mode="json")

    assert "driving_states" not in dumped
    assert "CAD/Braking" not in str(dumped)


def test_editing_an_unselected_state_does_not_change_the_hash() -> None:
    before = resolve(_with_states("testcase"), profile="car_dev").spec_hash()

    altered = _with_states("testcase")
    altered["driving_states"]["braking"]["flow"]["u_inf"] = 99.0
    after = resolve(altered, profile="car_dev").spec_hash()

    assert before == after


def test_editing_the_selected_state_does_change_the_hash() -> None:
    before = resolve(_with_states("testcase"), profile="car_dev").spec_hash()

    altered = _with_states("testcase")
    altered["driving_states"]["testcase"]["flow"]["u_inf"] = 17.0

    assert resolve(altered, profile="car_dev").spec_hash() != before


def test_cli_overrides_still_win_over_the_state() -> None:
    spec = resolve(
        _with_states("testcase"),
        profile="car_dev",
        overrides={"physics": {"corner_radius": 8.0}},
    )
    assert spec.physics.corner_radius == 8.0


def test_an_unknown_state_names_the_ones_that_exist() -> None:
    with pytest.raises(UnknownDrivingStateError, match="braking, testcase"):
        resolve(_with_states("wet"), profile="car_dev")


def test_defining_states_but_selecting_none_is_an_error() -> None:
    with pytest.raises(UnknownDrivingStateError, match="selects none"):
        resolve(_with_states(None), profile="car_dev")


def test_a_case_without_states_is_unaffected() -> None:
    spec = resolve(dict(BASE_CASE), profile="car_dev")
    assert spec.driving_state is None
    assert spec.flow.u_inf == 10.0


# --- the shipped car case -------------------------------------------------


def test_the_car_case_selects_a_state_that_it_defines() -> None:
    case = yaml.safe_load(
        Path("cases/car/config.yaml").read_text(encoding="utf-8")
    )
    assert case["driving_state"] in case["driving_states"]


def test_the_car_case_resolves_to_the_requested_corner() -> None:
    case = yaml.safe_load(
        Path("cases/car/config.yaml").read_text(encoding="utf-8")
    )
    spec = resolve(case, profile="car_dev")

    assert spec.flow.u_inf == 15.0
    assert spec.physics.corner_radius == 4.0
    assert spec.physics.mode is Mode.CORNERING
    assert spec.domain.kind == "annulus"
    assert spec.flow.direction is FlowDirection.MINUS_X
    assert spec.geometry.symmetric is False
    assert spec.omega_signed == pytest.approx(15.0 / 4.0)
