from __future__ import annotations

from pathlib import Path

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.domain.box import BoxDomainBuilder
from simdev.render.context import WALL_FUNCTIONS, build_bcs, inlet_turbulence
from simdev.render.render import render_case

BOUNDS = ((0.0, -0.1945, 0.05), (1.044, 0.1945, 0.338))

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044, "c_of_r": [0.5, 0.0, 0.0]},
    "geometry": {
        "kind": "ahmed",
        "symmetric": True,
        "ahmed": {},
        "patches": [
            {"name": "body", "role": "body"},
            {"name": "ground", "role": "ground"},
            {"name": "symmetry", "role": "symmetry"},
            {"name": "inlet", "role": "inlet"},
            {"name": "outlet", "role": "outlet"},
            {"name": "farfield", "role": "farfield"},
        ],
    },
}


def _spec(overrides: dict | None = None, **kw: object):
    return resolve(deep_merge(BASE, overrides or {}), profile="dev", **kw)


def _render(tmp_path: Path, spec=None) -> Path:
    spec = spec or _spec()
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    render_case(spec, domain, {"body": Path("body.stl")}, tmp_path)
    return tmp_path


def _bcs(spec):
    k, omega = inlet_turbulence(spec)
    return build_bcs(spec, k, omega, k / omega, WALL_FUNCTIONS[spec.physics.wall_treatment])


def test_static_ground_is_no_slip() -> None:
    bcs = _bcs(_spec())
    ground = next(b for b in bcs["U"] if b.patch == "ground")
    assert ground.entries["type"] == "noSlip"


def test_moving_ground_translates_at_freestream() -> None:
    bcs = _bcs(_spec({"ground": {"motion": "moving"}}))
    ground = next(b for b in bcs["U"] if b.patch == "ground")
    assert ground.entries["type"] == "fixedValue"
    assert "40.0" in ground.entries["value"]


def test_body_is_always_no_slip() -> None:
    body = next(b for b in _bcs(_spec())["U"] if b.patch == "body")
    assert body.entries["type"] == "noSlip"


def test_outlet_uses_inlet_outlet_for_backflow() -> None:
    outlet = next(b for b in _bcs(_spec())["U"] if b.patch == "outlet")
    assert outlet.entries["type"] == "inletOutlet"


def test_farfield_is_slip_not_no_slip() -> None:
    farfield = next(b for b in _bcs(_spec())["U"] if b.patch == "farfield")
    assert farfield.entries["type"] == "slip"


def test_wall_functions_follow_the_treatment() -> None:
    low = _bcs(_spec(wall_treatment="low_y_plus"))
    body_nut = next(b for b in low["nut"] if b.patch == "body")
    assert body_nut.entries["type"] == "nutLowReWallFunction"


def test_every_field_covers_every_patch() -> None:
    spec = _spec()
    names = {p.name for p in spec.geometry.patches}
    for field, bcs in _bcs(spec).items():
        assert {b.patch for b in bcs} == names, field


def test_force_coeffs_uses_the_effective_reference_area(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    # Half model: 0.112 / 2
    assert "Aref            0.056;" in text


def test_force_coeffs_mag_u_inf_matches_the_inlet(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    assert "magUInf         40.0;" in text


def test_force_coeffs_lists_only_force_patches(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    # Read the patches entry itself. Splitting on "forceCoeffs" would not work:
    # the block name and its own "type forceCoeffs;" both match.
    patches = next(l for l in text.splitlines() if l.strip().startswith("patches"))
    assert "body" in patches
    assert "ground" not in patches
    assert "symmetry" not in patches


def test_each_force_patch_gets_its_own_coefficient_monitor(tmp_path: Path) -> None:
    """The aggregate says the car oscillates; only per-patch says which part.

    Same Aref and CofR as the aggregate on every one of them, so the per-patch
    coefficients sum to the total and a share can be read straight off.
    """
    text = (_render(tmp_path) / "system" / "controlDict").read_text()

    assert "forceCoeffs_body" in text
    # Non-force walls must not acquire one: the ground carries a wall function
    # but no reported force, and a Cd for it would be meaningless.
    assert "forceCoeffs_ground" not in text
    assert "forceCoeffs_symmetry" not in text

    block = text[text.index("forceCoeffs_body") :]
    block = block[: block.index("\n    }")]
    assert "patches         (body);" in block
    assert "Aref            0.056;" in block


def test_control_dict_end_time_is_the_iteration_cap(tmp_path: Path) -> None:
    spec = _spec()
    text = (_render(tmp_path, spec) / "system" / "controlDict").read_text()
    assert f"endTime         {spec.solve.max_iterations};" in text


def test_div_scheme_is_second_order_stabilised(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "fvSchemes").read_text()
    assert "div(phi,U)      bounded Gauss linearUpwind grad(U);" in text


def test_steady_case_has_residual_control(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "fvSolution").read_text()
    assert "residualControl" in text


def test_pressure_uses_gamg(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "fvSolution").read_text()
    assert "GAMG" in text


def test_transport_properties_carries_nu(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "constant" / "transportProperties").read_text()
    assert "1.5e-05" in text


def test_case_spec_json_is_written(tmp_path: Path) -> None:
    assert (_render(tmp_path) / "caseSpec.json").exists()


def test_field_files_exist_for_every_field(tmp_path: Path) -> None:
    out = _render(tmp_path)
    for field in ("U", "p", "k", "omega", "nut"):
        assert (out / "0" / field).exists()


def test_fields_include_constraint_types_for_parallel_runs(tmp_path: Path) -> None:
    """procBoundary* patches exist only in the decomposed mesh.

    No case-level field file can enumerate them, so without the constraint
    types include every parallel simpleFoam aborts with "Cannot find
    patchField entry for procBoundary0to1".
    """
    case = _render(tmp_path)
    for field in ("U", "p", "k", "omega", "nut"):
        text = (case / "0" / field).read_text()
        assert '#includeEtc "caseDicts/setConstraintTypes"' in text, field


# --- the ground's own wall treatment ---------------------------------------


def test_ground_uses_spalding_while_the_body_keeps_the_case_treatment() -> None:
    """The ground is the one wall whose y+ nobody gets to design.

    It is a blockMesh patch, so snappy never surface-refines it, and its layer
    coverage is bimodal - layered under the car where the shells reach the
    floor, bare beyond it. nutLowReWallFunction sets nu_t = 0 at the wall,
    which is right at y+ ~1 and wrong at the y+ 7-12 the far field actually
    runs at. Spalding is valid across the whole range and costs nothing.
    """
    bcs = _bcs(_spec(wall_treatment="low_y_plus"))
    ground = next(b for b in bcs["nut"] if b.patch == "ground")
    body = next(b for b in bcs["nut"] if b.patch == "body")
    assert ground.entries["type"] == "nutUSpaldingWallFunction"
    assert body.entries["type"] == "nutLowReWallFunction"


def test_ground_keeps_the_blended_k_and_omega_wall_functions() -> None:
    """Only nut is overridden. kLowReWallFunction and omegaWallFunction both
    blend across the sublayer already, so they pair with Spalding correctly -
    swapping them too would be a change with no argument behind it."""
    bcs = _bcs(_spec(wall_treatment="low_y_plus"))
    assert next(b for b in bcs["k"] if b.patch == "ground").entries["type"] == (
        "kLowReWallFunction"
    )
    assert next(b for b in bcs["omega"] if b.patch == "ground").entries["type"] == (
        "omegaWallFunction"
    )


def test_ground_uses_spalding_under_every_wall_treatment() -> None:
    """The override is about the patch, not the case. Whatever the vehicle
    runs, the floor's y+ is an outcome rather than a choice, so it gets the
    treatment that is valid at any y+."""
    for treatment in ("low_y_plus", "high_y_plus", "spalding"):
        bcs = _bcs(_spec(wall_treatment=treatment))
        ground = next(b for b in bcs["nut"] if b.patch == "ground")
        assert ground.entries["type"] == "nutUSpaldingWallFunction", treatment


# --- area-weighted y+ ------------------------------------------------------


def test_controldict_area_averages_y_plus_on_every_wall(tmp_path: Path) -> None:
    """The yPlus function object reports an unweighted face mean.

    Small faces are fine cells are low y+, so on any patch whose cell size
    varies the plain mean is biased low - measured at 7.40 against an
    area-weighted 12.40 on this pipeline's own ground patch. The gate has to
    read the weighted number, and only a surfaceFieldValue produces it.
    """
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    for patch in ("body", "ground"):
        assert f"yPlusArea_{patch}" in text, patch
    assert "operation       areaAverage;" in text
    assert "fields          (yPlus);" in text


def test_area_average_is_declared_after_the_field_it_reads(tmp_path: Path) -> None:
    """Function objects execute in dictionary order and surfaceFieldValue
    reads yPlus out of the registry, so the yPlus object has to have run
    first. Declared the other way round it silently finds nothing."""
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    assert text.index("\n    yPlus\n") < text.index("yPlusArea_")


def test_area_average_is_not_written_for_non_wall_patches(tmp_path: Path) -> None:
    """y+ on an inlet is meaningless and the patch carries no wall function."""
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    for patch in ("inlet", "outlet", "farfield", "symmetry"):
        assert f"yPlusArea_{patch}" not in text, patch


# --- time-averaged fields ---------------------------------------------------


def test_controldict_time_averages_p_and_u(tmp_path: Path) -> None:
    """A steady case with a limit cycle has no fixed point, so a field written
    at one iteration is one arbitrary phase of the oscillation. Comparing two
    designs off single timesteps compares two arbitrary phases of two cycles
    and manufactures flow-structure differences that are pure sampling."""
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    assert "type            fieldAverage;" in text
    assert "mean        on;" in text
    for field in ("p", "U"):
        assert f"            {field}\n" in text, field


def test_field_averaging_starts_where_the_force_window_starts(tmp_path: Path) -> None:
    """<field>Mean and the reported Cd/Cl must describe the same iterations,
    or the slice and the coefficient are answers to different questions.
    dev profile: 300 iterations, 50-iteration window -> start at 250."""
    spec = _spec({"solve": {"max_iterations": 300, "plateau_window": 50}})
    text = (_render(tmp_path, spec) / "system" / "controlDict").read_text()
    assert "timeStart       250;" in text


def test_field_averaging_start_never_goes_negative(tmp_path: Path) -> None:
    """A smoke run whose window is longer than the run itself would otherwise
    ask OpenFOAM to start averaging before iteration 0."""
    spec = _spec({"solve": {"max_iterations": 20, "plateau_window": 50}})
    text = (_render(tmp_path, spec) / "system" / "controlDict").read_text()
    assert "timeStart       0;" in text


def test_no_field_average_object_when_the_list_is_empty(tmp_path: Path) -> None:
    """Empty disables it outright rather than rendering an object averaging
    nothing, which OpenFOAM rejects."""
    spec = _spec({"solve": {"average_fields": []}})
    text = (_render(tmp_path, spec) / "system" / "controlDict").read_text()
    assert "fieldAverage" not in text


def test_controldict_writes_forces_in_newtons(tmp_path: Path) -> None:
    """forceCoeffs reports coefficients only.

    F in newtons and M in newton-metres need their own object. Recovering
    them from Cd/Cs/Cl would mean assuming how OpenFOAM maps
    CmRoll/CmPitch/CmYaw onto Cartesian axes, and this pipeline does not
    assume conventions it can read directly.
    """
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    block = text[text.index("\n    forces\n") :]
    block = block[: block.index("\n    }")]

    assert "type            forces;" in block
    assert 'libs            ("libforces.so");' in block
    assert "rho             rhoInf;" in block
    assert "CofR            (0.5 0.0 0.0);" in block
    assert "body" in block


def test_the_forces_object_is_aggregate_only(tmp_path: Path) -> None:
    """One forces object, not one per patch.

    The per-patch split the report needs is a split of *coefficients*, which
    forceCoeffs_<patch> already supplies. Eleven more force objects would add
    eleven directories under postProcessing/ and no information.
    """
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    assert "forces_body" not in text
    assert text.count("type            forces;") == 1
