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
