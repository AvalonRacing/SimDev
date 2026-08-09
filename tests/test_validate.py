from __future__ import annotations

import copy

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.config.validate import ValidationError, estimate_y_plus, validate

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
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
    case = deep_merge(BASE, overrides or {})
    return resolve(case, profile="dev", **kw)


def test_valid_case_passes() -> None:
    assert validate(_spec()) == []


def test_half_model_without_symmetry_patch_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("symmetry" in e for e in exc.value.errors)


def test_full_model_with_symmetry_patch_is_rejected() -> None:
    # Yaw breaks symmetry, so a symmetry patch is now an error.
    with pytest.raises(ValidationError) as exc:
        validate(_spec({"physics": {"yaw_deg": 5.0}}))
    assert any("symmetry" in e for e in exc.value.errors)


def test_cornering_half_model_is_unconstructable() -> None:
    case = deep_merge(BASE, {"physics": {"mode": "cornering", "corner_radius": 8.0}})
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    # Symmetry patch present on a cornering case, and box domain unsupported.
    assert exc.value.errors


def test_cornering_without_corner_radius_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["physics"] = {"mode": "cornering"}
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("corner_radius" in e for e in exc.value.errors)


def test_cornering_requires_annulus_domain() -> None:
    case = copy.deepcopy(BASE)
    case["physics"] = {"mode": "cornering", "corner_radius": 8.0}
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("annulus" in e for e in exc.value.errors)


def test_missing_inlet_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "inlet"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("inlet" in e for e in exc.value.errors)


def test_no_body_patch_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "body"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("force" in e or "body" in e for e in exc.value.errors)


def test_wall_treatment_mismatch_is_rejected() -> None:
    # low_y_plus band with a first layer sized for wall functions.
    spec = _spec(
        {"mesh": {"first_layer_thickness": 3.0e-3}}, wall_treatment="low_y_plus"
    )
    with pytest.raises(ValidationError) as exc:
        validate(spec)
    assert any("y+" in e for e in exc.value.errors)


def test_rank_count_above_physical_cores_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        validate(_spec({"solve": {"n_ranks": 80}}))
    assert any("rank" in e.lower() or "core" in e.lower() for e in exc.value.errors)


def test_refinement_range_inverted_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        validate(
            _spec({"mesh": {"surface_refinement_min": 6, "surface_refinement_max": 3}})
        )
    assert any("refinement" in e for e in exc.value.errors)


def test_static_ground_on_a_vehicle_case_warns() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["kind"] = "stl"
    case["geometry"]["stl_dir"] = "geom"
    case["geometry"]["ahmed"] = None
    warnings = validate(resolve(case, profile="dev"))
    assert any("static ground" in w for w in warnings)


def test_estimate_y_plus_is_in_the_expected_band_for_high_y_plus() -> None:
    # Ahmed at 40 m/s with a 0.3 mm first layer should land in wall-function range.
    y = estimate_y_plus(_spec())
    assert 30.0 <= y <= 300.0
