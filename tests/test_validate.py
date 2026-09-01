from __future__ import annotations

import copy

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.config.validate import ValidationError, estimate_y_plus, validate

BASE: dict = {
    "name": "ahmed",
    # 0.6 mm at 40 m/s puts nu_t/nu at 10.7, inside the band validate()
    # judges, so cases below warn about their own subject and nothing else.
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.0006},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
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
    case = deep_merge(BASE, overrides or {})
    return resolve(case, profile="dev", **kw)


def test_valid_case_passes() -> None:
    """The dev profile's 15 mm surface cell holds 5 of the 7 requested layers.

    That clamp is reported as a warning, not an error: the case is buildable
    and correct, it just cannot carry the full high_y_plus stack at this
    resolution. Anything beyond that single warning is a regression.
    """
    warnings = validate(_spec())
    assert len(warnings) == 1
    assert "only 5 fit" in warnings[0]


def test_stack_that_cannot_fit_one_layer_is_rejected() -> None:
    """0.469 mm surface cells against a 1 mm first layer.

    This is the production profile paired with high_y_plus, and it is the
    combination that silently produced layerless meshes: snappy runs, drops
    the layers, and the failure only surfaces as a coverage number long after
    the cause has scrolled past.
    """
    with pytest.raises(ValidationError) as exc:
        validate(_spec({"mesh": {"base_cell_size": 0.03, "surface_refinement_max": 6}}))
    assert "not one layer fits" in str(exc.value)


def test_too_few_layers_warns_and_names_the_patch() -> None:
    """A thin stack is a warning, not a rejection.

    A small appendage legitimately carries fewer layers than the main body -
    refining it shrinks its layer budget - and whether the wall treatment
    survives there is something the y+ gate measures after the run rather
    than something a flat-plate correlation can settle before it.
    """
    warnings = validate(
        _spec({"mesh": {"base_cell_size": 0.06, "surface_refinement_max": 4}})
    )
    assert any("barely represents a boundary layer" in w for w in warnings)
    assert any("'body'" in w for w in warnings)


def test_per_patch_refinement_overrides_the_case_wide_level() -> None:
    spec = _spec(
        {
            "geometry": {
                "patches": [
                    {"name": "body", "role": "body"},
                    {"name": "stilts", "role": "body", "refinement_max": 5},
                    {"name": "ground", "role": "ground"},
                    {"name": "symmetry", "role": "symmetry"},
                    {"name": "inlet", "role": "inlet"},
                    {"name": "outlet", "role": "outlet"},
                    {"name": "farfield", "role": "farfield"},
                ]
            }
        }
    )
    body = next(p for p in spec.geometry.patches if p.name == "body")
    stilts = next(p for p in spec.geometry.patches if p.name == "stilts")

    assert spec.patch_refinement(body)[1] == spec.mesh.surface_refinement_max
    assert spec.patch_refinement(stilts)[1] == 5
    # Finer cells mean a smaller layer budget, so fewer layers fit.
    assert spec.surface_cell_size_for(stilts) < spec.surface_cell_size_for(body)
    assert spec.n_layers_for(stilts) < spec.n_layers_for(body)


def test_layer_count_is_never_raised_above_the_request() -> None:
    """A cell with room to spare still gets exactly what was asked for."""
    spec = _spec({"mesh": {"base_cell_size": 1.0, "surface_refinement_max": 1}})
    assert spec.n_layers_effective == spec.mesh.n_layers


def test_effective_stack_always_fits_the_budget() -> None:
    for base, level in ((0.12, 3), (0.2, 3), (0.5, 4), (1.0, 2)):
        spec = _spec(
            {"mesh": {"base_cell_size": base, "surface_refinement_max": level}}
        )
        stack = spec.layer_stack_thickness(spec.n_layers_effective)
        assert stack <= spec.layer_budget, (base, level, stack, spec.layer_budget)


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
    case["geometry"]["source_dir"] = "geom"
    case["geometry"]["ahmed"] = None
    warnings = validate(resolve(case, profile="dev"))
    assert any("static ground" in w for w in warnings)


def test_estimate_y_plus_is_in_the_expected_band_for_high_y_plus() -> None:
    # Ahmed at 40 m/s with a 0.3 mm first layer should land in wall-function range.
    y = estimate_y_plus(_spec())
    assert 30.0 <= y <= 300.0


# --- freestream turbulence -------------------------------------------------


def test_soupy_freestream_warns() -> None:
    """nu_t/nu = 179 at the inlet, from 40 m/s on a 10 mm length scale.

    Neither input looks wrong on its own, which is the point: only their
    product says the body is sitting in air 179 times more diffusive than it
    should be. Nothing in the case file shows that number, so validate() has
    to be the thing that says it.
    """
    warnings = validate(_spec({"flow": {"turbulence_length_scale": 0.01}}))
    soup = [w for w in warnings if "nu_t/nu" in w]
    assert len(soup) == 1
    assert "179" in soup[0]
    assert "turbulence_length_scale" in soup[0]


def test_clean_freestream_does_not_warn() -> None:
    """1.5 mm at 15 m/s puts nu_t/nu at 10.1, inside the band."""
    warnings = validate(
        _spec({"flow": {"u_inf": 15.0, "turbulence_length_scale": 0.0015}})
    )
    assert [w for w in warnings if "nu_t/nu" in w] == []


def test_a_freestream_too_quiet_to_sustain_turbulence_warns() -> None:
    """The other end of the band. A vanishing freestream nu_t lets the model
    laminarise in shear layers it should be resolving as turbulent."""
    warnings = validate(
        _spec({"flow": {"u_inf": 40.0, "turbulence_length_scale": 1.0e-6}})
    )
    assert len([w for w in warnings if "nu_t/nu" in w]) == 1


# --- force groups ------------------------------------------------------


def test_a_patch_in_no_group_is_an_error() -> None:
    spec = _spec({"post": {"groups": {"body": ["body"]}},
                  "geometry": {"patches": [
                      {"name": "body", "role": "body"},
                      {"name": "wing", "role": "body"},
                      {"name": "ground", "role": "ground"},
                      {"name": "inlet", "role": "inlet"},
                      {"name": "outlet", "role": "outlet"},
                      {"name": "farfield", "role": "farfield"},
                  ]}})
    with pytest.raises(ValidationError, match="is in no post group"):
        validate(spec)


def test_a_patch_in_two_groups_is_an_error() -> None:
    spec = _spec({"post": {"groups": {"a": ["body"], "b": ["body", "wing"]}},
                  "geometry": {"patches": [
                      {"name": "body", "role": "body"},
                      {"name": "wing", "role": "body"},
                      {"name": "ground", "role": "ground"},
                      {"name": "inlet", "role": "inlet"},
                      {"name": "outlet", "role": "outlet"},
                      {"name": "farfield", "role": "farfield"},
                  ]}})
    with pytest.raises(ValidationError, match="more than one post group"):
        validate(spec)


def test_no_groups_at_all_is_fine() -> None:
    """Ahmed declares none and must not be forced to."""
    validate(_spec())
