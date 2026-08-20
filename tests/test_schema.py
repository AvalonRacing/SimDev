from __future__ import annotations

import pytest

from simdev.config.schema import CaseSpec, Mode
from simdev.geometry.roles import PatchRole


def _spec(**physics_overrides: object) -> CaseSpec:
    physics = {"wall_treatment": "high_y_plus"}
    physics.update(physics_overrides)
    return CaseSpec.model_validate(
        {
            "name": "t",
            "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
            "ground": {"motion": "static"},
            "physics": physics,
            "domain": {},
            "mesh": {
                "base_cell_size": 0.04,
                "surface_refinement_min": 4,
                "surface_refinement_max": 5,
                "n_layers": 6,
                "first_layer_thickness": 3.0e-4,
            },
            "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
            "solve": {"max_iterations": 2000, "n_ranks": 8},
            "post": {"yplus_min": 30.0, "yplus_max": 300.0},
            "geometry": {
                "kind": "ahmed",
                "symmetric": True,
                "ahmed": {},
                "patches": [
                    {"name": "body", "role": "body"},
                    {"name": "ground", "role": "ground"},
                    {"name": "symm", "role": "symmetry"},
                ],
            },
        }
    )


def test_straight_zero_yaw_is_half_model() -> None:
    assert _spec().half_model is True


def test_yaw_alone_breaks_symmetry() -> None:
    # Yaw kills symmetry even in straight-line mode.
    assert _spec(yaw_deg=5.0).half_model is False


def test_cornering_is_never_half_model() -> None:
    spec = _spec(mode="cornering", corner_radius=8.0)
    assert spec.half_model is False


def test_a_ref_is_halved_only_for_half_model() -> None:
    assert _spec().a_ref_effective == pytest.approx(0.056)
    assert _spec(yaw_deg=5.0).a_ref_effective == pytest.approx(0.112)


def test_omega_rotation_from_corner_radius() -> None:
    spec = _spec(mode="cornering", corner_radius=8.0)
    assert spec.omega_rotation == pytest.approx(5.0)


def test_omega_rotation_is_none_for_straight() -> None:
    assert _spec().omega_rotation is None


def test_patches_with_role() -> None:
    assert _spec().patches_with_role(PatchRole.BODY) == ["body"]


def test_spec_hash_is_stable_and_sensitive() -> None:
    assert _spec().spec_hash() == _spec().spec_hash()
    assert _spec().spec_hash() != _spec(yaw_deg=5.0).spec_hash()


def test_mode_enum_values() -> None:
    assert Mode.CORNERING.value == "cornering"


# --- the refinement cap ---------------------------------------------------


def _capped_spec(cap):
    case = {
        "name": "t",
        "flow": {"u_inf": 15.0, "turbulence_length_scale": 0.02},
        "ground": {"motion": "moving"},
        "physics": {"wall_treatment": "high_y_plus"},
        "domain": {},
        "mesh": {
            "base_cell_size": 0.05,
            "surface_refinement_min": 1,
            "surface_refinement_max": 2,
            "n_layers": 3,
            "first_layer_thickness": 1.0e-3,
            "refinement_cap": cap,
        },
        "forces": {"a_ref_full": 0.02, "l_ref": 0.44},
        "solve": {"max_iterations": 50, "n_ranks": 4},
        "post": {"yplus_min": 30.0, "yplus_max": 300.0},
        "geometry": {
            "kind": "stl",
            "source_dir": "cad",
            "patches": [
                {"name": "body", "role": "body"},
                {"name": "wing", "role": "body",
                 "refinement_min": 5, "refinement_max": 6},
                {"name": "ground", "role": "ground"},
            ],
        },
    }
    return CaseSpec.model_validate(case)


def _patch(spec, name):
    return next(p for p in spec.geometry.patches if p.name == name)


def test_without_a_cap_a_patch_keeps_its_own_levels() -> None:
    spec = _capped_spec(None)
    assert spec.patch_refinement(_patch(spec, "wing")) == (5, 6)


def test_the_cap_overrides_per_patch_levels() -> None:
    """The point of it: per-patch levels override the profile, so coarsening
    the profile alone leaves the expensive patches expensive."""
    spec = _capped_spec(2)
    assert spec.patch_refinement(_patch(spec, "wing")) == (2, 2)


def test_the_cap_never_raises_a_level() -> None:
    spec = _capped_spec(4)
    assert spec.patch_refinement(_patch(spec, "body")) == (1, 2)


def test_the_cap_reaches_the_case_wide_surface_cell() -> None:
    uncapped = _capped_spec(None).surface_cell_size
    assert _capped_spec(1).surface_cell_size == pytest.approx(uncapped * 2)


def test_the_smoke_profile_caps_refinement() -> None:
    """The cap must actually bind, whatever the background happens to be.

    Asserting the cap's raw *number* pins it to one choice of base_cell_size
    and breaks on any re-basing while telling you nothing about whether the
    profile is still cheap. What matters is that a cap is set and that it sits
    at or below the profile's own surface refinement, so per-patch levels -
    which otherwise override the profile - are overridden in turn. That is the
    whole reason the cap exists. The absolute cell size it produces is pinned
    in tests/test_cell_sizes.py.
    """
    from simdev.config.profiles import RESOLUTION_PROFILES

    mesh = RESOLUTION_PROFILES["car_smoke"]["mesh"]
    assert mesh["refinement_cap"] is not None
    assert mesh["refinement_cap"] <= mesh["surface_refinement_max"]
