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
