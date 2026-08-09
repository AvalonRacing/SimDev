from __future__ import annotations

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.domain.box import BoxDomainBuilder, blockage_ratio, check_blockage

BOUNDS = ((0.0, -0.1945, 0.05), (1.044, 0.1945, 0.338))

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


def _spec(overrides: dict | None = None):
    return resolve(deep_merge(BASE, overrides or {}), profile="dev")


def _full_model_spec():
    case = deep_merge(BASE, {"physics": {"yaw_deg": 5.0}})
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    return resolve(case, profile="dev")


def test_streamwise_extent_uses_body_lengths() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    length = 1.044
    assert d.x_min == pytest.approx(0.0 - 5.0 * length)
    assert d.x_max == pytest.approx(1.044 + 10.0 * length)


def test_ground_is_at_z_zero() -> None:
    assert BoxDomainBuilder().build(_spec(), BOUNDS).z_min == 0.0


def test_half_model_starts_at_the_symmetry_plane() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    assert d.y_min == 0.0
    assert d.symmetry == "symmetry"


def test_full_model_is_symmetric_about_y_zero_with_no_symmetry_patch() -> None:
    d = BoxDomainBuilder().build(_full_model_spec(), BOUNDS)
    assert d.y_min == pytest.approx(-d.y_max)
    assert d.symmetry is None


def test_patch_names_come_from_the_spec() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    assert (d.inlet, d.outlet, d.ground, d.farfield) == (
        "inlet",
        "outlet",
        "ground",
        "farfield",
    )


def test_cell_counts_are_positive_and_roughly_cubic() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    nx, ny, nz = d.n_cells
    assert min(nx, ny, nz) >= 1
    sx, sy, sz = d.size
    assert (sx / nx) == pytest.approx(sy / ny, rel=0.35)
    assert (sx / nx) == pytest.approx(sz / nz, rel=0.35)


def test_blockage_ratio_uses_the_domain_cross_section() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    ratio = blockage_ratio(0.056, d)
    assert ratio == pytest.approx(0.056 / d.cross_section_area)


def test_check_blockage_flags_a_cramped_domain() -> None:
    d = BoxDomainBuilder().build(
        _spec({"domain": {"half_width_lengths": 0.3, "height_lengths": 0.4}}), BOUNDS
    )
    assert check_blockage(0.056, d, max_blockage=0.01)


def test_check_blockage_is_quiet_on_a_roomy_domain() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    assert check_blockage(0.056, d, max_blockage=0.01) == []
