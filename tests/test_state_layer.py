"""The library state as a merge layer, and CAD provenance in the spec."""

from __future__ import annotations

import copy

from simdev.config.resolve import resolve
from simdev.config.schema import CadProvenance, Mode

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

CORNER = {
    "flow": {"u_inf": 15.0},
    "physics": {"mode": "cornering", "corner_radius": 4.0},
    "domain": {"kind": "annulus"},
    "ground": {"motion": "static"},
}


def test_the_state_wins_over_the_case_file() -> None:
    spec = resolve(BASE_CASE, profile="car_dev", state=CORNER)
    assert spec.flow.u_inf == 15.0
    assert spec.physics.mode is Mode.CORNERING
    assert spec.domain.kind == "annulus"


def test_overrides_win_over_the_state() -> None:
    spec = resolve(
        BASE_CASE,
        profile="car_dev",
        overrides={"physics": {"corner_radius": 8.0}},
        state=CORNER,
    )
    assert spec.physics.corner_radius == 8.0


def test_the_state_is_not_mutated() -> None:
    before = copy.deepcopy(CORNER)
    resolve(BASE_CASE, profile="car_dev", state=CORNER)
    assert CORNER == before


def test_no_state_leaves_the_case_alone() -> None:
    assert resolve(BASE_CASE, profile="car_dev").flow.u_inf == 10.0


def test_cad_provenance_is_recorded_and_part_of_the_hash() -> None:
    def with_body(digest: str):
        return resolve(
            BASE_CASE,
            profile="car_dev",
            overrides={"cad": {"design": "v01", "state": "s", "parts": {"Body": digest}}},
            state=CORNER,
        )

    a, b = with_body("aa"), with_body("bb")
    assert a.cad == CadProvenance(design="v01", state="s", parts={"Body": "aa"})
    assert a.spec_hash() != b.spec_hash()


def test_no_cad_provenance_by_default() -> None:
    assert resolve(BASE_CASE, profile="car_dev").cad is None
