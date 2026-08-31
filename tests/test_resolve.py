from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simdev.config.resolve import deep_merge, load_case, resolve
from simdev.config.schema import WallTreatment

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {},
        "patches": [{"name": "body", "role": "body"}],
    },
}


def test_deep_merge_overrides_nested_leaves_only() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 9}}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": 9}, "b": 3}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_dev_profile_uses_eight_ranks() -> None:
    assert resolve(CASE, profile="dev").solve.n_ranks == 8


def test_production_profile_uses_forty_ranks() -> None:
    # 40 physical cores, never the 80 logical threads.
    assert resolve(CASE, profile="production").solve.n_ranks == 40


def test_dev_profile_is_coarser_than_production() -> None:
    dev = resolve(CASE, profile="dev")
    prod = resolve(CASE, profile="production")
    assert dev.mesh.base_cell_size > prod.mesh.base_cell_size
    assert dev.solve.max_iterations < prod.solve.max_iterations


def test_wall_treatment_sets_yplus_band() -> None:
    high = resolve(CASE, profile="dev", wall_treatment="high_y_plus")
    low = resolve(CASE, profile="dev", wall_treatment="low_y_plus")
    assert (high.post.yplus_min, high.post.yplus_max) == (30.0, 300.0)
    assert low.post.yplus_max <= 5.0
    assert low.mesh.first_layer_thickness < high.mesh.first_layer_thickness
    assert low.mesh.n_layers > high.mesh.n_layers


def test_cli_wall_treatment_beats_the_case_file() -> None:
    # The CLI picks which wall profile applies, so it must also win the field.
    # Otherwise the spec names one treatment but carries another's mesh numbers.
    case = deep_merge(CASE, {"physics": {"wall_treatment": "high_y_plus"}})
    spec = resolve(case, profile="dev", wall_treatment="low_y_plus")
    assert spec.physics.wall_treatment is WallTreatment.LOW_Y_PLUS
    assert spec.post.yplus_max <= 5.0
    assert spec.mesh.first_layer_thickness == pytest.approx(2.0e-5)


def test_case_file_wall_treatment_applies_when_no_cli_override() -> None:
    case = deep_merge(CASE, {"physics": {"wall_treatment": "low_y_plus"}})
    spec = resolve(case, profile="dev")
    assert spec.physics.wall_treatment is WallTreatment.LOW_Y_PLUS
    assert spec.post.yplus_max <= 5.0


def test_case_file_beats_profile() -> None:
    case = deep_merge(CASE, {"solve": {"n_ranks": 4}})
    assert resolve(case, profile="production").solve.n_ranks == 4


def test_cli_overrides_beat_case_file() -> None:
    case = deep_merge(CASE, {"solve": {"n_ranks": 4}})
    spec = resolve(case, profile="dev", overrides={"solve": {"n_ranks": 2}})
    assert spec.solve.n_ranks == 2


def test_load_case_reads_yaml(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    assert load_case(p, profile="dev").name == "ahmed"


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        resolve(CASE, profile="nonexistent")


def test_resolved_spec_has_no_missing_values() -> None:
    spec = resolve(CASE, profile="dev")
    assert spec.physics.wall_treatment in set(WallTreatment)
    assert spec.mesh.n_layers > 0
    assert spec.post.yplus_max > spec.post.yplus_min


# --- profile invariants -----------------------------------------------------


def test_write_interval_divides_max_iterations_in_every_profile() -> None:
    """simpleFoam does not force a write at endTime unless residualControl
    stops it first, so an interval that does not divide the run length writes
    fields and then never writes again.

    It has happened: 500 into 750 left the 13.28M run finishing cleanly with
    its newest field data 250 iterations stale, invisible until someone opened
    it in ParaView and read a coefficient off the wrong time. The rule was
    written into a comment and enforced by nothing, so the next time
    max_iterations moved it was free to break again.
    """
    from simdev.config.profiles import RESOLUTION_PROFILES

    for name, profile in RESOLUTION_PROFILES.items():
        solve = profile.get("solve", {})
        interval = solve.get("write_interval")
        if interval is None:
            continue  # writes once at the end, which is always consistent
        assert solve["max_iterations"] % interval == 0, (
            f"profile '{name}': write_interval {interval} does not divide "
            f"max_iterations {solve['max_iterations']}"
        )


def test_two_plateau_windows_fit_inside_every_profiles_run() -> None:
    """The drift test compares two consecutive windows. A profile whose run is
    shorter than 2 x plateau_window can never produce a verdict, so it would
    report not_judged forever however well it converged."""
    from simdev.config.profiles import DEFAULTS, RESOLUTION_PROFILES

    for name, profile in RESOLUTION_PROFILES.items():
        solve = profile.get("solve", {})
        window = solve.get("plateau_window", DEFAULTS["solve"]["plateau_window"])
        assert solve["max_iterations"] >= 2 * window, (
            f"profile '{name}': {solve['max_iterations']} iterations cannot "
            f"hold two {window}-iteration windows"
        )
