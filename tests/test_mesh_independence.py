from __future__ import annotations

from typing import Any

import pytest
import yaml

from scripts.mesh_independence import (
    LEVELS,
    LevelFailed,
    _level_overrides,
    monotonic_convergence,
    run_levels,
    within_tolerance,
)
from simdev.config.resolve import deep_merge, resolve

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
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


def _spec_for_level(settings: dict[str, Any]):
    return resolve(
        deep_merge(CASE, _level_overrides(settings)), profile="production"
    )


def test_monotonic_increasing_series_is_monotonic() -> None:
    assert monotonic_convergence([0.30, 0.33, 0.345]) is True


def test_monotonic_decreasing_series_is_monotonic() -> None:
    assert monotonic_convergence([0.40, 0.36, 0.348]) is True


def test_oscillating_series_is_not_monotonic() -> None:
    assert monotonic_convergence([0.30, 0.40, 0.33]) is False


def test_two_points_are_insufficient() -> None:
    assert monotonic_convergence([0.30, 0.35]) is False


def test_within_tolerance_accepts_a_ten_percent_error() -> None:
    assert within_tolerance(0.32, target=0.30, tolerance=0.10) is True


def test_within_tolerance_rejects_a_twenty_percent_error() -> None:
    assert within_tolerance(0.36, target=0.30, tolerance=0.10) is False


@pytest.mark.parametrize("value", [0.30, 0.33, 0.27])
def test_tolerance_band_is_symmetric(value: float) -> None:
    assert within_tolerance(value, target=0.30, tolerance=0.10) is True


def test_ladder_holds_refinement_depth_and_layers_constant() -> None:
    """Only base_cell_size may vary across a grid study.

    Changing the refinement depth as well compounds the two, and letting
    n_layers float means the near-wall treatment differs between levels -
    both confound the quantity the study is trying to isolate.
    """
    depths = {tuple(v["refinement"]) for v in LEVELS.values()}
    layers = {v["n_layers"] for v in LEVELS.values()}
    assert len(depths) == 1
    assert len(layers) == 1


def test_ladder_refines_at_a_constant_ratio() -> None:
    """Richardson extrapolation needs a constant, moderate refinement ratio."""
    cells = [
        v["base_cell_size"] / 2 ** v["refinement"][1] for v in LEVELS.values()
    ]
    ratios = [a / b for a, b in zip(cells, cells[1:])]
    assert all(r == pytest.approx(ratios[0], rel=0.02) for r in ratios)
    assert 1.2 < ratios[0] < 2.0


def test_ladder_levels_all_fit_the_pinned_layer_count() -> None:
    """Every level must actually deliver the layers the ladder pins.

    If the finest level cannot fit four layers the clamp silently drops it,
    and the study compares meshes with different near-wall resolution.
    """
    for name, settings in LEVELS.items():
        spec = _spec_for_level(settings)
        body = next(p for p in spec.geometry.patches if p.name == "body")
        assert spec.n_layers_for(body) == settings["n_layers"], name


def test_a_failed_level_aborts_instead_of_running_the_rest(
    tmp_path, monkeypatch
) -> None:
    """Exit code 2 means no result record; continuing wastes the other levels."""
    calls: list[str] = []

    def fake_main(argv):
        calls.append(argv[1])
        return 2

    monkeypatch.setattr("scripts.mesh_independence.cli_main", fake_main)
    case = tmp_path / "config.yaml"
    case.write_text(yaml.safe_dump(CASE), encoding="utf-8")

    with pytest.raises(LevelFailed):
        run_levels(case, tmp_path / "out")
    assert len(calls) == 1


def test_a_gate_flagged_level_still_counts(tmp_path, monkeypatch) -> None:
    """Exit 1 is a gate flag, not a failure: the record exists and is usable."""
    seen: list[str] = []

    def fake_main(argv):
        seen.append(argv[1])
        return 1

    monkeypatch.setattr("scripts.mesh_independence.cli_main", fake_main)
    monkeypatch.setattr("scripts.mesh_independence.aggregate", lambda dirs: dirs)
    case = tmp_path / "config.yaml"
    case.write_text(yaml.safe_dump(CASE), encoding="utf-8")

    result = run_levels(case, tmp_path / "out")
    assert len(seen) == len(LEVELS)
    assert len(result) == len(LEVELS)
