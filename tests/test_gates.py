from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.gates.convergence import check_convergence
from simdev.gates.mesh_quality import check_mesh_quality
from simdev.gates.yplus import check_y_plus
from simdev.run.parsers import CheckMeshResult, LayerInfo

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
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


def _spec(overrides: dict | None = None):
    return resolve(deep_merge(BASE, overrides or {}), profile="dev")


def _good_mesh() -> CheckMeshResult:
    return CheckMeshResult(
        n_cells=1000000,
        max_non_ortho=64.0,
        max_skewness=3.2,
        has_negative_volumes=False,
        failed_checks=[],
    )


def _layers(body_layers: float, ground_layers: float | None = None) -> dict[str, LayerInfo]:
    ground = body_layers if ground_layers is None else ground_layers
    return {
        "body": LayerInfo("body", 18000, body_layers, 3.5e-4),
        "ground": LayerInfo("ground", 22000, ground, 3.5e-4),
    }


def _requested(spec, n: int) -> dict[str, int]:
    """Layers asked of snappy on every wall patch."""
    return {"body": n, "ground": n}


def _forces(n: int, drift: float = 0.0) -> pd.DataFrame:
    it = np.arange(1, n + 1)
    return pd.DataFrame(
        {
            "Time": it,
            "Cd": 0.35 + drift * it + 1e-5 * np.sin(it),
            "Cl": -0.10 + drift * it,
        }
    )


def test_mesh_gate_passes_a_good_mesh() -> None:
    spec = _spec()
    n = spec.mesh.n_layers
    result = check_mesh_quality(
        _good_mesh(), _layers(n), spec, _requested(spec, n)
    )
    assert result.passed is True
    assert result.reasons == []


def test_mesh_gate_fails_on_negative_volumes() -> None:
    spec = _spec()
    bad = CheckMeshResult(1000, 60.0, 3.0, True, [])
    n = spec.mesh.n_layers
    result = check_mesh_quality(bad, _layers(n), spec, _requested(spec, n))
    assert result.passed is False
    assert any("negative" in r for r in result.reasons)


def test_mesh_gate_fails_on_excessive_non_orthogonality() -> None:
    spec = _spec()
    bad = CheckMeshResult(1000, 85.0, 3.0, False, [])
    n = spec.mesh.n_layers
    result = check_mesh_quality(bad, _layers(n), spec, _requested(spec, n))
    assert result.passed is False
    assert any("orthogonal" in r for r in result.reasons)


def test_mesh_gate_fails_on_collapsed_layers() -> None:
    spec = _spec()
    # Requested n_layers, achieved a small fraction of them.
    result = check_mesh_quality(
        _good_mesh(), _layers(1.0), spec, _requested(spec, spec.mesh.n_layers)
    )
    assert result.passed is False
    assert any("layer" in r for r in result.reasons)


def test_mesh_gate_fails_when_the_layer_table_is_missing() -> None:
    """An unparseable or absent layer table must not read as success.

    This is how the gate behaved against a real v2412 log: the table did not
    parse, every lookup returned None, and the stage reported state 'ok' with
    no reasons while the stilts carried 25% of their layers.
    """
    spec = _spec()
    result = check_mesh_quality(
        _good_mesh(), {}, spec, _requested(spec, spec.mesh.n_layers)
    )
    assert result.passed is False
    assert any("no layer data" in r for r in result.reasons)


def test_mesh_gate_judges_the_ground_too() -> None:
    """The ground is a wall, so it carries a wall function and needs layers.

    It used to be skipped because layers followed refinement == "high" and
    the ground is a blockMesh patch, which is how it ended up running with
    its first cell 60 mm off the floor and y+ around 2000.
    """
    spec = _spec()
    n = spec.mesh.n_layers
    result = check_mesh_quality(
        _good_mesh(), _layers(n, ground_layers=0.0), spec, _requested(spec, n)
    )
    assert result.passed is False
    assert any("ground" in r for r in result.reasons)


def test_mesh_gate_skips_patches_that_asked_for_no_layers() -> None:
    spec = _spec()
    n = spec.mesh.n_layers
    result = check_mesh_quality(
        _good_mesh(), _layers(n, ground_layers=0.0), spec, {"body": n, "ground": 0}
    )
    assert result.passed is True


def test_convergence_detects_a_plateau() -> None:
    spec = _spec()
    result = check_convergence(_forces(300), spec)
    assert result.converged is True
    assert result.means["Cd"] == pytest.approx(0.35, abs=1e-3)


def test_convergence_reports_the_mean_not_the_last_value() -> None:
    spec = _spec()
    df = _forces(300)
    df.loc[df.index[-1], "Cd"] = 99.0  # a single spike must not dominate
    result = check_convergence(df, spec)
    assert result.means["Cd"] < 5.0


def test_convergence_fails_on_a_drifting_signal() -> None:
    spec = _spec()
    result = check_convergence(_forces(300, drift=1e-3), spec)
    assert result.converged is False
    assert any("drift" in r or "plateau" in r for r in result.reasons)


def test_non_converged_run_still_reports_means() -> None:
    spec = _spec()
    result = check_convergence(_forces(300, drift=1e-3), spec)
    assert result.converged is False
    assert "Cd" in result.means and "Cl" in result.means


def test_convergence_fails_when_too_short_to_judge() -> None:
    spec = _spec()
    result = check_convergence(_forces(5), spec)
    assert result.converged is False


def test_convergence_window_is_the_trailing_slice() -> None:
    spec = _spec()
    result = check_convergence(_forces(300), spec)
    start, end = result.window
    assert end - start == spec.solve.plateau_window
    assert end == 300


def test_y_plus_gate_passes_inside_the_band() -> None:
    spec = _spec()
    df = pd.DataFrame(
        {"Time": [1], "patch": ["body"], "min": [35.0], "max": [280.0], "average": [95.0]}
    )
    assert check_y_plus(df, spec).passed is True


def test_y_plus_gate_fails_outside_the_band() -> None:
    spec = _spec()
    df = pd.DataFrame(
        {"Time": [1], "patch": ["body"], "min": [0.4], "max": [3.0], "average": [1.2]}
    )
    result = check_y_plus(df, spec)
    assert result.passed is False
    assert any("y+" in r for r in result.reasons)


def test_y_plus_gate_only_judges_force_patches() -> None:
    spec = _spec()
    df = pd.DataFrame(
        {
            "Time": [1, 1],
            "patch": ["body", "ground"],
            "min": [35.0, 0.1],
            "max": [280.0, 2.0],
            "average": [95.0, 1.0],
        }
    )
    assert check_y_plus(df, spec).passed is True


# --- trusting checkMesh's own verdict --------------------------------------
#
# checkMesh judges skewness and non-orthogonality against limits compiled into
# it, so without this switch mesh.max_skewness can only ever tighten the gate:
# raising it leaves checkMesh still failing the mesh at 4.6, and the config
# then describes something other than what the code does.


def _check_result(**overrides):
    fields = {
        "n_cells": 21908,
        "max_non_ortho": 68.5,
        "max_skewness": 4.55,
        "has_negative_volumes": False,
        "failed_checks": [
            "Max skewness = 4.5489768, 1 highly skew faces detected which "
            "may impair the quality of the results",
        ],
    }
    fields.update(overrides)
    return CheckMeshResult(**fields)


def _mesh_spec(**mesh_overrides):
    return _spec({"mesh": mesh_overrides})


def test_trusting_check_mesh_fails_even_when_the_spec_is_permissive() -> None:
    spec = _mesh_spec(max_skewness=20.0, trust_check_mesh_verdict=True)
    gate = check_mesh_quality(_check_result(), {}, spec, {})
    assert not gate.passed
    assert any("checkMesh reported" in r for r in gate.reasons)


def test_not_trusting_it_leaves_the_spec_as_the_authority() -> None:
    spec = _mesh_spec(max_skewness=20.0, trust_check_mesh_verdict=False)
    gate = check_mesh_quality(_check_result(), {}, spec, {})
    assert gate.passed, gate.reasons


def test_the_spec_threshold_still_bites_when_not_trusting_check_mesh() -> None:
    """Opting out is not opting out of skewness, only of checkMesh's number."""
    spec = _mesh_spec(max_skewness=3.0, trust_check_mesh_verdict=False)
    gate = check_mesh_quality(_check_result(), {}, spec, {})
    assert not gate.passed
    assert any("max skewness 4.55 exceeds 3.00" in r for r in gate.reasons)


def test_structural_failures_gate_whatever_the_case_says() -> None:
    """A blanket mute would let a smoke profile sail past a broken mesh."""
    spec = _mesh_spec(max_skewness=20.0, trust_check_mesh_verdict=False)
    result = _check_result(
        failed_checks=[
            "Max skewness = 4.5489768, 1 highly skew faces detected",
            "Number of not closed cells: 3",
            "The mesh has multiple regions",
        ]
    )

    gate = check_mesh_quality(result, {}, spec, {})

    assert not gate.passed
    assert any("not closed cells" in r for r in gate.reasons)
    assert any("multiple regions" in r for r in gate.reasons)
    assert not any("skew" in r.lower() for r in gate.reasons)


def test_negative_volumes_are_never_waived() -> None:
    spec = _mesh_spec(max_skewness=20.0, trust_check_mesh_verdict=False)
    gate = check_mesh_quality(
        _check_result(has_negative_volumes=True), {}, spec, {}
    )
    assert not gate.passed
    assert any("negative volume" in r for r in gate.reasons)


def test_non_orthogonality_verdict_is_also_spec_owned() -> None:
    spec = _mesh_spec(
        max_non_ortho=75.0, max_skewness=20.0, trust_check_mesh_verdict=False
    )
    result = _check_result(
        failed_checks=["Number of severely non-orthogonal faces: 12"]
    )
    assert check_mesh_quality(result, {}, spec, {}).passed


def test_production_profiles_still_trust_check_mesh() -> None:
    from simdev.config.profiles import RESOLUTION_PROFILES

    for name in ("production", "car", "car_dev"):
        mesh = RESOLUTION_PROFILES[name].get("mesh", {})
        assert mesh.get("trust_check_mesh_verdict", True) is True, name
    assert (
        RESOLUTION_PROFILES["car_smoke"]["mesh"]["trust_check_mesh_verdict"]
        is False
    )
