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
    """A mesh whose *average* non-orthogonality is high is genuinely bad."""
    spec = _spec()
    bad = CheckMeshResult(
        1000, 85.0, 3.0, False, [], n_faces=6000, mean_non_ortho=40.0
    )
    n = spec.mesh.n_layers
    result = check_mesh_quality(bad, _layers(n), spec, _requested(spec, n))
    assert result.passed is False
    assert any("orthogonal" in r for r in result.reasons)


def test_mesh_gate_tolerates_a_handful_of_bad_faces_in_a_large_mesh() -> None:
    """The production mesh's real numbers: 58 non-ortho and 27 skew faces out
    of 21.1 M, average non-orthogonality 9.6. checkMesh itself passes the
    non-orthogonality check on it. Gating on the single worst face makes a
    mesh of this size unpassable for no physical reason."""
    spec = _spec({"mesh": {"trust_check_mesh_verdict": False}})
    n = spec.mesh.n_layers
    production = CheckMeshResult(
        n_cells=6624853,
        max_non_ortho=71.47,
        max_skewness=6.73,
        has_negative_volumes=False,
        failed_checks=[],
        n_faces=21080791,
        mean_non_ortho=9.58,
        n_severely_non_ortho=58,
        n_highly_skew=27,
    )
    result = check_mesh_quality(production, _layers(n), spec, _requested(spec, n))
    assert result.passed is True, result.reasons


def test_mesh_gate_fails_when_bad_faces_are_a_real_fraction() -> None:
    """Same maxima as the production mesh, but 2% of faces are severely
    non-orthogonal rather than 0.0003%."""
    spec = _spec({"mesh": {"trust_check_mesh_verdict": False}})
    n = spec.mesh.n_layers
    bad = CheckMeshResult(
        n_cells=1000000,
        max_non_ortho=71.47,
        max_skewness=3.0,
        has_negative_volumes=False,
        failed_checks=[],
        n_faces=1000000,
        mean_non_ortho=12.0,
        n_severely_non_ortho=20000,
        n_highly_skew=0,
    )
    result = check_mesh_quality(bad, _layers(n), spec, _requested(spec, n))
    assert result.passed is False
    assert any("orthogonal" in r for r in result.reasons)


def test_mesh_gate_fails_when_skew_faces_are_a_real_fraction() -> None:
    spec = _spec({"mesh": {"trust_check_mesh_verdict": False}})
    n = spec.mesh.n_layers
    bad = CheckMeshResult(
        n_cells=1000000,
        max_non_ortho=60.0,
        max_skewness=8.0,
        has_negative_volumes=False,
        failed_checks=[],
        n_faces=1000000,
        mean_non_ortho=9.0,
        n_severely_non_ortho=0,
        n_highly_skew=50000,
    )
    result = check_mesh_quality(bad, _layers(n), spec, _requested(spec, n))
    assert result.passed is False
    assert any("skew" in r for r in result.reasons)


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


# --- limit cycles ---------------------------------------------------------
#
# Steady RANS on a massively separated cornering open-wheel car has no fixed
# point to find. It settles to a stationary mean and then oscillates about it
# forever. The measured car case (docs/linux-migration.md, "Still open")
# converges in the mean - Cd 1.0635 +/- 0.35 %, Cl -1.726 +/- 0.9 % over
# rolling 200-iteration windows - while carrying a physical limit cycle of
# +/-2.2 % in Cd and +/-7.9 % in Cl. Judging that oscillation as scatter and
# calling it "not converged" measures the amplitude of real physics, not the
# state of the solve, so drift and amplitude are two separate questions with
# two separate tolerances.


def _limit_cycle(
    n: int, period: float = 180.0, drift: float = 0.0
) -> pd.DataFrame:
    """A settled mean with the car case's measured oscillation on top."""
    it = np.arange(1, n + 1)
    phase = 2.0 * np.pi * it / period
    return pd.DataFrame(
        {
            "Time": it,
            "Cd": 1.0635 * (1.0 + 0.022 * np.sin(phase) + drift * it),
            "Cl": -1.726 * (1.0 + 0.079 * np.sin(phase + 0.7) + drift * it),
        }
    )


# The window has to be at least one oscillation period long for either test
# to mean anything, so these pin it explicitly rather than inheriting a
# profile's.
CYCLE: dict = {"solve": {"plateau_window": 180}}


def test_a_settled_limit_cycle_converges() -> None:
    """The mean has stopped moving; the oscillation about it is the physics."""
    spec = _spec(CYCLE)
    result = check_convergence(_limit_cycle(1200), spec)

    assert result.converged is True, result.reasons
    assert result.means["Cd"] == pytest.approx(1.0635, rel=5e-3)
    assert result.means["Cl"] == pytest.approx(-1.726, rel=5e-3)


def test_the_oscillation_is_reported_even_though_it_passes() -> None:
    """Amplitude is a number the engineer needs, not a reason to fail."""
    spec = _spec(CYCLE)
    result = check_convergence(_limit_cycle(1200), spec)

    assert result.converged is True, result.reasons
    assert result.amplitudes["Cl"] > result.amplitudes["Cd"]
    # std of a sine of amplitude A is A/sqrt(2).
    assert result.amplitudes["Cl"] == pytest.approx(0.079 / np.sqrt(2), rel=0.1)
    assert result.amplitudes["Cd"] == pytest.approx(0.022 / np.sqrt(2), rel=0.1)


def test_a_limit_cycle_that_is_still_drifting_fails() -> None:
    """A moving mean is the thing the gate exists to catch."""
    spec = _spec(CYCLE)
    result = check_convergence(_limit_cycle(1200, drift=1e-4), spec)

    assert result.converged is False
    assert any("drifting" in r for r in result.reasons)


def test_an_oscillation_beyond_the_amplitude_bound_fails() -> None:
    """A settled mean does not excuse an unbounded wobble."""
    spec = _spec({"solve": {"plateau_window": 180, "amplitude_tol": 0.01}})
    result = check_convergence(_limit_cycle(1200), spec)

    assert result.converged is False
    assert any("oscillates" in r for r in result.reasons)


def test_drift_is_judged_on_the_mean_not_on_scatter_within_the_window() -> None:
    """Where in the cycle the run stopped must not decide the verdict.

    A least-squares slope fitted inside a single window reads a stationary
    limit cycle as drift, because a sine sampled over part of a period
    genuinely has a slope - that is what produced a '+10.92 % drift' on a
    mean that had not moved at all. Two consecutive window means each average
    the cycle away instead, so every stopping point agrees.
    """
    spec = _spec(CYCLE)
    window = spec.solve.plateau_window

    for stop in range(1000, 1000 + window, 37):
        result = check_convergence(
            _limit_cycle(stop, period=float(window)), spec
        )
        assert result.converged is True, (stop, result.reasons)


def test_two_whole_windows_are_needed_before_a_verdict() -> None:
    """One window gives a mean but nothing to compare it against."""
    spec = _spec(CYCLE)
    window = spec.solve.plateau_window

    result = check_convergence(_limit_cycle(window + 10), spec)

    assert result.converged is False
    assert any(str(2 * window) in r for r in result.reasons)
    assert "Cd" in result.means  # still reported, never withheld


# --- convergence not judged -----------------------------------------------
#
# drift_tol None is for a run whose stopping point is chosen rather than
# reached: a fixed-cost sample used to compare meshes against each other,
# where every run carries the same bias and only the delta is read. The drift
# is still measured, because a number nobody looks at is worse than a number
# that fails - it just stops deciding the verdict.

# amplitude_tol is lifted alongside, and only because _forces() drifts Cl
# through zero: std/|mean| is 11.5% there, so the stock 10% bound fires for a
# reason that has nothing to do with drift and would mask what these check.
# test_amplitude_still_gates_when_drift_is_not_judged covers the bound itself.
UNJUDGED: dict = {
    "solve": {"plateau_window": 50, "drift_tol": None, "amplitude_tol": 1.0}
}
JUDGED: dict = {
    "solve": {"plateau_window": 50, "drift_tol": 0.002, "amplitude_tol": 1.0}
}


def test_unset_drift_tol_does_not_fail_a_drifting_run() -> None:
    drifting = _forces(250, drift=1e-3)

    assert check_convergence(drifting, _spec(JUDGED)).converged is False
    assert check_convergence(drifting, _spec(UNJUDGED)).converged is True


def test_unset_drift_tol_still_measures_and_reports_the_drift() -> None:
    """The number has to survive, or turning the gate off hides the problem."""
    result = check_convergence(_forces(250, drift=1e-3), _spec(UNJUDGED))

    assert any("not judged" in r for r in result.reasons)
    assert any("NOT judged" in r for r in result.reasons)
    # the measured drift is still in there, with a sign and a magnitude
    assert any("Cd mean moved" in r for r in result.reasons)


def test_unset_drift_tol_reports_the_mean_over_the_window() -> None:
    spec = _spec(UNJUDGED)
    result = check_convergence(_forces(250), spec)

    assert result.window == (200, 250)
    assert result.means["Cd"] == pytest.approx(0.35, abs=1e-3)


def test_amplitude_still_gates_when_drift_is_not_judged() -> None:
    """Turning off the convergence test must not turn off the stability one."""
    spec = _spec(
        {"solve": {"plateau_window": 180, "drift_tol": None, "amplitude_tol": 0.01}}
    )
    result = check_convergence(_limit_cycle(1200), spec)

    assert result.converged is False
    assert any("oscillates" in r for r in result.reasons)


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


def test_check_mesh_skew_verdict_yields_to_the_extent_criteria() -> None:
    """The production case: 27 skew faces of 21.1 M, and checkMesh fails it.

    checkMesh judges skewness against a hardcoded 4 and reports a failed check
    for a single bad face, so on any production mesh its verdict is the worst
    face by another route - exactly what the extent criteria were introduced
    to stop gating on. Where the gate has a denominator it is the authority,
    and checkMesh's verdict on the two quantities the spec owns is recorded
    rather than fatal.
    """
    spec = _mesh_spec(trust_check_mesh_verdict=True)
    result = _check_result(
        n_cells=6_624_853,
        n_faces=21_080_791,
        max_non_ortho=71.47,
        mean_non_ortho=9.58,
        max_skewness=6.73,
        n_severely_non_ortho=58,
        n_highly_skew=27,
        failed_checks=[
            "Max skewness = 6.731082, 27 highly skew faces detected which "
            "may impair the quality of the results",
        ],
    )
    gate = check_mesh_quality(result, {}, spec, {})
    assert gate.passed, gate.reasons
    assert any("checkMesh reported" in r for r in gate.reasons)


def test_check_mesh_structural_verdict_still_gates_a_large_mesh() -> None:
    """Demotion is only ever for the two quantities the spec measures itself."""
    spec = _mesh_spec(trust_check_mesh_verdict=True)
    result = _check_result(
        n_faces=21_080_791,
        mean_non_ortho=9.58,
        n_severely_non_ortho=58,
        n_highly_skew=27,
        failed_checks=[
            "Max skewness = 6.731082, 27 highly skew faces detected",
            "Number of not closed cells: 3",
        ],
    )
    gate = check_mesh_quality(result, {}, spec, {})
    assert not gate.passed
    assert any("not closed cells" in r for r in gate.reasons)

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
