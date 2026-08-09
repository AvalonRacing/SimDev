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


def _layers(body_layers: float) -> dict[str, LayerInfo]:
    return {
        "body": LayerInfo("body", 18000, body_layers, 3.5e-4),
        "ground": LayerInfo("ground", 22000, 0.0, 0.0),
    }


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
    result = check_mesh_quality(_good_mesh(), _layers(spec.mesh.n_layers), spec)
    assert result.passed is True
    assert result.reasons == []


def test_mesh_gate_fails_on_negative_volumes() -> None:
    spec = _spec()
    bad = CheckMeshResult(1000, 60.0, 3.0, True, [])
    result = check_mesh_quality(bad, _layers(spec.mesh.n_layers), spec)
    assert result.passed is False
    assert any("negative" in r for r in result.reasons)


def test_mesh_gate_fails_on_excessive_non_orthogonality() -> None:
    spec = _spec()
    bad = CheckMeshResult(1000, 85.0, 3.0, False, [])
    result = check_mesh_quality(bad, _layers(spec.mesh.n_layers), spec)
    assert result.passed is False
    assert any("orthogonal" in r for r in result.reasons)


def test_mesh_gate_fails_on_collapsed_layers() -> None:
    spec = _spec()
    # Requested n_layers, achieved a small fraction of them.
    result = check_mesh_quality(_good_mesh(), _layers(1.0), spec)
    assert result.passed is False
    assert any("layer" in r for r in result.reasons)


def test_mesh_gate_ignores_patches_that_requested_no_layers() -> None:
    spec = _spec()
    result = check_mesh_quality(_good_mesh(), _layers(spec.mesh.n_layers), spec)
    assert "ground" not in " ".join(result.reasons)


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
