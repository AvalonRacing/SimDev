from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, write_status
from simdev.stages.post import post
from simdev.stages.prepare import prepare

FIXTURES = Path(__file__).parent / "fixtures"

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "symmetric": True,
        "ahmed": {"include_stilts": False},
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


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    case = tmp_path / "config.yaml"
    case.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    target = tmp_path / "run"
    prepare(case, target, profile="dev")

    for name, fixture in (
        ("forceCoeffs", "coefficient.dat"),
        ("yPlus", "yPlus.dat"),
    ):
        out = target / "postProcessing" / name / "0"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURES / fixture, out / fixture)

    write_status(
        target, StageStatus("mesh", "ok", "h", [], {"n_cells": 1122334})
    )
    return target


def _mark_solve(run_dir: Path, state: str) -> None:
    from simdev.stages.common import load_spec

    write_status(
        run_dir,
        StageStatus("solve", state, load_spec(run_dir).spec_hash(), [], {}),
    )


def test_post_writes_a_result(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    assert (run_dir / "results" / "result.json").exists()
    assert record.n_cells == 1122334


def test_post_runs_after_a_non_converged_solve(run_dir: Path) -> None:
    # A run that produced numbers must not vanish because it did not plateau.
    _mark_solve(run_dir, "gate_failed")
    record = post(run_dir)
    assert record.converged is False
    assert record.cd_mean != 0.0


def test_post_refuses_after_a_failed_solve(run_dir: Path) -> None:
    _mark_solve(run_dir, "failed")
    with pytest.raises(StageError):
        post(run_dir)


def test_post_applies_the_y_plus_gate(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    # Fixture body y+ averages 92.7, inside the high_y_plus band.
    assert record.yplus_passed is True
    assert record.yplus["body"] == pytest.approx(92.7)


def test_post_writes_plots(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    post(run_dir)
    assert (run_dir / "results" / "forces.png").exists()


def test_post_records_the_spec_hash(run_dir: Path) -> None:
    from simdev.stages.common import load_spec

    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    assert record.spec_hash == load_spec(run_dir).spec_hash()


def test_post_status_is_recorded(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    post(run_dir)
    assert read_status(run_dir, "post").state == "ok"


def test_post_writes_the_residual_plot(run_dir: Path) -> None:
    """postProcessing subdirectories carry the function object's *name*.

    controlDict registers the solverInfo function object as 'residuals', so
    the data lands in postProcessing/residuals/<time>/solverInfo.dat. Globbing
    the type instead found nothing, and the tolerant FileNotFoundError guard
    turned that into a silently missing plot.
    """
    out = run_dir / "postProcessing" / "residuals" / "0"
    out.mkdir(parents=True, exist_ok=True)
    out.joinpath("solverInfo.dat").write_text(
        "# Time\tUx_initial\tp_initial\n"
        "1\t1.0e-02\t2.0e-02\n"
        "2\t1.0e-03\t2.0e-03\n"
        "3\t1.0e-04\t2.0e-04\n",
        encoding="utf-8",
    )
    _mark_solve(run_dir, "ok")
    post(run_dir)
    assert (run_dir / "results" / "residuals.png").exists()
