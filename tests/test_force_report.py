from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from simdev.report.forces import axles_from_prepare, build_force_report
from simdev.run.status import StageStatus, write_status
from simdev.stages.common import load_spec
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
    for name, fixture in (("forces", "force.dat"), ("forces", "moment.dat")):
        out = target / "postProcessing" / name / "0"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURES / fixture, out / fixture)
    return target


def test_the_report_carries_dimensional_forces(run_dir: Path) -> None:
    report = build_force_report(run_dir, load_spec(run_dir), (2, 3))
    # force.dat totals average 2.5, 0.5, -25.0 over Time 2..3.
    assert report.force is not None
    assert report.force[2] == pytest.approx(-25.0)
    assert report.moment is not None
    assert report.moment[1] == pytest.approx(2.5)


def test_a_run_without_a_forces_object_still_reports(run_dir: Path) -> None:
    """A run made before the forces object existed is not a failed run.

    It has coefficients, a y+ verdict and a flow field. It just has no
    newtons, and the report says so rather than refusing.
    """
    shutil.rmtree(run_dir / "postProcessing" / "forces")
    report = build_force_report(run_dir, load_spec(run_dir), (2, 3))
    assert report.force is None
    assert report.cop is None
    assert any("forces" in reason for reason in report.reasons)


def test_axles_come_from_what_prepare_measured(tmp_path: Path) -> None:
    write_status(
        tmp_path,
        StageStatus(
            "prepare", "ok", "h", [],
            {"wheels": {
                "FL": {"origin": [0.30, 0.1, 0.03]},
                "FR": {"origin": [0.30, -0.1, 0.03]},
                "RL": {"origin": [-0.10, 0.1, 0.03]},
                "RR": {"origin": [-0.10, -0.1, 0.03]},
            }},
        ),
    )
    assert axles_from_prepare(tmp_path) == pytest.approx((0.30, -0.10))


def test_no_wheels_means_no_balance(tmp_path: Path) -> None:
    """The Ahmed body has no axles and must not get an invented wheelbase."""
    write_status(tmp_path, StageStatus("prepare", "ok", "h", [], {"wheels": {}}))
    assert axles_from_prepare(tmp_path) is None
