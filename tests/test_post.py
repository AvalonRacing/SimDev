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


# --- area-weighted y+ reaches the result -----------------------------------


def _write_area_yplus(run_dir: Path, patch: str, value: float) -> None:
    out = run_dir / "postProcessing" / f"yPlusArea_{patch}" / "0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "surfaceFieldValue.dat").write_text(
        f"# Region type : patch {patch}\n"
        f"# Time\tareaAverage(yPlus)\n"
        f"250\t{value}\n",
        encoding="utf-8",
    )


def test_post_judges_and_reports_the_area_weighted_y_plus(run_dir: Path) -> None:
    """The weighted number is what the gate judged, so it is the number the
    result record has to carry - otherwise the report and the verdict disagree.
    """
    _mark_solve(run_dir, "ok")
    _write_area_yplus(run_dir, "body", 87.5)
    record = post(run_dir)
    assert record.yplus["body"] == pytest.approx(87.5)


def test_post_survives_the_gates_non_numeric_detail(run_dir: Path) -> None:
    """The gate records which basis it judged on as a string. ResultRecord.yplus
    is a mapping of floats, so post has to select rather than coerce - it used
    to float() everything the gate returned."""
    _mark_solve(run_dir, "ok")
    _write_area_yplus(run_dir, "body", 87.5)
    record = post(run_dir)
    assert all(isinstance(v, float) for v in record.yplus.values())
    assert "weighting" not in record.yplus
    assert not any(k.endswith("_facemean_yplus") for k in record.yplus)


def test_result_record_carries_the_verdict_not_just_a_boolean(run_dir: Path) -> None:
    """result.json is what a sweep, an aggregate or a human actually reads, so
    the tri-state has to reach it. A boolean alone cannot distinguish a run
    that passed a plateau test from one nobody tested."""
    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    assert record.verdict in ("converged", "not_converged", "not_judged")
    assert record.converged is (record.verdict == "converged")

    import json
    stored = json.loads((run_dir / "results" / "result.json").read_text())
    assert stored["verdict"] == record.verdict


def test_post_writes_the_component_force_plots(run_dir: Path) -> None:
    """The per-patch function objects have been writing to disk since the
    'eleven force traces' commit with nothing reading them. The attribution
    plot is what makes them useful, so post has to produce it."""
    _mark_solve(run_dir, "ok")
    for patch, cl in (("body", -1.0), ("stilts", -0.05)):
        d = run_dir / "postProcessing" / f"forceCoeffs_{patch}" / "0"
        d.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"{t}\t0.3\t{cl + 0.01 * (t % 7)}" for t in range(1, 120))
        (d / "coefficient.dat").write_text(f"# Time\tCd\tCl\n{rows}\n", encoding="utf-8")

    post(run_dir)
    assert (run_dir / "results" / "components_Cl.png").exists()
    assert (run_dir / "results" / "components_Cd.png").exists()


def test_post_still_works_with_no_component_objects(run_dir: Path) -> None:
    """Runs meshed before those function objects existed have none."""
    _mark_solve(run_dir, "ok")
    post(run_dir)
    assert (run_dir / "results" / "result.json").exists()
