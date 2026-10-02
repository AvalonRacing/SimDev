from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from simdev.run.status import StageStatus, write_status
from simdev.ui import runview
from simdev.ui.housekeeping import delete_run, strip_mesh

FIXTURES = Path(__file__).parent / "fixtures"

SOLVER_INFO = """# Solver information
# Time          \tU_solver        \tUx_initial      \tUx_final        \tp_solver        \tp_initial       \tp_final
1               \tsmoothSolver\t1.0e+00\t6.8e-02\tGAMG\t1.0e+00\t4.9e-02
2               \tsmoothSolver\t1.2e-02\t9.2e-04\tGAMG\t7.1e-02\t3.4e-03
"""

SIMPLEFOAM_LOG = """Time = 1

ExecutionTime = 10.0 s  ClockTime = 10 s

Time = 2

ExecutionTime = 25.0 s  ClockTime = 25 s

Time = 3

ExecutionTime = 40.0 s  ClockTime = 40 s
"""


@pytest.fixture
def run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "r1"
    coeffs = run_dir / "postProcessing" / "forceCoeffs" / "0"
    coeffs.mkdir(parents=True)
    shutil.copy(FIXTURES / "coefficient.dat", coeffs / "coefficient.dat")
    body = run_dir / "postProcessing" / "forceCoeffs_Body" / "0"
    body.mkdir(parents=True)
    shutil.copy(FIXTURES / "coefficient.dat", body / "coefficient.dat")
    residuals = run_dir / "postProcessing" / "residuals" / "0"
    residuals.mkdir(parents=True)
    (residuals / "solverInfo.dat").write_text(SOLVER_INFO)
    logs = run_dir / "logs"
    logs.mkdir()
    (logs / "log.simpleFoam").write_text(SIMPLEFOAM_LOG)
    (logs / "log.snappyHexMesh").write_text(
        "\n".join([f"line {i}" for i in range(40)] + ["--> FOAM FATAL ERROR: bad"] +
                  [f"after {i}" for i in range(40)])
    )
    images = run_dir / "results" / "images" / "cp_x"
    images.mkdir(parents=True)
    (images / "cp_x_01_+0.100.png").write_bytes(b"png")
    (run_dir / "results" / "forces.png").write_bytes(b"png")
    return run_dir


def test_force_series_is_incremental(run: Path) -> None:
    full = runview.force_series(run)
    assert full["iteration"] == [1, 2, 3, 4, 5]
    assert full["Cd"][0] == pytest.approx(0.512)
    assert runview.force_series(run, after=3)["iteration"] == [4, 5]


def test_force_series_of_an_unsolved_run_is_empty(tmp_path: Path) -> None:
    assert runview.force_series(tmp_path) == {"iteration": [], "Cd": [], "Cl": []}


def test_component_series(run: Path) -> None:
    assert list(runview.component_series(run)) == ["Body"]


def test_residual_series_keeps_initial_residuals(run: Path) -> None:
    series = runview.residual_series(run)
    assert series["iteration"] == [1, 2]
    assert set(series) == {"iteration", "Ux", "p"}
    assert series["p"][1] == pytest.approx(7.1e-2)


def test_progress_from_the_solver_log(run: Path) -> None:
    p = runview.progress(run, max_iterations=103)
    assert p["iteration"] == 3
    assert p["seconds_per_iteration"] == pytest.approx(15.0)
    assert p["eta_seconds"] == pytest.approx(100 * 15.0)


def test_stages_of_a_running_job(run: Path) -> None:
    write_status(run, StageStatus(stage="prepare", state="ok", input_hash="h"))
    write_status(run, StageStatus(stage="mesh", state="ok", input_hash="h"))
    views = runview.stages(run, running=True, since=0.0)
    assert [v.state for v in views] == ["ok", "ok", "running", "pending", "pending"]


def test_a_stale_failure_is_not_shown_while_retrying(run: Path) -> None:
    write_status(run, StageStatus(stage="prepare", state="ok", input_hash="h"))
    write_status(run, StageStatus(stage="mesh", state="failed", input_hash="h", reasons=["x"]))
    old = (run / "status" / "mesh.json")
    os.utime(old, (1000, 1000))
    views = runview.stages(run, running=True, since=2000.0)
    assert views[1].state == "running"


def test_forced_stages_before_the_restart_count_as_pending(run: Path) -> None:
    for stage in ("prepare", "mesh", "solve"):
        write_status(run, StageStatus(stage=stage, state="ok", input_hash="h"))
        os.utime(run / "status" / f"{stage}.json", (1000, 1000))
    views = runview.stages(run, running=True, since=2000.0, forced_from="mesh")
    assert [v.state for v in views][:3] == ["ok", "running", "pending"]


def test_errors_include_fatal_log_lines_with_context(run: Path) -> None:
    write_status(run, StageStatus(stage="mesh", state="failed", input_hash="h",
                                  reasons=["snappyHexMesh failed"]))
    found = runview.errors(run)
    assert found[0].source == "stage mesh"
    log_error = next(e for e in found if e.source == "logs/log.snappyHexMesh")
    assert "FOAM FATAL ERROR" in log_error.message
    assert len(log_error.context) == 51


def test_logs_and_tail(run: Path) -> None:
    assert runview.log_names(run) == ["log.simpleFoam", "log.snappyHexMesh"]
    assert runview.tail(run, "log.snappyHexMesh", lines=2) == ["after 38", "after 39"]
    with pytest.raises(KeyError):
        runview.tail(run, "../../etc/passwd")


def test_images_grouped_by_folder(run: Path) -> None:
    found = runview.images(run)
    assert found["summary"] == ["results/forces.png"]
    assert found["groups"] == {"cp_x": ["results/images/cp_x/cp_x_01_+0.100.png"]}


@pytest.mark.parametrize("rel", ["../r2/x", "/etc/passwd", "results/../../r2/x", "results", "a\x00b"])
def test_safe_file_refuses_anything_outside_or_not_a_file(run: Path, rel: str) -> None:
    (run.parent / "r2").mkdir()
    (run.parent / "r2" / "x").write_text("secret")
    with pytest.raises(KeyError):
        runview.safe_file(run, rel)


def test_safe_file_serves_a_file_inside(run: Path) -> None:
    assert runview.safe_file(run, "results/forces.png") == (run / "results" / "forces.png").resolve()


def test_list_runs(run: Path) -> None:
    (run / "caseSpec.json").write_text("{}")
    (run.parent / "not-a-run").mkdir()
    assert runview.list_runs(run.parent) == [run]


def test_strip_mesh_keeps_results(run: Path) -> None:
    for folder in ("processor0/0", "constant/polyMesh", "100", "0"):
        (run / folder).mkdir(parents=True)
        (run / folder / "f").write_bytes(b"x" * 10)
    strip_mesh(run)
    assert not (run / "processor0").exists()
    assert not (run / "constant" / "polyMesh").exists()
    assert not (run / "100").exists()
    assert (run / "0").exists()
    assert (run / "results" / "forces.png").exists()
    assert (run / "postProcessing").exists()


def test_delete_run(run: Path) -> None:
    delete_run(run)
    assert not run.exists()


def test_force_series_with_truncated_line_drops_nan(tmp_path: Path) -> None:
    """A solver mid-write leaves a truncated last line; pandas pads with NaN."""
    import json
    import shutil
    run_dir = tmp_path / "runs" / "r1"
    coeffs = run_dir / "postProcessing" / "forceCoeffs" / "0"
    coeffs.mkdir(parents=True)
    # Copy fixture and add a truncated line
    shutil.copy(FIXTURES / "coefficient.dat", coeffs / "coefficient.dat")
    with open(coeffs / "coefficient.dat", "a") as f:
        f.write("6               0.3")  # Truncated, will have NaN columns
    result = runview.force_series(run_dir)
    # Should drop the truncated row, keeping only [1..5]
    assert result["iteration"] == [1, 2, 3, 4, 5]
    # Verify json.dumps works (no bare NaN values)
    json.dumps(result, allow_nan=False)


def test_residual_series_with_truncated_line_drops_nan(tmp_path: Path) -> None:
    """Residual series should also drop rows with NaN."""
    import json
    run_dir = tmp_path / "runs" / "r1"
    residuals = run_dir / "postProcessing" / "residuals" / "0"
    residuals.mkdir(parents=True)
    solver_info = SOLVER_INFO + "3               \tpartialSolver\t1.1e-03\t9.9e-05"  # Truncated
    (residuals / "solverInfo.dat").write_text(solver_info)
    result = runview.residual_series(run_dir)
    # Should drop the truncated row
    assert result["iteration"] == [1, 2]
    # Verify json.dumps works
    json.dumps(result, allow_nan=False)


def test_safe_file_with_nul_byte_raises_keyerror(run: Path) -> None:
    """NUL byte in path should raise KeyError, not ValueError."""
    with pytest.raises(KeyError):
        runview.safe_file(run, "a\x00b")


def test_stages_with_half_written_status_file(tmp_path: Path) -> None:
    """Half-written (truncated) JSON status file should not raise."""
    run_dir = tmp_path / "runs" / "r1"
    status_dir = run_dir / "status"
    status_dir.mkdir(parents=True)
    (status_dir / "prepare.json").write_text("{")  # Truncated JSON
    # Should not raise; prepare should be pending or treated as missing
    views = runview.stages(run_dir, running=True, since=0.0)
    assert len(views) == 5


def test_errors_with_half_written_status_file(tmp_path: Path) -> None:
    """Corrupted status file in errors() should not raise."""
    run_dir = tmp_path / "runs" / "r1"
    status_dir = run_dir / "status"
    status_dir.mkdir(parents=True)
    (status_dir / "mesh.json").write_text("{")  # Truncated JSON
    # Should not raise; errors() should return empty or skip the stage
    errors_found = runview.errors(run_dir)
    assert isinstance(errors_found, list)
