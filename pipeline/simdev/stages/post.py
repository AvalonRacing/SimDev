from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from simdev.gates.convergence import check_convergence
from simdev.gates.yplus import check_y_plus
from simdev.report.plots import plot_force_history, plot_residuals
from simdev.report.results import ResultRecord, write_result
from simdev.run.parsers import read_force_coeffs, read_y_plus
from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, should_skip, write_status
from simdev.stages.common import find_latest, load_spec

STAGE = "post"


def post(run_dir: Path, force: bool = False) -> ResultRecord:
    run_dir = Path(run_dir)
    spec = load_spec(run_dir)

    solve_status = read_status(run_dir, "solve")
    if solve_status is None:
        raise StageError(["stage 'solve' has not been run"])
    if solve_status.state == "failed":
        raise StageError(
            ["stage 'solve' failed outright; there is nothing to post-process"]
        )
    # 'gate_failed' is acceptable: a non-converged run still produced numbers.

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        from simdev.report.results import read_result

        return read_result(run_dir)

    forces = read_force_coeffs(find_latest(run_dir, "forceCoeffs/*/coefficient.dat"))
    convergence = check_convergence(forces, spec)

    y_plus_df = read_y_plus(find_latest(run_dir, "yPlus/*/yPlus.dat"))
    y_plus_gate = check_y_plus(y_plus_df, spec)

    mesh_status = read_status(run_dir, "mesh")
    n_cells = int(mesh_status.detail.get("n_cells", 0)) if mesh_status else 0

    results_dir = run_dir / "results"
    plot_force_history(forces, results_dir / "forces.png", convergence.window)
    try:
        residuals = read_force_coeffs(find_latest(run_dir, "solverInfo/*/solverInfo.dat"))
        plot_residuals(residuals, results_dir / "residuals.png")
    except FileNotFoundError:
        pass

    record = ResultRecord(
        case_name=spec.name,
        spec_hash=spec.spec_hash(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        converged=convergence.converged,
        cd_mean=convergence.means.get("Cd", float("nan")),
        cd_std=convergence.stds.get("Cd", float("nan")),
        cl_mean=convergence.means.get("Cl", float("nan")),
        cl_std=convergence.stds.get("Cl", float("nan")),
        window_start=convergence.window[0],
        window_end=convergence.window[1],
        n_iterations=convergence.n_iterations,
        n_cells=n_cells,
        yplus_passed=y_plus_gate.passed,
        yplus={
            k.replace("_avg_yplus", ""): float(v)
            for k, v in y_plus_gate.detail.items()
        },
        reasons=[*convergence.reasons, *y_plus_gate.reasons],
    )
    write_result(run_dir, record)

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok" if y_plus_gate.passed else "gate_failed",
            input_hash=spec.spec_hash(),
            reasons=record.reasons,
            detail={"cd_mean": record.cd_mean, "cl_mean": record.cl_mean},
        ),
    )
    return record
