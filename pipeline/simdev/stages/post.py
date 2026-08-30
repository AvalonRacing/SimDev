from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from simdev.gates.convergence import check_convergence
from simdev.gates.yplus import check_y_plus
from simdev.report.plots import (
    plot_component_forces,
    plot_force_history,
    plot_residuals,
)
from simdev.report.results import ResultRecord, write_result
from simdev.run.parsers import (
    read_component_coeffs,
    read_force_coeffs,
    read_y_plus,
    read_y_plus_area,
)
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
    # The weighted number is what the gate judges; the yPlus function object's
    # own mean is unweighted and reads low wherever cell size varies across a
    # patch. Empty for runs meshed before those function objects existed, and
    # check_y_plus falls back per patch and says so.
    y_plus_gate = check_y_plus(y_plus_df, spec, read_y_plus_area(run_dir))

    mesh_status = read_status(run_dir, "mesh")
    n_cells = int(mesh_status.detail.get("n_cells", 0)) if mesh_status else 0

    results_dir = run_dir / "results"
    plot_force_history(forces, results_dir / "forces.png", convergence.window)

    # Attribution. The aggregate above says the car oscillates; only these say
    # which part of it does, and a limit cycle fed by the rear wing stalling
    # and one fed by a front tyre wake want opposite fixes. Empty on runs
    # meshed before the per-patch function objects existed.
    components = read_component_coeffs(run_dir)
    if components:
        for coefficient in ("Cl", "Cd"):
            plot_component_forces(
                components,
                results_dir / f"components_{coefficient}.png",
                convergence.window,
                coefficient,
            )
    try:
        # postProcessing/ subdirectories are named after the *function object*
        # (controlDict calls it 'residuals'), not after its type - the file
        # inside is what is named solverInfo.dat. Globbing the type silently
        # found nothing and the residual plot was never written.
        residuals = read_force_coeffs(find_latest(run_dir, "residuals/*/solverInfo.dat"))
        plot_residuals(residuals, results_dir / "residuals.png")
    except FileNotFoundError:
        pass

    record = ResultRecord(
        case_name=spec.name,
        spec_hash=spec.spec_hash(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        verdict=convergence.verdict,
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
        # Selected, not coerced. The gate's detail also carries each patch's
        # unweighted face mean and a string naming which basis it judged on,
        # and float()-ing everything it returns used to be how a new detail
        # key crashed the whole post stage.
        yplus={
            k[: -len("_avg_yplus")]: float(v)
            for k, v in y_plus_gate.detail.items()
            if k.endswith("_avg_yplus")
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
            detail={
                "cd_mean": record.cd_mean,
                "cl_mean": record.cl_mean,
                "verdict": record.verdict,
            },
        ),
    )
    return record
