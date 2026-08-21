from __future__ import annotations

from pathlib import Path

from simdev.gates.convergence import ConvergenceResult, check_convergence
from simdev.run.parsers import read_force_coeffs
from simdev.run.runner import Runner
from simdev.run.status import StageStatus, should_skip, write_status
from simdev.stages.common import find_latest, load_spec, require_stage

STAGE = "solve"


def solve(
    run_dir: Path, force: bool = False, runner: Runner | None = None
) -> ConvergenceResult:
    run_dir = Path(run_dir)
    require_stage(run_dir, "mesh")
    spec = load_spec(run_dir)

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        from simdev.run.status import read_status

        previous = read_status(run_dir, STAGE)
        assert previous is not None
        return ConvergenceResult(
            converged=True,
            reasons=["skipped: unchanged input"],
            means={k: float(v) for k, v in previous.detail.items() if k.endswith("_mean")},
        )

    runner = runner or Runner(run_dir)
    # The mesh stage already decomposed; reuse that decomposition.
    runner.run_parallel(["simpleFoam"], spec.solve.n_ranks, name="simpleFoam")

    forces = read_force_coeffs(find_latest(run_dir, "forceCoeffs/*/coefficient.dat"))
    result = check_convergence(forces, spec)

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok" if result.converged else "gate_failed",
            input_hash=spec.spec_hash(),
            reasons=result.reasons,
            detail={
                **{f"{k}_mean": v for k, v in result.means.items()},
                **{f"{k}_std": v for k, v in result.stds.items()},
                # std/|mean|, recorded rather than left to be recomputed: on a
                # limit-cycle case this is how wide the cycle is, and it is
                # the number that says whether a delta between two runs is a
                # design effect or just where each one stopped.
                **{f"{k}_amplitude": v for k, v in result.amplitudes.items()},
                "n_iterations": result.n_iterations,
                "window_start": result.window[0],
                "window_end": result.window[1],
            },
        ),
    )

    # A non-converged run is recorded and flagged, never silently passed and
    # never fatal: the post stage still reports the means, marked non-converged.
    return result
