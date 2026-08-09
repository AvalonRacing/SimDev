from __future__ import annotations

from pathlib import Path

from simdev.gates.base import GateResult
from simdev.gates.mesh_quality import check_mesh_quality
from simdev.run.parsers import parse_check_mesh, parse_layer_summary
from simdev.run.runner import Runner, StageError
from simdev.run.status import StageStatus, should_skip, write_status
from simdev.stages.common import load_spec, require_stage

STAGE = "mesh"


def mesh(run_dir: Path, force: bool = False, runner: Runner | None = None) -> GateResult:
    run_dir = Path(run_dir)
    require_stage(run_dir, "prepare")
    spec = load_spec(run_dir)

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        return GateResult(passed=True, reasons=[], detail={"skipped": "unchanged input"})

    runner = runner or Runner(run_dir)

    runner.run(["blockMesh"], name="blockMesh")
    runner.run(["surfaceFeatureExtract"], name="surfaceFeatureExtract")
    # -force removes any existing processor* directories. Reaching this line
    # means we have already decided to (re)mesh, and decomposePar aborts
    # rather than overwrite - so without it the second run of any stage in an
    # existing run directory fails, including one asked for with --force.
    runner.run(["decomposePar", "-force"], name="decomposePar")
    snappy = runner.run_parallel(
        ["snappyHexMesh", "-overwrite"], spec.solve.n_ranks, name="snappyHexMesh"
    )
    check = runner.run_parallel(["checkMesh"], spec.solve.n_ranks, name="checkMesh")

    layers = parse_layer_summary(snappy.log_path.read_text(encoding="utf-8"))
    quality = parse_check_mesh(check.log_path.read_text(encoding="utf-8"))
    gate = check_mesh_quality(quality, layers, spec)

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok" if gate.passed else "gate_failed",
            input_hash=spec.spec_hash(),
            reasons=gate.reasons,
            detail=dict(gate.detail),
        ),
    )

    if not gate.passed:
        raise StageError(
            ["mesh quality gate failed", *gate.reasons, "refusing to start the solve"]
        )

    return gate
