from __future__ import annotations

import shutil
from pathlib import Path

from simdev.gates.base import GateResult
from simdev.gates.mesh_quality import check_mesh_quality
from simdev.run.parsers import parse_check_mesh, parse_layer_summary, parse_renumber_band
from simdev.run.runner import Runner, StageError
from simdev.geometry.roles import traits
from simdev.run.status import StageStatus, read_status, should_skip, write_status
from simdev.stages.common import load_spec, require_stage

STAGE = "mesh"


def _requested_layers(run_dir: Path, spec) -> dict[str, int]:
    """What prepare actually asked snappy for, per patch.

    Read back from the prepare record rather than recomputed: the ground's
    layer count depends on the domain and its refinement regions, which this
    stage does not build. Falls back to the patch-only calculation for run
    directories prepared before this was recorded.
    """
    status = read_status(run_dir, "prepare")
    recorded = (status.detail.get("requested_layers") if status else None) or {}
    if recorded:
        return {name: int(n) for name, n in recorded.items()}
    return {
        p.name: spec.n_layers_for(p)
        for p in spec.geometry.patches
        if traits(p.role).refinement == "high"
    }


def restore_zero_dir(run_dir: Path, n_ranks: int) -> None:
    """Re-seed processor*/0 from the rendered fields, after snappy.

    decomposePar has to run before snappyHexMesh, so the fields it writes
    describe the *background* mesh. They carry no patchField for the surfaces
    snappy is about to create, and their processor-boundary entries name a
    decomposition snappy then rebalances away. simpleFoam reads them and
    aborts with "Cannot find patchField entry for body".

    OpenFOAM's own parallel tutorials fix this with `restore0Dir -processor`
    after meshing. Same idea without the shell dependency: every rendered
    field is uniform, so each processor can take the case-level file verbatim
    and let OpenFOAM match patches by name and synthesise the
    processor-boundary entries from the mesh it actually has.
    """
    source = run_dir / "0"
    for rank in range(n_ranks):
        target = run_dir / f"processor{rank}" / "0"
        target.mkdir(parents=True, exist_ok=True)
        for field in sorted(source.iterdir()):
            if field.is_file():
                shutil.copy2(field, target / field.name)


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
    restore_zero_dir(run_dir, spec.solve.n_ranks)

    # After restore_zero_dir, never before it. renumberMesh renumbers the
    # fields alongside the mesh, so it has to read them - and until the zero
    # directory is re-seeded those are the *background* mesh's fields, with no
    # patchField for the surfaces snappy just created. It would abort on
    # exactly the "Cannot find patchField entry for body" that restore_zero_dir
    # exists to prevent.
    band = None
    if spec.mesh.renumber:
        renumber = runner.run_parallel(
            ["renumberMesh", "-overwrite"], spec.solve.n_ranks, name="renumberMesh"
        )
        band = parse_renumber_band(renumber.log_path.read_text(encoding="utf-8"))

    check = runner.run_parallel(["checkMesh"], spec.solve.n_ranks, name="checkMesh")

    layers = parse_layer_summary(snappy.log_path.read_text(encoding="utf-8"))
    quality = parse_check_mesh(check.log_path.read_text(encoding="utf-8"))
    gate = check_mesh_quality(quality, layers, spec, _requested_layers(run_dir, spec))

    detail = dict(gate.detail)
    if band is not None:
        # Recorded so a slow solve can be told from a badly ordered mesh
        # without re-running anything. A band that barely moved means
        # renumbering found nothing to fix; the solve rate then has another
        # cause and this is not it.
        detail["renumber_band"] = {"before": band[0], "after": band[1]}

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok" if gate.passed else "gate_failed",
            input_hash=spec.spec_hash(),
            reasons=gate.reasons,
            detail=detail,
        ),
    )

    if not gate.passed:
        raise StageError(
            ["mesh quality gate failed", *gate.reasons, "refusing to start the solve"]
        )

    return gate
