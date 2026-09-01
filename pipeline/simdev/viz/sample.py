"""Get the slices and surfaces out of a decomposed run, in parallel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from simdev.config.schema import CaseSpec
from simdev.geometry.roles import traits
from simdev.render.context import solver_function_names
from simdev.render.render import env
from simdev.run.runner import Runner, StageError

SAMPLE_DICT = "system/sampleSurfaces"


def render_sample_dict(
    spec: CaseSpec,
    out_dir: Path,
    slices: Sequence[Mapping[str, Any]],
) -> Path:
    """Write system/sampleSurfaces for this run's plane list.

    Takes no Domain and no geometry files: the dictionary needs only the
    patch lists and the function-object names, and both follow from the spec
    alone. That is what lets the images stage render this from a run
    directory without rebuilding the case.
    """
    context = {
        "spec": spec,
        "force_patches": [
            p.name for p in spec.geometry.patches if traits(p.role).in_forces
        ],
        "solver_function_names": solver_function_names(spec),
        "slices": list(slices),
    }

    target = Path(out_dir) / SAMPLE_DICT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        env().get_template("sampleSurfaces.jinja").render(**context),
        encoding="utf-8",
    )
    return target


def run_sampling(run_dir: Path, n_ranks: int) -> Path:
    """Cut every plane and patch at the latest written time.

    Returns the directory the surfaces landed in. Raises rather than guessing
    when nothing was written: an empty sample directory after a successful
    postProcess means the fields it wanted were not in the time directory,
    which is a different problem from a failed command and wants a different
    fix.
    """
    run_dir = Path(run_dir)
    before = _force_coeff_times(run_dir)

    Runner(run_dir).run_parallel(
        ["postProcess", "-dict", SAMPLE_DICT, "-latestTime"],
        n_ranks,
        name="postProcess.sample",
    )

    # Belt and braces on the -dict merge. If a solve-time object slipped
    # through the suppression list it would have written here, and a silently
    # corrupted force history is worth one directory listing to rule out.
    if _force_coeff_times(run_dir) != before:
        raise StageError([
            "the sampling pass wrote new forceCoeffs output, which means a "
            "solve-time function object was not disabled. The force history "
            "may now be wrong: check system/sampleSurfaces against "
            "render/context.py::solver_function_names before trusting "
            "results/report.tsv"
        ])

    roots = sorted((run_dir / "postProcessing" / "surfaces").glob("*"))
    if not roots:
        raise StageError([
            "postProcess wrote no surfaces. The usual cause is that the "
            "latest time directory holds no pMean/UMean - a run stopped "
            "before fieldAverage's timeStart has no averaged fields to cut"
        ])
    return roots[-1]


def _force_coeff_times(run_dir: Path) -> set[str]:
    root = Path(run_dir) / "postProcessing" / "forceCoeffs"
    return {p.name for p in root.glob("*")} if root.exists() else set()


def find_sample(samples_root: Path, name: str) -> Path | None:
    """The .vtp or .vtk a named surface produced, whichever the build wrote."""
    for suffix in (".vtp", ".vtk"):
        candidate = Path(samples_root) / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None
