"""Pictures of a solved run, framed identically to every other run's."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from simdev.report.results import read_result
from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, write_status
from simdev.stages.common import load_spec
from simdev.viz.plan import build_render_plan
from simdev.viz.sample import find_sample, render_sample_dict, run_sampling
from simdev.viz.sheet import write_contact_sheet
from simdev.viz.views import DEFAULT_VIEWS_PATH, load_views

STAGE = "images"

RENDERER = Path(__file__).resolve().parent.parent / "viz" / "pv_render.py"


def _render(interpreter: str, plan_path: Path) -> dict[str, Any]:
    """Hand the plan to the ParaView interpreter and read back what it did."""
    result = subprocess.run(
        [interpreter, str(RENDERER), str(plan_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StageError([
            f"the renderer failed under {interpreter}",
            result.stderr.strip()[-2000:] or "no stderr",
            "run 'simdev doctor' to check the interpreter can import "
            "paraview.simple. Note that pvpython and pvbatch are NOT the "
            "answer here - they hang on this machine",
        ])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise StageError([
            "the renderer wrote no summary line",
            result.stdout.strip()[-2000:] or "no stdout",
        ]) from None


def _window(run_dir: Path) -> tuple[int, int]:
    """The averaging window, from results/result.json - NOT from post's
    status detail.

    post's StageStatus.detail carries only cd_mean, cl_mean and verdict (see
    stages/post.py). window_start/window_end live on the ResultRecord that
    `post` writes to results/result.json, which is the authoritative record.
    Falls back to (0, 0) when post has not run: images does not require post
    to have run, and a missing window is a stamp of "0-0", not a StageError.
    """
    try:
        record = read_result(run_dir)
    except FileNotFoundError:
        return (0, 0)
    return (record.window_start, record.window_end)


def images(
    run_dir: Path,
    force: bool = False,
    axes: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Sample the run, then draw it. Never gates: these are pictures."""
    run_dir = Path(run_dir)
    spec = load_spec(run_dir)

    solve_status = read_status(run_dir, "solve")
    if solve_status is None:
        raise StageError(["stage 'solve' has not been run"])
    if solve_status.state == "failed":
        raise StageError(["stage 'solve' failed outright; there is nothing to draw"])
    # 'gate_failed' is fine, exactly as in post: a run that did not plateau
    # still has a flow field worth looking at, and the stamp says so.

    prepare_status = read_status(run_dir, "prepare")
    if prepare_status is None or "datum" not in prepare_status.detail:
        raise StageError([
            "no datum in status/prepare.json: this run was prepared before "
            "the images stage existed. Re-run 'simdev prepare --force' on it "
            "- it re-renders dictionaries and re-measures geometry, it does "
            "not touch the mesh or the solution"
        ])

    views_path = Path(spec.post.views) if spec.post.views else DEFAULT_VIEWS_PATH
    views = load_views(views_path)
    datum = tuple(prepare_status.detail["datum"])

    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)

    window = _window(run_dir)

    plan = build_render_plan(
        u_inf=spec.flow.u_inf,
        views=views,
        datum=datum,
        frame=prepare_status.detail.get("corner_frame"),
        samples_dir=results / "samples",
        images_dir=results / "images",
        stamp={
            "run": run_dir.name,
            "spec_hash": spec.spec_hash()[:8],
            "window": f"{window[0]}-{window[1]}",
            "mean": bool(spec.solve.average_fields),
        },
    )

    if axes:
        plan["slices"] = [s for s in plan["slices"] if s["axis"] in set(axes)]
    if fields:
        wanted = set(fields)
        for entry in (*plan["slices"], *plan["surfaces"]):
            entry["images"] = [i for i in entry["images"] if i["field"] in wanted]

    reasons: list[str] = []
    lo, hi = prepare_status.detail["geometry_bounds"]
    for index, axis in enumerate("xyz"):
        if axis not in views.axes:
            continue
        offsets = views.axes[axis].offsets()
        low, high = datum[index] + offsets[0], datum[index] + offsets[-1]
        if low > lo[index] or high < hi[index]:
            # NOT silently extended. An auto-fitted range produces a different
            # picture wearing the same file name, and then two runs that look
            # comparable are not.
            reasons.append(
                f"the {axis} slice range {low:+.3f}..{high:+.3f} does not "
                f"cover the geometry {lo[index]:+.3f}..{hi[index]:+.3f}; "
                f"widen planes.{axis} in {views_path.name} if that matters"
            )

    # Written before anything runs, so the dictionary is on disk to read even
    # when the OpenFOAM call is what fails.
    render_sample_dict(spec, run_dir, plan["slices"])

    sampled_at = time.monotonic()
    samples_root = run_sampling(run_dir, spec.solve.n_ranks)
    sample_seconds = time.monotonic() - sampled_at

    for entry in plan["slices"]:
        found = find_sample(samples_root, entry["name"])
        entry["sample"] = str(found) if found else None

    # ONE combined 'vehicle' surface, not one per patch (Task 13). The seven
    # surface views are all "overall" views of the whole car, so every
    # surface entry gets the same sample - the merged patch surface that
    # system/sampleSurfaces (viz/sample.py::render_sample_dict) writes as a
    # single `patch`-type surface listing every force patch. Picking whichever
    # per-patch sample sorted first would draw one part of the car - the
    # bodywork alone, say - labelled as the entire vehicle.
    found = find_sample(samples_root, "vehicle")
    for entry in plan["surfaces"]:
        entry["sample"] = str(found) if found else None

    plan_path = results / "render_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (results / "views.yaml").write_text(
        views_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    drawn_at = time.monotonic()
    summary = _render(spec.post.paraview_python, plan_path)
    render_seconds = time.monotonic() - drawn_at

    if summary.get("missing"):
        reasons.append(
            f"{len(summary['missing'])} surfaces produced no sample and were "
            f"not drawn (first: {summary['missing'][0]})"
        )

    record = {
        "run": run_dir.name,
        "spec_hash": spec.spec_hash(),
        "views_digest": views.digest,
        "datum": list(datum),
        "images_written": summary.get("written", 0),
        "sample_seconds": round(sample_seconds, 1),
        "render_seconds": round(render_seconds, 1),
        "reasons": reasons,
    }
    (results / "images.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_contact_sheet(results, plan, record)

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            # Always ok when it completed. These are pictures; a short slice
            # range is a note, not a failed run.
            state="ok",
            input_hash=spec.spec_hash(),
            reasons=reasons,
            detail={
                "images_written": record["images_written"],
                "views_digest": views.digest,
                "sample_seconds": record["sample_seconds"],
                "render_seconds": record["render_seconds"],
            },
        ),
    )
    return record
