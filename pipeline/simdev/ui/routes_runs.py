"""Runs: the list, one run's page, its live data, files and actions."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse

from simdev.ui import forms, runview
from simdev.ui.context import back, ctx, render
from simdev.ui.housekeeping import delete_run, strip_mesh
from simdev.ui.jobrun import STAGES
from simdev.ui.queue import QueueError

router = APIRouter()


def _run_dir(request: Request, name: str) -> Path:
    if not forms.RUN_NAME.fullmatch(name) or ".." in name:
        raise HTTPException(status_code=404)
    path = ctx(request).config.runs_root / name
    if not path.is_dir():
        raise HTTPException(status_code=404)
    return path


def _active(job) -> bool:
    return job is not None and job.status in ("queued", "running")


def _stage_views(run_dir: Path, job):
    return runview.stages(
        run_dir,
        running=job is not None and job.status == "running",
        since=job.started_at if job is not None else None,
        forced_from=job.force_from if job is not None else None,
    )


@router.get("/runs")
def runs(request: Request, error: str | None = None, message: str | None = None):
    c = ctx(request)
    rows = []
    for run_dir in runview.list_runs(c.config.runs_root):
        job = c.queue.latest_for(run_dir.name)
        rows.append({"name": run_dir.name, "job": job, "stages": _stage_views(run_dir, job)})
    return render(request, "runs.html", rows=rows, error=error, message=message)


@router.get("/runs/{name}")
def run_page(request: Request, name: str, error: str | None = None, message: str | None = None):
    run_dir = _run_dir(request, name)
    job = ctx(request).queue.latest_for(name)
    stages = _stage_views(run_dir, job)
    failed = (job is not None and job.status == "failed") or any(s.state == "failed" for s in stages)
    return render(
        request, "run.html", name=name, job=job, stages=stages,
        errors=runview.errors(run_dir) if failed else [],
        images=runview.images(run_dir), logs=runview.log_names(run_dir),
        live=job is not None and job.status == "running", stage_names=STAGES,
        active=_active(job), error=error, message=message,
    )


@router.get("/runs/{name}/partials/stages")
def stages_partial(request: Request, name: str):
    run_dir = _run_dir(request, name)
    job = ctx(request).queue.latest_for(name)
    return render(request, "_stages.html", name=name, stages=_stage_views(run_dir, job),
                  live=job is not None and job.status == "running")


@router.get("/api/runs/{name}/forces")
def forces(request: Request, name: str, after: int = 0):
    run_dir = _run_dir(request, name)
    return {
        "total": runview.force_series(run_dir, after),
        "components": runview.component_series(run_dir, after),
        "convergence": runview.convergence(run_dir),
    }


@router.get("/api/runs/{name}/residuals")
def residuals(request: Request, name: str, after: int = 0):
    return runview.residual_series(_run_dir(request, name), after)


@router.get("/runs/{name}/logs/{log}")
def log_tail(request: Request, name: str, log: str):
    run_dir = _run_dir(request, name)
    try:
        lines = runview.tail(run_dir, log)
    except KeyError:
        raise HTTPException(status_code=404) from None
    return render(request, "_log.html", name=name, log=log, lines=lines)


@router.get("/runs/{name}/files/{rel:path}")
def run_file(request: Request, name: str, rel: str):
    try:
        return FileResponse(runview.safe_file(_run_dir(request, name), rel))
    except KeyError:
        raise HTTPException(status_code=404) from None


def _requeue(request: Request, name: str, force_from: str | None):
    _run_dir(request, name)
    queue = ctx(request).queue
    job = queue.latest_for(name)
    if job is None:
        return back(f"/runs/{name}", error="only runs queued from this UI can be resumed here")
    try:
        queue.requeue(job.id, force_from=force_from)
    except QueueError as error:
        return back(f"/runs/{name}", error=str(error))
    return back("/", message=f"{name} queued again" + (f" from {force_from}" if force_from else ""))


@router.post("/runs/{name}/resume")
def resume(request: Request, name: str):
    return _requeue(request, name, None)


@router.post("/runs/{name}/rerun")
def rerun(request: Request, name: str, stage: str = Form(...)):
    if stage not in STAGES:
        raise HTTPException(status_code=400, detail=f"unknown stage {stage!r}")
    return _requeue(request, name, stage)


@router.post("/runs/{name}/strip")
def strip(request: Request, name: str):
    run_dir = _run_dir(request, name)
    if _active(ctx(request).queue.latest_for(name)):
        return back(f"/runs/{name}", error="the run is still queued or running")
    freed = strip_mesh(run_dir)
    return back(f"/runs/{name}", message=f"mesh stripped, {freed / 1e9:.1f} GB freed")


@router.post("/runs/{name}/delete")
def delete(request: Request, name: str):
    run_dir = _run_dir(request, name)
    if _active(ctx(request).queue.latest_for(name)):
        return back(f"/runs/{name}", error="the run is still queued or running")
    delete_run(run_dir)
    return back("/runs", message=f"deleted {name}")
