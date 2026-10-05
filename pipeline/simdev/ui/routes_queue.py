"""Home (the queue), job actions, and the New Run form."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request

from simdev.config.validate import ValidationError, validate
from simdev.stages.common import load_spec
from simdev.ui import forms, runview
from simdev.ui.context import Context, back, ctx, render
from simdev.ui.queue import JobSpec, QueueError

router = APIRouter()


def _running_items(c: Context) -> list[dict[str, Any]]:
    items = []
    for job in c.queue.running():
        run_dir = Path(job.run_dir)
        try:
            max_iterations = load_spec(run_dir).solve.max_iterations
        except (FileNotFoundError, KeyError, ValueError):
            max_iterations = None
        items.append({
            "job": job,
            "stages": runview.stages(
                run_dir, running=True, since=job.started_at, forced_from=job.force_from
            ),
            "progress": runview.progress(run_dir, max_iterations),
            "max_iterations": max_iterations,
        })
    return items


def _queue_values(request: Request) -> dict[str, Any]:
    c = ctx(request)
    return {
        "running": _running_items(c),
        "queued": c.queue.queued(),
        "finished": c.queue.finished(),
        "core_budget": c.queue.setting("core_budget"),
    }


@router.get("/")
def home(request: Request, error: str | None = None, message: str | None = None):
    return render(request, "queue.html", error=error, message=message, **_queue_values(request))


@router.get("/partials/queue")
def queue_partial(request: Request):
    return render(request, "_queue.html", **_queue_values(request))


@router.post("/jobs/{job_id}/cancel")
def cancel(request: Request, job_id: int):
    ctx(request).worker.cancel(job_id)
    return back("/")


@router.post("/jobs/{job_id}/move")
def move(request: Request, job_id: int, direction: str = Form(...)):
    try:
        ctx(request).queue.move(job_id, -1 if direction == "up" else 1)
    except QueueError as error:
        return back("/", error=str(error))
    return back("/")


# --- New Run ---------------------------------------------------------------


def _form_page(
    request: Request,
    *,
    pair: tuple[str, str] | None = None,
    profile: str = forms.DEFAULT_PROFILE,
    run_name: str | None = None,
    values: dict[str, Any] | None = None,
    job=None,
    note: str | None = None,
    error: str | None = None,
    status_code: int = 200,
):
    c = ctx(request)
    pairs = c.library.pairs()
    if not pairs:
        return render(request, "new_run.html", pairs=[], error=error, status_code=status_code)
    if pair not in pairs:
        pair = pairs[0]
    design, state = pair
    rows: list[dict[str, Any]] = []
    try:
        base = forms.resolve_run(c.library, c.config.case_path, design, state, profile, {})
        rows = forms.field_rows(base, values)
    except forms.FormError as problem:
        error = error or str(problem)
    if run_name is None:
        run_name = forms.default_run_name(c.config.runs_root, c.queue, design, state, profile)
    return render(
        request, "new_run.html", pairs=pairs, selected=pair, profiles=forms.profiles(),
        profile=profile, run_name=run_name, fields=rows, job=job, state=state,
        note=note if note is not None else (job.note if job else None),
        error=error, status_code=status_code,
    )


@router.get("/runs/new")
def new_run(request: Request, pair: str | None = None,
            profile: str = forms.DEFAULT_PROFILE, job: int | None = None):
    c = ctx(request)
    if job is not None:
        existing = c.queue.get(job)
        if existing is None or existing.status != "queued":
            return back("/", error="this job has already started and can no longer be edited")
        return _form_page(
            request, pair=(existing.design, existing.state), profile=existing.profile,
            run_name=existing.run_name, values=existing.overrides, job=existing,
        )
    return _form_page(request, pair=forms.split_pair(pair), profile=profile)


@router.get("/runs/new/fields")
def new_run_fields(request: Request, pair: str = "", profile: str = forms.DEFAULT_PROFILE,
                   job_id: str = "", run_name: str = ""):
    """The part of the form that depends on the pair and profile (htmx)."""
    c = ctx(request)
    chosen = forms.split_pair(pair)
    try:
        if chosen is None:
            raise forms.FormError("choose a design and a driving state")
        base = forms.resolve_run(c.library, c.config.case_path, *chosen, profile, {})
    except forms.FormError as error:
        return render(request, "_fields.html", fields=[], run_name=run_name, error=str(error))
    if not job_id:
        run_name = forms.default_run_name(c.config.runs_root, c.queue, *chosen, profile)
    return render(request, "_fields.html", fields=forms.field_rows(base), run_name=run_name,
                  state=chosen[1], profile=profile)


def _build(c: Context, form) -> tuple[str, str, str, dict[str, Any], Any, list[str]]:
    chosen = forms.split_pair(form.get("pair"))
    if chosen is None:
        raise forms.FormError("choose a design and a driving state")
    design, state = chosen
    profile = form.get("profile") or forms.DEFAULT_PROFILE
    base = forms.resolve_run(c.library, c.config.case_path, design, state, profile, {})
    overrides = forms.changed_overrides(form, base)
    spec = forms.resolve_run(c.library, c.config.case_path, design, state, profile, overrides)
    warnings = validate(spec)
    return design, state, profile, overrides, spec, warnings


@router.post("/runs/preview")
async def preview(request: Request):
    form = await request.form()
    try:
        _, _, _, overrides, spec, warnings = _build(ctx(request), form)
    except (forms.FormError, ValidationError) as error:
        return render(request, "_preview.html", error=str(error))
    return render(
        request, "_preview.html", numbers=forms.preview_numbers(spec), overrides=overrides,
        warnings=warnings, spec_json=json.dumps(spec.model_dump(mode="json"), indent=2),
    )


@router.post("/runs")
async def queue_run(request: Request):
    form = await request.form()
    c = ctx(request)
    job_id = int(form["job_id"]) if form.get("job_id") else None
    name = str(form.get("run_name") or "").strip()
    note = str(form.get("note") or "").strip()[:2000] or None
    try:
        if job_id is not None:
            existing = c.queue.get(job_id)
            if existing is None or existing.status != "queued":
                raise QueueError("this job has already started and can no longer be edited")
        design, state, profile, overrides, spec, _ = _build(c, form)
        budget = c.queue.setting("core_budget")
        if spec.solve.n_ranks > budget:
            # The worker would never find room for it and would skip it
            # forever without a word, so say so now.
            raise forms.FormError(
                f"this run needs {spec.solve.n_ranks} cores but the core budget is "
                f"{budget}; lower solve.n_ranks or raise the budget on the System page"
            )
        run_dir = forms.check_run_name(name, c.config.runs_root, c.queue, exclude_id=job_id)
        job_spec = JobSpec(
            run_name=name, run_dir=str(run_dir), case_path=str(c.config.case_path),
            design=design, state=state, profile=profile, n_ranks=spec.solve.n_ranks,
            overrides=overrides, note=note,
        )
        if job_id is None:
            c.queue.enqueue(job_spec)
        else:
            c.queue.update(job_id, job_spec)
    except (forms.FormError, ValidationError, QueueError) as error:
        return _form_page(
            request,
            pair=forms.split_pair(form.get("pair")),
            profile=form.get("profile") or forms.DEFAULT_PROFILE,
            run_name=name,
            note=note,
            values={f.key: form.get(f.key) for f in forms.FIELDS if form.get(f.key)},
            job=c.queue.get(job_id) if job_id else None,
            error=str(error),
            status_code=400,
        )
    return back("/", message=f"{'saved' if job_id else 'queued'} {name}")
