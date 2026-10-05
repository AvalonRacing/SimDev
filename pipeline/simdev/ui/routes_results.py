"""The results table: every run's numbers, its note and its reference."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import PlainTextResponse

from simdev.ui import forms, results
from simdev.ui.context import ctx, render
from simdev.ui.notes import write_note

router = APIRouter()


def valid_run_name(name: str) -> bool:
    return bool(forms.RUN_NAME.fullmatch(name)) and ".." not in name


def _table(request: Request):
    rows = results.load_rows(ctx(request).config.runs_root)
    columns = results.columns_for(results.group_names(rows))
    return rows, columns, {r.name: r for r in rows}


def _matches(row, state: str, design: str, verdict: str, q: str) -> bool:
    text = f"{row.name} {row.note.note}".lower()
    return ((not state or row.state == state) and (not design or row.design == design)
            and (not verdict or row.verdict == verdict) and (not q or q.lower() in text))


@router.get("/results")
def results_page(request: Request, state: str = "", design: str = "", verdict: str = "",
                 q: str = ""):
    rows, columns, by_name = _table(request)
    shown = [r for r in rows if _matches(r, state, design, verdict, q)]
    return render(
        request, "results.html", rows=shown, columns=columns, by_name=by_name,
        names=[r.name for r in rows], delta=results.delta,
        states=sorted({r.state for r in rows if r.state}),
        designs=sorted({r.design for r in rows if r.design}),
        filters={"state": state, "design": design, "verdict": verdict, "q": q},
    )


@router.post("/runs/{name}/note")
def save_note(request: Request, name: str, note: str = Form(""), compare_with: str = Form("")):
    if not valid_run_name(name):
        raise HTTPException(status_code=404)
    run_dir = ctx(request).config.runs_root / name
    if not run_dir.is_dir():
        raise HTTPException(status_code=404)
    reference = compare_with.strip() or None
    if reference is not None and (not valid_run_name(reference) or reference == name):
        raise HTTPException(status_code=400, detail="a run cannot be compared with that")
    # Allowed for runs started from the shell too: this is metadata, never results.
    write_note(run_dir, note, reference)
    rows, columns, by_name = _table(request)
    return render(request, "_result_row.html", row=by_name[name], columns=columns,
                  by_name=by_name, names=list(by_name), delta=results.delta)


@router.get("/results.tsv")
def results_tsv(request: Request):
    rows, columns, by_name = _table(request)
    return PlainTextResponse(results.to_tsv(rows, columns, by_name),
                             media_type="text/tab-separated-values; charset=utf-8")
