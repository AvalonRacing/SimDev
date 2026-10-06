"""The compare page and the data its script asks for."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from simdev.ui import results, runview
from simdev.ui.context import ctx, render
from simdev.ui.delta import DeltaError, delta_png
from simdev.ui.imageindex import build_index, cp_station
from simdev.ui.routes_results import valid_run_name
from simdev.ui.summary import finite, load_summary

router = APIRouter()
MAX_PANES = 4


def _run_dir(request: Request, name: str) -> Path:
    if not valid_run_name(name):
        raise HTTPException(status_code=404)
    path = ctx(request).config.runs_root / name
    if not path.is_dir():
        raise HTTPException(status_code=404)
    return path


@router.get("/compare")
def compare_page(request: Request, runs: list[str] = Query([]), ref: str = ""):
    root = ctx(request).config.runs_root
    wanted = [n.strip() for item in runs for n in item.split(",") if n.strip()]
    names, missing = [], []
    for name in wanted:
        if valid_run_name(name) and (root / name).is_dir() and name not in names:
            names.append(name)
        else:
            missing.append(name)
    names, dropped = names[:MAX_PANES], names[MAX_PANES:]
    ref = ref if ref in names else (names[0] if names else "")
    # Only runs with pictures: this page is the picture comparison, not the numbers.
    offered = [p.name for p in runview.list_runs(root)
               if valid_run_name(p.name) and (p / "results" / "render_plan.json").is_file()]
    # The current selection always round-trips through Show, pictures or not.
    offered += [n for n in names if n not in offered]
    payload = {"runs": names, "ref": ref, "limits": {"cp": 0.2, "cpt": 0.2},
               "missing": missing}
    return render(
        request, "compare.html", ref=ref, runs=names, offered=offered,
        missing=missing, dropped=dropped, max_panes=MAX_PANES,
        data_json=json.dumps(payload).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"),
    )


@router.get("/api/runs/{name}/index")
def run_index(request: Request, name: str):
    return build_index(_run_dir(request, name))


@router.get("/api/runs/{name}/summary")
def run_summary(request: Request, name: str):
    summary = load_summary(_run_dir(request, name))
    if summary is None:
        raise HTTPException(status_code=404, detail="no results yet")
    row = results.load_row(_run_dir(request, name))
    # NaN is not JSON; the browser's parser rejects the whole response.
    patches = {p: {k: finite(v) for k, v in d.items()} for p, d in summary.patches.items()}
    return {"window": list(summary.window), "values": row.values, "noise": row.noise,
            "patches": patches, "groups_map": summary.groups_map}


@router.get("/api/runs/{name}/cplines/{station}")
def run_cplines(request: Request, name: str, station: str):
    try:
        data = cp_station(_run_dir(request, name), station)
    except KeyError:
        raise HTTPException(status_code=404) from None
    except ValueError as error:
        raise HTTPException(status_code=404, detail=f"unreadable cp data: {error}") from None
    # NaN/inf are not JSON; they would turn the response into a 500.
    data["u_inf"] = finite(data.get("u_inf"))
    data["patches"] = {
        patch: {k: [finite(v) for v in values] for k, values in series.items()}
        for patch, series in data["patches"].items()
    }
    return data


@router.get("/api/delta/{name}.png")
def delta_picture(request: Request, name: str, ref: str, view: str, field: str,
                  limit: float | None = None):
    pane_dir, ref_dir = _run_dir(request, name), _run_dir(request, ref)
    try:
        return FileResponse(delta_png(pane_dir, ref_dir, view, field, limit),
                            media_type="image/png")
    except DeltaError as error:
        headers = {"Retry-After": "2"} if error.status == 503 else None
        return JSONResponse({"error": str(error)}, status_code=error.status, headers=headers)
