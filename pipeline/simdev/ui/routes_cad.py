"""The CAD library pages: driving states, designs and their slots.

Routes are plain `def` so FastAPI runs them in its thread pool: the STEP check
behind every upload can take tens of seconds and must not stall the server.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from simdev.cad.library import LibraryError
from simdev.config.resolve import deep_merge
from simdev.ui.context import back, ctx, render

router = APIRouter()


def params_from_form(
    mode: str,
    u_inf: str,
    corner_radius: str | None,
    corner_direction: str | None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The state parameters a form can set, merged over what the state had.

    Domain and ground follow from the mode exactly as in the shipped states:
    a corner runs in a static-ground annulus (the frame transform sweeps the
    road), a straight in a moving-ground box. Anything else a state.yaml
    carries is kept.
    """
    try:
        speed = float(u_inf)
    except (TypeError, ValueError):
        raise LibraryError("speed u_inf must be a number in m/s") from None

    if mode == "cornering":
        try:
            radius = float(corner_radius or "")
        except ValueError:
            raise LibraryError("a cornering state needs a corner radius in metres") from None
        own = {
            "flow": {"u_inf": speed},
            "physics": {"mode": "cornering", "corner_radius": radius,
                        "corner_direction": corner_direction or "left"},
            "domain": {"kind": "annulus"},
            "ground": {"motion": "static"},
        }
    elif mode == "straight":
        own = {
            "flow": {"u_inf": speed},
            "physics": {"mode": "straight"},
            "domain": {"kind": "box"},
            "ground": {"motion": "moving"},
        }
    else:
        raise LibraryError("choose straight or cornering")

    params = deep_merge(existing or {}, own)
    if mode == "straight":
        params["physics"].pop("corner_radius", None)
        params["physics"].pop("corner_direction", None)
    return params


@contextmanager
def _saved(request: Request, uploads: list[UploadFile] | None) -> Iterator[dict[str, Path]]:
    """Uploads on disk under their own names, inside the library root so the
    library's rename into place stays on one filesystem."""
    root = ctx(request).config.cad_root
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".upload-") as folder:
        saved: dict[str, Path] = {}
        for index, upload in enumerate(uploads or []):
            if not upload.filename:
                continue
            name = Path(upload.filename).name
            path = Path(folder) / f"{index}-{name}"
            with path.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            saved[name] = path
        yield saved


def _page(request: Request, error: str | None = None, message: str | None = None,
          status_code: int = 200):
    library = ctx(request).library
    rows = []
    for state in library.states():
        params = state.params
        physics = params.get("physics", {})
        rows.append({
            "state": state,
            "mode": physics.get("mode", "straight"),
            "u_inf": params.get("flow", {}).get("u_inf", ""),
            "corner_radius": physics.get("corner_radius", ""),
            "corner_direction": physics.get("corner_direction", "left"),
            "slots": library.slots_for_state(state.name),
        })
    return render(
        request, "cad.html", states=rows, broken=library.broken_states(),
        designs=library.designs(),
        state_names=[r["state"].name for r in rows],
        error=error, message=message, status_code=status_code,
    )


def _do(request: Request, action: Callable[[], Any], message: str):
    try:
        action()
    except LibraryError as error:
        return _page(request, error=str(error), status_code=400)
    return back("/cad", message=message)


@router.get("/cad")
def cad(request: Request, error: str | None = None, message: str | None = None):
    return _page(request, error=error, message=message)


# --- states ----------------------------------------------------------------


@router.post("/cad/states")
def create_state(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    mode: str = Form(...),
    u_inf: str = Form(...),
    corner_radius: str = Form(""),
    corner_direction: str = Form("left"),
    files: list[UploadFile] | None = File(None),
):
    with _saved(request, files) as saved:
        return _do(
            request,
            lambda: ctx(request).library.create_state(
                name, description,
                params_from_form(mode, u_inf, corner_radius, corner_direction), saved,
            ),
            f"driving state {name} added",
        )


@router.post("/cad/states/{name}/edit")
def edit_state(
    request: Request,
    name: str,
    description: str = Form(""),
    mode: str = Form(...),
    u_inf: str = Form(...),
    corner_radius: str = Form(""),
    corner_direction: str = Form("left"),
):
    library = ctx(request).library

    def action() -> None:
        existing = library.state(name).params
        library.update_state(
            name, description,
            params_from_form(mode, u_inf, corner_radius, corner_direction, existing),
        )

    return _do(request, action, f"driving state {name} updated")


@router.post("/cad/states/{name}/parts")
def replace_state_parts(request: Request, name: str, files: list[UploadFile] | None = File(None)):
    with _saved(request, files) as saved:
        return _do(request, lambda: ctx(request).library.replace_state_parts(name, saved),
                   f"parts of {name} replaced")


@router.post("/cad/states/{name}/rename")
def rename_state(request: Request, name: str, new_name: str = Form(...)):
    return _do(request, lambda: ctx(request).library.rename_state(name, new_name),
               f"{name} renamed to {new_name}")


@router.post("/cad/states/{name}/delete")
def delete_state(request: Request, name: str):
    return _do(request, lambda: ctx(request).library.delete_state(name), f"{name} deleted")


# --- designs and slots -----------------------------------------------------


@router.post("/cad/designs")
def create_design(request: Request, name: str = Form(...)):
    return _do(request, lambda: ctx(request).library.create_design(name), f"design {name} added")


@router.post("/cad/designs/{design}/slots")
def add_slot(request: Request, design: str, state: str = Form(...),
             files: list[UploadFile] | None = File(None)):
    with _saved(request, files) as saved:
        return _do(request, lambda: ctx(request).library.add_slot(design, state, saved),
                   f"{design}: Body and Wing added for {state}")


@router.post("/cad/designs/{design}/slots/{state}/move")
def move_slot(request: Request, design: str, state: str, to: str = Form(...),
              overwrite: bool = Form(False)):
    return _do(request, lambda: ctx(request).library.move_slot(design, state, to, overwrite),
               f"{design}: moved from {state} to {to}")


@router.post("/cad/designs/{design}/slots/{state}/replace")
def replace_slot(request: Request, design: str, state: str,
                 files: list[UploadFile] | None = File(None)):
    with _saved(request, files) as saved:
        return _do(request, lambda: ctx(request).library.replace_slot_parts(design, state, saved),
                   f"{design}/{state}: parts replaced")


@router.post("/cad/designs/{design}/slots/{state}/delete")
def delete_slot(request: Request, design: str, state: str):
    return _do(request, lambda: ctx(request).library.delete_slot(design, state),
               f"{design}/{state} deleted")


@router.post("/cad/designs/{design}/rename")
def rename_design(request: Request, design: str, new_name: str = Form(...)):
    return _do(request, lambda: ctx(request).library.rename_design(design, new_name),
               f"{design} renamed to {new_name}")


@router.post("/cad/designs/{design}/delete")
def delete_design(request: Request, design: str):
    return _do(request, lambda: ctx(request).library.delete_design(design), f"{design} deleted")
