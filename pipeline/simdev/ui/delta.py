"""Delta pictures on request: build the helper's request, run it once, cache.

One delta at a time. A delta is seconds of one core; it must never compete
with a running solve for more than that.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from simdev.stages.common import load_spec
from simdev.ui.imageindex import load_plan, reroot

PLANE_FIELDS = ("cp", "cpt", "U")
SURFACE_FIELDS = ("cp",)
HELPER = Path(__file__).resolve().parent.parent / "viz" / "delta.py"
DEFAULT_INTERPRETER = "/usr/bin/python3"
TIMEOUT = 600
_LOCK = threading.Lock()


class DeltaError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def default_limit(field: str, u_inf: float) -> float:
    return round(0.1 * u_inf, 6) if field == "U" else 0.2


def _plan(run_dir: Path) -> dict[str, Any]:
    plan = load_plan(run_dir)
    if plan is None:
        raise DeltaError(f"{run_dir.name} has no pictures yet (no render_plan.json)", 404)
    return plan


def _spec_hash(run_dir: Path) -> str:
    try:
        return json.loads((run_dir / "results" / "result.json").read_text("utf-8")).get(
            "spec_hash", "")
    except (OSError, ValueError):
        return ""


def build_request(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float) -> dict:
    pane_plan, ref_plan = _plan(pane_dir), _plan(ref_dir)
    if pane_plan.get("views_digest") != ref_plan.get("views_digest"):
        raise DeltaError("the two runs were pictured with different post_views.yaml; "
                         "their planes are not the same planes", 409)
    if not limit or limit <= 0:
        raise DeltaError("the colour limit must be positive", 422)
    if view.startswith("surface_"):
        if field not in SURFACE_FIELDS:
            raise DeltaError(f"no surface delta for {field}", 422)
        name = view[len("surface_"):]
        entries = {s["name"]: s for s in pane_plan.get("surfaces", [])}
        ref_entries = {s["name"]: s for s in ref_plan.get("surfaces", [])}
        if name not in entries or name not in ref_entries:
            raise DeltaError(f"no surface view {name}", 404)
        return {
            "kind": "surface", "field": field, "limit": limit,
            "resolution": pane_plan["resolution"], "camera": entries[name]["camera"],
            "pane": {"sample": str(reroot(entries[name]["sample"], pane_dir)),
                     "frame": pane_plan["frame"]},
            "ref": {"sample": str(reroot(ref_entries[name]["sample"], ref_dir)),
                    "frame": ref_plan["frame"]},
        }
    if field not in PLANE_FIELDS:
        raise DeltaError(f"no plane delta for {field}", 422)
    slices = {s["name"]: s for s in ref_plan.get("slices", [])}
    pane_slices = {s["name"]: s for s in pane_plan.get("slices", [])}
    if view not in slices or view not in pane_slices:
        raise DeltaError(f"no plane {view}", 404)
    entry = slices[view]
    return {
        "kind": "plane", "field": field, "limit": limit,
        "resolution": ref_plan["resolution"],
        "slice": {k: entry[k] for k in ("point", "normal", "camera")},
        "pane": {"sample": str(reroot(pane_slices[view]["sample"], pane_dir)),
                 "frame": pane_plan["frame"]},
        "ref": {"sample": str(reroot(entry["sample"], ref_dir)), "frame": ref_plan["frame"]},
    }


def _interpreter(run_dir: Path) -> str:
    try:
        return load_spec(run_dir).post.paraview_python or DEFAULT_INTERPRETER
    except (FileNotFoundError, KeyError, ValueError, AttributeError):
        return DEFAULT_INTERPRETER


def _fresh_cache(pane_dir: Path, ref_dir: Path) -> Path:
    folder = pane_dir / "ui" / "delta" / ref_dir.name
    key = hashlib.sha1(json.dumps([
        _spec_hash(pane_dir), _spec_hash(ref_dir),
        _plan(pane_dir).get("views_digest"), _plan(ref_dir).get("views_digest"),
    ]).encode()).hexdigest()
    stamp = folder / "key.txt"
    if not stamp.is_file() or stamp.read_text() != key:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        stamp.write_text(key)
    return folder


def delta_png(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float | None,
              runner: Callable[..., Any] = subprocess.run) -> Path:
    pane_dir, ref_dir = Path(pane_dir), Path(ref_dir)
    if limit is None:
        limit = default_limit(field, float(_plan(pane_dir)["frame"]["u_inf"]))
    request = build_request(pane_dir, ref_dir, view, field, limit)
    with _LOCK:
        folder = _fresh_cache(pane_dir, ref_dir)
        out = folder / f"{view}_{field}_{limit:g}.png"
        if out.is_file():
            return out
        request["out"] = str(out)
        request["cache_vtp"] = str(folder / "surface.vtp")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(request, handle)
        try:
            done = runner(
                [_interpreter(pane_dir), str(HELPER), handle.name],
                capture_output=True, text=True, timeout=TIMEOUT,
                env={"HOME": os.environ.get("HOME", ""), "PATH": "/usr/bin:/bin"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DeltaError(f"the delta helper could not run: {error}", 500) from None
        finally:
            os.unlink(handle.name)
        try:
            summary = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            summary = {"ok": False, "error": (done.stderr or "no output")[-1500:]}
        if done.returncode != 0 or not summary.get("ok") or not out.is_file():
            raise DeltaError(
                f"{summary.get('error') or 'the delta helper failed'} - "
                "run 'simdev doctor' to check the ParaView interpreter", 500)
        return out
