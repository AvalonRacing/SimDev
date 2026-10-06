"""Delta pictures on request: build the helper's request, run it once, cache.

One delta at a time. A delta is seconds of one core; it must never compete
with a running solve for more than that.
"""

from __future__ import annotations

import hashlib
import json
import math
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
# Part of the cache key: bump it whenever viz/delta.py changes what it draws,
# so pictures cached by an older helper are recomputed. Bump note: 2 is the
# helper as merged with the compare viewer.
HELPER_VERSION = 2
# Same picture frame if every number agrees to 0.1 mm (or 1e-4 in a unit
# vector). Not round-off: each run measures its datum from its own Chassis
# tessellation, so two runs of one state already differ by a few um (2e-6 m
# between c02_combo12 and its base). An attitude change of even 0.1 deg moves
# a normal by ~2e-3, and the 1e-4 m slice-probe tolerance is the same size.
FRAMING_TOL = 1e-4
_LOCK = threading.Lock()
BUSY = "another delta is being computed - try again in a moment"


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


def _close(a: Any, b: Any) -> bool:
    """Same framing: numbers within FRAMING_TOL, the rest equal."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= FRAMING_TOL
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


NOT_THE_SAME = ("the two runs were pictured differently (driving state, attitude or camera "
                "convention) - the planes are not the same planes; re-render or compare like with like")


def build_request(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float) -> dict:
    pane_plan, ref_plan = _plan(pane_dir), _plan(ref_dir)
    if pane_plan.get("views_digest") != ref_plan.get("views_digest"):
        raise DeltaError("the two runs were pictured with different post_views.yaml; "
                         "their planes are not the same planes", 409)
    if not (math.isfinite(limit) and limit > 0):
        raise DeltaError("the colour limit must be positive and finite", 422)
    if view.startswith("surface_"):
        if field not in SURFACE_FIELDS:
            raise DeltaError(f"no surface delta for {field}", 422)
        name = view[len("surface_"):]
        entries = {s["name"]: s for s in pane_plan.get("surfaces", [])}
        ref_entries = {s["name"]: s for s in ref_plan.get("surfaces", [])}
        if name not in entries or name not in ref_entries:
            raise DeltaError(f"no surface view {name}", 404)
        if not _close(entries[name].get("camera"), ref_entries[name].get("camera")):
            raise DeltaError(NOT_THE_SAME, 409)
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
    if not all(_close(entry.get(k), pane_slices[view].get(k)) for k in ("point", "normal")):
        raise DeltaError(NOT_THE_SAME, 409)
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


def _cache_folder(pane_dir: Path, ref_dir: Path) -> Path:
    ref_name = ref_dir.name
    if not ref_name or ref_name in (".", ".."):
        raise DeltaError(f"invalid reference run name: {ref_name}", 404)
    folder = pane_dir / "ui" / "delta" / ref_name
    # Verify folder is a direct child of <pane>/ui/delta before rmtree
    try:
        folder.resolve().relative_to((pane_dir / "ui" / "delta").resolve())
    except ValueError:
        raise DeltaError("cache folder would be outside ui/delta", 404) from None
    return folder


def _cache_key(pane_dir: Path, ref_dir: Path) -> str:
    pane_plan, ref_plan = _plan(pane_dir), _plan(ref_dir)
    return hashlib.sha1(json.dumps([
        HELPER_VERSION, _spec_hash(pane_dir), _spec_hash(ref_dir),
        pane_plan.get("views_digest"), ref_plan.get("views_digest"),
        pane_plan.get("stamp"), ref_plan.get("stamp"),
    ], sort_keys=True, default=str).encode()).hexdigest()


def _key_matches(folder: Path, key: str) -> bool:
    try:
        return (folder / "key.txt").read_text() == key
    except OSError:
        return False


def _fresh_cache(folder: Path, key: str) -> Path:
    if not _key_matches(folder, key):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "key.txt").write_text(key)
    return folder


def delta_png(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float | None,
              runner: Callable[..., Any] = subprocess.run) -> Path:
    pane_dir, ref_dir = Path(pane_dir), Path(ref_dir)
    if limit is None:
        limit = default_limit(field, float(_plan(pane_dir)["frame"]["u_inf"]))
    limit = round(float(limit), 6)
    request = build_request(pane_dir, ref_dir, view, field, limit)
    folder, key = _cache_folder(pane_dir, ref_dir), _cache_key(pane_dir, ref_dir)
    out = folder / f"{view}_{field}_{limit:g}.png"
    # A cached picture never waits behind a running compute.
    if _key_matches(folder, key) and out.is_file():
        return out
    # Never queue: a request that waited would hold a server thread (and a
    # browser connection) for a picture the user has likely scrolled past.
    if not _LOCK.acquire(blocking=False):
        raise DeltaError(BUSY, 503)
    try:
        _fresh_cache(folder, key)
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
    finally:
        _LOCK.release()
