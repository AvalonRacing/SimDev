"""Which pictures a run has, for the viewer, read from its render plan.

render_plan.json is the authoritative list - it names every picture with its
plane, offset and field - but it records absolute paths, and the runs folder
is reached through more than one path on this machine (~/runs is a symlink
to /mnt/data/runs). Every recorded path is therefore re-rooted onto the run
directory being looked at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simdev.ui.summary import load_summary
from simdev.viz.cplines import read_station

ANCHORS = ("results", "postProcessing")


def reroot(recorded: str, run_dir: Path) -> Path:
    parts = Path(recorded).parts
    for i, part in enumerate(parts):
        if part in ANCHORS:
            return Path(run_dir).joinpath(*parts[i:])
    return Path(recorded)


def load_plan(run_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(run_dir) / "results" / "render_plan.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _rel(recorded: str, run_dir: Path) -> str | None:
    path = reroot(recorded, run_dir)
    if not path.is_file():
        return None
    run_dir_resolved = Path(run_dir).resolve()
    try:
        path_resolved = path.resolve()
        if not path_resolved.is_relative_to(run_dir_resolved):
            return None
        return str(path_resolved.relative_to(run_dir_resolved))
    except ValueError:
        return None


def build_index(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    plan = load_plan(run_dir) or {}
    planes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in plan.get("slices", []):
        axis = entry.get("axis")
        name = entry.get("name")
        offset = entry.get("offset")
        if axis is None or name is None or offset is None:
            continue
        for image in entry.get("images", []):
            field = image.get("field")
            out = image.get("out")
            if field is None or out is None:
                continue
            rel = _rel(out, run_dir)
            if rel is None:
                continue
            planes.setdefault(axis, {}).setdefault(field, []).append(
                {"name": name, "offset": offset, "rel": rel})
    for fields in planes.values():
        for items in fields.values():
            items.sort(key=lambda p: p["offset"])
    surfaces: dict[str, dict[str, str]] = {}
    for entry in plan.get("surfaces", []):
        name = entry.get("name")
        if name is None:
            continue
        for image in entry.get("images", []):
            field = image.get("field")
            out = image.get("out")
            if field is None or out is None:
                continue
            rel = _rel(out, run_dir)
            if rel is not None:
                surfaces.setdefault(field, {})[name] = rel
    stations = []
    for s in (plan.get("cp_lines") or {}).get("stations", []):
        name = s.get("name")
        offset = s.get("offset")
        if name is not None and offset is not None:
            stations.append({"name": name, "offset": offset})
    summary = load_summary(run_dir)
    job_state = ""
    try:
        job_state = json.loads((run_dir / "ui" / "job.json").read_text("utf-8")).get("state", "")
    except (OSError, ValueError, AttributeError):
        pass
    return {
        "run": run_dir.name,
        "views_digest": plan.get("views_digest"),
        "state": job_state,
        "u_inf": (plan.get("frame") or {}).get("u_inf"),
        "window": list(summary.window) if summary else None,
        "planes": planes,
        "surfaces": surfaces,
        "stations": stations,
        "groups": summary.groups_map if summary else {},
    }


def cp_station(run_dir: Path, station: str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    plan = load_plan(run_dir) or {}
    for entry in (plan.get("cp_lines") or {}).get("stations", []):
        if entry.get("name") == station:
            path = reroot(entry.get("csv", ""), run_dir)
            break
    else:
        raise KeyError(station)
    u_inf = float((plan.get("frame") or {}).get("u_inf") or 0.0)
    q = 0.5 * u_inf * u_inf
    points: dict[str, tuple[list, list, list]] = {}
    if path.is_file():
        run_dir_resolved = run_dir.resolve()
        try:
            path_resolved = path.resolve()
            if path_resolved.is_relative_to(run_dir_resolved):
                points = read_station(path)
        except ValueError:
            pass
    return {
        "u_inf": u_inf,
        "patches": {patch: {"x": xs, "z": zs, "cp": [p / q for p in ps] if q else []}
                    for patch, (xs, zs, ps) in points.items()},
    }
