"""A read-only view of one run directory, for the pages.

Knows nothing about the queue. Everything is read from the files the pipeline
already writes, every time it is asked, so a run made from the shell looks
exactly like one made from the browser.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from simdev.gates.convergence import check_convergence
from simdev.run.parsers import is_fatal_line, read_component_coeffs, read_force_coeffs
from simdev.run.status import read_status
from simdev.stages.common import load_spec
from simdev.ui.jobrun import STAGES

TAIL_BYTES = 256 * 1024
CONTEXT_LINES = 25

_TIME = re.compile(r"^Time = (\d+)\s*$", re.MULTILINE)
_EXEC = re.compile(r"^ExecutionTime = ([\d.eE+-]+) s", re.MULTILINE)


def _read_status(run_dir: Path, stage: str):
    """Read status safely, returning None if the file is corrupted."""
    try:
        return read_status(run_dir, stage)
    except (ValueError, TypeError):
        # Half-written or invalid status file
        return None


@dataclass(frozen=True)
class StageView:
    name: str
    state: str
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    finished_at: float | None = None
    seconds: float | None = None


@dataclass(frozen=True)
class ErrorView:
    source: str
    message: str
    context: list[str] = field(default_factory=list)


def stages(
    run_dir: Path,
    running: bool = False,
    since: float | None = None,
    forced_from: str | None = None,
) -> list[StageView]:
    """One entry per pipeline stage.

    A status file older than `since` (the job's start) is still current when
    the stage was skipped as unchanged - except a failure, which is being
    retried, and a stage the job forces, which is being redone.
    """
    run_dir = Path(run_dir)
    forced_index = STAGES.index(forced_from) if forced_from else len(STAGES)
    views: list[StageView] = []
    previous_end = since
    running_shown = False

    for index, name in enumerate(STAGES):
        path = run_dir / "status" / f"{name}.json"
        status = _read_status(run_dir, name) if path.is_file() else None
        mtime = path.stat().st_mtime if status else None
        old = since is not None and mtime is not None and mtime < since
        current = status is not None and not (
            old and (status.state == "failed" or index >= forced_index)
        )

        if current:
            seconds = None
            if previous_end is not None and mtime is not None and mtime >= previous_end and not old:
                seconds = mtime - previous_end
            views.append(
                StageView(name, status.state, status.reasons, status.detail, mtime, seconds)
            )
            if mtime is not None and not old:
                previous_end = mtime
        elif running and not running_shown:
            views.append(StageView(name, "running"))
            running_shown = True
        else:
            views.append(StageView(name, "pending"))
    return views


def _read_frames(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(read_force_coeffs(path))
        except (pd.errors.EmptyDataError, ValueError):
            continue
    if not frames:
        return pd.DataFrame()
    # A restarted solve writes a new time directory that repeats iterations.
    frame = pd.concat(frames).drop_duplicates("Time", keep="last").sort_values("Time")
    return frame.reset_index(drop=True)


def force_frame(run_dir: Path) -> pd.DataFrame:
    root = Path(run_dir) / "postProcessing" / "forceCoeffs"
    return _read_frames(sorted(root.glob("*/coefficient.dat")))


def _series(frame: pd.DataFrame, after: int, columns: dict[str, str]) -> dict[str, list]:
    if frame.empty:
        return {"iteration": [], **{name: [] for name in columns}}
    frame = frame[frame["Time"] > after]
    # Drop rows with NaN in Time or any requested column
    cols_to_check = ["Time"] + list(columns.values())
    frame = frame.dropna(subset=cols_to_check, how="any")
    return {
        "iteration": frame["Time"].astype(int).tolist(),
        **{name: frame[column].astype(float).tolist() for name, column in columns.items()},
    }


def force_series(run_dir: Path, after: int = 0) -> dict[str, list]:
    return _series(force_frame(run_dir), after, {"Cd": "Cd", "Cl": "Cl"})


def component_series(run_dir: Path, after: int = 0) -> dict[str, dict[str, list]]:
    return {
        patch: _series(frame.reset_index(drop=True), after, {"Cd": "Cd", "Cl": "Cl"})
        for patch, frame in read_component_coeffs(Path(run_dir)).items()
    }


def residual_series(run_dir: Path, after: int = 0) -> dict[str, list]:
    root = Path(run_dir) / "postProcessing" / "residuals"
    frame = _read_frames(sorted(root.glob("*/solverInfo.dat")))
    if frame.empty:
        return {"iteration": []}
    columns = {
        column[: -len("_initial")]: column
        for column in frame.columns
        if column.endswith("_initial")
    }
    return _series(frame, after, columns)


def convergence(run_dir: Path) -> dict[str, Any] | None:
    """The plateau gate applied to what exists so far, for the live plot."""
    try:
        spec = load_spec(Path(run_dir))
    except (FileNotFoundError, KeyError, ValueError):
        # No caseSpec yet, or one written by an older pipeline: no gate to show.
        return None
    frame = force_frame(run_dir)
    if frame.empty:
        return None
    result = check_convergence(frame, spec)
    start, end = result.window
    times = frame["Time"].astype(int).tolist()
    window = [times[start], times[max(end - 1, start)]] if end > start else None
    return {
        "verdict": result.verdict,
        "reasons": result.reasons,
        "window": window,
        "drift_tol": spec.solve.drift_tol,
        "max_iterations": spec.solve.max_iterations,
    }


def _tail_text(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - TAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace")


def progress(run_dir: Path, max_iterations: int | None) -> dict[str, float | int | None]:
    path = Path(run_dir) / "logs" / "log.simpleFoam"
    empty = {"iteration": None, "seconds_per_iteration": None, "eta_seconds": None}
    if not path.is_file():
        return empty

    samples: list[tuple[int, float]] = []
    iteration: int | None = None
    for line in _tail_text(path).splitlines():
        if (match := _TIME.match(line)) is not None:
            iteration = int(match.group(1))
        elif (match := _EXEC.match(line)) is not None and iteration is not None:
            samples.append((iteration, float(match.group(1))))
    if not samples:
        return {**empty, "iteration": iteration}

    recent = samples[-20:]
    per_iteration = None
    if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
        per_iteration = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
    last = samples[-1][0]
    eta = None
    if per_iteration is not None and max_iterations:
        eta = max(0, max_iterations - last) * per_iteration
    return {"iteration": last, "seconds_per_iteration": per_iteration, "eta_seconds": eta}


def log_names(run_dir: Path) -> list[str]:
    folder = Path(run_dir) / "logs"
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file())


def tail(run_dir: Path, name: str, lines: int = 200) -> list[str]:
    # Only names from the listing: the name arrives from a URL.
    if name not in log_names(run_dir):
        raise KeyError(name)
    return _tail_text(Path(run_dir) / "logs" / name).splitlines()[-lines:]


def errors(run_dir: Path) -> list[ErrorView]:
    run_dir = Path(run_dir)
    found: list[ErrorView] = []
    for name in STAGES:
        status = _read_status(run_dir, name)
        if status is not None and status.state == "failed":
            found.append(ErrorView(f"stage {name}", "; ".join(status.reasons) or "failed"))
    for name in log_names(run_dir):
        log_path = run_dir / "logs" / name
        # Stream log file line-by-line to avoid memory issues with large logs
        preceding = deque(maxlen=CONTEXT_LINES)
        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if is_fatal_line(line):
                        # Collect following lines
                        context = list(preceding) + [line]
                        for _ in range(CONTEXT_LINES):
                            try:
                                context.append(next(f).rstrip("\n"))
                            except StopIteration:
                                break
                        found.append(ErrorView(f"logs/{name}", line.strip(), context))
                        break
                    preceding.append(line)
        except (OSError, IOError):
            # Log file disappeared or can't be read
            continue
    return found


def images(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    results = run_dir / "results"
    summary = sorted(
        str(p.relative_to(run_dir)) for p in results.glob("*.png")
    ) if results.is_dir() else []
    groups: dict[str, list[str]] = {}
    root = results / "images"
    if root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            pictures = sorted(str(p.relative_to(run_dir)) for p in folder.glob("*.png"))
            if pictures:
                groups[folder.name] = pictures
    return {"summary": summary, "groups": groups}


def safe_file(run_dir: Path, rel: str) -> Path:
    """A file inside the run directory, or KeyError - never anything else."""
    try:
        root = Path(run_dir).resolve()
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise KeyError(rel)
        return target
    except (ValueError, OSError):
        # NUL byte or other path issues
        raise KeyError(rel)


def list_runs(runs_root: Path) -> list[Path]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    runs = [
        p for p in root.iterdir()
        if p.is_dir() and ((p / "caseSpec.json").is_file() or (p / "ui" / "job.json").is_file())
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
