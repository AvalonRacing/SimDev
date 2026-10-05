"""The job queue: the only state the UI owns.

How a run is going - stages, verdicts, coefficients - is never stored here;
it is read from the run directory every time. This table only says which
runs were asked for, in what order, and what became of the process.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JOB_STATUSES = ("queued", "running", "done", "gate_failed", "failed", "cancelled")
FINISHED = ("done", "gate_failed", "failed", "cancelled")
DEFAULT_SETTINGS = {"core_budget": 40, "max_parallel": 1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name    TEXT NOT NULL,
    run_dir     TEXT NOT NULL,
    case_path   TEXT NOT NULL,
    design      TEXT NOT NULL,
    state       TEXT NOT NULL,
    profile     TEXT NOT NULL,
    n_ranks     INTEGER NOT NULL,
    overrides   TEXT NOT NULL DEFAULT '{}',
    force_from  TEXT,
    position    REAL NOT NULL,
    status      TEXT NOT NULL,
    pid         INTEGER,
    pgid        INTEGER,
    exit_code   INTEGER,
    error       TEXT,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    note        TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class QueueError(ValueError):
    pass


@dataclass(frozen=True)
class JobSpec:
    run_name: str
    run_dir: str
    case_path: str
    design: str
    state: str
    profile: str
    n_ranks: int
    overrides: dict[str, Any] = field(default_factory=dict)
    force_from: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Job:
    id: int
    run_name: str
    run_dir: str
    case_path: str
    design: str
    state: str
    profile: str
    n_ranks: int
    overrides: dict[str, Any]
    force_from: str | None
    position: float
    status: str
    pid: int | None
    pgid: int | None
    exit_code: int | None
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    note: str | None = None


class Queue:
    def __init__(self, db_path: Path, clock: Callable[[], float] = time.time) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the web threads and the worker, serialised
        # by the lock. Autocommit, so every statement is durable on return.
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._clock = clock
        self._add_note_column()

    def _add_note_column(self) -> None:
        # A database made before change notes existed. Once migrated, an older
        # server process still running against it cannot read job rows (its Job
        # has no `note`), so deploying this means restarting the UI service.
        with self._lock:
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(jobs)")}
            if "note" in columns:
                return
            try:
                self._db.execute("ALTER TABLE jobs ADD COLUMN note TEXT")
            except sqlite3.OperationalError as problem:
                # Another process migrated between the check and the ALTER.
                if "duplicate column name" not in str(problem):
                    raise

    # --- reading -------------------------------------------------------------

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        values = dict(row)
        values["overrides"] = json.loads(values["overrides"])
        return Job(**values)

    def _all(self, sql: str, args: tuple = ()) -> list[Job]:
        with self._lock:
            return [self._job(r) for r in self._db.execute(sql, args).fetchall()]

    def get(self, job_id: int) -> Job | None:
        jobs = self._all("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return jobs[0] if jobs else None

    def queued(self) -> list[Job]:
        return self._all("SELECT * FROM jobs WHERE status = 'queued' ORDER BY position")

    def running(self) -> list[Job]:
        return self._all("SELECT * FROM jobs WHERE status = 'running' ORDER BY started_at")

    def finished(self, limit: int = 20) -> list[Job]:
        marks = ",".join("?" * len(FINISHED))
        return self._all(
            f"SELECT * FROM jobs WHERE status IN ({marks}) "
            "ORDER BY finished_at DESC LIMIT ?",
            (*FINISHED, limit),
        )

    def latest_for(self, run_name: str) -> Job | None:
        jobs = self._all(
            "SELECT * FROM jobs WHERE run_name = ? ORDER BY id DESC LIMIT 1", (run_name,)
        )
        return jobs[0] if jobs else None

    def name_taken(self, run_name: str, exclude_id: int | None = None) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM jobs WHERE run_name = ? AND status IN ('queued', 'running') "
                "AND id IS NOT ?",
                (run_name, exclude_id),
            ).fetchone()
        return row is not None

    def next_fitting(self, free_cores: int) -> Job | None:
        for job in self.queued():
            if job.n_ranks <= free_cores:
                return job
        return None

    # --- writing -------------------------------------------------------------

    def enqueue(self, spec: JobSpec) -> Job:
        with self._lock:
            if self.name_taken(spec.run_name):
                raise QueueError(
                    f"a queued or running job already uses the name {spec.run_name!r}"
                )
            (top,) = self._db.execute(
                "SELECT COALESCE(MAX(position), 0) FROM jobs"
            ).fetchone()
            cursor = self._db.execute(
                "INSERT INTO jobs (run_name, run_dir, case_path, design, state, profile, "
                "n_ranks, overrides, force_from, position, status, created_at, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                (
                    spec.run_name, spec.run_dir, spec.case_path, spec.design, spec.state,
                    spec.profile, spec.n_ranks, json.dumps(spec.overrides, sort_keys=True),
                    spec.force_from, top + 1, self._clock(), spec.note,
                ),
            )
            return self.get(cursor.lastrowid)

    def update(self, job_id: int, spec: JobSpec) -> Job:
        with self._lock:
            job = self.get(job_id)
            if job is None or job.status != "queued":
                raise QueueError("this job has already started and can no longer be edited")
            if self.name_taken(spec.run_name, exclude_id=job_id):
                raise QueueError(
                    f"a queued or running job already uses the name {spec.run_name!r}"
                )
            self._db.execute(
                "UPDATE jobs SET run_name = ?, run_dir = ?, case_path = ?, design = ?, "
                "state = ?, profile = ?, n_ranks = ?, overrides = ?, force_from = ?, note = ? "
                "WHERE id = ? AND status = 'queued'",
                (
                    spec.run_name, spec.run_dir, spec.case_path, spec.design, spec.state,
                    spec.profile, spec.n_ranks, json.dumps(spec.overrides, sort_keys=True),
                    spec.force_from, spec.note, job_id,
                ),
            )
            return self.get(job_id)

    def move(self, job_id: int, direction: int) -> None:
        with self._lock:
            jobs = self.queued()
            ids = [j.id for j in jobs]
            if job_id not in ids:
                raise QueueError("only queued jobs can be reordered")
            i = ids.index(job_id)
            j = i + direction
            if not 0 <= j < len(jobs):
                return
            for a, b in ((jobs[i], jobs[j]), (jobs[j], jobs[i])):
                self._db.execute(
                    "UPDATE jobs SET position = ? WHERE id = ?", (b.position, a.id)
                )

    def _transition(self, job_id: int, expected: str, **values: Any) -> None:
        columns = ", ".join(f"{k} = ?" for k in values)
        with self._lock:
            cursor = self._db.execute(
                f"UPDATE jobs SET {columns} WHERE id = ? AND status = ?",
                (*values.values(), job_id, expected),
            )
            if cursor.rowcount == 0:
                raise QueueError(f"job {job_id} is not {expected}")

    def mark_running(self, job_id: int, pid: int | None = None, pgid: int | None = None) -> None:
        # The worker claims a job before it starts the process, so a job can be
        # running with no pid yet; set_process fills it in once the process exists.
        self._transition(
            job_id, "queued", status="running", pid=pid, pgid=pgid, started_at=self._clock()
        )

    def set_process(self, job_id: int, pid: int, pgid: int) -> None:
        self._transition(job_id, "running", pid=pid, pgid=pgid)

    def mark_finished(
        self, job_id: int, status: str, exit_code: int | None, error: str | None
    ) -> None:
        if status not in FINISHED:
            raise QueueError(f"{status!r} is not a finished status")
        self._transition(
            job_id, "running", status=status, exit_code=exit_code, error=error,
            finished_at=self._clock(),
        )

    def mark_cancelled(self, job_id: int) -> None:
        self._transition(job_id, "queued", status="cancelled", finished_at=self._clock())

    def requeue(self, job_id: int, force_from: str | None = None) -> Job:
        job = self.get(job_id)
        if job is None or job.status in ("queued", "running"):
            raise QueueError("only a finished job can be resumed or re-run")
        return self.enqueue(
            JobSpec(
                run_name=job.run_name, run_dir=job.run_dir, case_path=job.case_path,
                design=job.design, state=job.state, profile=job.profile,
                n_ranks=job.n_ranks, overrides=job.overrides, force_from=force_from,
                note=job.note,
            )
        )

    # --- settings ------------------------------------------------------------

    def setting(self, key: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return int(row["value"]) if row else DEFAULT_SETTINGS[key]

    def set_setting(self, key: str, value: int) -> None:
        if key not in DEFAULT_SETTINGS:
            raise QueueError(f"unknown setting {key!r}")
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(int(value)))
            )
