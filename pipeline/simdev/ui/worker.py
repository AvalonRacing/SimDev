"""Starts queued jobs, watches them, and records how they ended.

Each job runs as the leader of its own process group (start_new_session), so
one killpg reaches the job runner, mpirun and every rank. The service unit
uses KillMode=process for the same reason in reverse: restarting the web
server must not kill a solve, and recover() adopts the survivors.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from simdev.cad.library import REPO_ROOT
from simdev.ui.jobrun import OUTCOME_FILE, UI_LOG, classify, read_outcome, write_job_file
from simdev.ui.notes import write_initial_note
from simdev.ui.queue import Job, Queue

log = logging.getLogger(__name__)

PROCESS_FILE = "ui/process.json"
RESTARTED = "service restarted while the run was in progress; resume it to continue"


def default_command(job: Job) -> list[str]:
    return [sys.executable, "-m", "simdev.ui.jobrun", job.run_dir]


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        # Our own job groups never raise PermissionError; a group we may not
        # signal belongs to someone else and is not ours.
        return False
    return True


def current_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def write_process_file(run_dir: Path, pgid: int) -> None:
    (run_dir / "ui").mkdir(parents=True, exist_ok=True)
    (run_dir / PROCESS_FILE).write_text(
        json.dumps({"pgid": pgid, "boot_id": current_boot_id()}), encoding="utf-8"
    )


def _owns_group(job: Job) -> bool:
    """A running job is ours to adopt only if its recorded group survived this boot."""
    try:
        recorded = json.loads((Path(job.run_dir) / PROCESS_FILE).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(recorded, dict) or recorded.get("boot_id") != current_boot_id():
        return False
    return bool(job.pgid) and recorded.get("pgid") == job.pgid and group_alive(job.pgid)


class Worker:
    def __init__(
        self,
        queue: Queue,
        command: Callable[[Job], list[str]] = default_command,
        cwd: Path = REPO_ROOT,
        kill_grace: float = 30.0,
        cad_root: Path | None = None,
    ) -> None:
        self.queue = queue
        self._command = command
        self._cwd = cwd
        self._kill_grace = kill_grace
        self._cad_root = cad_root
        self._procs: dict[int, subprocess.Popen] = {}
        self._adopted: set[int] = set()
        self._cancelling: dict[int, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle -----------------------------------------------------------

    def recover(self) -> None:
        with self._lock:
            for job in self.queue.running():
                if job.id in self._procs:
                    continue
                if _owns_group(job):
                    self._adopted.add(job.id)
                else:
                    self.queue.mark_finished(job.id, "failed", None, RESTARTED)

    def start(self) -> threading.Thread:
        self.recover()
        self._thread = threading.Thread(target=self._loop, name="simdev-worker", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(2.0):
            try:
                self.tick()
            except Exception:
                # One bad tick must not end the worker; the next one retries.
                log.exception("worker tick failed")

    # --- one iteration -------------------------------------------------------

    def tick(self) -> None:
        with self._lock:
            self._reap()
            self._escalate()
            self._start_next()

    def _reap(self) -> None:
        for job in self.queue.running():
            proc = self._procs.get(job.id)
            code: int | None = None
            if proc is not None:
                code = proc.poll()
                if code is None:
                    continue
            elif job.id not in self._adopted:
                if job.pgid is None:
                    # Claimed but never given a process by this worker: the
                    # service stopped between the claim and the start.
                    self.queue.mark_finished(job.id, "failed", None, RESTARTED)
                continue
            # The leader exiting is not the end: ranks may outlive it.
            if job.pgid and group_alive(job.pgid):
                continue

            cancelled = job.id in self._cancelling
            run_dir = Path(job.run_dir)
            try:
                status, error = classify(run_dir, read_outcome(run_dir), cancelled)
            except Exception as problem:
                log.exception("could not classify job %s", job.id)
                status, error = "failed", f"could not classify the result: {problem}"
            self.queue.mark_finished(job.id, status, code, error)
            # Only now forget the job, so a failed mark_finished is retried.
            self._procs.pop(job.id, None)
            self._adopted.discard(job.id)
            self._cancelling.pop(job.id, None)

    def _escalate(self) -> None:
        now = time.monotonic()
        for job_id, deadline in list(self._cancelling.items()):
            if now < deadline:
                continue
            job = self.queue.get(job_id)
            if job is not None and job.pgid:
                try:
                    os.killpg(job.pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            self._cancelling[job_id] = float("inf")

    def _start_next(self) -> None:
        running = self.queue.running()
        if len(running) >= self.queue.setting("max_parallel"):
            return
        free = self.queue.setting("core_budget") - sum(j.n_ranks for j in running)
        job = self.queue.next_fitting(free)
        if job is not None:
            self._spawn(job)

    def _spawn(self, job: Job) -> None:
        # Claim the job before anything can fail, so a job that cannot start is
        # finished as failed once instead of staying queued and being retried
        # (and blocking the queue) on every tick, and so a process that did
        # start can never be started a second time.
        self.queue.mark_running(job.id)
        run_dir = Path(job.run_dir)
        try:
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            write_job_file(job, str(self._cad_root) if self._cad_root else None)
            try:
                write_initial_note(run_dir, job.note)
            except OSError as error:
                # The note is a label; losing it must not cost the run.
                log.warning("could not write the note of job %s: %s", job.id, error)
            (run_dir / OUTCOME_FILE).unlink(missing_ok=True)
            with open(run_dir / UI_LOG, "a", encoding="utf-8") as output:
                proc = subprocess.Popen(
                    self._command(job),
                    cwd=self._cwd,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except Exception as error:
            log.exception("could not start job %s", job.id)
            self.queue.mark_finished(job.id, "failed", None, f"could not start: {error}")
            return

        self._procs[job.id] = proc
        # start_new_session makes the child its own group leader: pgid == pid.
        self.queue.set_process(job.id, proc.pid, proc.pid)
        try:
            write_process_file(run_dir, proc.pid)
        except OSError:
            # Only recover() after a restart needs the record; without it the
            # job is reported as lost then, which is better than not running it.
            log.exception("could not record the process of job %s", job.id)

    # --- requests from the UI ------------------------------------------------

    def cancel(self, job_id: int) -> None:
        with self._lock:
            job = self.queue.get(job_id)
            if job is None:
                return
            if job.status == "queued":
                self.queue.mark_cancelled(job_id)
            elif job.status == "running" and job_id not in self._cancelling:
                if job.pgid:
                    try:
                        os.killpg(job.pgid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                self._cancelling[job_id] = time.monotonic() + self._kill_grace
