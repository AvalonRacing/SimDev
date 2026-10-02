"""Starts queued jobs, watches them, and records how they ended.

Each job runs as the leader of its own process group (start_new_session), so
one killpg reaches the job runner, mpirun and every rank. The service unit
uses KillMode=process for the same reason in reverse: restarting the web
server must not kill a solve, and recover() adopts the survivors.
"""

from __future__ import annotations

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
from simdev.ui.queue import Job, Queue

log = logging.getLogger(__name__)

RESTARTED = "service restarted while the run was in progress; resume it to continue"


def default_command(job: Job) -> list[str]:
    return [sys.executable, "-m", "simdev.ui.jobrun", job.run_dir]


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
                if job.pgid and group_alive(job.pgid):
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
            if proc is not None:
                code = proc.poll()
                if code is None:
                    continue
                del self._procs[job.id]
            elif job.id in self._adopted:
                if job.pgid and group_alive(job.pgid):
                    continue
                self._adopted.discard(job.id)
                code = None
            else:
                continue

            cancelled = self._cancelling.pop(job.id, None) is not None
            run_dir = Path(job.run_dir)
            status, error = classify(run_dir, read_outcome(run_dir), cancelled)
            self.queue.mark_finished(job.id, status, code, error)

    def _escalate(self) -> None:
        now = time.monotonic()
        for job_id, deadline in list(self._cancelling.items()):
            if now < deadline:
                continue
            job = self.queue.get(job_id)
            if job is not None and job.pgid:
                try:
                    os.killpg(job.pgid, signal.SIGKILL)
                except ProcessLookupError:
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
        run_dir = Path(job.run_dir)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        write_job_file(job, str(self._cad_root) if self._cad_root else None)
        (run_dir / OUTCOME_FILE).unlink(missing_ok=True)

        with open(run_dir / UI_LOG, "a", encoding="utf-8") as output:
            try:
                proc = subprocess.Popen(
                    self._command(job),
                    cwd=self._cwd,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as error:
                self.queue.mark_running(job.id, 0, 0)
                self.queue.mark_finished(job.id, "failed", None, f"could not start: {error}")
                return

        self._procs[job.id] = proc
        # start_new_session makes the child its own group leader: pgid == pid.
        self.queue.mark_running(job.id, proc.pid, proc.pid)

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
                    except ProcessLookupError:
                        pass
                self._cancelling[job_id] = time.monotonic() + self._kill_grace
