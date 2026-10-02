"""The worker against a stub job, so no OpenFOAM is needed.

The stub's behaviour is chosen by the job's design name: ok, gate, crash,
sleep (spawns a grandchild and waits, for cancel tests).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from simdev.ui.queue import JobSpec, Queue
from simdev.ui.worker import Worker, group_alive

STUB = r"""
import json, os, subprocess, sys, time
from pathlib import Path
run_dir, mode = Path(sys.argv[1]), sys.argv[2]
status = run_dir / "status"
status.mkdir(parents=True, exist_ok=True)
def mark(stage, state):
    (status / f"{stage}.json").write_text(json.dumps(
        {"stage": stage, "state": state, "input_hash": "h", "reasons": [], "detail": {}}))
if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (run_dir / "grandchild.pid").write_text(str(child.pid))
    time.sleep(120)
if mode == "crash":
    mark("prepare", "ok"); mark("mesh", "failed")
    print("error: snappyHexMesh failed with exit code 1", flush=True)
    steps = [{"argv": ["run"], "exit": 2}]
else:
    for stage in ("prepare", "mesh", "solve", "post", "images"):
        mark(stage, "gate_failed" if (mode == "gate" and stage == "solve") else "ok")
    steps = [{"argv": ["run"], "exit": 1 if mode == "gate" else 0}, {"argv": ["images"], "exit": 0}]
(run_dir / "ui").mkdir(exist_ok=True)
(run_dir / "ui" / "outcome.json").write_text(json.dumps({"steps": steps}))
"""


def stub_command(job) -> list[str]:
    return [sys.executable, "-c", STUB, job.run_dir, job.design]


def spec(tmp_path: Path, name: str, mode: str = "ok", n_ranks: int = 40) -> JobSpec:
    return JobSpec(
        run_name=name, run_dir=str(tmp_path / "runs" / name), case_path="case.yaml",
        design=mode, state="corner", profile="dev", n_ranks=n_ranks,
    )


def wait_until(condition, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


@pytest.fixture
def queue(tmp_path: Path) -> Queue:
    return Queue(tmp_path / "ui.db")


def worker_for(queue: Queue, **kwargs) -> Worker:
    return Worker(queue, command=stub_command, kill_grace=1.0, **kwargs)


def run_to_end(worker: Worker, queue: Queue, job_id: int) -> str:
    def finished() -> bool:
        worker.tick()
        return queue.get(job_id).status not in ("queued", "running")

    wait_until(finished)
    return queue.get(job_id).status


def drain(worker: Worker, queue: Queue) -> None:
    """Cancel everything, queued first so finishing jobs do not start the next."""
    for job in queue.queued():
        worker.cancel(job.id)
    for job in queue.running():
        worker.cancel(job.id)
    wait_until(lambda: (worker.tick(), not queue.running())[1])


@pytest.mark.parametrize("mode, expected", [("ok", "done"), ("gate", "gate_failed")])
def test_a_job_runs_to_its_verdict(tmp_path, queue, mode, expected) -> None:
    job = queue.enqueue(spec(tmp_path, "a", mode))
    assert run_to_end(worker_for(queue), queue, job.id) == expected
    assert (Path(job.run_dir) / "ui" / "job.json").is_file()


def test_a_crash_is_failed_with_the_cli_message(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a", "crash"))
    assert run_to_end(worker_for(queue), queue, job.id) == "failed"
    assert queue.get(job.id).error == "snappyHexMesh failed with exit code 1"


def test_a_failed_job_does_not_stop_the_queue(tmp_path, queue) -> None:
    first = queue.enqueue(spec(tmp_path, "a", "crash"))
    second = queue.enqueue(spec(tmp_path, "b", "ok"))
    worker = worker_for(queue)
    run_to_end(worker, queue, first.id)
    assert run_to_end(worker, queue, second.id) == "done"


def test_one_at_a_time_by_default(tmp_path, queue) -> None:
    queue.enqueue(spec(tmp_path, "a", "sleep", n_ranks=8))
    queue.enqueue(spec(tmp_path, "b", "sleep", n_ranks=8))
    worker = worker_for(queue)
    worker.tick()
    worker.tick()
    try:
        assert len(queue.running()) == 1
    finally:
        drain(worker, queue)


def test_the_core_budget_allows_parallel_jobs_that_fit(tmp_path, queue) -> None:
    queue.set_setting("max_parallel", 2)
    queue.enqueue(spec(tmp_path, "a", "sleep", n_ranks=20))
    queue.enqueue(spec(tmp_path, "b", "sleep", n_ranks=30))
    queue.enqueue(spec(tmp_path, "c", "sleep", n_ranks=20))
    worker = worker_for(queue)
    for _ in range(3):
        worker.tick()
    try:
        assert sorted(j.run_name for j in queue.running()) == ["a", "c"]
    finally:
        drain(worker, queue)


def test_cancel_kills_the_whole_process_group(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a", "sleep"))
    worker = worker_for(queue)
    worker.tick()
    pid_file = Path(job.run_dir) / "grandchild.pid"
    wait_until(pid_file.exists)
    grandchild = int(pid_file.read_text())

    worker.cancel(job.id)
    assert run_to_end(worker, queue, job.id) == "cancelled"
    wait_until(lambda: not _pid_alive(grandchild))


def test_cancel_of_a_queued_job(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a"))
    worker_for(queue).cancel(job.id)
    assert queue.get(job.id).status == "cancelled"


def test_recover_adopts_a_live_job_and_finishes_it(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a"))
    Path(job.run_dir).mkdir(parents=True)
    survivor = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1)"], start_new_session=True
    )
    queue.mark_running(job.id, survivor.pid, survivor.pid)

    worker = worker_for(queue)
    worker.recover()
    assert queue.get(job.id).status == "running"
    survivor.wait()
    worker.tick()
    # No outcome file was written by this stand-in, so it cannot be "done".
    assert queue.get(job.id).status == "failed"


def test_recover_fails_a_job_whose_process_is_gone(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a"))
    gone = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    gone.wait()
    queue.mark_running(job.id, gone.pid, gone.pid)

    worker_for(queue).recover()
    assert queue.get(job.id).status == "failed"
    assert "restarted" in queue.get(job.id).error


def test_group_alive() -> None:
    assert group_alive(os.getpgid(0))
    assert not group_alive(2**22 + 12345)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it is dead for our purposes.
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().split()[2] != "Z"
    except FileNotFoundError:
        return False
