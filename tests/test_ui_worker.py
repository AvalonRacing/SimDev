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
from simdev.ui.worker import Worker, current_boot_id, group_alive

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
if mode == "stubborn":
    code = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "open(sys.argv[1] + '/grandchild.pid', 'w').write(str(__import__('os').getpid()))\n"
        "time.sleep(120)\n"
    )
    subprocess.Popen([sys.executable, "-c", code, str(run_dir)])
    time.sleep(120)
if mode == "crash":
    mark("prepare", "ok"); mark("mesh", "failed")
    print("error: snappyHexMesh failed with exit code 1", flush=True)
    steps = [{"argv": ["run"], "exit": 2}]
else:
    for stage in ("prepare", "mesh", "solve", "post", "images"):
        mark(stage, "gate_failed" if (mode == "gate" and stage == "solve") else "ok")
    steps = [{"argv": ["run"], "exit": 1 if mode == "gate" else 0}, {"argv": ["images"], "exit": 0}]
if mode == "badstatus":
    (status / "solve.json").write_text("{")
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
    _write_process_file(job, survivor.pid, current_boot_id())

    try:
        worker = worker_for(queue)
        worker.recover()
        assert queue.get(job.id).status == "running"
        survivor.wait()
        worker.tick()
        # No outcome file was written by this stand-in, so it cannot be "done".
        assert queue.get(job.id).status == "failed"
    finally:
        survivor.kill()
        survivor.wait()


def test_recover_does_not_adopt_a_group_from_another_boot(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a"))
    Path(job.run_dir).mkdir(parents=True)
    survivor = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        queue.mark_running(job.id, survivor.pid, survivor.pid)
        _write_process_file(job, survivor.pid, "some-earlier-boot")
        worker_for(queue).recover()
        assert queue.get(job.id).status == "failed"
        assert "restarted" in queue.get(job.id).error
    finally:
        survivor.kill()
        survivor.wait()


def test_recover_does_not_adopt_a_job_without_a_process_record(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a"))
    Path(job.run_dir).mkdir(parents=True)
    survivor = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        queue.mark_running(job.id, survivor.pid, survivor.pid)
        worker_for(queue).recover()
        assert queue.get(job.id).status == "failed"
    finally:
        survivor.kill()
        survivor.wait()


def _write_process_file(job, pgid: int, boot_id: str) -> None:
    import json

    (Path(job.run_dir) / "ui").mkdir(parents=True, exist_ok=True)
    (Path(job.run_dir) / "ui" / "process.json").write_text(
        json.dumps({"pgid": pgid, "boot_id": boot_id})
    )


def test_recover_fails_a_job_whose_process_is_gone(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a"))
    gone = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    gone.wait()
    queue.mark_running(job.id, gone.pid, gone.pid)

    worker_for(queue).recover()
    assert queue.get(job.id).status == "failed"
    assert "restarted" in queue.get(job.id).error


def test_group_alive(monkeypatch) -> None:
    assert group_alive(os.getpgid(0))
    assert not group_alive(2**22 + 12345)

    def denied(pgid, sig):
        raise PermissionError

    monkeypatch.setattr(os, "killpg", denied)
    assert not group_alive(os.getpgid(0))


def test_a_truncated_status_file_fails_the_job_instead_of_sticking(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a", "badstatus"))
    assert run_to_end(worker_for(queue), queue, job.id) == "failed"
    assert queue.get(job.id).error.startswith("could not classify the result")


def _start_stubborn(tmp_path, queue):
    job = queue.enqueue(spec(tmp_path, "a", "stubborn"))
    queue.enqueue(spec(tmp_path, "b", "ok"))
    worker = worker_for(queue)
    worker.tick()
    pid_file = Path(job.run_dir) / "grandchild.pid"
    wait_until(pid_file.exists)
    wait_until(lambda: pid_file.read_text() != "")
    return worker, job, int(pid_file.read_text())


def test_cancel_escalates_to_sigkill_while_ranks_survive_the_leader(tmp_path, queue) -> None:
    worker, job, grandchild = _start_stubborn(tmp_path, queue)
    try:
        worker.cancel(job.id)
        leader = queue.get(job.id).pid
        wait_until(lambda: not _pid_alive(leader))
        worker.tick()
        # The leader is gone but the grandchild ignores SIGTERM: still running.
        assert queue.get(job.id).status == "running"
        assert _pid_alive(grandchild)
        wait_until(lambda: (worker.tick(), queue.get(job.id).status != "running")[1])
        assert queue.get(job.id).status == "cancelled"
        wait_until(lambda: not _pid_alive(grandchild))
    finally:
        _kill(grandchild)
        drain(worker, queue)


def test_the_next_job_waits_for_the_whole_group_to_die(tmp_path, queue) -> None:
    worker, job, grandchild = _start_stubborn(tmp_path, queue)
    try:
        worker.cancel(job.id)
        leader = queue.get(job.id).pid
        wait_until(lambda: not _pid_alive(leader))
        worker.tick()
        assert [j.run_name for j in queue.running()] == ["a"]
        assert [j.run_name for j in queue.queued()] == ["b"]
    finally:
        _kill(grandchild)
        drain(worker, queue)


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


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


def test_a_job_that_cannot_start_fails_and_the_queue_moves_on(tmp_path, queue) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a regular file where the run directory's parent should be")
    broken = queue.enqueue(
        JobSpec(
            run_name="a", run_dir=str(blocker / "a"), case_path="case.yaml",
            design="ok", state="corner", profile="dev", n_ranks=40,
        )
    )
    second = queue.enqueue(spec(tmp_path, "b", "ok"))
    worker = worker_for(queue)
    worker.tick()
    failed = queue.get(broken.id)
    assert failed.status == "failed"
    assert failed.error.startswith("could not start: ")
    assert run_to_end(worker, queue, second.id) == "done"


def test_a_lost_process_record_does_not_start_the_job_twice(tmp_path, queue, monkeypatch) -> None:
    import simdev.ui.worker as worker_module

    def disk_full(run_dir, pgid):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(worker_module, "write_process_file", disk_full)
    starts = []

    def counting(job):
        starts.append(job.id)
        return stub_command(job)

    job = queue.enqueue(spec(tmp_path, "a", "ok"))
    worker = Worker(queue, command=counting, kill_grace=1.0)
    worker.tick()
    running = queue.get(job.id)
    assert running.status == "running"
    assert running.pgid == running.pid and running.pid is not None
    assert run_to_end(worker, queue, job.id) == "done"
    for _ in range(3):
        worker.tick()
    assert starts == [job.id]


def test_a_claimed_job_without_a_process_is_lost(tmp_path, queue) -> None:
    job = queue.enqueue(spec(tmp_path, "a"))
    queue.mark_running(job.id)
    worker_for(queue).recover()
    assert queue.get(job.id).status == "failed"
    assert "restarted" in queue.get(job.id).error


def test_a_tick_fails_a_claimed_job_this_worker_never_started(tmp_path, queue) -> None:
    worker = worker_for(queue)
    worker.recover()
    job = queue.enqueue(spec(tmp_path, "a"))
    queue.mark_running(job.id)
    worker.tick()
    assert queue.get(job.id).status == "failed"
    assert "restarted" in queue.get(job.id).error


def test_the_worker_writes_the_job_note_into_the_run(tmp_path: Path) -> None:
    import json

    queue = Queue(tmp_path / "q.db")
    run_dir = tmp_path / "runs" / "r1"
    job = queue.enqueue(JobSpec("r1", str(run_dir), "case.yaml", "ok", "corner", "car_dev", 1,
                                note="new rear deck"))
    worker = Worker(queue, command=stub_command, cwd=tmp_path)
    worker._spawn(job)
    worker._procs[job.id].wait(timeout=60)
    assert json.loads((run_dir / "ui" / "note.json").read_text())["note"] == "new rear deck"
