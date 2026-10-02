from __future__ import annotations

from pathlib import Path

import pytest

from simdev.ui.queue import JobSpec, Queue, QueueError


def spec(name: str, n_ranks: int = 40, **extra) -> JobSpec:
    return JobSpec(
        run_name=name, run_dir=f"/runs/{name}", case_path="cases/car/config.yaml",
        design="v01", state="corner", profile="car_dev", n_ranks=n_ranks, **extra,
    )


@pytest.fixture
def queue(tmp_path: Path) -> Queue:
    ticks = iter(range(1, 10_000))
    return Queue(tmp_path / "ui.db", clock=lambda: float(next(ticks)))


def test_jobs_come_out_in_the_order_they_went_in(queue: Queue) -> None:
    a, b = queue.enqueue(spec("a")), queue.enqueue(spec("b"))
    assert [j.id for j in queue.queued()] == [a.id, b.id]
    assert queue.next_fitting(40).id == a.id


def test_overrides_round_trip(queue: Queue) -> None:
    job = queue.enqueue(spec("a", overrides={"flow.u_inf": 18.0}))
    assert queue.get(job.id).overrides == {"flow.u_inf": 18.0}


def test_move_swaps_neighbours(queue: Queue) -> None:
    a, b, c = (queue.enqueue(spec(n)) for n in "abc")
    queue.move(c.id, -1)
    assert [j.run_name for j in queue.queued()] == ["a", "c", "b"]
    queue.move(a.id, -1)  # already first: no-op
    assert [j.run_name for j in queue.queued()] == ["a", "c", "b"]


def test_a_name_cannot_be_queued_twice(queue: Queue) -> None:
    queue.enqueue(spec("a"))
    with pytest.raises(QueueError, match="already uses the name"):
        queue.enqueue(spec("a"))


def test_next_fitting_skips_jobs_that_do_not_fit(queue: Queue) -> None:
    queue.enqueue(spec("big", n_ranks=40))
    small = queue.enqueue(spec("small", n_ranks=20))
    assert queue.next_fitting(20).id == small.id
    assert queue.next_fitting(10) is None


def test_lifecycle(queue: Queue) -> None:
    job = queue.enqueue(spec("a"))
    queue.mark_running(job.id, pid=123, pgid=123)
    running = queue.get(job.id)
    assert (running.status, running.pid, running.started_at is not None) == ("running", 123, True)
    queue.mark_finished(job.id, "gate_failed", 1, None)
    done = queue.get(job.id)
    assert done.status == "gate_failed"
    assert queue.finished()[0].id == job.id


def test_a_started_job_cannot_be_edited_or_moved(queue: Queue) -> None:
    job = queue.enqueue(spec("a"))
    queue.mark_running(job.id, pid=1, pgid=1)
    with pytest.raises(QueueError, match="already started"):
        queue.update(job.id, spec("a", n_ranks=8))
    with pytest.raises(QueueError, match="only queued"):
        queue.move(job.id, 1)
    assert queue.get(job.id).n_ranks == 40


def test_transitions_are_checked(queue: Queue) -> None:
    job = queue.enqueue(spec("a"))
    with pytest.raises(QueueError):
        queue.mark_finished(job.id, "done", 0, None)
    queue.mark_cancelled(job.id)
    with pytest.raises(QueueError):
        queue.mark_running(job.id, pid=1, pgid=1)


def test_requeue_makes_a_new_job_for_the_same_run(queue: Queue) -> None:
    job = queue.enqueue(spec("a"))
    queue.mark_running(job.id, 1, 1)
    queue.mark_finished(job.id, "failed", 2, "boom")
    again = queue.requeue(job.id, force_from="mesh")
    assert again.id != job.id
    assert (again.run_name, again.status, again.force_from) == ("a", "queued", "mesh")
    assert queue.latest_for("a").id == again.id


def test_requeue_refuses_an_active_job(queue: Queue) -> None:
    job = queue.enqueue(spec("a"))
    with pytest.raises(QueueError):
        queue.requeue(job.id)


def test_settings_default_and_persist(tmp_path: Path) -> None:
    queue = Queue(tmp_path / "ui.db")
    assert queue.setting("core_budget") == 40
    assert queue.setting("max_parallel") == 1
    queue.set_setting("max_parallel", 2)
    assert Queue(tmp_path / "ui.db").setting("max_parallel") == 2
