from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from tests.ui_support import make_client  # noqa: E402

FORM = {"pair": "v01/corner", "profile": "car_dev", "run_name": "r1",
        "flow.u_inf": "15", "physics.corner_radius": "4", "physics.corner_direction": "left",
        "solve.n_ranks": "4", "solve.max_iterations": ""}


def test_home_page_renders(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        assert client.get("/").status_code == 200


def test_new_run_form_lists_only_pairs_with_a_slot(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        page = client.get("/runs/new").text
        assert 'value="v01/corner"' in page
        assert 'value="v01/straight"' not in page


def test_queueing_stores_only_changed_overrides(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        response = client.post("/runs", data=FORM, follow_redirects=False)
        assert response.status_code == 303
        job = app.state.ctx.queue.latest_for("r1")
        assert job.overrides == {"solve.n_ranks": 4}
        assert job.n_ranks == 4
        assert job.run_dir == str(tmp_path / "runs" / "r1")
        assert "r1" in client.get("/").text


def test_a_bad_number_is_shown_and_nothing_is_queued(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        response = client.post("/runs", data={**FORM, "flow.u_inf": "fast"})
        assert response.status_code == 400
        assert "is not a number" in response.text
        assert app.state.ctx.queue.queued() == []


def test_a_path_in_the_run_name_is_refused(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        response = client.post("/runs", data={**FORM, "run_name": "../evil"})
        assert response.status_code == 400
        assert not (tmp_path / "evil").exists()


def test_preview_shows_the_resolved_numbers(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        text = client.post("/runs/preview", data=FORM).text
        assert "3.750 rad/s" in text


def test_editing_a_job_that_started_is_refused(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        client.post("/runs", data=FORM, follow_redirects=False)
        (tmp_path / "runs" / "r1").mkdir(parents=True)
        queue = app.state.ctx.queue
        job = queue.latest_for("r1")
        queue.mark_running(job.id, 1, 1)
        response = client.post("/runs", data={**FORM, "job_id": str(job.id), "solve.n_ranks": "2"})
        assert response.status_code == 400
        assert "already started" in response.text
        assert queue.get(job.id).n_ranks == 4


def test_reorder_and_cancel(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        client.post("/runs", data=FORM, follow_redirects=False)
        client.post("/runs", data={**FORM, "run_name": "r2"}, follow_redirects=False)
        queue = app.state.ctx.queue
        r2 = queue.latest_for("r2")
        client.post(f"/jobs/{r2.id}/move", data={"direction": "up"}, follow_redirects=False)
        assert [j.run_name for j in queue.queued()] == ["r2", "r1"]
        client.post(f"/jobs/{r2.id}/cancel", follow_redirects=False)
        assert queue.get(r2.id).status == "cancelled"


def test_a_hostile_run_name_cannot_break_out_of_the_confirm_dialog(tmp_path: Path) -> None:
    from simdev.ui.queue import JobSpec

    name = "x');alert(1);('"
    with make_client(tmp_path) as (client, app):
        queue = app.state.ctx.queue
        job = queue.enqueue(JobSpec(name, str(tmp_path / "runs" / "x"), "c", "v01", "corner", "car_dev", 4))
        queue.mark_running(getattr(job, "id", job), 1, 1)
        for url in ("/", "/partials/queue"):
            page = client.get(url).text
            assert "confirm('Cancel x');alert(1)" not in page
            assert "confirm(&#39;" not in page
            assert "\\u0027);alert(1);(\\u0027" in page


def test_a_run_wider_than_the_core_budget_is_refused(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        app.state.ctx.queue.set_setting("core_budget", 2)
        response = client.post("/runs", data=FORM)
        assert response.status_code == 400
        assert "needs 4 cores but the core budget is 2" in response.text
        assert app.state.ctx.queue.queued() == []


def test_editing_a_job_beyond_the_core_budget_is_refused(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        client.post("/runs", data=FORM, follow_redirects=False)
        job = app.state.ctx.queue.latest_for("r1")
        app.state.ctx.queue.set_setting("core_budget", 6)
        response = client.post("/runs", data={**FORM, "job_id": str(job.id), "solve.n_ranks": "8"})
        assert response.status_code == 400
        assert "needs 8 cores but the core budget is 6" in response.text
        assert app.state.ctx.queue.get(job.id).n_ranks == 4


def test_a_queued_job_wider_than_the_budget_says_why_it_waits(tmp_path: Path) -> None:
    from simdev.ui.queue import JobSpec

    with make_client(tmp_path) as (client, app):
        # A resumed job keeps the width it was queued with, whatever the budget is now.
        app.state.ctx.queue.enqueue(JobSpec("wide", str(tmp_path / "runs" / "wide"),
                                            "case.yaml", "v01", "corner", "car_dev", 64))
        assert "needs 64 cores, budget is 40" in client.get("/").text


def test_the_change_note_is_queued_and_prefilled_when_editing(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        client.post("/runs", data={**FORM, "note": "rear deck +3 mm"}, follow_redirects=False)
        job = app.state.ctx.queue.latest_for("r1")
        assert job.note == "rear deck +3 mm"
        assert "rear deck +3 mm" in client.get(f"/runs/new?job={job.id}").text


def test_the_new_run_form_offers_a_change_note(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        assert 'name="note"' in client.get("/runs/new").text
