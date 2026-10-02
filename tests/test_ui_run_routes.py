from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from simdev.run.status import StageStatus, write_status  # noqa: E402
from simdev.ui.queue import JobSpec  # noqa: E402
from tests.ui_support import make_client  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def fake_run(runs_root: Path, name: str = "r1") -> Path:
    run_dir = runs_root / name
    coeffs = run_dir / "postProcessing" / "forceCoeffs" / "0"
    coeffs.mkdir(parents=True)
    shutil.copy(FIXTURES / "coefficient.dat", coeffs / "coefficient.dat")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "log.snappyHexMesh").write_text("ok\n--> FOAM FATAL ERROR: bad\n")
    (run_dir / "results").mkdir()
    (run_dir / "results" / "forces.png").write_bytes(b"\x89PNG fake")
    (run_dir / "caseSpec.json").write_text("{}")
    write_status(run_dir, StageStatus(stage="prepare", state="ok", input_hash="h"))
    write_status(run_dir, StageStatus(stage="mesh", state="failed", input_hash="h",
                                      reasons=["snappyHexMesh failed with exit code 1"]))
    return run_dir


def add_job(app, name: str, status: str) -> int:
    queue = app.state.ctx.queue
    job = queue.enqueue(JobSpec(name, str(app.state.ctx.config.runs_root / name),
                                "case.yaml", "v01", "corner", "car_dev", 8))
    if status in ("running", "failed"):
        queue.mark_running(job.id, 1, 1)
    if status == "failed":
        queue.mark_finished(job.id, "failed", 2, "snappyHexMesh failed with exit code 1")
    return job.id


def test_the_runs_list_includes_shell_runs(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        fake_run(tmp_path / "runs")
        assert "r1" in client.get("/runs").text


def test_the_run_page_shows_the_failure_and_the_log_context(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        fake_run(tmp_path / "runs")
        add_job(app, "r1", "failed")
        text = client.get("/runs/r1").text
        assert "snappyHexMesh failed with exit code 1" in text
        assert "FOAM FATAL ERROR" in text


def test_forces_api_is_incremental(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        fake_run(tmp_path / "runs")
        data = client.get("/api/runs/r1/forces?after=3").json()
        assert data["total"]["iteration"] == [4, 5]


def test_files_are_served_from_inside_the_run_only(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        fake_run(tmp_path / "runs")
        (tmp_path / "secret.txt").write_text("secret")
        assert client.get("/runs/r1/files/results/forces.png").status_code == 200
        assert client.get("/runs/r1/files/..%2F..%2Fsecret.txt").status_code == 404
        assert client.get("/runs/r1/logs/..%2F..%2Fsecret.txt").status_code == 404
        assert client.get("/runs/..%2Fsecret.txt").status_code == 404


def test_log_tail(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        fake_run(tmp_path / "runs")
        assert "FOAM FATAL ERROR" in client.get("/runs/r1/logs/log.snappyHexMesh").text


def test_resume_requeues_the_same_run(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        fake_run(tmp_path / "runs")
        add_job(app, "r1", "failed")
        client.post("/runs/r1/rerun", data={"stage": "mesh"}, follow_redirects=False)
        job = app.state.ctx.queue.latest_for("r1")
        assert (job.status, job.force_from) == ("queued", "mesh")


def test_strip_and_delete_are_refused_while_running(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        run_dir = fake_run(tmp_path / "runs")
        add_job(app, "r1", "running")
        response = client.post("/runs/r1/delete", follow_redirects=True)
        assert "still queued or running" in response.text
        assert run_dir.exists()


def test_delete_removes_the_run(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        run_dir = fake_run(tmp_path / "runs")
        add_job(app, "r1", "failed")
        client.post("/runs/r1/delete", follow_redirects=False)
        assert not run_dir.exists()


@pytest.mark.parametrize("action", ["strip", "delete"])
def test_a_run_started_from_the_shell_is_read_only(tmp_path: Path, action: str) -> None:
    with make_client(tmp_path) as (client, app):
        run_dir = fake_run(tmp_path / "runs")
        mesh = run_dir / "constant" / "polyMesh"
        mesh.mkdir(parents=True)
        (mesh / "points").write_text("()")
        response = client.post(f"/runs/r1/{action}", follow_redirects=True)
        assert "started from the shell" in response.text
        assert (mesh / "points").is_file()
        page = client.get("/runs/r1").text
        assert "/runs/r1/strip" not in page
        assert "/runs/r1/delete" not in page


def test_a_hostile_job_name_cannot_break_out_of_the_confirm_dialogs(tmp_path: Path) -> None:
    # Names reaching the page through a URL are restricted by RUN_NAME, so the
    # route cannot be fed a hostile name; render the template directly instead.
    name = "x');alert(1);('"
    with make_client(tmp_path) as (client, app):
        job = add_job(app, "r1", "running")
        stub = app.state.ctx.queue.get(job)
        page = app.state.ctx.templates.get_template("run.html").render(
            name=name, job=stub, stages=[], errors=[], images={"summary": [], "groups": {}},
            logs=[], live=False, stage_names=[], active=False,
        )
        assert "confirm('" not in page
        assert "confirm(&#39;" not in page
        assert page.count("\\u0027);alert(1);(\\u0027") == 3
