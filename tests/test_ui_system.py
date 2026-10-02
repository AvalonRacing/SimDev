from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from simdev.ui.serve import resolve_host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", " "])
def test_every_interface_is_refused(host: str) -> None:
    with pytest.raises(ValueError, match="every interface"):
        resolve_host(host)


def test_an_explicit_address_is_used() -> None:
    assert resolve_host("127.0.0.1") == "127.0.0.1"


def test_the_default_is_the_tailscale_address() -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="100.70.1.2\n", stderr="")

    assert resolve_host(None, which=lambda name: "/usr/bin/tailscale", run=fake_run) == "100.70.1.2"


def test_tailscale_down_is_explained() -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Tailscale is stopped.")

    with pytest.raises(ValueError, match="Tailscale is stopped"):
        resolve_host(None, which=lambda name: "/usr/bin/tailscale", run=fake_run)


def test_no_tailscale_at_all_is_explained(monkeypatch) -> None:
    monkeypatch.setattr("simdev.ui.serve.SNAP_TAILSCALE", Path("/nonexistent/tailscale"))
    with pytest.raises(ValueError, match="--host 127.0.0.1"):
        resolve_host(None, which=lambda name: None)


def test_system_page_and_settings(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from tests.ui_support import make_client

    with make_client(tmp_path) as (client, app):
        assert "core budget" in client.get("/system").text.lower()
        client.post("/system/settings", data={"core_budget": "40", "max_parallel": "2"},
                    follow_redirects=False)
        assert app.state.ctx.queue.setting("max_parallel") == 2
        response = client.post("/system/settings", data={"core_budget": "0", "max_parallel": "1"},
                               follow_redirects=True)
        assert "between" in response.text
        assert app.state.ctx.queue.setting("core_budget") == 40


def test_the_budget_cannot_drop_below_a_queued_job(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from simdev.ui.queue import JobSpec
    from tests.ui_support import make_client

    monkeypatch.setattr("os.cpu_count", lambda: 64)
    with make_client(tmp_path) as (client, app):
        queue = app.state.ctx.queue
        queue.enqueue(JobSpec("narrow", str(tmp_path / "runs" / "narrow"),
                              "case.yaml", "v01", "corner", "car_dev", 8))
        queue.enqueue(JobSpec("wide", str(tmp_path / "runs" / "wide"),
                              "case.yaml", "v01", "corner", "car_dev", 32))
        response = client.post("/system/settings", data={"core_budget": "16", "max_parallel": "1"},
                               follow_redirects=True)
        assert "below the 32 cores the queued run wide needs" in response.text
        assert queue.setting("core_budget") == 40
        client.post("/system/settings", data={"core_budget": "32", "max_parallel": "1"},
                    follow_redirects=False)
        assert queue.setting("core_budget") == 32


def _finished_run(app, tmp_path: Path) -> Path:
    from simdev.ui.queue import JobSpec

    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "caseSpec.json").write_text("{}")
    queue = app.state.ctx.queue
    job = queue.enqueue(JobSpec("r1", str(run_dir), "case.yaml", "v01", "corner", "car_dev", 8))
    queue.mark_running(job.id, 1, 1)
    queue.mark_finished(job.id, "failed", 2, "x")
    return run_dir


@pytest.mark.parametrize("header", ["origin", "referer"])
def test_a_post_from_another_site_is_refused(tmp_path: Path, header: str) -> None:
    pytest.importorskip("fastapi")
    from tests.ui_support import make_client

    with make_client(tmp_path) as (client, app):
        run_dir = _finished_run(app, tmp_path)
        response = client.post("/runs/r1/delete", headers={header: "http://evil.example/page"},
                               follow_redirects=False)
        assert response.status_code == 403
        assert run_dir.exists()
        # Reading is never refused: links from elsewhere must still open pages.
        assert client.get("/runs/r1", headers={header: "http://evil.example/"}).status_code == 200


def test_a_post_from_the_ui_itself_is_allowed(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from tests.ui_support import make_client

    with make_client(tmp_path) as (client, app):
        run_dir = _finished_run(app, tmp_path)
        response = client.post("/runs/r1/delete", headers={"origin": "http://testserver"},
                               follow_redirects=False)
        assert response.status_code == 303
        assert not run_dir.exists()


def test_a_different_port_is_another_site(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from tests.ui_support import make_client

    with make_client(tmp_path) as (client, app):
        run_dir = _finished_run(app, tmp_path)
        response = client.post("/runs/r1/delete", headers={"origin": "http://testserver:8001"},
                               follow_redirects=False)
        assert response.status_code == 403
        assert run_dir.exists()


def test_only_the_allowed_host_names_are_answered(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from tests.ui_support import make_client

    with make_client(tmp_path, allowed_hosts=["testserver"]) as (client, app):
        assert client.get("/").status_code == 200
        assert client.get("/", headers={"host": "rebound.example"}).status_code == 400


def test_the_allowed_hosts_include_the_magicdns_names() -> None:
    from simdev.ui.serve import allowed_hosts, tailscale_names

    def fake_run(argv, **kwargs):
        assert argv[1:] == ["status", "--json"]
        return subprocess.CompletedProcess(
            argv, 0, stdout='{"Self": {"DNSName": "z8.tail1234.ts.net."}}', stderr=""
        )

    names = tailscale_names(which=lambda name: "/usr/bin/tailscale", run=fake_run)
    assert names == ["z8.tail1234.ts.net", "z8"]
    assert allowed_hosts("100.70.1.2", 8000, names) == [
        "100.70.1.2", "100.70.1.2:8000", "localhost", "127.0.0.1", "z8.tail1234.ts.net", "z8",
    ]


def test_magicdns_names_are_best_effort(monkeypatch) -> None:
    from simdev.ui.serve import tailscale_names

    def broken(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="stopped")

    assert tailscale_names(which=lambda name: "/usr/bin/tailscale", run=broken) == []
    monkeypatch.setattr("simdev.ui.serve.SNAP_TAILSCALE", Path("/nonexistent/tailscale"))
    assert tailscale_names(which=lambda name: None) == []
