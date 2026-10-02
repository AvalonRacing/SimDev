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
