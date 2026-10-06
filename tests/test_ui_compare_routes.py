from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from tests.ui_support import make_client  # noqa: E402


def make_run(root: Path, name: str, cl: float = -1.0) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "caseSpec.json").write_text("{}")
    (run / "results" / "result.json").write_text(json.dumps(
        {"cd_mean": 0.8, "cl_mean": cl, "window_start": 1, "window_end": 2, "spec_hash": "s"}))
    return run


def test_compare_page_keeps_order_and_reference(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b", cl=-1.2)
        page = client.get("/compare?runs=b,a&ref=a").text
        data = json.loads(page.split('<script id="compare-data" type="application/json">')[1]
                          .split("</script>")[0])
        assert data["runs"] == ["b", "a"] and data["ref"] == "a"
        assert sorted(data["all_runs"]) == ["a", "b"]
        assert "Δ against a" in page


def test_ticked_checkboxes_arrive_as_repeated_parameters(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        assert client.get("/compare?runs=a&runs=b").status_code == 200


def test_unknown_and_hostile_names_are_dropped_with_a_warning(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        page = client.get("/compare?runs=a,nope,..%2Fetc").text
        assert "not found" in page


def test_index_summary_and_cplines_apis(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        assert client.get("/api/runs/a/index").json()["planes"] == {}
        assert client.get("/api/runs/a/summary").json()["window"] == [1, 2]
        patch = tmp_path / "runs/a/postProcessing/forceCoeffs_Body/0/coefficient.dat"
        patch.parent.mkdir(parents=True)
        patch.write_text("# Time Cd Cs Cl CmRoll CmPitch CmYaw\n5 0.1 0 -0.1 0 0 0\n")
        body = client.get("/api/runs/a/summary").text  # window 1-2 holds no rows: NaN
        assert "NaN" not in body and json.loads(body)["patches"]["Body"]["Cd"] is None
        assert client.get("/api/runs/a/cplines/y_+0.000").status_code == 404


def test_delta_endpoint_reports_errors_as_json(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        response = client.get("/api/delta/b.png?ref=a&view=x_+0.000&field=cp")
        assert response.status_code == 404
        assert "render_plan" in response.json()["error"]


def test_delta_endpoint_serves_the_png(tmp_path: Path, monkeypatch) -> None:
    from simdev.ui import routes_compare

    picture = tmp_path / "d.png"
    picture.write_bytes(b"\x89PNG")
    monkeypatch.setattr(routes_compare, "delta_png", lambda *a, **k: picture)
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        response = client.get("/api/delta/b.png?ref=a&view=x_+0.000&field=cp&limit=0.1")
        assert response.status_code == 200 and response.content == b"\x89PNG"
