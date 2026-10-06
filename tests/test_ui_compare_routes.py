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


def make_pictured_run(root: Path, name: str) -> Path:
    run = make_run(root, name)
    (run / "results" / "render_plan.json").write_text("{}")
    return run


def test_compare_page_keeps_order_and_reference(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        make_pictured_run(tmp_path / "runs", "b")
        page = client.get("/compare?runs=b,a&ref=a").text
        data = _payload(page)
        assert data["runs"] == ["b", "a"] and data["ref"] == "a"
        assert "all_runs" not in data


def test_compare_page_is_pictures_only(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        page = client.get("/compare?runs=a").text
        assert "Δ against" not in page and "<h2>Numbers</h2>" not in page
        assert 'id="bars"' not in page


def test_picker_lists_runs_with_pictures_and_preselects_the_chosen(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        make_pictured_run(tmp_path / "runs", "b")
        make_run(tmp_path / "runs", "numbers_only")
        page = client.get("/compare?runs=b,a&ref=a").text
        assert page.count('<select name="runs"') == 4
        assert '<select name="ref"' in page
        assert '<option value="b" selected>' in page and '<option value="a" selected>' in page
        assert "numbers_only" not in page


def test_empty_picker_shows_a_hint_not_a_results_link(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        page = client.get("/compare").text
        assert '<select name="runs"' in page and "Tick runs" not in page
        assert "Choose runs" in page


def test_run_without_pictures_is_not_offered(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        page = client.get("/compare?runs=a").text
        assert '<option value="a"' in page and '<option value="b"' not in page


def test_current_selection_without_pictures_round_trips_through_the_picker(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        page = client.get("/compare?runs=b,a&ref=b").text
        assert '<option value="b" selected>' in page and '<option value="a" selected>' in page
        assert page.count('<option value="b" selected>') == 2  # pane 1 and REF


def test_picker_sends_repeated_runs_parameters(tmp_path: Path) -> None:
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


def test_hostile_directory_name_cannot_break_out_of_the_data_script(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        make_pictured_run(tmp_path / "runs", "x</script><img src=x onerror=alert(1)>")
        page = client.get("/compare?runs=a").text
        assert "</script><img" not in page
        data = json.loads(page.split('<script id="compare-data" type="application/json">')[1]
                          .split("</script>")[0])


def test_payload_escapes_angle_brackets(tmp_path: Path, monkeypatch) -> None:
    from simdev.ui import routes_compare

    monkeypatch.setattr(routes_compare.runview, "list_runs",
                        lambda root: [root / "a", root / "<b>"])
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        (tmp_path / "runs" / "<b>" / "results").mkdir(parents=True)
        (tmp_path / "runs" / "<b>" / "results" / "render_plan.json").write_text("{}")
        page = client.get("/compare?runs=a").text
        assert "<b>" not in page.split('id="compare-data"')[1].split("</script>")[0]


def test_cplines_nan_becomes_null_and_corrupt_csv_is_404(tmp_path: Path, monkeypatch) -> None:
    from simdev.ui import routes_compare

    nan = float("nan")
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        monkeypatch.setattr(routes_compare, "cp_station", lambda *a: {
            "u_inf": 10.0, "patches": {"Body": {"x": [0.0, nan], "z": [float("inf"), 1.0],
                                                  "cp": [nan, 0.5]}}})
        body = client.get("/api/runs/a/cplines/s").text
        assert "NaN" not in body and "Infinity" not in body
        assert json.loads(body)["patches"]["Body"]["cp"] == [None, 0.5]

        def corrupt(*a):
            raise ValueError("bad row")
        monkeypatch.setattr(routes_compare, "cp_station", corrupt)
        response = client.get("/api/runs/a/cplines/s")
        assert response.status_code == 404 and "bad row" in response.json()["detail"]


def test_the_compare_script_is_served(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        script = client.get("/static/compare.js")
        assert script.status_code == 200 and "SimdevCompare" in script.text


def _payload(page: str) -> dict:
    return json.loads(page.split('<script id="compare-data" type="application/json">')[1]
                      .split("</script>")[0])


def test_reference_defaults_to_the_first_run_and_falls_back(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        make_pictured_run(tmp_path / "runs", "b")
        assert _payload(client.get("/compare?runs=b,a").text)["ref"] == "b"
        assert _payload(client.get("/compare?runs=b,a&ref=a").text)["ref"] == "a"
        assert _payload(client.get("/compare?runs=b,a&ref=zzz").text)["ref"] == "b"


def test_blank_picker_selects_are_ignored(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        page = client.get("/compare?runs=a&runs=&runs=&runs=&ref=")
        assert _payload(page.text)["runs"] == ["a"] and "not found" not in page.text


def test_runs_beyond_four_are_named_in_the_warning(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        for name in "abcdef":
            make_pictured_run(tmp_path / "runs", name)
        page = client.get("/compare?runs=a,b,c,d,e,f").text
        assert _payload(page)["runs"] == ["a", "b", "c", "d"]
        assert "max 4 runs: left out e, f" in page


def test_a_busy_delta_answers_503_with_retry_after(tmp_path: Path, monkeypatch) -> None:
    from simdev.ui import routes_compare
    from simdev.ui.delta import DeltaError

    def busy(*a, **k):
        raise DeltaError("another delta is being computed", 503)

    monkeypatch.setattr(routes_compare, "delta_png", busy)
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        response = client.get("/api/delta/b.png?ref=a&view=x_+0.000&field=cp")
        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"


def test_nav_links_to_compare(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        assert 'href="/compare"' in client.get("/").text


def test_empty_compare_page_includes_the_restore_script(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_pictured_run(tmp_path / "runs", "a")
        empty = client.get("/compare").text
        assert "/static/compare-restore.js" in empty
        assert "/static/compare-restore.js" not in client.get("/compare?runs=a").text
        assert "/static/compare-state.js" in client.get("/compare?runs=a").text
        script = client.get("/static/compare-restore.js")
        assert script.status_code == 200 and "location.replace" in script.text
        assert client.get("/static/compare-state.js").status_code == 200


def test_results_page_remembers_its_filters(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        page = client.get("/results").text
        assert "simdev.results.v1" in page and "location.replace" in page
        assert "/results?clear=1" in page
