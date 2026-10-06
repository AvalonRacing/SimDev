from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from simdev.ui.notes import read_note  # noqa: E402
from tests.ui_support import make_client  # noqa: E402


def make_run(root: Path, name: str, cl: float = -1.0) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "caseSpec.json").write_text("{}")
    (run / "results" / "result.json").write_text(json.dumps(
        {"cd_mean": 0.8, "cl_mean": cl, "window_start": 1, "window_end": 2,
         "verdict": "converged"}))
    return run


def test_results_lists_runs_with_their_numbers(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        page = client.get("/results").text
        assert "base" in page and "-1.0000" in page
        assert 'href="/results"' in client.get("/").text


def test_editing_note_and_reference_inline(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        new = make_run(tmp_path / "runs", "new", cl=-1.2)
        response = client.post("/runs/new/note", data={"note": "<b>deck</b>", "compare_with": "base"})
        assert response.status_code == 200
        assert "&lt;b&gt;deck&lt;/b&gt;" in response.text
        assert "Δ" in response.text
        assert read_note(new).compare_with == "base"


def test_shell_runs_can_be_annotated(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        run = make_run(tmp_path / "runs", "shellrun")
        assert app.state.ctx.queue.latest_for("shellrun") is None
        assert client.post("/runs/shellrun/note", data={"note": "x", "compare_with": ""}).status_code == 200
        assert read_note(run).note == "x"


@pytest.mark.parametrize("bad", ["new", "../base", "a/b", "nope!"])
def test_bad_references_are_refused(tmp_path: Path, bad: str) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "new")
        response = client.post("/runs/new/note", data={"note": "", "compare_with": bad})
        assert response.status_code == 400


def test_a_deleted_reference_shows_missing(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "new")
        (tmp_path / "runs" / "new" / "ui").mkdir()
        (tmp_path / "runs" / "new" / "ui" / "note.json").write_text(
            json.dumps({"note": "", "compare_with": "gone"}))
        page = client.get("/results").text
        assert "missing" in page


def test_tsv_download(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        response = client.get("/results.tsv")
        assert response.headers["content-type"].startswith("text/tab-separated-values")
        assert response.text.splitlines()[1].startswith("base\t")


def test_the_run_page_links_to_its_comparison(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        make_run(tmp_path / "runs", "newer")
        client.post("/runs/newer/note", data={"note": "", "compare_with": "base"})
        assert "/compare?runs=base,newer" in client.get("/runs/newer").text


def test_a_garbled_report_does_not_break_the_results_page(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        run = make_run(tmp_path / "runs", "base")
        (run / "results" / "report.tsv").write_bytes(b"run\tdriving_state\nbase\t\xff\xfe\n")
        response = client.get("/results")
        assert response.status_code == 200 and "base" in response.text
