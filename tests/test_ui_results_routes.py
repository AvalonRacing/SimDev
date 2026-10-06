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


def test_results_table_is_not_linked_to_the_pictures_and_the_note_comes_last(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        page = client.get("/results").text
        table = page.split("<table")[1]
        # (the column picker above the table has checkboxes; the table itself has none)
        assert 'type="checkbox"' not in table and "compare-form" not in page
        assert "/compare" not in table
        head = page.split("<thead>")[1].split("</thead>")[0]
        assert head.index("design") < head.index("compare with") < head.index("Cl") < head.index("note")


def make_job(run: Path, state: str, design: str) -> None:
    (run / "ui").mkdir(exist_ok=True)
    (run / "ui" / "job.json").write_text(json.dumps({"state": state, "design": design}))


def test_the_table_shows_the_design_with_the_run_name_on_hover(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_job(make_run(tmp_path / "runs", "wing-v2-tc10-low-20261006"), "tc10-low", "wing-v2")
        make_run(tmp_path / "runs", "shellrun")
        page = client.get("/results").text
        assert ('<a href="/runs/wing-v2-tc10-low-20261006" '
                'title="wing-v2-tc10-low-20261006">wing-v2</a>') in page
        # started from the shell: no design on record, so the run name stands in
        assert '<a href="/runs/shellrun" title="shellrun">shellrun</a>' in page


def test_the_driving_state_is_its_own_column(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_job(make_run(tmp_path / "runs", "base"), "tc10-low", "wing-v2")
        page = client.get("/results").text
        head = page.split("<thead>")[1].split("</thead>")[0]
        assert 'class="c-state' in head
        assert 'class="c-state" title="tc10-low">tc10-low</td>' in page
        assert '<span class="muted">tc10-low</span>' not in page


def test_filters_still_match_the_full_run_name_and_the_state(tmp_path: Path) -> None:
    def shown(client, query: str) -> set[str]:
        # the compare-with selects list every run, so look at the row ids
        page = client.get(f"/results?{query}").text
        return {n for n in ("wing-v2-tc10-low-1", "other-tc10-high-1") if f'id="row-{n}"' in page}

    with make_client(tmp_path) as (client, app):
        make_job(make_run(tmp_path / "runs", "wing-v2-tc10-low-1"), "tc10-low", "wing-v2")
        make_job(make_run(tmp_path / "runs", "other-tc10-high-1"), "tc10-high", "other")
        assert shown(client, "q=tc10-low") == {"wing-v2-tc10-low-1"}
        assert shown(client, "state=tc10-high") == {"other-tc10-high-1"}
        assert shown(client, "design=wing-v2") == {"wing-v2-tc10-low-1"}


def test_tsv_keeps_the_run_name_first_and_adds_the_design(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_job(make_run(tmp_path / "runs", "base"), "tc10-low", "wing-v2")
        lines = client.get("/results.tsv").text.splitlines()
        assert lines[0].split("\t")[:3] == ["run", "design", "state"]
        assert lines[1].split("\t")[:3] == ["base", "wing-v2", "tc10-low"]


def test_tsv_cols_exports_only_the_listed_columns(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        head = client.get("/results.tsv?cols=cl,nonsense").text.splitlines()[0].split("\t")
        assert "Cl" in head and "Cd" not in head and "state" not in head
        assert "verdict" not in head and "note" not in head
        assert head[:2] == ["run", "design"] and "compare_with" in head
        everything = client.get("/results.tsv").text.splitlines()[0].split("\t")
        assert {"Cl", "Cd", "state", "note", "verdict"} <= set(everything)
        wanted = client.get("/results.tsv?cols=state,verdict,note,cd").text.splitlines()[0].split("\t")
        assert {"state", "verdict", "note", "Cd", "n_iterations"} <= set(wanted) and "Cl" not in wanted
        none = client.get("/results.tsv?cols=").text.splitlines()[0].split("\t")
        assert "Cl" not in none and none[:2] == ["run", "design"]


def test_the_page_offers_column_ticks_and_every_cell_carries_its_column_class(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        new = make_run(tmp_path / "runs", "new", cl=-1.2)
        page = client.get("/results").text
        assert "/static/results.js" in page and 'id="column-picker"' in page
        assert 'data-key="cl"' in page.split('id="column-picker"')[1].split("</details>")[0]
        assert '<col class="c-cl" data-key="cl"' in page and '<th class="c-cl num' in page
        # the compare-with and design columns cannot be hidden
        picker = page.split('id="column-picker"')[1].split("</details>")[0]
        assert 'data-key="design"' not in picker and 'data-key="compare"' not in picker
        swapped = client.post("/runs/new/note", data={"note": "", "compare_with": "base"}).text
        # the delta row and the run row both carry the class the hiding rule targets
        assert swapped.count('class="c-cl num') == 2 and 'class="c-design' in swapped


def test_default_hidden_columns_are_hidden_by_the_served_page(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        page = client.get("/results").text
        style = page.split('<style id="column-hide">')[1].split("</style>")[0]
        assert ".c-fx { display: none; }" in style and ".c-cl " not in style
