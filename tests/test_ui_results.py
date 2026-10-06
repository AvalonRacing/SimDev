from __future__ import annotations

import json
from pathlib import Path

from simdev.ui.notes import write_note
from simdev.ui.results import columns_for, delta, group_names, load_row, load_rows, to_tsv


def make_run(root: Path, name: str, **result) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "caseSpec.json").write_text("{}")
    payload = {"cd_mean": 0.80, "cl_mean": -1.00, "window_start": 1, "window_end": 2,
               "verdict": "converged", "n_iterations": 400, "n_cells": 1000, **result}
    (run / "results" / "result.json").write_text(json.dumps(payload))
    return run


def test_an_old_result_without_groups_or_forces_loads_with_empty_cells(tmp_path: Path) -> None:
    row = load_row(make_run(tmp_path, "old"))
    assert row.has_result
    assert row.values["cl"] == -1.0
    assert row.values["fx"] is None and row.values["cop_x"] is None
    assert row.values["eff"] == 1.0 / 0.8


def test_groups_become_columns(tmp_path: Path) -> None:
    row = load_row(make_run(tmp_path, "g", groups={"body": {"Cd": 0.5, "Cl": -0.9},
                                                    "wing": {"Cd": 0.1, "Cl": -0.2}}))
    assert group_names([row]) == ["body", "wing"]
    keys = [c.key for c in columns_for(["body", "wing"])]
    assert keys[:5] == ["cl", "cd", "fx", "fz", "fy"]
    assert "cl_body" in keys and "cd_wing" in keys
    assert row.values["cl_body"] == -0.9


def test_delta_tones_follow_better_direction_and_noise(tmp_path: Path) -> None:
    ref = load_row(make_run(tmp_path, "ref", cl_mean=-1.00, cd_mean=0.80))
    run = load_row(make_run(tmp_path, "new", cl_mean=-1.10, cd_mean=0.81,
                            balance_front_pct=41.0))
    cols = columns_for([])
    # no histories: noise unknown, so tones go by direction alone
    cells = delta(run, ref, cols)
    assert cells["cl"].tone == "better" and round(cells["cl"].value, 6) == -0.1
    assert cells["cd"].tone == "worse"
    assert cells["balance"].tone == "none"  # ref has no balance


def test_a_delta_inside_the_noise_is_greyed(tmp_path: Path) -> None:
    from dataclasses import replace

    ref = load_row(make_run(tmp_path, "ref"))
    run = load_row(make_run(tmp_path, "new", cl_mean=-1.005))
    run = replace(run, noise={**run.noise, "cl": 0.01})
    ref = replace(ref, noise={**ref.noise, "cl": 0.01})
    assert delta(run, ref, columns_for([]))["cl"].tone == "noise"


def test_rows_carry_note_state_and_runs_without_results(tmp_path: Path) -> None:
    run = make_run(tmp_path, "a")
    (run / "results" / "report.tsv").write_text("run\tdriving_state\na\ttc10-low\n")
    write_note(run, "<script>x</script>", "b")
    (tmp_path / "pending").mkdir()
    (tmp_path / "pending" / "caseSpec.json").write_text("{}")
    rows = {r.name: r for r in load_rows(tmp_path)}
    assert rows["a"].state == "tc10-low"
    assert rows["a"].note.compare_with == "b"
    assert rows["pending"].has_result is False


def test_tsv_has_a_delta_line_under_each_referenced_run(tmp_path: Path) -> None:
    ref = load_row(make_run(tmp_path, "ref"))
    run_dir = make_run(tmp_path, "new", cl_mean=-1.1)
    write_note(run_dir, "deck", "ref")
    run = load_row(run_dir)
    text = to_tsv([run, ref], columns_for([]), {"ref": ref, "new": run})
    lines = text.splitlines()
    assert lines[0].split("\t")[:4] == ["run", "design", "state", "note"]
    assert lines[1].startswith("new\t")
    assert lines[2].startswith("Δ new − ref\t")
    assert lines[3].startswith("ref\t")


def test_an_undecodable_report_does_not_break_the_row(tmp_path: Path) -> None:
    run = make_run(tmp_path, "a")
    (run / "results" / "report.tsv").write_bytes(b"run\tdriving_state\na\t\xff\xfe\x80bad\n")
    row = load_row(run)
    assert row.state == "" and row.has_result


def test_columns_follow_the_benchmark_sheet_order() -> None:
    # Aeroexcel TC10: cz_a cx_a Fx Fz Fy, group Cz then group Cx, COP, efficiency, balance.
    keys = [c.key for c in columns_for(["body", "other", "wing"])]
    assert keys == ["cl", "cd", "fx", "fz", "fy",
                    "cl_body", "cl_wing", "cd_body", "cd_wing",
                    "cop_x", "cop_y", "cop_z", "eff", "balance",
                    "cl_other", "cd_other", "cs"]


def test_table_column_groups_are_contiguous() -> None:
    from simdev.ui.results import table_columns

    labels = [c.group for c in table_columns(columns_for(["body", "wing", "rear"]))]
    collapsed = [g for i, g in enumerate(labels) if i == 0 or labels[i - 1] != g]
    assert len(collapsed) == len(set(collapsed)), collapsed
    assert "other groups" in collapsed and next(
        c for c in table_columns(columns_for([])) if c.key == "cs").group == "coefficients"
