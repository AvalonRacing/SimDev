from __future__ import annotations

from pathlib import Path

import pytest

from simdev.report.tsv import (
    FIXED_COLUMNS,
    aggregate_reports,
    read_report,
    report_columns,
    write_report,
)

ROW: dict[str, object] = {
    "run": "car-01",
    "case_name": "car",
    "driving_state": "testcase",
    "spec_hash": "abc123",
    "timestamp": "2026-09-01T10:00:00+00:00",
    "verdict": "converged",
    "converged": True,
    "window_start": 300,
    "window_end": 400,
    "n_iterations": 400,
    "cd_amplitude": 0.022,
    "cl_amplitude": 0.079,
    "yplus_passed": True,
    "n_cells": 20259975,
    "Fx": 12.34,
    "Fy": -0.56,
    "Fz": -78.9,
    "Mx": 0.12,
    "My": 3.45,
    "Mz": -0.67,
    "cd": 0.99,
    "cd_std": 0.02,
    "cl": -1.62,
    "cl_std": 0.13,
    "cs": -0.04,
    "COP_x": 0.043,
    "COP_y": 0.001,
    "COP_z": 0.279,
    "balance_front_pct": 46.2,
    "cd_body": 0.4,
    "cl_body": -0.9,
    "cd_wing": 0.3,
    "cl_wing": -0.5,
    "cd_other": 0.29,
    "cl_other": -0.22,
}


def test_the_column_order_is_a_contract() -> None:
    """Pinned deliberately. A sheet with pasted rows breaks if columns move.

    New columns are APPENDED, never inserted. If this test fails because you
    inserted one, move it to the end instead of updating the expectation.
    """
    assert FIXED_COLUMNS[:5] == (
        "run", "case_name", "driving_state", "spec_hash", "timestamp",
    )
    assert FIXED_COLUMNS[5:9] == (
        "verdict", "converged", "window_start", "window_end",
    )
    assert FIXED_COLUMNS[-4:] == ("COP_x", "COP_y", "COP_z", "balance_front_pct")


def test_group_columns_follow_the_configured_order() -> None:
    columns = report_columns(["body", "wing", "other"])
    assert columns[len(FIXED_COLUMNS):] == [
        "cd_body", "cl_body", "cd_wing", "cl_wing", "cd_other", "cl_other",
    ]


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    write_report(tmp_path, ROW, ["body", "wing", "other"])
    text = (tmp_path / "results" / "report.tsv").read_text(encoding="utf-8")
    header, data = text.strip().splitlines()
    assert header.split("\t")[0] == "run"
    assert len(header.split("\t")) == len(data.split("\t"))
    assert read_report(tmp_path)["balance_front_pct"] == "46.2"


def test_an_absent_value_is_an_empty_field_not_nan(tmp_path: Path) -> None:
    """"nan" in a spreadsheet cell is a value. An empty cell is a gap.

    A COP that could not be computed must not arrive as text that Excel will
    happily average.
    """
    row = dict(ROW, COP_x=None, COP_z=float("nan"))
    write_report(tmp_path, row, ["body", "wing", "other"])
    fields = read_report(tmp_path)
    assert fields["COP_x"] == ""
    assert fields["COP_z"] == ""


def test_aggregate_puts_one_row_per_run(tmp_path: Path) -> None:
    for name in ("car-01", "car-02"):
        run = tmp_path / name
        write_report(run, dict(ROW, run=name), ["body", "wing", "other"])
    header, rows = aggregate_reports([tmp_path / "car-01", tmp_path / "car-02"])
    assert header[0] == "run"
    assert [r[0] for r in rows] == ["car-01", "car-02"]


def test_aggregate_skips_a_run_with_no_report(tmp_path: Path) -> None:
    write_report(tmp_path / "car-01", ROW, ["body", "wing", "other"])
    header, rows = aggregate_reports([tmp_path / "car-01", tmp_path / "nothing"])
    assert len(rows) == 1
