from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdev.report.results import ResultRecord, aggregate, read_result, write_result


def _record(**overrides: object) -> ResultRecord:
    base = dict(
        case_name="ahmed",
        spec_hash="abc123",
        timestamp="2026-08-09T12:00:00Z",
        verdict="converged",
        converged=True,
        cd_mean=0.347,
        cd_std=0.0004,
        cl_mean=-0.102,
        cl_std=0.0006,
        window_start=100,
        window_end=300,
        n_iterations=300,
        n_cells=1122334,
        yplus_passed=True,
        yplus={"body": 95.4},
        reasons=[],
    )
    base.update(overrides)
    return ResultRecord(**base)


def test_write_result_produces_json_and_csv(tmp_path: Path) -> None:
    json_path, csv_path = write_result(tmp_path, _record())
    assert json_path.exists() and csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["spec_hash"] == "abc123"


def test_result_round_trips(tmp_path: Path) -> None:
    write_result(tmp_path, _record())
    assert read_result(tmp_path).cd_mean == pytest.approx(0.347)


def test_csv_holds_exactly_one_row(tmp_path: Path) -> None:
    _, csv_path = write_result(tmp_path, _record())
    lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2  # header plus one row


def test_rewriting_replaces_rather_than_appends(tmp_path: Path) -> None:
    # The benchmark pipeline appended to a shared table and corrupted it.
    write_result(tmp_path, _record())
    _, csv_path = write_result(tmp_path, _record(cd_mean=0.9))
    lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert read_result(tmp_path).cd_mean == pytest.approx(0.9)


def test_aggregate_combines_independent_runs(tmp_path: Path) -> None:
    dirs = []
    for i, cd in enumerate([0.34, 0.35, 0.36]):
        d = tmp_path / f"run{i}"
        d.mkdir()
        write_result(d, _record(cd_mean=cd, spec_hash=f"hash{i}"))
        dirs.append(d)
    df = aggregate(dirs)
    assert len(df) == 3
    assert set(df["spec_hash"]) == {"hash0", "hash1", "hash2"}


def test_non_converged_record_is_written_and_flagged(tmp_path: Path) -> None:
    write_result(
        tmp_path,
        _record(verdict="not_converged", converged=False, reasons=["Cd still drifting"]),
    )
    loaded = read_result(tmp_path)
    assert loaded.converged is False
    assert loaded.verdict == "not_converged"
    assert loaded.reasons == ["Cd still drifting"]


def test_an_unjudged_record_round_trips_as_unjudged(tmp_path: Path) -> None:
    """The state that a boolean could not hold. It has to survive the write,
    or the distinction is lost exactly where it is needed - in the file a
    sweep reads back."""
    write_result(tmp_path, _record(verdict="not_judged", converged=False))
    loaded = read_result(tmp_path)
    assert loaded.verdict == "not_judged"
    assert loaded.converged is False
