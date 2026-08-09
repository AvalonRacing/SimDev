from __future__ import annotations

import sys
from pathlib import Path

import pytest

from simdev.run.runner import (
    CommandResult,
    Runner,
    StageError,
    parallel_argv,
)
from simdev.run.status import (
    StageStatus,
    read_status,
    should_skip,
    write_status,
)


def test_parallel_argv_wraps_with_mpirun_and_parallel_flag() -> None:
    assert parallel_argv(["simpleFoam"], 8) == [
        "mpirun",
        "-np",
        "8",
        "simpleFoam",
        "-parallel",
    ]


def test_successful_command_writes_a_log(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    result = runner.run([sys.executable, "-c", "print('hello')"], name="greet")
    assert result.returncode == 0
    assert result.log_path.exists()
    assert "hello" in result.log_path.read_text(encoding="utf-8")


def test_failing_command_raises(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    with pytest.raises(StageError) as exc:
        runner.run([sys.executable, "-c", "raise SystemExit(3)"], name="boom")
    assert any("exit code 3" in r for r in exc.value.reasons)


def test_failing_command_can_be_tolerated(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    result = runner.run(
        [sys.executable, "-c", "raise SystemExit(3)"], name="boom", check=False
    )
    assert isinstance(result, CommandResult)
    assert result.returncode == 3


def test_foam_fatal_on_a_zero_exit_is_still_a_failure(tmp_path: Path) -> None:
    # The expensive lesson: OpenFOAM exits 0 on partial failure.
    runner = Runner(tmp_path)
    script = "print('--> FOAM FATAL ERROR: keyword nu undefined')"
    with pytest.raises(StageError) as exc:
        runner.run([sys.executable, "-c", script], name="sneaky")
    assert any("FOAM FATAL" in r for r in exc.value.reasons)


def test_status_round_trips(tmp_path: Path) -> None:
    status = StageStatus(
        stage="mesh", state="ok", input_hash="abc123", reasons=[], detail={"n_cells": 10}
    )
    write_status(tmp_path, status)
    loaded = read_status(tmp_path, "mesh")
    assert loaded is not None
    assert loaded.state == "ok"
    assert loaded.input_hash == "abc123"


def test_read_status_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_status(tmp_path, "mesh") is None


def test_should_skip_only_on_matching_hash(tmp_path: Path) -> None:
    write_status(
        tmp_path, StageStatus("mesh", "ok", "abc123", [], {})
    )
    assert should_skip(tmp_path, "mesh", "abc123", force=False) is True
    assert should_skip(tmp_path, "mesh", "different", force=False) is False


def test_force_defeats_skipping(tmp_path: Path) -> None:
    write_status(tmp_path, StageStatus("mesh", "ok", "abc123", [], {}))
    assert should_skip(tmp_path, "mesh", "abc123", force=True) is False


def test_failed_stage_is_never_skipped(tmp_path: Path) -> None:
    write_status(tmp_path, StageStatus("mesh", "gate_failed", "abc123", ["bad"], {}))
    assert should_skip(tmp_path, "mesh", "abc123", force=False) is False
