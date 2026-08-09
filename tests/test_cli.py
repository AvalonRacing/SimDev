from __future__ import annotations

from pathlib import Path

import pytest

from simdev.cli import main

CASE = Path("cases/ahmed/config.yaml")


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_prepare_builds_a_case(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert main(["prepare", str(CASE), "--run-dir", str(run_dir), "--profile", "dev"]) == 0
    assert (run_dir / "system" / "controlDict").exists()


def test_prepare_accepts_a_wall_treatment_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    main(
        [
            "prepare",
            str(CASE),
            "--run-dir",
            str(run_dir),
            "--profile",
            "dev",
            "--wall-treatment",
            "low_y_plus",
        ]
    )
    text = (run_dir / "0" / "nut").read_text(encoding="utf-8")
    assert "nutLowReWallFunction" in text


def test_invalid_case_returns_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: broken\n", encoding="utf-8")
    assert main(["prepare", str(bad), "--run-dir", str(tmp_path / "r")]) != 0


def test_doctor_reports_without_crashing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert "simpleFoam" in out
    assert "surfaceFeatures" in out
    assert code in (0, 1)


def test_unknown_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["nonsense"])
    assert exc.value.code != 0
