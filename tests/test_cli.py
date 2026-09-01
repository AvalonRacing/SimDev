from __future__ import annotations

from pathlib import Path

import pytest

from simdev.cli import main, _parser

CASE = Path(__file__).parent / "fixtures" / "ahmed.yaml"


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
    assert "surfaceFeatureExtract" in out
    assert code in (0, 1)


def test_doctor_checks_step_import(capsys: pytest.CaptureFixture[str]) -> None:
    """gmsh fails at import on a missing system library, not at install.

    A venv pip reports as complete can still be unable to read a STEP file,
    and doctor is what gets run when a fresh machine misbehaves.
    """
    main(["doctor"])
    assert "gmsh (STEP import)" in capsys.readouterr().out


def test_unknown_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["nonsense"])
    assert exc.value.code != 0


def test_report_stacks_runs_on_read(tmp_path: Path, capsys) -> None:
    from simdev.report.tsv import write_report

    for name in ("car-01", "car-02"):
        write_report(tmp_path / name, {"run": name, "cd": 0.99}, ["body"])
    code = main([
        "report", str(tmp_path / "car-01"), str(tmp_path / "car-02"),
        "--out", str(tmp_path / "summary.tsv"),
    ])
    assert code == 0
    text = (tmp_path / "summary.tsv").read_text(encoding="utf-8")
    assert len(text.strip().splitlines()) == 3  # header + two runs


def test_report_says_which_runs_it_skipped(tmp_path: Path, capsys) -> None:
    """Silently dropping a run from a comparison table is how one goes missing."""
    from simdev.report.tsv import write_report

    write_report(tmp_path / "car-01", {"run": "car-01"}, [])
    main(["report", str(tmp_path / "car-01"), str(tmp_path / "car-99")])
    assert "car-99" in capsys.readouterr().err
