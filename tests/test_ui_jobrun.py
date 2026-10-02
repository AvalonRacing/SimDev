from __future__ import annotations

import json
from pathlib import Path

from simdev.run.status import StageStatus, write_status
from simdev.ui.jobrun import (
    STAGES,
    build_steps,
    classify,
    first_error,
    read_outcome,
    run_job,
)

JOB = {
    "case": "/repo/cases/car/config.yaml", "profile": "car_dev", "design": "v01",
    "state": "corner", "overrides": {"flow.u_inf": 18.0, "physics.corner_direction": "right"},
    "force_from": None, "cad_root": None,
}


def write_job(run_dir: Path, **changes) -> None:
    (run_dir / "ui").mkdir(parents=True, exist_ok=True)
    (run_dir / "ui" / "job.json").write_text(json.dumps({**JOB, **changes}))


def statuses(run_dir: Path, **states: str) -> None:
    for stage, state in states.items():
        write_status(run_dir, StageStatus(stage=stage, state=state, input_hash="h"))


def test_a_normal_job_is_run_then_images(tmp_path: Path) -> None:
    steps = build_steps(JOB, tmp_path)
    assert steps[0][:2] == ["run", JOB["case"]]
    assert ["--design", "v01"] == steps[0][steps[0].index("--design"):][:2]
    assert "flow.u_inf=18.0" in steps[0]
    assert 'physics.corner_direction="right"' in steps[0]
    assert steps[1] == ["images", str(tmp_path)]


def test_rerun_from_a_stage_forces_it_and_everything_after(tmp_path: Path) -> None:
    steps = build_steps({**JOB, "force_from": "solve"}, tmp_path)
    assert steps == [
        ["solve", str(tmp_path), "--force"],
        ["post", str(tmp_path), "--force"],
        ["images", str(tmp_path), "--force"],
    ]


def test_rerun_from_prepare_passes_the_case_again(tmp_path: Path) -> None:
    steps = build_steps({**JOB, "force_from": "prepare"}, tmp_path)
    assert steps[0][0] == "prepare" and steps[0][-1] == "--force"
    assert [s[0] for s in steps] == list(STAGES)


def test_run_job_records_each_step(tmp_path: Path) -> None:
    write_job(tmp_path)
    calls = []

    def fake_main(argv):
        calls.append(argv[0])
        if argv[0] == "run":
            statuses(tmp_path, prepare="ok", mesh="ok", solve="gate_failed", post="ok")
            return 1  # a gate failed: not a crash, images still run
        statuses(tmp_path, images="ok")
        return 0

    outcome = run_job(tmp_path, main=fake_main)
    assert calls == ["run", "images"]
    assert [s["exit"] for s in outcome["steps"]] == [1, 0]
    assert read_outcome(tmp_path) == outcome


def test_images_are_skipped_when_post_never_ran(tmp_path: Path) -> None:
    write_job(tmp_path)
    outcome = run_job(tmp_path, main=lambda argv: 2)
    assert [s["argv"][0] for s in outcome["steps"]] == ["run"]


def test_an_exception_is_an_outcome_not_a_hang(tmp_path: Path) -> None:
    write_job(tmp_path)

    def explode(argv):
        raise RuntimeError("kaboom")

    outcome = run_job(tmp_path, main=explode)
    assert outcome["steps"][0]["exit"] == 3


def test_classify(tmp_path: Path) -> None:
    ok = {"steps": [{"argv": ["run"], "exit": 0}, {"argv": ["images"], "exit": 0}]}
    statuses(tmp_path, **{s: "ok" for s in STAGES})
    assert classify(tmp_path, ok) == ("done", None)

    statuses(tmp_path, solve="gate_failed")
    assert classify(tmp_path, ok) == ("gate_failed", None)

    assert classify(tmp_path, ok, cancelled=True) == ("cancelled", None)
    assert classify(tmp_path, None)[0] == "failed"


def test_classify_a_crash_uses_the_cli_error_line(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "simdev-ui.log").write_text(
        "$ simdev run ...\nerror: snappyHexMesh failed with exit code 1\n"
    )
    statuses(tmp_path, prepare="ok", mesh="failed")
    crashed = {"steps": [{"argv": ["run"], "exit": 2}]}
    assert classify(tmp_path, crashed) == ("failed", "snappyHexMesh failed with exit code 1")
    assert first_error(tmp_path) == "snappyHexMesh failed with exit code 1"


def test_classify_a_gate_stop_is_gate_failed_with_the_cli_error(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "simdev-ui.log").write_text(
        "=== simdev-ui job started ===\n$ simdev run ...\n"
        "error: mesh quality gate failed; layer coverage low; refusing to start the solve\n"
    )
    statuses(tmp_path, prepare="ok", mesh="gate_failed")
    stopped = {"steps": [{"argv": ["run"], "exit": 2}]}
    assert classify(tmp_path, stopped) == (
        "gate_failed",
        "mesh quality gate failed; layer coverage low; refusing to start the solve",
    )


def test_classify_names_stages_that_never_ran(tmp_path: Path) -> None:
    statuses(tmp_path, prepare="ok", mesh="ok")
    status, error = classify(tmp_path, {"steps": [{"argv": ["run"], "exit": 0}]})
    assert status == "failed"
    assert "solve" in error


def test_system_exit_is_caught_and_recorded(tmp_path: Path) -> None:
    write_job(tmp_path)

    def exit_with_code(argv):
        raise SystemExit(2)

    outcome = run_job(tmp_path, main=exit_with_code)
    assert outcome["steps"][0]["exit"] == 2
    assert read_outcome(tmp_path) == outcome


def test_system_exit_none_becomes_zero(tmp_path: Path) -> None:
    write_job(tmp_path)

    def exit_no_code(argv):
        raise SystemExit()

    outcome = run_job(tmp_path, main=exit_no_code)
    assert outcome["steps"][0]["exit"] == 0


def test_system_exit_non_int_becomes_two(tmp_path: Path) -> None:
    write_job(tmp_path)

    def exit_string(argv):
        raise SystemExit("error message")

    outcome = run_job(tmp_path, main=exit_string)
    assert outcome["steps"][0]["exit"] == 2


def test_missing_job_file_is_an_outcome(tmp_path: Path) -> None:
    outcome = run_job(tmp_path, main=lambda argv: 0)
    assert outcome["steps"] == []
    assert "error" in outcome
    assert "job file not found" in outcome["error"]
    assert classify(tmp_path, outcome) == ("failed", outcome["error"])


def test_malformed_job_json_is_an_outcome(tmp_path: Path) -> None:
    (tmp_path / "ui").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ui" / "job.json").write_text("not valid json {")

    outcome = run_job(tmp_path, main=lambda argv: 0)
    assert outcome["steps"] == []
    assert "error" in outcome
    assert "malformed" in outcome["error"]
    assert classify(tmp_path, outcome) == ("failed", outcome["error"])


def test_unknown_force_from_is_an_outcome(tmp_path: Path) -> None:
    write_job(tmp_path, force_from="nonexistent_stage")

    outcome = run_job(tmp_path, main=lambda argv: 0)
    assert outcome["steps"] == []
    assert "error" in outcome
    assert "unknown force_from" in outcome["error"]
    assert classify(tmp_path, outcome) == ("failed", outcome["error"])


def test_stale_outcome_json_is_deleted_at_start(tmp_path: Path) -> None:
    write_job(tmp_path)

    # Write a stale outcome file
    (tmp_path / "ui").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ui" / "outcome.json").write_text(json.dumps({"steps": [{"argv": ["stale"], "exit": 1}]}))

    # Verify it exists before
    assert (tmp_path / "ui" / "outcome.json").is_file()

    # Track when main is first called to check outcome is gone by then
    outcome_existed_at_call = []

    def fake_main(argv):
        outcome_existed_at_call.append((tmp_path / "ui" / "outcome.json").is_file())
        return 0

    outcome = run_job(tmp_path, main=fake_main)

    # The outcome file should not have existed when main was first called
    assert outcome_existed_at_call[0] is False
    # And the new outcome should be written
    assert read_outcome(tmp_path) == outcome


def test_first_error_ignores_errors_before_last_marker(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "simdev-ui.log").write_text(
        "$ simdev run ...\nerror: old error from previous run\n"
        "=== simdev-ui job started ===\n"
        "$ simdev run ...\nerror: current error\n"
    )

    # Should find the current error after the marker, not the old one
    assert first_error(tmp_path) == "current error"


def test_first_error_with_multiple_markers_uses_last_one(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "simdev-ui.log").write_text(
        "=== simdev-ui job started ===\n"
        "error: first attempt error\n"
        "=== simdev-ui job started ===\n"
        "error: second attempt error\n"
    )

    # Should use the second marker
    assert first_error(tmp_path) == "second attempt error"
