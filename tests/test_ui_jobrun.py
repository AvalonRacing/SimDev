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


def test_classify_names_stages_that_never_ran(tmp_path: Path) -> None:
    statuses(tmp_path, prepare="ok", mesh="ok")
    status, error = classify(tmp_path, {"steps": [{"argv": ["run"], "exit": 0}]})
    assert status == "failed"
    assert "solve" in error
