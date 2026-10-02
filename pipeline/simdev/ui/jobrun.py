"""What one queued job executes, and what its result means.

The worker starts `python -m simdev.ui.jobrun <run_dir>` as the leader of a
new process group. This module then calls the ordinary CLI in-process, so the
job is exactly what someone would type, and mpirun and its ranks inherit the
group - which is what lets a cancel reach them all.

The outcome is written to a file rather than returned through an exit code,
because after a service restart the worker adopts a job it did not start and
cannot wait() on: the file is the only way it learns how that job ended.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from simdev.run.status import read_status

STAGES = ("prepare", "mesh", "solve", "post", "images")
JOB_FILE = "ui/job.json"
OUTCOME_FILE = "ui/outcome.json"
UI_LOG = "logs/simdev-ui.log"


def write_job_file(job: Any, cad_root: str | None = None) -> Path:
    run_dir = Path(job.run_dir)
    path = run_dir / JOB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "case": job.case_path, "profile": job.profile, "design": job.design,
                "state": job.state, "overrides": job.overrides,
                "force_from": job.force_from, "cad_root": cad_root,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def build_steps(job: dict[str, Any], run_dir: Path) -> list[list[str]]:
    run_dir_s = str(run_dir)
    prepare_args = [
        job["case"], "--run-dir", run_dir_s, "--profile", job["profile"],
        "--design", job["design"], "--state", job["state"],
    ]
    if job.get("cad_root"):
        prepare_args += ["--cad-root", job["cad_root"]]
    for key, value in sorted(job.get("overrides", {}).items()):
        # JSON is valid YAML, and parse_overrides reads values as YAML.
        prepare_args += ["--set", f"{key}={json.dumps(value)}"]

    force_from = job.get("force_from")
    if force_from is None:
        return [["run", *prepare_args], ["images", run_dir_s]]

    steps: list[list[str]] = []
    for stage in STAGES[STAGES.index(force_from):]:
        if stage == "prepare":
            steps.append(["prepare", *prepare_args, "--force"])
        else:
            steps.append([stage, run_dir_s, "--force"])
    return steps


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_job(run_dir: Path, main: Callable[[list[str]], int] | None = None) -> dict[str, Any]:
    if main is None:
        from simdev.cli import main

    run_dir = Path(run_dir)
    job = json.loads((run_dir / JOB_FILE).read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = []

    for argv in build_steps(job, run_dir):
        if argv[0] == "images":
            post = read_status(run_dir, "post")
            if post is None or post.state == "failed":
                continue
        print(f"$ simdev {' '.join(argv)}", flush=True)
        try:
            code = main(argv)
        except Exception:
            # The CLI catches the errors it expects; anything else is a bug,
            # and it is still an outcome rather than a job that never ends.
            traceback.print_exc()
            code = 3
        sys.stdout.flush()
        steps.append({"argv": argv, "exit": code})
        # 0 is success, 1 is "finished but a gate failed" - keep going so the
        # result is still post-processed and pictured. Anything else stops.
        if code not in (0, 1):
            break

    outcome = {"steps": steps}
    _write_json(run_dir / OUTCOME_FILE, outcome)
    return outcome


def read_outcome(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / OUTCOME_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def first_error(run_dir: Path) -> str | None:
    """The CLI's own `error: ...` line, which is the best one-line summary."""
    path = Path(run_dir) / UI_LOG
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("error: "):
            return line[len("error: "):].strip()
    return None


def classify(
    run_dir: Path, outcome: dict[str, Any] | None, cancelled: bool = False
) -> tuple[str, str | None]:
    if cancelled:
        return "cancelled", None
    if outcome is None:
        return "failed", (
            "the job ended without recording an outcome - it was killed, "
            "or the service lost track of it"
        )

    run_dir = Path(run_dir)
    statuses = {stage: read_status(run_dir, stage) for stage in STAGES}
    hard = [s for s in outcome["steps"] if s["exit"] not in (0, 1)]
    failed = [name for name, st in statuses.items() if st is not None and st.state == "failed"]

    if hard or failed:
        detail = first_error(run_dir)
        if detail is None:
            detail = (
                f"stage {failed[0]} failed" if failed
                else f"simdev {hard[0]['argv'][0]} exited with code {hard[0]['exit']}"
            )
        return "failed", detail
    if any(st is not None and st.state == "gate_failed" for st in statuses.values()):
        return "gate_failed", None
    missing = [name for name, st in statuses.items() if st is None]
    if missing:
        return "failed", "the pipeline stopped before " + ", ".join(missing)
    return "done", None


if __name__ == "__main__":
    run_job(Path(sys.argv[1]))
