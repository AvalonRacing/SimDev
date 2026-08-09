from __future__ import annotations

import json
from pathlib import Path

from simdev.config.schema import CaseSpec
from simdev.run.runner import StageError
from simdev.run.status import read_status


def load_spec(run_dir: Path) -> CaseSpec:
    payload = json.loads((Path(run_dir) / "caseSpec.json").read_text(encoding="utf-8"))
    return CaseSpec.model_validate(payload["spec"])


def require_stage(run_dir: Path, stage: str) -> None:
    status = read_status(run_dir, stage)
    if status is None:
        raise StageError([f"stage '{stage}' has not been run"])
    if status.state != "ok":
        raise StageError(
            [f"stage '{stage}' did not succeed (state={status.state})", *status.reasons]
        )


def find_latest(run_dir: Path, pattern: str) -> Path:
    matches = sorted(
        (Path(run_dir) / "postProcessing").glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"no file matching 'postProcessing/{pattern}' under {run_dir}"
        )
    return matches[-1]
