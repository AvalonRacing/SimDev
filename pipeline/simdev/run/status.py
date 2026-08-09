from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATES = ("ok", "failed", "gate_failed")


@dataclass(frozen=True)
class StageStatus:
    stage: str
    state: str
    input_hash: str
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def _path(run_dir: Path, stage: str) -> Path:
    return Path(run_dir) / "status" / f"{stage}.json"


def write_status(run_dir: Path, status: StageStatus) -> Path:
    if status.state not in STATES:
        raise ValueError(f"unknown state {status.state!r}; expected one of {STATES}")
    target = _path(run_dir, status.stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
    return target


def read_status(run_dir: Path, stage: str) -> StageStatus | None:
    target = _path(run_dir, stage)
    if not target.exists():
        return None
    return StageStatus(**json.loads(target.read_text(encoding="utf-8")))


def should_skip(run_dir: Path, stage: str, input_hash: str, force: bool) -> bool:
    """Skip only on an unchanged input hash from a stage that succeeded.

    Never infers staleness from directory contents.
    """
    if force:
        return False
    status = read_status(run_dir, stage)
    if status is None:
        return False
    return status.state == "ok" and status.input_hash == input_hash
