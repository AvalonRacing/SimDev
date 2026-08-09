from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ResultRecord:
    case_name: str
    spec_hash: str
    timestamp: str
    converged: bool
    cd_mean: float
    cd_std: float
    cl_mean: float
    cl_std: float
    window_start: int
    window_end: int
    n_iterations: int
    n_cells: int
    yplus_passed: bool
    yplus: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _dir(run_dir: Path) -> Path:
    target = Path(run_dir) / "results"
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_result(run_dir: Path, record: ResultRecord) -> tuple[Path, Path]:
    """Write this run's record. Never appends to a shared file."""
    target = _dir(run_dir)
    json_path = target / "result.json"
    csv_path = target / "result.csv"

    json_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    flat = {k: v for k, v in asdict(record).items() if not isinstance(v, (dict, list))}
    flat["reasons"] = " | ".join(record.reasons)
    for patch, value in record.yplus.items():
        flat[f"yplus_{patch}"] = value
    pd.DataFrame([flat]).to_csv(csv_path, index=False)

    return json_path, csv_path


def read_result(run_dir: Path) -> ResultRecord:
    payload = json.loads(
        (Path(run_dir) / "results" / "result.json").read_text(encoding="utf-8")
    )
    return ResultRecord(**payload)


def aggregate(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Combine per-run records on read. This replaces shared-append tables."""
    rows = []
    for run_dir in run_dirs:
        record = read_result(run_dir)
        row = {k: v for k, v in asdict(record).items() if not isinstance(v, (dict, list))}
        row["run_dir"] = str(run_dir)
        rows.append(row)
    return pd.DataFrame(rows)
