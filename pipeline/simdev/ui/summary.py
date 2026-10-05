"""One run's numbers for the results table: what post wrote, plus noise.

Noise is how far the reported window mean would move depending on where
the run happened to stop - half the spread of the rolling mean of half the
window length, inside the window (the ab-report.py measure). It is what
decides whether a delta between two runs is real.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simdev.run.parsers import read_component_coeffs
from simdev.stages.common import load_spec
from simdev.ui.runview import force_frame

COEFFICIENTS = ("Cd", "Cl")


def rolling_noise(values: Sequence[float]) -> float:
    data = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if len(data) < 2:
        return float("nan")
    length = max(len(data) // 2, 1)
    means = np.convolve(data, np.ones(length) / length, mode="valid")
    return float((means.max() - means.min()) / 2)


@dataclass(frozen=True)
class RunSummary:
    name: str
    result: dict[str, Any]
    window: tuple[int, int]
    noise: dict[str, float]
    patches: dict[str, dict[str, float]]
    groups_map: dict[str, list[str]]


def _in_window(series: pd.Series, times: pd.Series, window: tuple[int, int]) -> pd.Series:
    # Same rows as report.forces.window_mean: by iteration number, inclusive.
    return series[(times >= window[0]) & (times <= window[1])]


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates("Time", keep="last").sort_values("Time")


def _groups(run_dir: Path) -> dict[str, list[str]]:
    try:
        return {g: list(p) for g, p in load_spec(run_dir).post.groups.items()}
    except (FileNotFoundError, KeyError, ValueError, AttributeError):
        return {}


def _signature(run_dir: Path) -> tuple:
    files = [run_dir / "results" / "result.json"]
    files += sorted((run_dir / "postProcessing").glob("forceCoeffs*/*/coefficient.dat"))
    return tuple((str(f), f.stat().st_mtime_ns) for f in files if f.exists())


def load_summary(run_dir: Path) -> RunSummary | None:
    run_dir = Path(run_dir)
    if not (run_dir / "results" / "result.json").is_file():
        return None
    try:
        return _load(str(run_dir), _signature(run_dir))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=256)
def _load(run_dir_s: str, _stamp: tuple) -> RunSummary:
    run_dir = Path(run_dir_s)
    result = json.loads((run_dir / "results" / "result.json").read_text(encoding="utf-8"))
    window = (int(result.get("window_start") or 0), int(result.get("window_end") or 0))

    noise: dict[str, float] = {c: float("nan") for c in COEFFICIENTS}
    total = force_frame(run_dir)
    if not total.empty:
        for c in COEFFICIENTS:
            if c in total:
                noise[c] = rolling_noise(_in_window(total[c], total["Time"], window).tolist())

    patches: dict[str, dict[str, float]] = {}
    series: dict[str, dict[str, pd.Series]] = {}
    for patch, frame in read_component_coeffs(run_dir).items():
        frame = _dedupe(frame)
        entry: dict[str, float] = {}
        for c in COEFFICIENTS:
            if c not in frame:
                continue
            values = _in_window(frame[c], frame["Time"], window)
            entry[c] = float(values.mean()) if len(values) else float("nan")
            entry[f"{c}_noise"] = rolling_noise(values.tolist())
            series.setdefault(patch, {})[c] = frame.set_index("Time")[c]
        patches[patch] = entry

    groups_map = _groups(run_dir)
    for group, members in groups_map.items():
        for c in COEFFICIENTS:
            parts = [series.get(p, {}).get(c) for p in members]
            if not parts or any(s is None for s in parts):
                noise[f"{c}_{group}"] = float("nan")
                continue
            summed = sum(parts[1:], parts[0]).dropna()
            values = _in_window(summed, pd.Series(summed.index, index=summed.index), window)
            noise[f"{c}_{group}"] = rolling_noise(values.tolist())

    return RunSummary(run_dir.name, result, window, noise, patches, groups_map)


def finite(value: Any) -> float | None:
    """A number for the table, or None for anything missing or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
