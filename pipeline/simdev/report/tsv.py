"""The paste target: one tab-separated line per run, and a way to stack them.

Tab-separated and one line per run because that is the shape the old
StarCCM pipeline's Auswertung_<version>.txt had, and there are sheets built
around it. Written per run and combined ON READ - never appended to a shared
file. See report/results.py for why that rule exists.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# THE COLUMN ORDER IS A CONTRACT. Someone has a sheet with formulas pointing
# at column N. New columns go on the END; nothing is ever inserted or
# reordered. tests/test_report_tsv.py pins it.
#
# The `verdict`..`n_cells` block is not decoration. A row that does not say
# which iterations it averaged and whether the run converged will eventually
# be compared against one that stopped mid-transient, and the difference will
# be believed. It travels with the numbers or it does not exist.
FIXED_COLUMNS: tuple[str, ...] = (
    # identity
    "run", "case_name", "driving_state", "spec_hash", "timestamp",
    # whether to trust the rest of the line
    "verdict", "converged", "window_start", "window_end", "n_iterations",
    "cd_amplitude", "cl_amplitude", "yplus_passed", "n_cells",
    # forces, newtons
    "Fx", "Fy", "Fz",
    # moments about c_of_r, newton-metres
    "Mx", "My", "Mz",
    # coefficients
    "cd", "cd_std", "cl", "cl_std", "cs",
    # centre of pressure, metres, and balance in percent.
    # THREE DIAGNOSTICS, NOT A POINT - see report/forces.py.
    "COP_x", "COP_y", "COP_z", "balance_front_pct",
)


def report_columns(group_names: Sequence[str]) -> list[str]:
    """The fixed prefix, then cd/cl per group in the order the config declares."""
    return [
        *FIXED_COLUMNS,
        *(f"{c}_{g}" for g in group_names for c in ("cd", "cl")),
    ]


def _format(value: object) -> str:
    """Empty for anything absent. NEVER the string "nan".

    A cell containing "nan" is a value a spreadsheet will happily average
    into a summary. An empty cell is a gap, which is what an uncomputable COP
    actually is.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_report(
    run_dir: Path, row: Mapping[str, object], group_names: Sequence[str]
) -> Path:
    """One header line and one data line. Overwrites; never appends."""
    columns = report_columns(group_names)
    target = Path(run_dir) / "results"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "report.tsv"
    path.write_text(
        "\t".join(columns)
        + "\n"
        + "\t".join(_format(row.get(c)) for c in columns)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_report(run_dir: Path) -> dict[str, str]:
    """The one data line of a run's report, as a column -> text mapping."""
    path = Path(run_dir) / "results" / "report.tsv"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: expected a header and one data line")
    return dict(zip(lines[0].split("\t"), lines[1].split("\t")))


def aggregate_reports(
    run_dirs: Iterable[Path],
) -> tuple[list[str], list[list[str]]]:
    """Stack per-run reports on read. Returns (header, rows).

    Columns are unioned in first-seen order, so a run with an extra group
    widens the table rather than being silently truncated to match the first
    run. A run with no report.tsv is skipped - it has not been post-processed
    - and the caller reports which.
    """
    header: list[str] = []
    found: list[dict[str, str]] = []

    for run_dir in run_dirs:
        try:
            fields = read_report(run_dir)
        except (FileNotFoundError, ValueError):
            continue
        for column in fields:
            if column not in header:
                header.append(column)
        found.append(fields)

    return header, [[fields.get(c, "") for c in header] for fields in found]
