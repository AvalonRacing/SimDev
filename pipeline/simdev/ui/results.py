"""The results table: one row per run, a delta row against its reference.

Modelled on the old pipeline's Excel sheet (benchmark_old_pipeline/
Aeroexcel.xlsx, sheet TC10) - per-row reference, change note, coloured
delta - but referencing by run name, not by row number, which is what
filled that sheet with #REF!.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from simdev.ui import runview
from simdev.ui.notes import RunNote, read_note
from simdev.ui.summary import finite, load_summary


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    better: int  # -1: lower is better, +1: higher is better, 0: no colour
    fmt: str
    noise_key: str | None = None


BASE_COLUMNS = (
    Column("cl", "Cl", -1, "{:+.4f}", "cl"),
    Column("cd", "Cd", -1, "{:.4f}", "cd"),
    Column("eff", "−Cl/Cd", +1, "{:.3f}", "eff"),
    Column("balance", "bal % F", 0, "{:.1f}"),
    Column("fx", "Fx", 0, "{:+.3f}"),
    Column("fy", "Fy", 0, "{:+.3f}"),
    Column("fz", "Fz", 0, "{:+.3f}"),
    Column("cop_x", "COP x", 0, "{:+.4f}"),
    Column("cop_y", "COP y", 0, "{:+.4f}"),
    Column("cop_z", "COP z", 0, "{:+.4f}"),
    Column("cs", "Cs", 0, "{:+.4f}"),
)

RESULT_KEYS = {"cl": "cl_mean", "cd": "cd_mean", "balance": "balance_front_pct",
               "fx": "fx", "fy": "fy", "fz": "fz", "cop_x": "cop_x", "cop_y": "cop_y",
               "cop_z": "cop_z", "cs": "cs_mean"}


def columns_for(groups: Sequence[str]) -> list[Column]:
    cols = list(BASE_COLUMNS)
    for g in groups:
        cols.append(Column(f"cl_{g}", f"Cl {g}", -1, "{:+.4f}", f"cl_{g}"))
        cols.append(Column(f"cd_{g}", f"Cd {g}", -1, "{:.4f}", f"cd_{g}"))
    return cols


@dataclass(frozen=True)
class Row:
    name: str
    state: str
    design: str
    note: RunNote
    has_result: bool
    verdict: str | None
    n_iterations: int | None
    n_cells: int | None
    values: dict[str, float | None]
    noise: dict[str, float | None]
    groups: tuple[str, ...]


@dataclass(frozen=True)
class DeltaCell:
    value: float | None
    noise: float | None
    tone: str


def _state_and_design(run_dir: Path) -> tuple[str, str]:
    state = design = ""
    try:
        job = json.loads((run_dir / "ui" / "job.json").read_text(encoding="utf-8"))
        state, design = str(job.get("state") or ""), str(job.get("design") or "")
    except (OSError, ValueError, AttributeError):
        pass
    report = run_dir / "results" / "report.tsv"
    if not state and report.is_file():
        try:
            rows = list(csv.DictReader(report.open(encoding="utf-8"), delimiter="\t"))
            state = rows[-1].get("driving_state") or "" if rows else ""
        except (OSError, csv.Error):
            pass
    return state, design


def _efficiency(cl: float | None, cd: float | None) -> float | None:
    return -cl / cd if cl is not None and cd else None


def _efficiency_noise(cl, cd, n_cl, n_cd) -> float | None:
    if None in (cl, cd, n_cl, n_cd) or not cl or not cd:
        return None
    return abs(cl / cd) * math.hypot(n_cl / cl, n_cd / cd)


def load_row(run_dir: Path) -> Row:
    run_dir = Path(run_dir)
    state, design = _state_and_design(run_dir)
    note = read_note(run_dir)
    summary = load_summary(run_dir)
    if summary is None:
        return Row(run_dir.name, state, design, note, False, None, None, None, {}, {}, ())
    result = summary.result
    values = {key: finite(result.get(src)) for key, src in RESULT_KEYS.items()}
    values["eff"] = _efficiency(values["cl"], values["cd"])
    groups = tuple(sorted((result.get("groups") or {}).keys()))
    for g in groups:
        entry = result["groups"].get(g) or {}
        values[f"cl_{g}"] = finite(entry.get("Cl"))
        values[f"cd_{g}"] = finite(entry.get("Cd"))
    noise: dict[str, float | None] = {
        "cl": finite(summary.noise.get("Cl")), "cd": finite(summary.noise.get("Cd")),
    }
    noise["eff"] = _efficiency_noise(values["cl"], values["cd"], noise["cl"], noise["cd"])
    for g in groups:
        noise[f"cl_{g}"] = finite(summary.noise.get(f"Cl_{g}"))
        noise[f"cd_{g}"] = finite(summary.noise.get(f"Cd_{g}"))
    return Row(
        run_dir.name, state or str(result.get("driving_state") or ""), design, note, True,
        result.get("verdict"), result.get("n_iterations"), result.get("n_cells"),
        values, noise, groups,
    )


def load_rows(runs_root: Path) -> list[Row]:
    return [load_row(p) for p in runview.list_runs(runs_root)]


def group_names(rows: Sequence[Row]) -> list[str]:
    return sorted({g for r in rows for g in r.groups})


def _tone(value: float, noise: float | None, better: int) -> str:
    if better == 0:
        return "neutral"
    if noise is not None and abs(value) < noise:
        return "noise"
    if value == 0:
        return "neutral"
    return "better" if value * better > 0 else "worse"


def delta(row: Row, ref: Row, columns: Sequence[Column]) -> dict[str, DeltaCell]:
    cells: dict[str, DeltaCell] = {}
    for col in columns:
        a, b = row.values.get(col.key), ref.values.get(col.key)
        if a is None or b is None:
            cells[col.key] = DeltaCell(None, None, "none")
            continue
        noise = None
        if col.noise_key:
            na, nb = row.noise.get(col.noise_key), ref.noise.get(col.noise_key)
            if na is not None and nb is not None:
                noise = math.hypot(na, nb)
        cells[col.key] = DeltaCell(a - b, noise, _tone(a - b, noise, col.better))
    return cells


def _cell(value: float | None) -> str:
    # Empty for "uncomputable", never "nan": a spreadsheet would average it in.
    return "" if value is None else f"{value:.6g}"


def to_tsv(rows: Sequence[Row], columns: Sequence[Column], by_name: Mapping[str, Row]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(["run", "state", "note", "compare_with", *[c.label for c in columns],
                     "verdict", "n_iterations", "n_cells"])
    for row in rows:
        writer.writerow([row.name, row.state, row.note.note, row.note.compare_with or "",
                         *[_cell(row.values.get(c.key)) for c in columns],
                         row.verdict or "", row.n_iterations or "", row.n_cells or ""])
        ref = by_name.get(row.note.compare_with or "")
        if ref is not None and row.has_result and ref.has_result:
            cells = delta(row, ref, columns)
            writer.writerow([f"Δ {row.name} − {ref.name}", "", "", "",
                             *[_cell(cells[c.key].value) for c in columns], "", "", ""])
    return out.getvalue()
