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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
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


_COL = {c.key: c for c in (
    Column("cl", "Cl", -1, "{:+.4f}", "cl"),
    Column("cd", "Cd", -1, "{:.4f}", "cd"),
    Column("fx", "Fx", 0, "{:+.3f}"),
    Column("fz", "Fz", 0, "{:+.3f}"),
    Column("fy", "Fy", 0, "{:+.3f}"),
    Column("cop_x", "COP x", 0, "{:+.4f}"),
    Column("cop_y", "COP y", 0, "{:+.4f}"),
    Column("cop_z", "COP z", 0, "{:+.4f}"),
    Column("eff", "−Cl/Cd", +1, "{:.3f}", "eff"),
    Column("balance", "bal % F", 0, "{:.1f}"),
    Column("cs", "Cs", 0, "{:+.4f}"),
)}

RESULT_KEYS = {"cl": "cl_mean", "cd": "cd_mean", "balance": "balance_front_pct",
               "fx": "fx", "fy": "fy", "fz": "fz", "cop_x": "cop_x", "cop_y": "cop_y",
               "cop_z": "cop_z", "cs": "cs_mean"}


def _group_col(coefficient: str, group: str) -> Column:
    label = coefficient.capitalize()
    return Column(f"{coefficient}_{group}", f"{label} {group}", -1,
                  "{:.4f}" if coefficient == "cd" else "{:+.4f}", f"{coefficient}_{group}")


def columns_for(groups: Sequence[str]) -> list[Column]:
    """The benchmark sheet's order: forces, group Cl then group Cd, COP, efficiency, balance."""
    named = [g for g in ("body", "wing") if g in groups]
    others = [g for g in groups if g not in named]
    cols = [_COL[k] for k in ("cl", "cd", "fx", "fz", "fy")]
    cols += [_group_col("cl", g) for g in named] + [_group_col("cd", g) for g in named]
    cols += [_COL[k] for k in ("cop_x", "cop_y", "cop_z", "eff", "balance")]
    for g in others:
        cols += [_group_col("cl", g), _group_col("cd", g)]
    return [*cols, _COL["cs"]]


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

    @property
    def shown_name(self) -> str:
        """What the table's first column shows: the design, the run name for shell runs."""
        return self.design or self.name


@dataclass(frozen=True)
class DeltaCell:
    value: float | None
    noise: float | None
    tone: str


@dataclass(frozen=True)
class TableColumn:
    """One column of the results table as the page lays it out."""
    key: str
    label: str
    kind: str  # design, compare, state, result, verdict, noise, note
    group: str
    width: int  # default width in px, the page lets the user change it
    default: bool = True  # shown until the user picks columns
    fixed: bool = False  # cannot be hidden
    num: bool = False
    fmt: str = ""
    start: bool = False  # first of its group: gets a separator on its left


_RESULT_GROUP = {"cl": "coefficients", "cd": "coefficients", "cs": "coefficients", "fx": "forces", "fz": "forces",
                 "fy": "forces", "cop_x": "COP", "cop_y": "COP", "cop_z": "COP",
                 "eff": "efficiency & balance", "balance": "efficiency & balance"}
_HIDDEN_BY_DEFAULT = {"forces", "COP"}


def _named_or_other(key: str) -> str:
    # body/wing group columns sit before COP, any further groups after balance.
    named = {f"{c}_{g}" for c in ("cl", "cd") for g in ("body", "wing")}
    return "groups" if key in named else "other groups"


def table_columns(columns: Sequence[Column]) -> list[TableColumn]:
    """Identity, the result columns, verdict and noise, the note - in display order."""
    cols = [
        TableColumn("design", "design", "design", "identity", 170, fixed=True),
        TableColumn("compare", "compare with", "compare", "identity", 150, fixed=True),
        TableColumn("state", "state", "state", "identity", 110),
    ]
    # Cs belongs with Cl and Cd, but columns_for puts it last: move it so every group is
    # contiguous and the page never shows a heading twice.
    shown = [c for c in columns if c.key in ("cl", "cd", "cs")] + [
        c for c in columns if c.key not in ("cl", "cd", "cs")]
    for c in shown:
        group = _RESULT_GROUP.get(c.key) or _named_or_other(c.key)
        cols.append(TableColumn(c.key, c.label, "result", group, 84, group not in _HIDDEN_BY_DEFAULT,
                                num=True, fmt=c.fmt))
    cols += [
        TableColumn("verdict", "verdict", "verdict", "verdict & noise", 110),
        TableColumn("noise", "noise Cl / Cd", "noise", "verdict & noise", 130, False, num=True),
        TableColumn("note", "note", "note", "note", 260),
    ]
    return [replace(c, start=i > 0 and cols[i - 1].group != c.group) for i, c in enumerate(cols)]


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
            with report.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            state = rows[-1].get("driving_state") or "" if rows else ""
        except (OSError, ValueError, csv.Error):
            # ValueError covers UnicodeDecodeError: a garbled report must not
            # take the whole Results page down.
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


def to_tsv(rows: Sequence[Row], columns: Sequence[Column], by_name: Mapping[str, Row],
           keys: Collection[str] | None = None) -> str:
    """`keys` limits the optional columns (state, note, verdict, result columns) to the
    ones the page shows; the run, its design and the reference always stay."""
    def wanted(key: str) -> bool:
        return keys is None or key in keys

    shown = [c for c in columns if wanted(c.key)]
    head = ["run", "design", *(["state"] if wanted("state") else []),
            *(["note"] if wanted("note") else []), "compare_with", *[c.label for c in shown],
            *(["verdict", "n_iterations", "n_cells"] if wanted("verdict") else [])]
    out = io.StringIO()
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(head)
    for row in rows:
        writer.writerow([row.name, row.shown_name, *([row.state] if wanted("state") else []),
                         *([row.note.note] if wanted("note") else []), row.note.compare_with or "",
                         *[_cell(row.values.get(c.key)) for c in shown],
                         *([row.verdict or "", row.n_iterations or "", row.n_cells or ""]
                           if wanted("verdict") else [])])
        ref = by_name.get(row.note.compare_with or "")
        if ref is not None and row.has_result and ref.has_result:
            cells = delta(row, ref, columns)
            writer.writerow([f"Δ {row.name} − {ref.name}", "",
                             *([""] if wanted("state") else []), *([""] if wanted("note") else []),
                             "", *[_cell(cells[c.key].value) for c in shown],
                             *(["", "", ""] if wanted("verdict") else [])])
    return out.getvalue()
