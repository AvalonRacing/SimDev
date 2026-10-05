# Results Table and Compare Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Results page that keeps the history of every run with change notes, per-run references and noise-aware Δ rows, and a Compare page that shows 1-4 runs side by side with synced plane/surface pictures, blink/swipe/fade overlays, on-request field deltas and overlaid charts.

**Architecture:** Everything is read from the run directories, as `simdev/ui/runview.py` already does. New pure-Python modules compute notes, noise, table rows and an image index; FastAPI routes render Jinja pages and serve JSON; the browser does sync/overlay/charts in one script. The only VTK code is a stand-alone helper (`simdev/viz/delta.py`) run under the ParaView interpreter, called by a small service with a disk cache and a lock.

**Tech Stack:** Python 3.14 venv (FastAPI, Jinja2, pandas, numpy, pytest), htmx 2.0.4 and uPlot 1.6.31 from the CDNs already in use, VTK + ParaView under `/usr/bin/python3` (helper only), PIL in both interpreters.

**Spec:** `docs/superpowers/specs/2026-10-06-compare-ui-design.md`

## Global Constraints

- Tests run with `.venv/bin/python -m pytest` from the repo root; UI test files start with `pytest.importorskip("fastapi")`.
- No new dependency in the venv. VTK and ParaView are imported **only** inside `pipeline/simdev/viz/delta.py`, and only inside functions (the module must import cleanly in the venv).
- `viz/delta.py` imports nothing from `simdev`.
- The helper is run with a clean environment: `{"HOME": <home>, "PATH": "/usr/bin:/bin"}` - ParaView hangs under the OpenFOAM environment.
- Interpreter for the helper: the run's `spec.post.paraview_python`, default `/usr/bin/python3`.
- Front-end libraries only from `https://cdn.jsdelivr.net/npm/uplot@1.6.31/...` and `https://unpkg.com/htmx.org@2.0.4/...` (already used).
- Note file: `<run>/ui/note.json` = `{"note": str, "compare_with": str|null, "updated_at": iso8601}`; note at most 2000 characters; written atomically.
- `compare_with` and every run name in a URL must match `simdev.ui.forms.RUN_NAME` and must not contain `..`.
- Delta fields: planes `cp`, `cpt`, `U`; surface `cp` only. Default limits cp ±0.2, cpt ±0.2, U ±10 % of u_inf. Diverging map with **20** bands (zero is a band edge). Solid in both runs: grey `(212, 212, 212)`; fluid/surface in one run only: black.
- Plane probe tolerance **1e-4 m**; surface interpolation radius **0.5 mm**, linear kernel, input arrays not passed through.
- Delta cache: `<pane run>/ui/delta/<ref run>/`, invalidated when either run's `spec_hash` or `views_digest` changes.
- Noise of one coefficient = half the spread (max − min) of the rolling mean of length `max(n // 2, 1)` over the `n` window samples; noise of a Δ = √(nA² + nB²).
- "Better": Cl lower (more negative), Cd lower, −Cl/Cd higher, group Cl/Cd lower. Balance, forces, COP and Cs are neutral.
- Comment style: sparse, explains *why*, like the surrounding code. Commit messages `feat(ui): ...` / `test(ui): ...` / `docs: ...`, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A `result.json` written by an older pipeline (no `groups`, no `fx`, no `cop_*`) must show empty cells, never a 500 - pinned in Task 4.
2. A restarted solve repeats iterations in `coefficient.dat`; noise must be computed on the de-duplicated history - pinned in Task 3.
3. Runs reached through a different path than the one `render_plan.json` recorded (`~/runs` is a symlink to `/mnt/data/runs`) must still find their pictures and samples - pinned in Task 6.
4. `compare_with` pointing at the run itself, at `../x`, or at a name with a slash must be refused with 400; a reference that was later deleted must show "missing", not crash - pinned in Task 5.
5. Two delta requests for the same picture at once must not run the helper twice or serve a half-written PNG - pinned in Task 8.

---

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/simdev/ui/notes.py` (new) | read/write `ui/note.json` |
| `pipeline/simdev/ui/queue.py` (modify) | `note` column, migration |
| `pipeline/simdev/ui/worker.py` (modify) | copy the job's note into the run on start |
| `pipeline/simdev/ui/routes_queue.py`, `templates/new_run.html` (modify) | note field in New Run |
| `pipeline/simdev/ui/summary.py` (new) | per-run summary: result values, window, noise, per-patch means |
| `pipeline/simdev/ui/results.py` (new) | table columns, rows, Δ cells, TSV |
| `pipeline/simdev/ui/routes_results.py`, `templates/results.html`, `templates/_result_row.html` (new) | Results page, inline note edits, TSV |
| `pipeline/simdev/ui/imageindex.py` (new) | picture/sample index from `render_plan.json`, path re-rooting |
| `pipeline/simdev/ui/routes_compare.py`, `templates/compare.html` (new) | Compare page, JSON APIs, delta endpoint |
| `pipeline/simdev/viz/delta.py` (new) | VTK helper: plane and surface deltas → PNG |
| `pipeline/simdev/ui/delta.py` (new) | request building, cache, lock, subprocess |
| `pipeline/simdev/ui/static/compare.js`, `static/style.css` (new / modify) | viewer and charts |
| `pipeline/simdev/ui/app.py`, `templates/base.html`, `templates/run.html`, `runview.py` (modify) | wiring, nav, run-page link, public `force_frame` |
| `docs/handbook.md`, `docs/ui-setup.md` (modify) | documentation |

---

### Task 1: Run notes

**Files:**
- Create: `pipeline/simdev/ui/notes.py`
- Test: `tests/test_ui_notes.py`

**Interfaces:**
- Produces:
  - `NOTE_FILE = "ui/note.json"`, `MAX_NOTE = 2000`
  - `@dataclass(frozen=True) class RunNote: note: str = ""; compare_with: str | None = None; updated_at: str | None = None; problem: str | None = None`
  - `read_note(run_dir: Path) -> RunNote`
  - `write_note(run_dir: Path, note: str, compare_with: str | None, now: datetime | None = None) -> RunNote`
  - `write_initial_note(run_dir: Path, text: str | None) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ui_notes.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from simdev.ui.notes import MAX_NOTE, NOTE_FILE, read_note, write_initial_note, write_note


def test_a_run_without_a_note_reads_empty(tmp_path: Path) -> None:
    note = read_note(tmp_path)
    assert note.note == "" and note.compare_with is None and note.problem is None


def test_write_then_read_round_trips_and_creates_ui(tmp_path: Path) -> None:
    when = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
    write_note(tmp_path, "  wing post -5 mm  ", "c02", now=when)
    note = read_note(tmp_path)
    assert note.note == "wing post -5 mm"
    assert note.compare_with == "c02"
    assert note.updated_at == "2026-10-06T10:00:00+00:00"
    assert not list((tmp_path / "ui").glob("*.tmp"))


def test_an_empty_reference_is_stored_as_none(tmp_path: Path) -> None:
    write_note(tmp_path, "x", "")
    assert json.loads((tmp_path / NOTE_FILE).read_text())["compare_with"] is None


def test_a_long_note_is_cut(tmp_path: Path) -> None:
    write_note(tmp_path, "a" * (MAX_NOTE + 50), None)
    assert len(read_note(tmp_path).note) == MAX_NOTE


def test_an_unreadable_file_reads_empty_with_a_problem(tmp_path: Path) -> None:
    (tmp_path / "ui").mkdir()
    (tmp_path / NOTE_FILE).write_text("{")
    note = read_note(tmp_path)
    assert note.note == "" and note.problem


def test_the_initial_note_never_overwrites_an_edited_one(tmp_path: Path) -> None:
    assert write_initial_note(tmp_path, "from the form") is True
    write_note(tmp_path, "edited in the table", None)
    assert write_initial_note(tmp_path, "from the form") is False
    assert read_note(tmp_path).note == "edited in the table"


def test_no_initial_note_without_text(tmp_path: Path) -> None:
    assert write_initial_note(tmp_path, "   ") is False
    assert not (tmp_path / NOTE_FILE).exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_notes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.ui.notes'`

- [ ] **Step 3: Implement**

```python
# pipeline/simdev/ui/notes.py
"""What changed in a run, and which run it is judged against.

Kept in the run directory, not in the queue database, so a run started from
the shell can carry a note too and the note goes wherever the run goes.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

NOTE_FILE = "ui/note.json"
MAX_NOTE = 2000


@dataclass(frozen=True)
class RunNote:
    note: str = ""
    compare_with: str | None = None
    updated_at: str | None = None
    # Set when a file existed but could not be read; the next save rewrites it.
    problem: str | None = None


def read_note(run_dir: Path) -> RunNote:
    path = Path(run_dir) / NOTE_FILE
    if not path.is_file():
        return RunNote()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not an object")
        return RunNote(
            note=str(data.get("note") or ""),
            compare_with=data.get("compare_with") or None,
            updated_at=data.get("updated_at"),
        )
    except (OSError, ValueError) as error:
        return RunNote(problem=f"{NOTE_FILE} could not be read ({error}); saving rewrites it")


def write_note(
    run_dir: Path, note: str, compare_with: str | None, now: datetime | None = None
) -> RunNote:
    folder = Path(run_dir) / "ui"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    record = RunNote(note=note.strip()[:MAX_NOTE], compare_with=compare_with or None,
                     updated_at=stamp)
    payload = {"note": record.note, "compare_with": record.compare_with,
               "updated_at": record.updated_at}
    # Temp file in the same folder, then rename: a reader never sees half a file.
    with tempfile.NamedTemporaryFile(
        "w", dir=folder, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    os.replace(handle.name, folder / "note.json")
    return record


def write_initial_note(run_dir: Path, text: str | None) -> bool:
    """The note typed into New Run, written once when the run starts.

    Never over an existing file: a resumed or re-run job must not undo an
    edit made in the results table.
    """
    if not text or not text.strip() or (Path(run_dir) / NOTE_FILE).exists():
        return False
    write_note(run_dir, text, None)
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ui_notes.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/ui/notes.py tests/test_ui_notes.py
git commit -m "feat(ui): change notes and references stored in the run directory

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Change note in the New Run form

**Files:**
- Modify: `pipeline/simdev/ui/queue.py` (SCHEMA, `JobSpec`, `Job`, `Queue.__init__`, `enqueue`, `update`, `requeue`)
- Modify: `pipeline/simdev/ui/worker.py` (`_spawn`)
- Modify: `pipeline/simdev/ui/routes_queue.py` (`_form_page`, `new_run`, `queue_run`)
- Modify: `pipeline/simdev/ui/templates/new_run.html`
- Test: `tests/test_ui_queue.py`, `tests/test_ui_worker.py`, `tests/test_ui_queue_routes.py` (append)

**Interfaces:**
- Consumes: `write_initial_note(run_dir, text)` from Task 1.
- Produces: `JobSpec.note: str | None = None`, `Job.note: str | None = None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_queue.py`:

```python
def test_a_job_keeps_its_note_through_edit_and_requeue(tmp_path) -> None:
    from simdev.ui.queue import JobSpec, Queue

    queue = Queue(tmp_path / "q.db")
    spec = JobSpec("r1", str(tmp_path / "r1"), "case.yaml", "v01", "corner", "car_dev", 4,
                   note="wing post -5 mm")
    job = queue.enqueue(spec)
    assert job.note == "wing post -5 mm"
    edited = queue.update(job.id, JobSpec(**{**spec.__dict__, "note": "wing post -6 mm"}))
    assert edited.note == "wing post -6 mm"
    queue.mark_running(job.id)
    queue.mark_finished(job.id, "done", 0, None)
    assert queue.requeue(job.id).note == "wing post -6 mm"


def test_a_database_from_before_notes_gains_the_column(tmp_path) -> None:
    import sqlite3

    from simdev.ui.queue import Queue

    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.executescript(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT NOT NULL, "
        "run_dir TEXT NOT NULL, case_path TEXT NOT NULL, design TEXT NOT NULL, "
        "state TEXT NOT NULL, profile TEXT NOT NULL, n_ranks INTEGER NOT NULL, "
        "overrides TEXT NOT NULL DEFAULT '{}', force_from TEXT, position REAL NOT NULL, "
        "status TEXT NOT NULL, pid INTEGER, pgid INTEGER, exit_code INTEGER, error TEXT, "
        "created_at REAL NOT NULL, started_at REAL, finished_at REAL);"
        "INSERT INTO jobs (run_name, run_dir, case_path, design, state, profile, n_ranks, "
        "position, status, created_at) VALUES ('old', '/x', 'c', 'v01', 'corner', 'p', 4, 1, "
        "'done', 0);"
    )
    old.commit()
    old.close()
    job = Queue(db).latest_for("old")
    assert job is not None and job.note is None
```

Append to `tests/test_ui_worker.py`:

```python
def test_the_worker_writes_the_job_note_into_the_run(tmp_path: Path) -> None:
    import json

    queue = Queue(tmp_path / "q.db")
    run_dir = tmp_path / "runs" / "r1"
    job = queue.enqueue(JobSpec("r1", str(run_dir), "case.yaml", "ok", "corner", "car_dev", 1,
                                note="new rear deck"))
    worker = Worker(queue, command=stub_command, cwd=tmp_path)
    worker._spawn(job)
    worker._procs[job.id].wait(timeout=60)
    assert json.loads((run_dir / "ui" / "note.json").read_text())["note"] == "new rear deck"
```

Append to `tests/test_ui_queue_routes.py`:

```python
def test_the_change_note_is_queued_and_prefilled_when_editing(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        client.post("/runs", data={**FORM, "note": "rear deck +3 mm"}, follow_redirects=False)
        job = app.state.ctx.queue.latest_for("r1")
        assert job.note == "rear deck +3 mm"
        assert "rear deck +3 mm" in client.get(f"/runs/new?job={job.id}").text


def test_the_new_run_form_offers_a_change_note(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        assert 'name="note"' in client.get("/runs/new").text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_queue.py tests/test_ui_worker.py tests/test_ui_queue_routes.py -v -k "note"`
Expected: FAIL (`TypeError: JobSpec.__init__() got an unexpected keyword argument 'note'`)

- [ ] **Step 3: Implement the queue changes**

In `pipeline/simdev/ui/queue.py`:

1. In `SCHEMA`, change the last column line of `jobs` from `    finished_at REAL` to:
```sql
    finished_at REAL,
    note        TEXT
```
2. Add `note: str | None = None` as the last field of `JobSpec` and as the last field of `Job` (after `finished_at`).
3. At the end of `Queue.__init__`, after `self._clock = clock`:
```python
        with self._lock:
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(jobs)")}
            if "note" not in columns:
                # A database made before change notes existed.
                self._db.execute("ALTER TABLE jobs ADD COLUMN note TEXT")
```
4. In `enqueue`, replace the INSERT with:
```python
            cursor = self._db.execute(
                "INSERT INTO jobs (run_name, run_dir, case_path, design, state, profile, "
                "n_ranks, overrides, force_from, position, status, created_at, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                (
                    spec.run_name, spec.run_dir, spec.case_path, spec.design, spec.state,
                    spec.profile, spec.n_ranks, json.dumps(spec.overrides, sort_keys=True),
                    spec.force_from, top + 1, self._clock(), spec.note,
                ),
            )
```
5. In `update`, replace the UPDATE with:
```python
            self._db.execute(
                "UPDATE jobs SET run_name = ?, run_dir = ?, case_path = ?, design = ?, "
                "state = ?, profile = ?, n_ranks = ?, overrides = ?, force_from = ?, note = ? "
                "WHERE id = ? AND status = 'queued'",
                (
                    spec.run_name, spec.run_dir, spec.case_path, spec.design, spec.state,
                    spec.profile, spec.n_ranks, json.dumps(spec.overrides, sort_keys=True),
                    spec.force_from, spec.note, job_id,
                ),
            )
```
6. In `requeue`, add `note=job.note,` to the `JobSpec(...)` call.

- [ ] **Step 4: Implement the worker change**

In `pipeline/simdev/ui/worker.py` add the import `from simdev.ui.notes import write_initial_note` and, in `_spawn`, directly after `write_job_file(job, ...)`:

```python
            write_initial_note(run_dir, job.note)
```

- [ ] **Step 5: Implement the form**

In `pipeline/simdev/ui/templates/new_run.html`, after `<div id="fields">{% include "_fields.html" %}</div>` (outside it: htmx replaces `#fields` when the pair changes, and the note must survive that):

```html
  <label>What changed in the geometry? <span class="muted">optional, shown in the results table</span>
    <textarea name="note" rows="2" maxlength="2000">{{ note or '' }}</textarea>
  </label>
```

In `pipeline/simdev/ui/routes_queue.py`:

1. `_form_page` gets a keyword parameter `note: str | None = None` and passes `note=note if note is not None else (job.note if job else None)` into the `render(...)` call.
2. In `queue_run`, after `name = ...`, add `note = str(form.get("note") or "").strip()[:2000] or None`; add `note=note,` to the `JobSpec(...)` call; and add `note=note,` to the `_form_page(...)` call in the `except` branch.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ui_queue.py tests/test_ui_worker.py tests/test_ui_queue_routes.py -v`
Expected: all pass, including the 5 new tests.

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/ui/queue.py pipeline/simdev/ui/worker.py pipeline/simdev/ui/routes_queue.py pipeline/simdev/ui/templates/new_run.html tests/test_ui_queue.py tests/test_ui_worker.py tests/test_ui_queue_routes.py
git commit -m "feat(ui): change note in the New Run form, copied into the run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Run summary and noise

**Files:**
- Modify: `pipeline/simdev/ui/runview.py` (rename `_force_frame` → `force_frame`, update its two callers in the same file)
- Create: `pipeline/simdev/ui/summary.py`
- Test: `tests/test_ui_summary.py`

**Interfaces:**
- Consumes: `simdev.ui.runview.force_frame(run_dir) -> pd.DataFrame` (de-duplicated by `Time`), `simdev.run.parsers.read_component_coeffs(run_dir) -> dict[str, pd.DataFrame]`, `simdev.stages.common.load_spec`.
- Produces:
  - `rolling_noise(values: Sequence[float]) -> float`
  - `@dataclass(frozen=True) class RunSummary: name: str; result: dict[str, Any]; window: tuple[int, int]; noise: dict[str, float]; patches: dict[str, dict[str, float]]; groups_map: dict[str, list[str]]`
    - `noise` keys: `"Cd"`, `"Cl"`, `"Cd_<group>"`, `"Cl_<group>"`
    - `patches[patch]` keys: `"Cd"`, `"Cl"`, `"Cd_noise"`, `"Cl_noise"`
  - `load_summary(run_dir: Path) -> RunSummary | None` (None without `results/result.json`; cached on file mtimes)

- [ ] **Step 1: Make `force_frame` public**

Run: `sed -i 's/\b_force_frame\b/force_frame/g' pipeline/simdev/ui/runview.py`
Then: `.venv/bin/python -m pytest tests/test_ui_runview.py tests/test_ui_run_routes.py -q`
Expected: all pass.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ui_summary.py
from __future__ import annotations

import json
import math
from pathlib import Path

from simdev.ui.summary import load_summary, rolling_noise

HEADER = "# Time Cd Cs Cl CmRoll CmPitch CmYaw\n"


def write_coeffs(path: Path, rows: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "".join(f"{t} {cd} 0 {cl} 0 0 0\n" for t, cd, cl in rows))


def make_run(root: Path, name: str = "r1", window=(3, 6)) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "results" / "result.json").write_text(json.dumps({
        "cd_mean": 0.5, "cl_mean": -1.0, "window_start": window[0], "window_end": window[1],
        "verdict": "converged",
    }))
    return run


def test_rolling_noise_is_half_the_spread_of_the_half_window_mean() -> None:
    # n = 4, rolling length 2 -> means 1.5, 2.5, 3.5 -> spread 2 -> noise 1
    assert rolling_noise([1, 2, 3, 4]) == 1.0
    assert rolling_noise([5, 5, 5, 5]) == 0.0
    assert math.isnan(rolling_noise([1.0]))


def test_no_result_no_summary(tmp_path: Path) -> None:
    (tmp_path / "r1").mkdir()
    assert load_summary(tmp_path / "r1") is None


def test_noise_uses_only_the_window_and_ignores_repeated_iterations(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    rows = [(1, 9, 9), (2, 9, 9), (3, 1, -1), (4, 2, -2), (5, 3, -3), (6, 4, -4)]
    write_coeffs(run / "postProcessing/forceCoeffs/0/coefficient.dat", rows)
    # A restart repeats 5 and 6 with the same values in a second time directory.
    write_coeffs(run / "postProcessing/forceCoeffs/5/coefficient.dat", rows[-2:])
    summary = load_summary(run)
    assert summary.window == (3, 6)
    assert summary.noise["Cd"] == 1.0
    assert summary.noise["Cl"] == 1.0


def test_patches_and_groups(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    write_coeffs(run / "postProcessing/forceCoeffs/0/coefficient.dat",
                 [(t, 1, -1) for t in range(1, 7)])
    write_coeffs(run / "postProcessing/forceCoeffs_Body/0/coefficient.dat",
                 [(t, 0.25 * t, -0.5) for t in range(1, 7)])
    write_coeffs(run / "postProcessing/forceCoeffs_Wing/0/coefficient.dat",
                 [(t, 0.1, -0.5) for t in range(1, 7)])
    summary = load_summary(run)
    assert summary.patches["Body"]["Cd"] == (0.75 + 1.0 + 1.25 + 1.5) / 4
    assert summary.patches["Wing"]["Cl_noise"] == 0.0
    assert summary.groups_map == {}  # no caseSpec.json: no groups, no crash


def test_a_run_without_histories_still_has_a_summary(tmp_path: Path) -> None:
    summary = load_summary(make_run(tmp_path))
    assert summary.result["cl_mean"] == -1.0
    assert math.isnan(summary.noise["Cl"])
    assert summary.patches == {}
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_summary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.ui.summary'`

- [ ] **Step 4: Implement**

```python
# pipeline/simdev/ui/summary.py
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
    return _load(str(run_dir), _signature(run_dir))


@lru_cache(maxsize=256)
def _load(run_dir_s: str, _signature: tuple) -> RunSummary:
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
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ui_summary.py tests/test_ui_runview.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/ui/runview.py pipeline/simdev/ui/summary.py tests/test_ui_summary.py
git commit -m "feat(ui): per-run summary with window noise for coefficients, patches and groups

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Results table model

**Files:**
- Create: `pipeline/simdev/ui/results.py`
- Test: `tests/test_ui_results.py`

**Interfaces:**
- Consumes: `load_summary`, `RunSummary`, `finite` (Task 3); `read_note`, `RunNote` (Task 1); `runview.list_runs(runs_root) -> list[Path]`.
- Produces:
  - `@dataclass(frozen=True) class Column: key: str; label: str; better: int; fmt: str; noise_key: str | None = None`
  - `columns_for(groups: Sequence[str]) -> list[Column]`
  - `@dataclass(frozen=True) class Row: name: str; state: str; design: str; note: RunNote; has_result: bool; verdict: str | None; n_iterations: int | None; n_cells: int | None; values: dict[str, float | None]; noise: dict[str, float | None]; groups: tuple[str, ...]`
  - `@dataclass(frozen=True) class DeltaCell: value: float | None; noise: float | None; tone: str` (`"better" | "worse" | "noise" | "neutral" | "none"`)
  - `load_row(run_dir: Path) -> Row`, `load_rows(runs_root: Path) -> list[Row]`
  - `delta(row: Row, ref: Row, columns: Sequence[Column]) -> dict[str, DeltaCell]`
  - `group_names(rows: Sequence[Row]) -> list[str]`
  - `to_tsv(rows: Sequence[Row], columns: Sequence[Column], by_name: Mapping[str, Row]) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ui_results.py
from __future__ import annotations

import json
from pathlib import Path

from simdev.ui.notes import write_note
from simdev.ui.results import columns_for, delta, group_names, load_row, load_rows, to_tsv


def make_run(root: Path, name: str, **result) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "caseSpec.json").write_text("{}")
    payload = {"cd_mean": 0.80, "cl_mean": -1.00, "window_start": 1, "window_end": 2,
               "verdict": "converged", "n_iterations": 400, "n_cells": 1000, **result}
    (run / "results" / "result.json").write_text(json.dumps(payload))
    return run


def test_an_old_result_without_groups_or_forces_loads_with_empty_cells(tmp_path: Path) -> None:
    row = load_row(make_run(tmp_path, "old"))
    assert row.has_result
    assert row.values["cl"] == -1.0
    assert row.values["fx"] is None and row.values["cop_x"] is None
    assert row.values["eff"] == 1.0 / 0.8


def test_groups_become_columns(tmp_path: Path) -> None:
    row = load_row(make_run(tmp_path, "g", groups={"body": {"Cd": 0.5, "Cl": -0.9},
                                                    "wing": {"Cd": 0.1, "Cl": -0.2}}))
    assert group_names([row]) == ["body", "wing"]
    keys = [c.key for c in columns_for(["body", "wing"])]
    assert keys[:3] == ["cl", "cd", "eff"]
    assert "cl_body" in keys and "cd_wing" in keys
    assert row.values["cl_body"] == -0.9


def test_delta_tones_follow_better_direction_and_noise(tmp_path: Path) -> None:
    ref = load_row(make_run(tmp_path, "ref", cl_mean=-1.00, cd_mean=0.80))
    run = load_row(make_run(tmp_path, "new", cl_mean=-1.10, cd_mean=0.81,
                            balance_front_pct=41.0))
    cols = columns_for([])
    # no histories: noise unknown, so tones go by direction alone
    cells = delta(run, ref, cols)
    assert cells["cl"].tone == "better" and round(cells["cl"].value, 6) == -0.1
    assert cells["cd"].tone == "worse"
    assert cells["balance"].tone == "none"  # ref has no balance


def test_a_delta_inside_the_noise_is_greyed(tmp_path: Path) -> None:
    from dataclasses import replace

    ref = load_row(make_run(tmp_path, "ref"))
    run = load_row(make_run(tmp_path, "new", cl_mean=-1.005))
    run = replace(run, noise={**run.noise, "cl": 0.01})
    ref = replace(ref, noise={**ref.noise, "cl": 0.01})
    assert delta(run, ref, columns_for([]))["cl"].tone == "noise"


def test_rows_carry_note_state_and_runs_without_results(tmp_path: Path) -> None:
    run = make_run(tmp_path, "a")
    (run / "results" / "report.tsv").write_text("run\tdriving_state\na\ttc10-low\n")
    write_note(run, "<script>x</script>", "b")
    (tmp_path / "pending").mkdir()
    (tmp_path / "pending" / "caseSpec.json").write_text("{}")
    rows = {r.name: r for r in load_rows(tmp_path)}
    assert rows["a"].state == "tc10-low"
    assert rows["a"].note.compare_with == "b"
    assert rows["pending"].has_result is False


def test_tsv_has_a_delta_line_under_each_referenced_run(tmp_path: Path) -> None:
    ref = load_row(make_run(tmp_path, "ref"))
    run_dir = make_run(tmp_path, "new", cl_mean=-1.1)
    write_note(run_dir, "deck", "ref")
    run = load_row(run_dir)
    text = to_tsv([run, ref], columns_for([]), {"ref": ref, "new": run})
    lines = text.splitlines()
    assert lines[0].split("\t")[:4] == ["run", "state", "note", "compare_with"]
    assert lines[1].startswith("new\t")
    assert lines[2].startswith("Δ new − ref\t")
    assert lines[3].startswith("ref\t")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.ui.results'`

- [ ] **Step 3: Implement**

```python
# pipeline/simdev/ui/results.py
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/ui/results.py tests/test_ui_results.py
git commit -m "feat(ui): results table rows, noise-aware delta cells and TSV

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Results page

**Files:**
- Create: `pipeline/simdev/ui/routes_results.py`, `pipeline/simdev/ui/templates/results.html`, `pipeline/simdev/ui/templates/_result_row.html`
- Modify: `pipeline/simdev/ui/app.py` (register router, `num` filter), `pipeline/simdev/ui/templates/base.html` (nav), `pipeline/simdev/ui/templates/run.html` (compare link), `pipeline/simdev/ui/routes_runs.py` (`run_page` passes the note), `pipeline/simdev/ui/static/style.css`
- Test: `tests/test_ui_results_routes.py`

**Interfaces:**
- Consumes: Tasks 1, 4; `forms.RUN_NAME`; `context.ctx/render`.
- Produces: routes `GET /results`, `POST /runs/{name}/note`, `GET /results.tsv`; Jinja filter `num(value, fmt)`; helper `valid_run_name(name: str) -> bool` in `routes_results.py` (reused by Task 9).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ui_results_routes.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from simdev.ui.notes import read_note  # noqa: E402
from tests.ui_support import make_client  # noqa: E402


def make_run(root: Path, name: str, cl: float = -1.0) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "caseSpec.json").write_text("{}")
    (run / "results" / "result.json").write_text(json.dumps(
        {"cd_mean": 0.8, "cl_mean": cl, "window_start": 1, "window_end": 2,
         "verdict": "converged"}))
    return run


def test_results_lists_runs_with_their_numbers(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        page = client.get("/results").text
        assert "base" in page and "-1.0000" in page
        assert 'href="/results"' in client.get("/").text


def test_editing_note_and_reference_inline(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        new = make_run(tmp_path / "runs", "new", cl=-1.2)
        response = client.post("/runs/new/note", data={"note": "<b>deck</b>", "compare_with": "base"})
        assert response.status_code == 200
        assert "&lt;b&gt;deck&lt;/b&gt;" in response.text
        assert "Δ" in response.text
        assert read_note(new).compare_with == "base"


def test_shell_runs_can_be_annotated(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        run = make_run(tmp_path / "runs", "shellrun")
        assert app.state.ctx.queue.latest_for("shellrun") is None
        assert client.post("/runs/shellrun/note", data={"note": "x", "compare_with": ""}).status_code == 200
        assert read_note(run).note == "x"


@pytest.mark.parametrize("bad", ["new", "../base", "a/b", "nope!"])
def test_bad_references_are_refused(tmp_path: Path, bad: str) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "new")
        response = client.post("/runs/new/note", data={"note": "", "compare_with": bad})
        assert response.status_code == 400


def test_a_deleted_reference_shows_missing(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "new")
        (tmp_path / "runs" / "new" / "ui").mkdir()
        (tmp_path / "runs" / "new" / "ui" / "note.json").write_text(
            json.dumps({"note": "", "compare_with": "gone"}))
        page = client.get("/results").text
        assert "missing" in page


def test_tsv_download(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        response = client.get("/results.tsv")
        assert response.headers["content-type"].startswith("text/tab-separated-values")
        assert response.text.splitlines()[1].startswith("base\t")


def test_the_run_page_links_to_its_comparison(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "base")
        make_run(tmp_path / "runs", "new")
        client.post("/runs/new/note", data={"note": "", "compare_with": "base"})
        assert "/compare?runs=base,new" in client.get("/runs/new").text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_results_routes.py -v`
Expected: FAIL (404 on `/results`)

- [ ] **Step 3: Implement the routes**

```python
# pipeline/simdev/ui/routes_results.py
"""The results table: every run's numbers, its note and its reference."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import PlainTextResponse

from simdev.ui import forms, results
from simdev.ui.context import ctx, render
from simdev.ui.notes import write_note

router = APIRouter()


def valid_run_name(name: str) -> bool:
    return bool(forms.RUN_NAME.fullmatch(name)) and ".." not in name


def _table(request: Request):
    rows = results.load_rows(ctx(request).config.runs_root)
    columns = results.columns_for(results.group_names(rows))
    return rows, columns, {r.name: r for r in rows}


def _matches(row, state: str, design: str, verdict: str, q: str) -> bool:
    text = f"{row.name} {row.note.note}".lower()
    return ((not state or row.state == state) and (not design or row.design == design)
            and (not verdict or row.verdict == verdict) and (not q or q.lower() in text))


@router.get("/results")
def results_page(request: Request, state: str = "", design: str = "", verdict: str = "",
                 q: str = ""):
    rows, columns, by_name = _table(request)
    shown = [r for r in rows if _matches(r, state, design, verdict, q)]
    return render(
        request, "results.html", rows=shown, columns=columns, by_name=by_name,
        names=[r.name for r in rows], delta=results.delta,
        states=sorted({r.state for r in rows if r.state}),
        designs=sorted({r.design for r in rows if r.design}),
        filters={"state": state, "design": design, "verdict": verdict, "q": q},
    )


@router.post("/runs/{name}/note")
def save_note(request: Request, name: str, note: str = Form(""), compare_with: str = Form("")):
    if not valid_run_name(name):
        raise HTTPException(status_code=404)
    run_dir = ctx(request).config.runs_root / name
    if not run_dir.is_dir():
        raise HTTPException(status_code=404)
    reference = compare_with.strip() or None
    if reference is not None and (not valid_run_name(reference) or reference == name):
        raise HTTPException(status_code=400, detail="a run cannot be compared with that")
    # Allowed for runs started from the shell too: this is metadata, never results.
    write_note(run_dir, note, reference)
    rows, columns, by_name = _table(request)
    return render(request, "_result_row.html", row=by_name[name], columns=columns,
                  by_name=by_name, names=list(by_name), delta=results.delta)


@router.get("/results.tsv")
def results_tsv(request: Request):
    rows, columns, by_name = _table(request)
    return PlainTextResponse(results.to_tsv(rows, columns, by_name),
                             media_type="text/tab-separated-values; charset=utf-8")
```

In `pipeline/simdev/ui/app.py`: import `routes_results`, add `routes_results.router` to the router tuple, and register a filter:

```python
def number(value: float | None, fmt: str = "{:.4f}") -> str:
    return "" if value is None else fmt.format(value)
```

with `templates.env.filters.update(duration=duration, overrides=overrides_text, clock=clock, num=number)`.

In `routes_runs.run_page`, add `note=read_note(run_dir)` to the `render(...)` call (import `from simdev.ui.notes import read_note`).

- [ ] **Step 4: Implement the templates**

`pipeline/simdev/ui/templates/_result_row.html`:

```html
{% set ref = by_name.get(row.note.compare_with or '') %}
<tbody id="row-{{ row.name }}" class="result{% if not row.has_result %} pending{% endif %}">
<tr>
  <td><input type="checkbox" name="runs" value="{{ row.name }}" form="compare-form" {% if not row.has_result %}disabled{% endif %}></td>
  <td><a href="/runs/{{ row.name }}">{{ row.name }}</a><br><span class="muted">{{ row.state }}</span></td>
  <td><textarea class="note-edit" name="note" rows="1" maxlength="2000"
        hx-post="/runs/{{ row.name }}/note" hx-trigger="change" hx-include="closest tbody"
        hx-target="closest tbody" hx-swap="outerHTML">{{ row.note.note }}</textarea>
      {% if row.note.problem %}<div class="warn">{{ row.note.problem }}</div>{% endif %}</td>
  <td><select class="note-edit" name="compare_with"
        hx-post="/runs/{{ row.name }}/note" hx-trigger="change" hx-include="closest tbody"
        hx-target="closest tbody" hx-swap="outerHTML">
      <option value="">–</option>
      {% for n in names if n != row.name %}<option {% if n == row.note.compare_with %}selected{% endif %}>{{ n }}</option>{% endfor %}
      {% if row.note.compare_with and not ref %}<option selected value="{{ row.note.compare_with }}">{{ row.note.compare_with }} (missing)</option>{% endif %}
    </select></td>
  {% if row.has_result %}
    {% for c in columns %}<td class="num">{{ row.values.get(c.key)|num(c.fmt) }}</td>{% endfor %}
    <td><span class="badge {{ 'ok' if row.verdict == 'converged' else 'gate_failed' }}">{{ row.verdict }}</span></td>
    <td class="num">{{ row.noise.get('cl')|num('{:.4f}') }} / {{ row.noise.get('cd')|num('{:.4f}') }}</td>
  {% else %}
    <td colspan="{{ columns|length + 2 }}" class="muted">no results yet</td>
  {% endif %}
</tr>
{% if ref and row.has_result and ref.has_result %}
{% set cells = delta(row, ref, columns) %}
<tr class="delta">
  <td></td><td colspan="3" class="muted">Δ against {{ ref.name }}</td>
  {% for c in columns %}{% set d = cells[c.key] %}
  <td class="num {{ d.tone }}" {% if d.noise is not none %}title="Δ {{ d.value|num('{:+.4f}') }} ± {{ d.noise|num('{:.4f}') }}"{% endif %}>{{ d.value|num(c.fmt) }}</td>
  {% endfor %}
  <td colspan="2"></td>
</tr>
{% elif row.note.compare_with and not ref %}
<tr class="delta"><td></td><td colspan="{{ columns|length + 5 }}" class="muted">reference {{ row.note.compare_with }} missing</td></tr>
{% endif %}
</tbody>
```

`pipeline/simdev/ui/templates/results.html`:

```html
{% extends "base.html" %}
{% block title %}Results · SimDev{% endblock %}
{% block main %}
<h1>Results</h1>
<form method="get" class="row">
  <select name="state"><option value="">all states</option>{% for s in states %}<option {% if s == filters.state %}selected{% endif %}>{{ s }}</option>{% endfor %}</select>
  <select name="design"><option value="">all designs</option>{% for d in designs %}<option {% if d == filters.design %}selected{% endif %}>{{ d }}</option>{% endfor %}</select>
  <select name="verdict"><option value="">any verdict</option>{% for v in ['converged', 'not_converged', 'not_judged'] %}<option {% if v == filters.verdict %}selected{% endif %}>{{ v }}</option>{% endfor %}</select>
  <input name="q" value="{{ filters.q }}" placeholder="name or note">
  <button>Filter</button>
  <a class="button" href="/results.tsv">Copy as TSV</a>
</form>
<form id="compare-form" method="get" action="/compare" class="row">
  <button class="primary">Open ticked runs in viewer</button>
  <span class="muted">the first ticked run is the reference</span>
</form>
<div class="scroll">
<table class="results">
  <thead><tr><th></th><th>run</th><th>note</th><th>compare with</th>
    {% for c in columns %}<th class="num">{{ c.label }}</th>{% endfor %}
    <th>verdict</th><th class="num">noise Cl / Cd</th></tr></thead>
  {% for row in rows %}{% include "_result_row.html" %}{% else %}
  <tbody><tr><td colspan="{{ columns|length + 6 }}" class="muted">No runs yet.</td></tr></tbody>
  {% endfor %}
</table>
</div>
{% endblock %}
```

In `base.html` add `<a href="/results">Results</a>` after the Runs link.

In `run.html`, after `<h1>{{ name }}</h1>`:

```html
{% if note and note.compare_with %}
<p><a href="/compare?runs={{ note.compare_with }},{{ name }}">Compare with reference {{ note.compare_with }}</a></p>
{% endif %}
```

Append to `style.css`:

```css
.scroll { overflow-x: auto; }
table.results td.num, table.results th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.results textarea { width: 16rem; }
tbody.result { border-top: 2px solid var(--line); }
tbody.pending { opacity: 0.6; }
tr.delta td { font-size: 12px; }
tr.delta td.better { color: var(--ok); font-weight: 600; }
tr.delta td.worse { color: var(--bad); font-weight: 600; }
tr.delta td.noise { color: var(--muted); }
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ui_results_routes.py tests/test_ui_run_routes.py tests/test_ui_queue_routes.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/ui/routes_results.py pipeline/simdev/ui/templates/results.html pipeline/simdev/ui/templates/_result_row.html pipeline/simdev/ui/app.py pipeline/simdev/ui/templates/base.html pipeline/simdev/ui/templates/run.html pipeline/simdev/ui/routes_runs.py pipeline/simdev/ui/static/style.css tests/test_ui_results_routes.py
git commit -m "feat(ui): results page with inline notes, references, delta rows and TSV

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Image index and data APIs

**Files:**
- Create: `pipeline/simdev/ui/imageindex.py`
- Test: `tests/test_ui_imageindex.py`

**Interfaces:**
- Consumes: `load_summary` (Task 3), `simdev.viz.cplines.read_station(path) -> dict[patch, (xs, zs, ps)]`.
- Produces:
  - `reroot(recorded: str, run_dir: Path) -> Path` - maps an absolute path written into `render_plan.json` onto the current run directory (keeps everything from the first `results/` or `postProcessing/` segment).
  - `load_plan(run_dir: Path) -> dict | None`
  - `build_index(run_dir: Path) -> dict` with keys `run`, `views_digest`, `state`, `u_inf`, `window`, `planes`, `surfaces`, `stations`, `groups`; `planes[axis][field]` = list of `{"name": "x_-0.100", "offset": -0.1, "rel": "results/images/..."}` sorted by offset; `surfaces[field][view]` = rel; `stations` = list of `{"name", "offset"}`.
  - `cp_station(run_dir: Path, station: str) -> dict` → `{"u_inf": float, "patches": {patch: {"x": [...], "z": [...], "cp": [...]}}}`; raises `KeyError` for an unknown station.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ui_imageindex.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdev.ui.imageindex import build_index, cp_station, reroot

OLD = "/somewhere/else/runs/r1"


def make_run(root: Path) -> Path:
    run = root / "r1"
    for rel in ("results/images/cp_x/cp_x_2_-0.100.png", "results/images/cp_x/cp_x_1_+0.000.png",
                "results/images/surface_cp/top.png"):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_bytes(b"png")
    (run / "results/cp_lines").mkdir(parents=True)
    (run / "results/cp_lines/y_+0.000.csv").write_text("patch,x,z,pMean\nBody,0.1,0.05,18\n")
    plan = {
        "views_digest": "abc", "frame": {"mode": "cornering", "u_inf": 6.0, "omega": 3.0,
                                         "origin": [0, 2, 0]},
        "slices": [
            {"name": "x_+0.000", "axis": "x", "offset": 0.0, "sample": f"{OLD}/postProcessing/surfaces/400/x_+0.000.vtp",
             "images": [{"field": "cp", "out": f"{OLD}/results/images/cp_x/cp_x_1_+0.000.png"},
                        {"field": "U", "out": f"{OLD}/results/images/U_x/U_x_1_+0.000.png"}]},
            {"name": "x_-0.100", "axis": "x", "offset": -0.1, "sample": "",
             "images": [{"field": "cp", "out": f"{OLD}/results/images/cp_x/cp_x_2_-0.100.png"}]},
        ],
        "surfaces": [{"name": "top", "sample": "", "images": [
            {"field": "cp", "out": f"{OLD}/results/images/surface_cp/top.png"}]}],
        "cp_lines": {"stations": [{"name": "y_+0.000", "offset": 0.0,
                                   "csv": f"{OLD}/results/cp_lines/y_+0.000.csv"}]},
    }
    (run / "results/render_plan.json").write_text(json.dumps(plan))
    (run / "results/result.json").write_text(json.dumps({"window_start": 300, "window_end": 400}))
    return run


def test_reroot_keeps_the_part_inside_the_run(tmp_path: Path) -> None:
    assert reroot(f"{OLD}/results/images/a.png", tmp_path) == tmp_path / "results/images/a.png"
    assert reroot(f"{OLD}/postProcessing/surfaces/400/x.vtp", tmp_path) == \
        tmp_path / "postProcessing/surfaces/400/x.vtp"


def test_index_lists_existing_pictures_sorted_and_rerooted(tmp_path: Path) -> None:
    index = build_index(make_run(tmp_path))
    planes = index["planes"]["x"]["cp"]
    assert [p["offset"] for p in planes] == [-0.1, 0.0]
    assert planes[0]["rel"] == "results/images/cp_x/cp_x_2_-0.100.png"
    assert "U" not in index["planes"]["x"]  # the U picture does not exist on disk
    assert index["surfaces"]["cp"]["top"] == "results/images/surface_cp/top.png"
    assert index["views_digest"] == "abc" and index["u_inf"] == 6.0
    assert index["window"] == [300, 400]
    assert index["stations"] == [{"name": "y_+0.000", "offset": 0.0}]


def test_a_run_without_pictures_has_an_empty_index(tmp_path: Path) -> None:
    (tmp_path / "r2").mkdir()
    index = build_index(tmp_path / "r2")
    assert index["planes"] == {} and index["surfaces"] == {}


def test_cp_station_converts_to_cp(tmp_path: Path) -> None:
    data = cp_station(make_run(tmp_path), "y_+0.000")
    assert data["patches"]["Body"]["cp"] == [18 / (0.5 * 36)]
    with pytest.raises(KeyError):
        cp_station(tmp_path / "r1", "y_+9.000")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_imageindex.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.ui.imageindex'`

- [ ] **Step 3: Implement**

```python
# pipeline/simdev/ui/imageindex.py
"""Which pictures a run has, for the viewer, read from its render plan.

render_plan.json is the authoritative list - it names every picture with its
plane, offset and field - but it records absolute paths, and the runs folder
is reached through more than one path on this machine (~/runs is a symlink
to /mnt/data/runs). Every recorded path is therefore re-rooted onto the run
directory being looked at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simdev.ui.summary import load_summary
from simdev.viz.cplines import read_station

ANCHORS = ("results", "postProcessing")


def reroot(recorded: str, run_dir: Path) -> Path:
    parts = Path(recorded).parts
    for i, part in enumerate(parts):
        if part in ANCHORS:
            return Path(run_dir).joinpath(*parts[i:])
    return Path(recorded)


def load_plan(run_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(run_dir) / "results" / "render_plan.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _rel(recorded: str, run_dir: Path) -> str | None:
    path = reroot(recorded, run_dir)
    if not path.is_file():
        return None
    return str(path.relative_to(run_dir))


def build_index(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    plan = load_plan(run_dir) or {}
    planes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in plan.get("slices", []):
        for image in entry.get("images", []):
            rel = _rel(image["out"], run_dir)
            if rel is None:
                continue
            planes.setdefault(entry["axis"], {}).setdefault(image["field"], []).append(
                {"name": entry["name"], "offset": entry["offset"], "rel": rel})
    for fields in planes.values():
        for items in fields.values():
            items.sort(key=lambda p: p["offset"])
    surfaces: dict[str, dict[str, str]] = {}
    for entry in plan.get("surfaces", []):
        for image in entry.get("images", []):
            rel = _rel(image["out"], run_dir)
            if rel is not None:
                surfaces.setdefault(image["field"], {})[entry["name"]] = rel
    stations = [{"name": s["name"], "offset": s["offset"]}
                for s in (plan.get("cp_lines") or {}).get("stations", [])]
    summary = load_summary(run_dir)
    job_state = ""
    try:
        job_state = json.loads((run_dir / "ui" / "job.json").read_text("utf-8")).get("state", "")
    except (OSError, ValueError, AttributeError):
        pass
    return {
        "run": run_dir.name,
        "views_digest": plan.get("views_digest"),
        "state": job_state,
        "u_inf": (plan.get("frame") or {}).get("u_inf"),
        "window": list(summary.window) if summary else None,
        "planes": planes,
        "surfaces": surfaces,
        "stations": stations,
        "groups": summary.groups_map if summary else {},
    }


def cp_station(run_dir: Path, station: str) -> dict[str, Any]:
    plan = load_plan(run_dir) or {}
    for entry in (plan.get("cp_lines") or {}).get("stations", []):
        if entry["name"] == station:
            path = reroot(entry["csv"], Path(run_dir))
            break
    else:
        raise KeyError(station)
    u_inf = float((plan.get("frame") or {}).get("u_inf") or 0.0)
    q = 0.5 * u_inf * u_inf
    points = read_station(path) if path.is_file() else {}
    return {
        "u_inf": u_inf,
        "patches": {patch: {"x": xs, "z": zs, "cp": [p / q for p in ps] if q else []}
                    for patch, (xs, zs, ps) in points.items()},
    }
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ui_imageindex.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/ui/imageindex.py tests/test_ui_imageindex.py
git commit -m "feat(ui): picture index and cp stations from the render plan, re-rooted

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Delta helper (VTK side)

**Files:**
- Create: `pipeline/simdev/viz/delta.py`
- Test: `tests/test_delta_helper.py`

**Interfaces:**
- Produces (pure, importable in the venv):
  - `BANDS = 20`, `SOLID = (212, 212, 212)`, `MOVED = (0, 0, 0)`
  - `diverging_table(n: int = BANDS) -> list[tuple[int, int, int]]`
  - `frame_grid(slice_entry: dict, resolution: Sequence[int]) -> tuple[np.ndarray, tuple[int, int]]` - points `(ny*nx, 3)`, row 0 at the **bottom**
  - `derived_fields(p, U, valid, pts, frame) -> dict[str, np.ndarray]` (`cp`, `cpt`, `U`, NaN where invalid)
  - `delta_rgb(delta: np.ndarray, solid: np.ndarray, moved: np.ndarray, limit: float) -> np.ndarray` uint8 `(ny, nx, 3)`, row 0 at the **top**
  - `draw_colour_bar(rgb: np.ndarray, limit: float, label: str) -> np.ndarray`
- Produces (VTK, run under the ParaView interpreter):
  - `plane_delta(request: dict) -> dict`, `surface_delta(request: dict) -> dict`, `main(argv) -> int`
  - CLI: `python delta.py request.json` → writes `request["out"]` atomically (temp + rename), prints one JSON line `{"ok": true, "out": ..., "seconds": ..., "max_abs": ..., "moved_pct": ...}`; on failure prints `{"ok": false, "error": ...}` and exits 1.
- Request (plane): `{"kind": "plane", "field": "cp"|"cpt"|"U", "limit": float, "resolution": [w, h], "slice": {"point","normal","camera"}, "pane": {"sample": path, "frame": {...}}, "ref": {"sample": path, "frame": {...}}, "out": path}`
- Request (surface): `{"kind": "surface", "field": "cp", "limit": float, "resolution": [w, h], "camera": {...}, "pane": {...}, "ref": {...}, "cache_vtp": path, "out": path}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_delta_helper.py
"""The delta helper: pure parts in the venv, VTK parts under the ParaView
interpreter (skipped where that interpreter cannot import vtk)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from simdev.viz import delta as D

PVPY = "/usr/bin/python3"
HELPER = Path(D.__file__)
SLICE = {"point": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0],
         "camera": {"focal": [0.0, 0.0, 0.0], "position": [1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0], "parallel_scale": 0.1}}
STRAIGHT = {"mode": "straight", "u_inf": 10.0, "omega": 0.0, "origin": [0.0, 0.0, 0.0]}


def has_vtk() -> bool:
    try:
        return subprocess.run([PVPY, "-c", "import vtk"], capture_output=True,
                              env={"PATH": "/usr/bin:/bin"}).returncode == 0
    except OSError:
        return False


def test_the_table_is_symmetric_with_twenty_bands() -> None:
    table = D.diverging_table()
    assert len(table) == 20
    assert table[0][2] > table[0][0]   # most negative is blue
    assert table[-1][0] > table[-1][2]  # most positive is red
    assert table[9] != table[10]        # zero is a band edge, not a band


def test_frame_grid_lies_on_the_plane_and_covers_the_frame() -> None:
    pts, (ny, nx) = D.frame_grid(SLICE, [40, 30])
    assert (ny, nx) == (30, 40)
    assert np.allclose(pts[:, 0], 0.0)
    assert np.isclose(pts[:, 2].min(), -0.1) and np.isclose(pts[:, 2].max(), 0.1)
    assert np.isclose(np.ptp(pts[:, 1]), 0.2 * 40 / 30)
    assert pts[0, 2] < pts[-1, 2]  # row 0 at the bottom


def test_derived_fields_match_the_renderer() -> None:
    pts = np.zeros((2, 3))
    U = np.array([[10.0, 0, 0], [5.0, 0, 0]])
    p = np.array([0.0, 50.0])
    out = D.derived_fields(p, U, np.array([True, False]), pts, STRAIGHT)
    assert out["cp"][0] == 0.0 and out["cpt"][0] == 1.0 and out["U"][0] == 10.0
    assert np.isnan(out["cp"][1])


def test_delta_rgb_marks_solid_moved_and_flips_rows() -> None:
    delta = np.array([[0.001, np.nan], [0.5, np.nan]])
    solid = np.array([[False, True], [False, False]])
    moved = np.array([[False, False], [False, True]])
    rgb = D.delta_rgb(delta, solid, moved, 0.2)
    assert tuple(rgb[1, 1]) == D.SOLID          # data row 0 -> image bottom row
    assert tuple(rgb[0, 1]) == D.MOVED
    assert tuple(rgb[0, 0]) == D.diverging_table()[-1]  # clipped to the top band
    assert tuple(rgb[1, 0]) == D.diverging_table()[10]  # first positive band


def vtp(path: Path, points, polys, p, U) -> None:
    pts = " ".join(f"{a} {b} {c}" for a, b, c in points)
    conn = " ".join(str(i) for poly in polys for i in poly)
    offs = " ".join(str(3 * (i + 1)) for i in range(len(polys)))
    path.write_text(f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
<PolyData><Piece NumberOfPoints="{len(points)}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{len(polys)}">
<PointData>
<DataArray type="Float32" Name="pMean" format="ascii">{" ".join(map(str, p))}</DataArray>
<DataArray type="Float32" Name="UMean" NumberOfComponents="3" format="ascii">{" ".join(f"{u} 0 0" for u in U)}</DataArray>
</PointData>
<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts}</DataArray></Points>
<Polys><DataArray type="Int32" Name="connectivity" format="ascii">{conn}</DataArray>
<DataArray type="Int32" Name="offsets" format="ascii">{offs}</DataArray></Polys>
</Piece></PolyData></VTKFile>""")


SQUARE = [(0, -0.05, -0.05), (0, 0.05, -0.05), (0, 0.05, 0.05), (0, -0.05, 0.05)]


@pytest.mark.skipif(not has_vtk(), reason="the ParaView interpreter has no vtk")
def test_plane_delta_end_to_end(tmp_path: Path) -> None:
    vtp(tmp_path / "a.vtp", SQUARE, [(0, 1, 2), (0, 2, 3)], [10, 10, 10, 10], [10] * 4)
    vtp(tmp_path / "b.vtp", SQUARE, [(0, 1, 2), (0, 2, 3)], [15, 15, 15, 15], [10] * 4)
    request = {"kind": "plane", "field": "cp", "limit": 0.2, "resolution": [40, 30],
               "slice": SLICE, "pane": {"sample": str(tmp_path / "b.vtp"), "frame": STRAIGHT},
               "ref": {"sample": str(tmp_path / "a.vtp"), "frame": STRAIGHT},
               "out": str(tmp_path / "d.png")}
    (tmp_path / "req.json").write_text(json.dumps(request))
    done = subprocess.run([PVPY, str(HELPER), str(tmp_path / "req.json")], capture_output=True,
                          text=True, env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert done.returncode == 0, done.stderr
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    assert summary["ok"] and abs(summary["max_abs"] - 5 / 50) < 1e-6
    assert (tmp_path / "d.png").stat().st_size > 0


@pytest.mark.skipif(not has_vtk(), reason="the ParaView interpreter has no vtk")
def test_surface_interpolation_masks_moved_surface(tmp_path: Path) -> None:
    moved = [(x + 0.01, y, z) for x, y, z in SQUARE]
    vtp(tmp_path / "pane.vtp", SQUARE, [(0, 1, 2), (0, 2, 3)], [5] * 4, [0] * 4)
    vtp(tmp_path / "ref.vtp", moved, [(0, 1, 2), (0, 2, 3)], [0] * 4, [0] * 4)
    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('d', {str(HELPER)!r})\n"
        "d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)\n"
        f"_, stats = d.surface_delta_field({str(tmp_path / 'pane.vtp')!r}, {str(tmp_path / 'ref.vtp')!r}, 50.0)\n"
        "print(json.dumps(stats))\n"
    )
    done = subprocess.run([PVPY, "-c", code], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1])["moved_pct"] == 100.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_delta_helper.py -v`
Expected: FAIL with `ImportError: cannot import name 'delta' from 'simdev.viz'`

- [ ] **Step 3: Implement**

```python
# pipeline/simdev/viz/delta.py
"""Field deltas between two runs, as pictures framed like the run's own.

Runs under the ParaView interpreter (post.paraview_python), like
viz/pv_render.py, and imports nothing from simdev. VTK and ParaView are
imported inside the functions that need them, so the pure parts - grid,
fields, colours - can be imported and tested from the venv.

Never decodes the existing PNGs: on banded pictures a small change turns
every shifted band edge into a one-band ring, which reads as noise.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

BANDS = 20  # even, so zero is a band edge and a small offset still shows
SOLID = (212, 212, 212)
MOVED = (0, 0, 0)
PROBE_TOLERANCE = 1e-4  # cut points sit up to ~50 um off the plane
SURFACE_RADIUS = 5e-4

_NEG = [(5, 48, 97), (33, 102, 172), (67, 147, 195), (146, 197, 222), (209, 229, 240),
        (247, 247, 247)]
_POS = [(247, 247, 247), (253, 219, 199), (244, 165, 130), (214, 96, 77), (178, 24, 43),
        (103, 0, 31)]


def _ramp(anchors, t: float) -> tuple[int, int, int]:
    x = t * (len(anchors) - 1)
    i = min(int(x), len(anchors) - 2)
    f = x - i
    a, b = anchors[i], anchors[i + 1]
    return tuple(int(round(a[k] + (b[k] - a[k]) * f)) for k in range(3))


def diverging_table(n: int = BANDS) -> list[tuple[int, int, int]]:
    half = n // 2
    # Band centres; the band touching zero is the palest but never white.
    neg = [_ramp(_NEG, (i + 0.5) / half * 0.9) for i in range(half)]
    pos = [_ramp(_POS, 0.1 + (i + 0.5) / half * 0.9) for i in range(half)]
    return neg + pos


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def frame_grid(slice_entry: dict, resolution: Sequence[int]) -> tuple[np.ndarray, tuple[int, int]]:
    cam = slice_entry["camera"]
    n = _unit(slice_entry["normal"])
    point = np.asarray(slice_entry["point"], dtype=float)
    focal = np.asarray(cam["focal"], dtype=float)
    focal = focal - np.dot(focal - point, n) * n
    view = _unit(focal - np.asarray(cam["position"], dtype=float))
    up = np.asarray(cam["up"], dtype=float)
    up = _unit(up - np.dot(up, n) * n)
    right = _unit(np.cross(view, up))
    nx, ny = int(resolution[0]), int(resolution[1])
    hh = float(cam["parallel_scale"])
    hw = hh * nx / ny
    uu, vv = np.meshgrid(np.linspace(-hw, hw, nx), np.linspace(-hh, hh, ny))
    pts = focal + uu[..., None] * right + vv[..., None] * up
    return pts.reshape(-1, 3), (ny, nx)


def derived_fields(p, U, valid, pts, frame) -> dict[str, np.ndarray]:
    """cp, cpt and |U_rel| exactly as viz/pv_render.py builds them."""
    w = float(frame.get("omega") or 0.0)
    ox, oy, _ = frame["origin"]
    u_inf = float(frame["u_inf"])
    q = 0.5 * u_inf * u_inf
    U = np.asarray(U, dtype=float)
    dx, dy = pts[:, 0] - ox, pts[:, 1] - oy
    rel = np.stack([U[:, 0] + w * dy, U[:, 1] - w * dx, U[:, 2]], axis=1)
    umag = np.linalg.norm(rel, axis=1)
    p = np.asarray(p, dtype=float)
    uff2 = (w * np.hypot(dx, dy)) ** 2 if w else u_inf * u_inf
    out = {"cp": p / q, "cpt": (p + 0.5 * umag**2 - 0.5 * uff2) / q + 1.0, "U": umag}
    valid = np.asarray(valid, dtype=bool)
    return {k: np.where(valid, v, np.nan) for k, v in out.items()}


def delta_rgb(delta: np.ndarray, solid: np.ndarray, moved: np.ndarray, limit: float) -> np.ndarray:
    table = np.array(diverging_table(), dtype=np.uint8)
    scaled = np.nan_to_num((delta + limit) / (2 * limit) * BANDS, nan=0.0)
    index = np.clip(np.floor(scaled).astype(int), 0, BANDS - 1)
    rgb = table[index]
    rgb[solid] = SOLID
    rgb[moved] = MOVED
    return rgb[::-1].copy()  # grid row 0 is the bottom, image row 0 the top


def draw_colour_bar(rgb: np.ndarray, limit: float, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    h, w = rgb.shape[:2]
    # Same place as the pipeline's own bar: bottom centre.
    x0, x1 = int(w * 0.37), int(w * 0.63)
    y0, y1 = int(h * 0.90), int(h * 0.93)
    table = diverging_table()
    step = (x1 - x0) / BANDS
    for i, colour in enumerate(table):
        draw.rectangle([x0 + i * step, y0, x0 + (i + 1) * step, y1], fill=colour)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0))
    # No text anchors: they need a FreeType font, which is not guaranteed.
    for text, x in ((f"{-limit:g}", x0), ("0", (x0 + x1) // 2), (f"+{limit:g}", x1)):
        draw.text((x - 4 * len(text), y0 - 14), text, fill=(0, 0, 0))
    draw.text(((x0 + x1) // 2 - 4 * len(label), y1 + 4), label, fill=(0, 0, 0))
    return np.asarray(image)


def _write_png(rgb: np.ndarray, out: Path) -> None:
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    Image.fromarray(rgb).save(tmp)
    os.replace(tmp, out)  # a reader never gets half a picture


def _read(path: str):
    import vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def _probe(path: str, pts: np.ndarray) -> dict[str, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(_read(path))
    tri.Update()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    target = vtk.vtkPolyData()
    target.SetPoints(points)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(target)
    probe.SetSourceData(tri.GetOutput())
    probe.SetComputeTolerance(False)
    probe.SetTolerance(PROBE_TOLERANCE)
    probe.Update()
    data = probe.GetOutput().GetPointData()
    return {
        "p": vtk_to_numpy(data.GetArray("pMean")),
        "U": vtk_to_numpy(data.GetArray("UMean")),
        "valid": vtk_to_numpy(data.GetArray("vtkValidPointMask")).astype(bool),
    }


def plane_delta(request: dict[str, Any]) -> dict[str, Any]:
    field, limit = request["field"], float(request["limit"])
    pts, shape = frame_grid(request["slice"], request["resolution"])
    pane = _probe(request["pane"]["sample"], pts)
    ref = _probe(request["ref"]["sample"], pts)
    a = derived_fields(pane["p"], pane["U"], pane["valid"], pts, request["pane"]["frame"])[field]
    b = derived_fields(ref["p"], ref["U"], ref["valid"], pts, request["ref"]["frame"])[field]
    a, b = a.reshape(shape), b.reshape(shape)
    delta = a - b
    solid = ~np.isfinite(a) & ~np.isfinite(b)
    moved = np.isfinite(a) ^ np.isfinite(b)
    rgb = draw_colour_bar(delta_rgb(delta, solid, moved, limit), limit, f"Δ {field}")
    _write_png(rgb, Path(request["out"]))
    both = np.isfinite(delta)
    return {"max_abs": float(np.abs(delta[both]).max()) if both.any() else None,
            "moved_pct": float(100 * moved.mean())}


def surface_delta_field(pane_path: str, ref_path: str, q: float):
    """Ref's pMean interpolated onto the pane's surface, as delta_cp.

    A pane point with no ref point within SURFACE_RADIUS gets NaN: surface
    that is new or moved in the pane, drawn black.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    pane, ref = _read(pane_path), _read(ref_path)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(ref)
    locator.BuildLocator()
    kernel = vtk.vtkLinearKernel()
    kernel.SetRadius(SURFACE_RADIUS)
    kernel.SetKernelFootprintToRadius()
    interp = vtk.vtkPointInterpolator()
    interp.SetInputData(pane)
    interp.SetSourceData(ref)
    interp.SetKernel(kernel)
    interp.SetLocator(locator)
    interp.PassPointArraysOff()  # else the pane's own pMean comes through under the same name
    interp.SetNullPointsStrategyToMaskPoints()
    interp.SetValidPointsMaskArrayName("has_ref")
    interp.Update()
    out = interp.GetOutput().GetPointData()
    p_pane = vtk_to_numpy(pane.GetPointData().GetArray("pMean"))
    p_ref = vtk_to_numpy(out.GetArray("pMean"))
    has = vtk_to_numpy(out.GetArray("has_ref")).astype(bool)
    d = np.where(has, (p_pane - p_ref) / q, np.nan).astype(np.float32)
    surface = vtk.vtkPolyData()
    surface.ShallowCopy(pane)
    array = numpy_to_vtk(d, deep=True)
    array.SetName("delta_cp")
    surface.GetPointData().AddArray(array)
    stats = {"moved_pct": float(100 * (~has).mean()),
             "max_abs": float(np.nanmax(np.abs(d))) if has.any() else None}
    return surface, stats


def surface_delta(request: dict[str, Any]) -> dict[str, Any]:
    import vtk

    limit = float(request["limit"])
    cache = Path(request["cache_vtp"])
    pane_s, ref_s = request["pane"]["sample"], request["ref"]["sample"]
    stats: dict[str, Any] = {}
    fresh = cache.is_file() and cache.stat().st_mtime > max(
        Path(pane_s).stat().st_mtime, Path(ref_s).stat().st_mtime)
    if not fresh:
        u_inf = float(request["pane"]["frame"]["u_inf"])
        surface, stats = surface_delta_field(pane_s, ref_s, 0.5 * u_inf * u_inf)
        cache.parent.mkdir(parents=True, exist_ok=True)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(cache.with_suffix(".tmp.vtp")))
        writer.SetInputData(surface)
        writer.Write()
        os.replace(cache.with_suffix(".tmp.vtp"), cache)

    from paraview.simple import (
        CreateRenderView, GetColorTransferFunction, Render, SaveScreenshot, Show,
        XMLPolyDataReader,
    )

    view = CreateRenderView()
    view.ViewSize = list(request["resolution"])
    view.OrientationAxesVisibility = 0
    try:
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    view.Background = [c / 255 for c in SOLID]
    display = Show(XMLPolyDataReader(FileName=[str(cache)]), view)
    display.ColorArrayName = ["POINTS", "delta_cp"]
    lut = GetColorTransferFunction("delta_cp")
    points = []
    for i, colour in enumerate(diverging_table()):
        rgb = [c / 255 for c in colour]
        points += [-limit + 2 * limit * i / BANDS, *rgb, -limit + 2 * limit * (i + 1) / BANDS, *rgb]
    lut.ColorSpace = "RGB"
    lut.RGBPoints = points
    lut.Discretize = 1
    lut.NumberOfTableValues = BANDS
    lut.AutomaticRescaleRangeMode = "Never"
    lut.NanColor = [c / 255 for c in MOVED]
    display.LookupTable = lut
    cam = request["camera"]
    view.CameraFocalPoint = cam["focal"]
    view.CameraPosition = cam["position"]
    view.CameraViewUp = cam["up"]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = cam["parallel_scale"]
    Render(view)
    out = Path(request["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.png")
    SaveScreenshot(str(raw), view, ImageResolution=list(request["resolution"]))
    from PIL import Image

    rgb = draw_colour_bar(np.asarray(Image.open(raw).convert("RGB")), limit, "Δ cp")
    raw.unlink()
    _write_png(rgb, out)
    return stats


def main(argv: Sequence[str]) -> int:
    started = time.time()
    try:
        request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        handler = {"plane": plane_delta, "surface": surface_delta}[request["kind"]]
        stats = handler(request)
    except Exception as error:  # the caller shows this text in the pane
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps({"ok": True, "out": request["out"],
                      "seconds": round(time.time() - started, 2), **stats}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_delta_helper.py -v`
Expected: 6 passed (the two VTK tests run on this machine; elsewhere they skip).

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/viz/delta.py tests/test_delta_helper.py
git commit -m "feat(viz): field-delta helper for planes and the cp surface

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Delta service

**Files:**
- Create: `pipeline/simdev/ui/delta.py`
- Test: `tests/test_ui_delta.py`

**Interfaces:**
- Consumes: `load_plan`, `reroot` (Task 6); the helper CLI contract (Task 7); `load_spec`.
- Produces:
  - `PLANE_FIELDS = ("cp", "cpt", "U")`, `SURFACE_FIELDS = ("cp",)`
  - `class DeltaError(Exception)` with attribute `status: int` (404 unknown view, 422 field/limit not allowed, 409 not comparable, 500 helper failure)
  - `default_limit(field: str, u_inf: float) -> float`
  - `build_request(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float) -> dict` (view is a plane name such as `x_-0.100` or `surface_<name>`)
  - `delta_png(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float | None, runner: Callable = subprocess.run) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ui_delta.py
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from simdev.ui import delta as service

FRAME = {"mode": "straight", "u_inf": 10.0, "omega": 0.0, "origin": [0, 0, 0]}


def make_run(root: Path, name: str, digest: str = "v1", spec_hash: str = "s") -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "postProcessing/surfaces/400").mkdir(parents=True)
    (run / "postProcessing/surfaces/400/x_+0.000.vtp").write_text("vtp")
    plan = {"views_digest": digest, "resolution": [16, 12], "frame": FRAME,
            "slices": [{"name": "x_+0.000", "axis": "x", "offset": 0.0,
                        "point": [0, 0, 0], "normal": [1, 0, 0], "camera": {"parallel_scale": 0.1},
                        "sample": f"/old/{name}/postProcessing/surfaces/400/x_+0.000.vtp",
                        "images": []}],
            "surfaces": [{"name": "top", "camera": {}, "sample": f"/old/{name}/postProcessing/patchSurfaces/400/vehicle.vtp", "images": []}]}
    (run / "results/render_plan.json").write_text(json.dumps(plan))
    (run / "results/result.json").write_text(json.dumps({"spec_hash": spec_hash}))
    return run


class FakeHelper:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        with self.lock:
            self.calls += 1
        request = json.loads(Path(argv[-1]).read_text())
        assert kwargs["env"]["PATH"] == "/usr/bin:/bin"
        Path(request["out"]).write_bytes(b"\x89PNG")
        class Done:
            returncode = 0
            stdout = json.dumps({"ok": True, "out": request["out"]}) + "\n"
            stderr = ""
        return Done()


def test_request_uses_the_reference_frame_and_rerooted_samples(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    request = service.build_request(pane, ref, "x_+0.000", "cp", 0.2)
    assert request["kind"] == "plane"
    assert request["pane"]["sample"] == str(pane / "postProcessing/surfaces/400/x_+0.000.vtp")
    assert request["ref"]["sample"] == str(ref / "postProcessing/surfaces/400/x_+0.000.vtp")
    assert request["resolution"] == [16, 12]


def test_refusals(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a", digest="v2")
    with pytest.raises(service.DeltaError) as e:
        service.build_request(pane, ref, "x_+0.000", "cp", 0.2)
    assert e.value.status == 409
    ref = make_run(tmp_path, "c")
    for view, field, status in (("x_+9.000", "cp", 404), ("x_+0.000", "lambda2", 422),
                                ("surface_top", "cpt", 422)):
        with pytest.raises(service.DeltaError) as e:
            service.build_request(pane, ref, view, field, 0.2)
        assert e.value.status == status


def test_default_limits() -> None:
    assert service.default_limit("cp", 15.0) == 0.2
    assert service.default_limit("U", 15.0) == 1.5


def test_result_is_cached_and_concurrent_requests_run_the_helper_once(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    paths = []
    threads = [threading.Thread(target=lambda: paths.append(
        service.delta_png(pane, ref, "x_+0.000", "cp", None, runner=helper))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert helper.calls == 1
    assert len(set(paths)) == 1 and paths[0].read_bytes() == b"\x89PNG"
    assert paths[0].is_relative_to(pane / "ui" / "delta" / "a")


def test_a_changed_reference_invalidates_the_cache(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    (ref / "results/result.json").write_text(json.dumps({"spec_hash": "changed"}))
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    assert helper.calls == 2


def test_helper_failure_is_reported(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")

    def failing(argv, **kwargs):
        class Done:
            returncode = 1
            stdout = json.dumps({"ok": False, "error": "KeyError: 'pMean'"}) + "\n"
            stderr = "trace"
        return Done()

    with pytest.raises(service.DeltaError) as e:
        service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=failing)
    assert e.value.status == 500 and "pMean" in str(e.value)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_delta.py -v`
Expected: FAIL with `ImportError: cannot import name 'delta' from 'simdev.ui'`

- [ ] **Step 3: Implement**

```python
# pipeline/simdev/ui/delta.py
"""Delta pictures on request: build the helper's request, run it once, cache.

One delta at a time. A delta is seconds of one core; it must never compete
with a running solve for more than that.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from simdev.stages.common import load_spec
from simdev.ui.imageindex import load_plan, reroot

PLANE_FIELDS = ("cp", "cpt", "U")
SURFACE_FIELDS = ("cp",)
HELPER = Path(__file__).resolve().parent.parent / "viz" / "delta.py"
DEFAULT_INTERPRETER = "/usr/bin/python3"
TIMEOUT = 600
_LOCK = threading.Lock()


class DeltaError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def default_limit(field: str, u_inf: float) -> float:
    return round(0.1 * u_inf, 6) if field == "U" else 0.2


def _plan(run_dir: Path) -> dict[str, Any]:
    plan = load_plan(run_dir)
    if plan is None:
        raise DeltaError(f"{run_dir.name} has no pictures yet (no render_plan.json)", 404)
    return plan


def _spec_hash(run_dir: Path) -> str:
    try:
        return json.loads((run_dir / "results" / "result.json").read_text("utf-8")).get(
            "spec_hash", "")
    except (OSError, ValueError):
        return ""


def build_request(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float) -> dict:
    pane_plan, ref_plan = _plan(pane_dir), _plan(ref_dir)
    if pane_plan.get("views_digest") != ref_plan.get("views_digest"):
        raise DeltaError("the two runs were pictured with different post_views.yaml; "
                         "their planes are not the same planes", 409)
    if not limit or limit <= 0:
        raise DeltaError("the colour limit must be positive", 422)
    if view.startswith("surface_"):
        if field not in SURFACE_FIELDS:
            raise DeltaError(f"no surface delta for {field}", 422)
        name = view[len("surface_"):]
        entries = {s["name"]: s for s in pane_plan.get("surfaces", [])}
        ref_entries = {s["name"]: s for s in ref_plan.get("surfaces", [])}
        if name not in entries or name not in ref_entries:
            raise DeltaError(f"no surface view {name}", 404)
        return {
            "kind": "surface", "field": field, "limit": limit,
            "resolution": pane_plan["resolution"], "camera": entries[name]["camera"],
            "pane": {"sample": str(reroot(entries[name]["sample"], pane_dir)),
                     "frame": pane_plan["frame"]},
            "ref": {"sample": str(reroot(ref_entries[name]["sample"], ref_dir)),
                    "frame": ref_plan["frame"]},
        }
    if field not in PLANE_FIELDS:
        raise DeltaError(f"no plane delta for {field}", 422)
    slices = {s["name"]: s for s in ref_plan.get("slices", [])}
    pane_slices = {s["name"]: s for s in pane_plan.get("slices", [])}
    if view not in slices or view not in pane_slices:
        raise DeltaError(f"no plane {view}", 404)
    entry = slices[view]
    return {
        "kind": "plane", "field": field, "limit": limit,
        "resolution": ref_plan["resolution"],
        "slice": {k: entry[k] for k in ("point", "normal", "camera")},
        "pane": {"sample": str(reroot(pane_slices[view]["sample"], pane_dir)),
                 "frame": pane_plan["frame"]},
        "ref": {"sample": str(reroot(entry["sample"], ref_dir)), "frame": ref_plan["frame"]},
    }


def _interpreter(run_dir: Path) -> str:
    try:
        return load_spec(run_dir).post.paraview_python or DEFAULT_INTERPRETER
    except (FileNotFoundError, KeyError, ValueError, AttributeError):
        return DEFAULT_INTERPRETER


def _fresh_cache(pane_dir: Path, ref_dir: Path) -> Path:
    folder = pane_dir / "ui" / "delta" / ref_dir.name
    key = hashlib.sha1(json.dumps([
        _spec_hash(pane_dir), _spec_hash(ref_dir),
        _plan(pane_dir).get("views_digest"), _plan(ref_dir).get("views_digest"),
    ]).encode()).hexdigest()
    stamp = folder / "key.txt"
    if not stamp.is_file() or stamp.read_text() != key:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        stamp.write_text(key)
    return folder


def delta_png(pane_dir: Path, ref_dir: Path, view: str, field: str, limit: float | None,
              runner: Callable[..., Any] = subprocess.run) -> Path:
    pane_dir, ref_dir = Path(pane_dir), Path(ref_dir)
    if limit is None:
        limit = default_limit(field, float(_plan(pane_dir)["frame"]["u_inf"]))
    request = build_request(pane_dir, ref_dir, view, field, limit)
    with _LOCK:
        folder = _fresh_cache(pane_dir, ref_dir)
        out = folder / f"{view}_{field}_{limit:g}.png"
        if out.is_file():
            return out
        request["out"] = str(out)
        request["cache_vtp"] = str(folder / "surface.vtp")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(request, handle)
        try:
            done = runner(
                [_interpreter(pane_dir), str(HELPER), handle.name],
                capture_output=True, text=True, timeout=TIMEOUT,
                env={"HOME": os.environ.get("HOME", ""), "PATH": "/usr/bin:/bin"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DeltaError(f"the delta helper could not run: {error}", 500) from None
        finally:
            os.unlink(handle.name)
        try:
            summary = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            summary = {"ok": False, "error": (done.stderr or "no output")[-1500:]}
        if done.returncode != 0 or not summary.get("ok") or not out.is_file():
            raise DeltaError(
                f"{summary.get('error') or 'the delta helper failed'} - "
                "run 'simdev doctor' to check the ParaView interpreter", 500)
        return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ui_delta.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/ui/delta.py tests/test_ui_delta.py
git commit -m "feat(ui): delta service with a per-pair cache and one helper at a time

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Compare page and its APIs

**Files:**
- Create: `pipeline/simdev/ui/routes_compare.py`, `pipeline/simdev/ui/templates/compare.html`
- Modify: `pipeline/simdev/ui/app.py` (register router)
- Test: `tests/test_ui_compare_routes.py`

**Interfaces:**
- Consumes: `valid_run_name` (Task 5), `results.load_row/columns_for/group_names/delta` (Task 4), `load_summary` (Task 3), `build_index/cp_station` (Task 6), `delta_png/DeltaError` (Task 8).
- Produces routes:
  - `GET /compare?runs=a,b&runs=c&ref=a` → page; runs keep their order, at most 4, unknown or invalid names listed as a warning and dropped; `ref` defaults to the first run.
  - `GET /api/runs/{name}/index` → `build_index`
  - `GET /api/runs/{name}/summary` → `{"window", "values", "noise", "patches", "groups_map"}`
  - `GET /api/runs/{name}/cplines/{station}` → `cp_station`
  - `GET /api/delta/{name}.png?ref=&view=&field=&limit=` → PNG, or JSON `{"error": ...}` with the `DeltaError.status`
- The page embeds `<script id="compare-data" type="application/json">{"runs": [...], "ref": "...", "limits": {"cp": 0.2, "cpt": 0.2}, "all_runs": [...]}</script>` for `compare.js` (Task 10); `all_runs` = every run with a `result.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ui_compare_routes.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from tests.ui_support import make_client  # noqa: E402


def make_run(root: Path, name: str, cl: float = -1.0) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "caseSpec.json").write_text("{}")
    (run / "results" / "result.json").write_text(json.dumps(
        {"cd_mean": 0.8, "cl_mean": cl, "window_start": 1, "window_end": 2, "spec_hash": "s"}))
    return run


def test_compare_page_keeps_order_and_reference(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b", cl=-1.2)
        page = client.get("/compare?runs=b,a&ref=a").text
        data = json.loads(page.split('<script id="compare-data" type="application/json">')[1]
                          .split("</script>")[0])
        assert data["runs"] == ["b", "a"] and data["ref"] == "a"
        assert sorted(data["all_runs"]) == ["a", "b"]
        assert "Δ against a" in page


def test_ticked_checkboxes_arrive_as_repeated_parameters(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        assert client.get("/compare?runs=a&runs=b").status_code == 200


def test_unknown_and_hostile_names_are_dropped_with_a_warning(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        page = client.get("/compare?runs=a,nope,..%2Fetc").text
        assert "not found" in page


def test_index_summary_and_cplines_apis(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        assert client.get("/api/runs/a/index").json()["planes"] == {}
        assert client.get("/api/runs/a/summary").json()["window"] == [1, 2]
        patch = tmp_path / "runs/a/postProcessing/forceCoeffs_Body/0/coefficient.dat"
        patch.parent.mkdir(parents=True)
        patch.write_text("# Time Cd Cs Cl CmRoll CmPitch CmYaw\n5 0.1 0 -0.1 0 0 0\n")
        body = client.get("/api/runs/a/summary").text  # window 1-2 holds no rows: NaN
        assert "NaN" not in body and json.loads(body)["patches"]["Body"]["Cd"] is None
        assert client.get("/api/runs/a/cplines/y_+0.000").status_code == 404


def test_delta_endpoint_reports_errors_as_json(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        response = client.get("/api/delta/b.png?ref=a&view=x_+0.000&field=cp")
        assert response.status_code == 404
        assert "render_plan" in response.json()["error"]


def test_delta_endpoint_serves_the_png(tmp_path: Path, monkeypatch) -> None:
    from simdev.ui import routes_compare

    picture = tmp_path / "d.png"
    picture.write_bytes(b"\x89PNG")
    monkeypatch.setattr(routes_compare, "delta_png", lambda *a, **k: picture)
    with make_client(tmp_path) as (client, app):
        make_run(tmp_path / "runs", "a")
        make_run(tmp_path / "runs", "b")
        response = client.get("/api/delta/b.png?ref=a&view=x_+0.000&field=cp&limit=0.1")
        assert response.status_code == 200 and response.content == b"\x89PNG"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_compare_routes.py -v`
Expected: FAIL (404 on `/compare`)

- [ ] **Step 3: Implement the routes**

```python
# pipeline/simdev/ui/routes_compare.py
"""The compare page and the data its script asks for."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from simdev.ui import results, runview
from simdev.ui.context import ctx, render
from simdev.ui.delta import DeltaError, delta_png
from simdev.ui.imageindex import build_index, cp_station
from simdev.ui.routes_results import valid_run_name
from simdev.ui.summary import finite, load_summary

router = APIRouter()
MAX_PANES = 4


def _run_dir(request: Request, name: str) -> Path:
    if not valid_run_name(name):
        raise HTTPException(status_code=404)
    path = ctx(request).config.runs_root / name
    if not path.is_dir():
        raise HTTPException(status_code=404)
    return path


@router.get("/compare")
def compare_page(request: Request, runs: list[str] = Query([]), ref: str = ""):
    root = ctx(request).config.runs_root
    wanted = [n.strip() for item in runs for n in item.split(",") if n.strip()]
    names, missing = [], []
    for name in wanted:
        if valid_run_name(name) and (root / name).is_dir() and name not in names:
            names.append(name)
        else:
            missing.append(name)
    names = names[:MAX_PANES]
    rows = [results.load_row(root / n) for n in names]
    columns = results.columns_for(results.group_names(rows))
    ref = ref if ref in names else (names[0] if names else "")
    by_name = {r.name: r for r in rows}
    all_runs = [p.name for p in runview.list_runs(root) if (p / "results" / "result.json").is_file()]
    payload = {"runs": names, "ref": ref, "limits": {"cp": 0.2, "cpt": 0.2}, "all_runs": all_runs}
    return render(
        request, "compare.html", rows=rows, columns=columns, ref=by_name.get(ref),
        missing=missing, delta=results.delta, data_json=json.dumps(payload),
    )


@router.get("/api/runs/{name}/index")
def run_index(request: Request, name: str):
    return build_index(_run_dir(request, name))


@router.get("/api/runs/{name}/summary")
def run_summary(request: Request, name: str):
    summary = load_summary(_run_dir(request, name))
    if summary is None:
        raise HTTPException(status_code=404, detail="no results yet")
    row = results.load_row(_run_dir(request, name))
    # NaN is not JSON; the browser's parser rejects the whole response.
    patches = {p: {k: finite(v) for k, v in d.items()} for p, d in summary.patches.items()}
    return {"window": list(summary.window), "values": row.values, "noise": row.noise,
            "patches": patches, "groups_map": summary.groups_map}


@router.get("/api/runs/{name}/cplines/{station}")
def run_cplines(request: Request, name: str, station: str):
    try:
        return cp_station(_run_dir(request, name), station)
    except KeyError:
        raise HTTPException(status_code=404) from None


@router.get("/api/delta/{name}.png")
def delta_picture(request: Request, name: str, ref: str, view: str, field: str,
                  limit: float | None = None):
    pane_dir, ref_dir = _run_dir(request, name), _run_dir(request, ref)
    try:
        return FileResponse(delta_png(pane_dir, ref_dir, view, field, limit),
                            media_type="image/png")
    except DeltaError as error:
        return JSONResponse({"error": str(error)}, status_code=error.status)
```

Note: FastAPI runs this sync endpoint in its thread pool, so a delta that takes 20 s does not block the other pages. Register `routes_compare.router` in `app.py`'s router tuple.

- [ ] **Step 4: Implement the template**

```html
{# pipeline/simdev/ui/templates/compare.html #}
{% extends "base.html" %}
{% block title %}Compare · SimDev{% endblock %}
{% block head %}
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.min.css">
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js"></script>
<script src="/static/compare.js" defer></script>
{% endblock %}
{% block main %}
<h1>Compare</h1>
{% if missing %}<div class="warn">not found or not allowed, left out: {{ missing|join(', ') }}</div>{% endif %}
{% if not rows %}
<p>No runs chosen. Tick runs on the <a href="/results">Results</a> page.</p>
{% else %}
<script id="compare-data" type="application/json">{{ data_json|safe }}</script>
<div id="strip" class="row"></div>
<div id="guards"></div>

<h2>Numbers</h2>
<div class="scroll"><table class="results">
  <thead><tr><th>run</th><th>note</th>{% for c in columns %}<th class="num">{{ c.label }}</th>{% endfor %}<th>verdict</th></tr></thead>
  {% for row in rows %}
  <tbody class="result">
  <tr><td><a href="/runs/{{ row.name }}">{{ row.name }}</a>{% if ref and row.name == ref.name %} <span class="badge">REF</span>{% endif %}<br><span class="muted">{{ row.state }}</span></td>
      <td>{{ row.note.note }}</td>
      {% for c in columns %}<td class="num">{{ row.values.get(c.key)|num(c.fmt) }}</td>{% endfor %}
      <td>{{ row.verdict or '' }}</td></tr>
  {% if ref and row.name != ref.name and row.has_result and ref.has_result %}
  {% set cells = delta(row, ref, columns) %}
  <tr class="delta"><td colspan="2" class="muted">Δ against {{ ref.name }}</td>
    {% for c in columns %}{% set d = cells[c.key] %}<td class="num {{ d.tone }}" {% if d.noise is not none %}title="± {{ d.noise|num('{:.4f}') }}"{% endif %}>{{ d.value|num(c.fmt) }}</td>{% endfor %}<td></td></tr>
  {% endif %}
  </tbody>
  {% endfor %}
</table></div>

<h2>Pictures</h2>
<div id="viewer"></div>

<h2>Forces</h2>
<div class="row">
  <label class="row">Show <select id="force-part"><option value="">whole car</option></select></label>
  <label class="row"><input type="checkbox" id="force-relative"> relative to window mean (%)</label>
</div>
<h3>Cl</h3><div id="plot-Cl" class="plot"></div>
<h3>Cd</h3><div id="plot-Cd" class="plot"></div>

<h2>Δ by component</h2>
<div id="bars"></div>

<h2>cp lines</h2>
<div class="row">
  <label class="row">Station <select id="cp-station"></select></label>
  <label class="row"><input type="checkbox" class="cp-patch" value="Body" checked> Body</label>
  <label class="row"><input type="checkbox" class="cp-patch" value="Wing" checked> Wing</label>
</div>
<canvas id="cp-plot" class="plot" width="1100" height="380"></canvas>
<canvas id="cp-outline" class="plot" width="1100" height="160"></canvas>
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ui_compare_routes.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/ui/routes_compare.py pipeline/simdev/ui/templates/compare.html pipeline/simdev/ui/app.py tests/test_ui_compare_routes.py
git commit -m "feat(ui): compare page with numbers against REF and the viewer's data APIs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Image viewer (compare.js, part 1)

**Files:**
- Create: `pipeline/simdev/ui/static/compare.js`
- Modify: `pipeline/simdev/ui/static/style.css`
- Test: manual check list (Step 4); `tests/test_ui_compare_routes.py` gets one smoke test that the script is served.

**Interfaces:**
- Consumes: `#compare-data` JSON, `GET /api/runs/{name}/index`, `GET /runs/{name}/files/{rel}`, `GET /api/delta/{name}.png?...` (Task 9).
- Produces: `window.SimdevCompare = { state, indexes, colourOf(run) }` used by Task 11; document events `simdev:ready` (indexes loaded), `simdev:ref` (REF changed or page re-rendered), `simdev:moved` (position or Δ changed); viewer DOM inside `#viewer`.

- [ ] **Step 1: Write the smoke test**

Append to `tests/test_ui_compare_routes.py`:

```python
def test_the_compare_script_is_served(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        script = client.get("/static/compare.js")
        assert script.status_code == 200 and "SimdevCompare" in script.text
```

Run: `.venv/bin/python -m pytest tests/test_ui_compare_routes.py::test_the_compare_script_is_served -v` → FAIL (404).

- [ ] **Step 2: Implement the viewer**

```javascript
// pipeline/simdev/ui/static/compare.js
// The compare page: synced pictures, overlays, on-request deltas (part 1),
// and the charts (part 2, below).
//
// Every picture of one view shares its frame pixel for pixel
// (post_views.yaml), so zoom and pan are kept in fractions of the picture
// and shared between synced panes.
(function () {
  const dataEl = document.getElementById("compare-data");
  if (!dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const PALETTE = ["#d0542c", "#2c6fd0", "#1f8a4c", "#8e44ad"];
  const DELTA_FIELDS = { plane: ["cp", "cpt", "U"], surface: ["cp"] };
  const SURFACE_VIEWS = ["front", "rear", "left", "right", "top", "bottom", "iso"];

  const state = {
    runs: data.runs.slice(),
    ref: data.ref,
    limits: Object.assign({ U: null }, data.limits),
    global: { kind: "x", field: "cp", pos: 0 },
    zoom: { s: 1, x: 0, y: 0 },
    layout: "side",
    overlay: { a: 0, b: 1, mode: "blink", showB: false, pos: 0.5, alpha: 0.5, timer: null, ms: 500 },
    panes: data.runs.slice(0, 3).map((run) => ({
      run, sync: true, delta: false, kind: "x", field: "cp", pos: 0, zoom: { s: 1, x: 0, y: 0 },
    })),
  };
  const indexes = {};
  // What render() built and update() refreshes in place: rebuilding the DOM
  // on every slider step would end the drag and drop keyboard focus.
  const live = { views: [], positions: [] };
  const colourOf = (run) => PALETTE[state.runs.indexOf(run) % PALETTE.length];
  window.SimdevCompare = { state, indexes, colourOf };

  // --- what a pane shows -----------------------------------------------------

  const settings = (pane) => (pane.sync ? state.global : pane);
  const zoomOf = (pane) => (pane.sync ? state.zoom : pane.zoom);

  function choices(kind, field) {
    // Positions come from REF's index: with the same views digest every run
    // has the same planes, and a run that lacks one shows a placeholder.
    const index = indexes[state.ref] || {};
    if (kind === "surface") return Object.keys((index.surfaces || {})[field] || {})
      .sort((a, b) => SURFACE_VIEWS.indexOf(a) - SURFACE_VIEWS.indexOf(b))
      .map((name) => ({ name, label: name }));
    return (((index.planes || {})[kind] || {})[field] || [])
      .map((p) => ({ name: p.name, label: `${kind} = ${p.offset >= 0 ? "+" : ""}${p.offset.toFixed(3)}` }));
  }

  function fieldsFor(kind) {
    const index = indexes[state.ref] || {};
    if (kind === "surface") return Object.keys(index.surfaces || {});
    return Object.keys((index.planes || {})[kind] || {});
  }

  function pictureOf(run, kind, field, name) {
    const index = indexes[run] || {};
    if (kind === "surface") return ((index.surfaces || {})[field] || {})[name] || null;
    const list = ((index.planes || {})[kind] || {})[field] || [];
    const hit = list.find((p) => p.name === name);
    return hit ? hit.rel : null;
  }

  function deltaAllowed(pane) {
    const s = settings(pane);
    const group = s.kind === "surface" ? "surface" : "plane";
    return pane.run !== state.ref && DELTA_FIELDS[group].includes(s.field);
  }

  function urlOf(pane) {
    const s = settings(pane);
    const list = choices(s.kind, s.field);
    const item = list[Math.min(s.pos, list.length - 1)];
    if (!item) return { url: null, label: "no pictures" };
    if (pane.delta && deltaAllowed(pane)) {
      const view = s.kind === "surface" ? `surface_${item.name}` : item.name;
      const limit = state.limits[s.field];
      const q = new URLSearchParams({ ref: state.ref, view, field: s.field });
      if (limit) q.set("limit", limit);
      return { url: `/api/delta/${encodeURIComponent(pane.run)}.png?${q}`, label: `Δ ${item.label}`, delta: true };
    }
    const rel = pictureOf(pane.run, s.kind, s.field, item.name);
    return rel ? { url: `/runs/${encodeURIComponent(pane.run)}/files/${rel}`, label: item.label }
               : { url: null, label: `${item.label}: no picture in this run` };
  }

  // --- loading pictures ------------------------------------------------------

  const deltaCache = new Map();  // url -> object URL, so blink is instant

  async function load(img, note, target) {
    note.textContent = "";
    if (!target.url) { img.removeAttribute("src"); note.textContent = target.label; return; }
    if (!target.delta) { img.src = target.url; return; }
    if (deltaCache.has(target.url)) { img.src = deltaCache.get(target.url); return; }
    note.textContent = "computing delta…";
    img.removeAttribute("src");
    try {
      const response = await fetch(target.url);
      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: response.statusText }));
        note.textContent = body.error || "delta failed";
        return;
      }
      const objectUrl = URL.createObjectURL(await response.blob());
      deltaCache.set(target.url, objectUrl);
      img.src = objectUrl;
      note.textContent = "";
    } catch (error) {
      note.textContent = String(error);
    }
  }

  function preload() {
    const s = state.global;
    const list = choices(s.kind, s.field);
    for (const pane of state.panes.filter((p) => p.sync && !p.delta)) {
      for (const step of [-2, -1, 1, 2]) {
        const item = list[s.pos + step];
        const rel = item && pictureOf(pane.run, s.kind, s.field, item.name);
        if (rel) new Image().src = `/runs/${encodeURIComponent(pane.run)}/files/${rel}`;
      }
    }
  }

  // --- zoom and pan ----------------------------------------------------------

  function applyZoom(img, zoom) {
    img.style.transform = `translate(${zoom.x * 100}%, ${zoom.y * 100}%) scale(${zoom.s})`;
  }

  function attachZoom(viewport, getZoom) {
    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      const zoom = getZoom();
      const box = viewport.getBoundingClientRect();
      const fx = (event.clientX - box.left) / box.width;
      const fy = (event.clientY - box.top) / box.height;
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
      const s = Math.min(20, Math.max(1, zoom.s * factor));
      // keep the point under the cursor where it is
      zoom.x = fx - (fx - zoom.x) * (s / zoom.s);
      zoom.y = fy - (fy - zoom.y) * (s / zoom.s);
      zoom.s = s;
      if (s === 1) { zoom.x = 0; zoom.y = 0; }
      refreshZoom();
    }, { passive: false });
    let drag = null;
    viewport.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".handle")) return;
      drag = { x: event.clientX, y: event.clientY };
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const zoom = getZoom();
      const box = viewport.getBoundingClientRect();
      zoom.x += (event.clientX - drag.x) / box.width;
      zoom.y += (event.clientY - drag.y) / box.height;
      drag = { x: event.clientX, y: event.clientY };
      refreshZoom();
    });
    viewport.addEventListener("pointerup", () => { drag = null; });
    viewport.addEventListener("dblclick", () => {
      Object.assign(getZoom(), { s: 1, x: 0, y: 0 });
      refreshZoom();
    });
  }

  function refreshZoom() {
    document.querySelectorAll("[data-pane]").forEach((img) => {
      const pane = state.panes[Number(img.dataset.pane)];
      if (pane) applyZoom(img, img.dataset.overlay ? state.zoom : zoomOf(pane));
    });
  }

  // --- controls --------------------------------------------------------------

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "on") for (const [e, f] of Object.entries(v)) node.addEventListener(e, f);
      else if (k === "text") node.textContent = v;
      else if (v === true) node.setAttribute(k, "");
      else if (v !== false && v != null) node.setAttribute(k, v);
    }
    for (const child of [].concat(children)) if (child) node.append(child);
    return node;
  }

  function select(options, value, onChange) {
    return el("select", { on: { change: (e) => onChange(e.target.value) } },
      options.map(([v, label]) => el("option", { value: v, selected: String(v) === String(value), text: label })));
  }

  function positionControls(target, onChange) {
    const kinds = [["x", "x-planes"], ["y", "y-planes"], ["z", "z-planes"], ["surface", "surface"]];
    const fields = fieldsFor(target.kind);
    if (!fields.includes(target.field) && fields.length) target.field = fields[0];
    const list = choices(target.kind, target.field);
    target.pos = Math.max(0, Math.min(target.pos, list.length - 1));
    const slider = el("input", { type: "range", min: 0, max: Math.max(0, list.length - 1), value: target.pos,
      on: { input: (e) => { target.pos = Number(e.target.value); onChange(false); } } });
    const label = el("span", { class: "muted", text: (list[target.pos] || {}).label || "–" });
    live.positions.push({ target, slider, label });
    return el("span", { class: "row" }, [
      select(fields.map((f) => [f, f]), target.field, (v) => { target.field = v; onChange(true); }),
      select(kinds, target.kind, (v) => { target.kind = v; target.pos = 0; onChange(true); }),
      slider, label,
    ]);
  }

  function toolbar() {
    const g = state.global;
    const limits = ["cp", "cpt", "U"].map((f) => el("label", { class: "row" }, [
      `Δ ${f} ±`,
      el("input", { type: "number", step: "any", min: 0, size: 5, value: state.limits[f] ?? "",
        placeholder: f === "U" ? "10% u∞" : "", on: { change: (e) => {
          state.limits[f] = e.target.value ? Number(e.target.value) : null; render(); } } }),
    ]));
    return el("div", { class: "toolbar" }, [
      el("div", { class: "row" }, [positionControls(g, (structural) => (structural ? render() : update())),
        el("span", { class: "muted", text: "← → step, Shift ×5, B blink, double-click resets zoom" })]),
      el("div", { class: "row" }, [
        el("label", { class: "row" }, [el("input", { type: "radio", name: "layout", checked: state.layout === "side",
          on: { change: () => { state.layout = "side"; render(); } } }), "side by side"]),
        el("label", { class: "row" }, [el("input", { type: "radio", name: "layout", checked: state.layout === "overlay",
          disabled: state.panes.length < 2, on: { change: () => { state.layout = "overlay"; render(); } } }), "overlay"]),
        el("button", { type: "button", disabled: state.panes.length >= 4,
          text: "+ add pane", title: "another pane of a run on this page", on: { click: addPane } }),
        // A run not yet on the page: reload with it, so the numbers table includes it too.
        select([["", "+ add run…"], ...data.all_runs.filter((r) => !state.runs.includes(r)).map((r) => [r, r])], "",
          (v) => { if (v) location.href = `/compare?${new URLSearchParams({ runs: [...state.runs, v].join(","), ref: state.ref })}`; }),
        ...limits,
      ]),
    ]);
  }

  function addPane() {
    const used = state.panes.map((p) => p.run);
    const run = state.runs.find((r) => !used.includes(r)) || state.runs[0];
    state.panes.push({ run, sync: true, delta: false, kind: state.global.kind, field: state.global.field,
      pos: state.global.pos, zoom: { s: 1, x: 0, y: 0 } });
    render();
  }

  function paneHeader(pane, i) {
    const isRef = pane.run === state.ref;
    return el("div", { class: "pane-head", style: `border-color:${colourOf(pane.run)}` }, [
      el("label", { class: "row" }, [el("input", { type: "checkbox", checked: pane.sync,
        on: { change: (e) => { pane.sync = e.target.checked;
          if (!pane.sync) Object.assign(pane, { kind: state.global.kind, field: state.global.field, pos: state.global.pos });
          render(); } } }), "sync"]),
      select(state.runs.map((r) => [r, r]), pane.run, (v) => { pane.run = v; render(); }),
      isRef ? el("span", { class: "badge", text: "REF" })
            : el("button", { type: "button", text: "make REF", on: { click: () => { state.ref = pane.run; render(); } } }),
      el("label", { class: "row" }, [el("input", { type: "checkbox", checked: pane.delta && deltaAllowed(pane),
        disabled: !deltaAllowed(pane), on: { change: (e) => { pane.delta = e.target.checked; update(); } } }), "Δ vs REF"]),
      state.panes.length > 1 ? el("button", { type: "button", text: "×", title: "remove pane",
        on: { click: () => { state.panes.splice(i, 1); render(); } } }) : null,
      pane.sync ? null : positionControls(pane, (structural) => (structural ? render() : update())),
    ]);
  }

  // --- layouts ---------------------------------------------------------------

  function sideBySide(root) {
    const grid = el("div", { class: `panes n${state.panes.length}` });
    state.panes.forEach((pane, i) => {
      const img = el("img", { "data-pane": i, draggable: "false", alt: "" });
      const note = el("div", { class: "pane-note" });
      const viewport = el("div", { class: "viewport" }, [img, note]);
      attachZoom(viewport, () => zoomOf(pane));
      grid.append(el("div", { class: "pane" }, [paneHeader(pane, i), viewport]));
      applyZoom(img, zoomOf(pane));
      live.views.push({ img, note, pane });
    });
    root.append(grid);
  }

  function overlay(root) {
    const o = state.overlay;
    o.a = Math.min(o.a, state.panes.length - 1);
    o.b = Math.min(o.b, state.panes.length - 1);
    const options = state.panes.map((p, i) => [i, `${i + 1}: ${p.run}${p.delta ? " (Δ)" : ""}`]);
    const imgA = el("img", { "data-pane": o.a, "data-overlay": "1", draggable: "false", alt: "" });
    const imgB = el("img", { "data-pane": o.b, "data-overlay": "1", draggable: "false", alt: "", class: "top" });
    const noteA = el("div", { class: "pane-note" });
    const handle = el("div", { class: "handle" });
    const viewport = el("div", { class: "viewport overlay" }, [imgA, imgB, handle, noteA]);
    const label = el("div", { class: "overlay-label" });

    function show() {
      const top = o.mode === "blink" ? (o.showB ? state.panes[o.b] : state.panes[o.a]) : state.panes[o.b];
      label.textContent = o.mode === "blink" ? `showing ${top.run}` : `${state.panes[o.a].run} | ${state.panes[o.b].run}`;
      label.style.color = colourOf(top.run);
      imgB.style.visibility = o.mode === "blink" && !o.showB ? "hidden" : "visible";
      imgB.style.opacity = o.mode === "fade" ? o.alpha : 1;
      imgB.style.clipPath = o.mode === "swipe" ? `inset(0 0 0 ${o.pos * 100}%)` : "none";
      handle.style.display = o.mode === "swipe" ? "block" : "none";
      handle.style.left = `${o.pos * 100}%`;
    }
    state.overlay.flip = () => { o.showB = !o.showB; show(); };

    let dragging = false;
    handle.addEventListener("pointerdown", (e) => { dragging = true; handle.setPointerCapture(e.pointerId); });
    handle.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const box = viewport.getBoundingClientRect();
      o.pos = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
      show();
    });
    handle.addEventListener("pointerup", () => { dragging = false; });
    attachZoom(viewport, () => state.zoom);

    const modes = ["blink", "swipe", "fade"].map((m) => el("label", { class: "row" }, [
      el("input", { type: "radio", name: "omode", checked: o.mode === m,
        on: { change: () => { o.mode = m; show(); } } }), m]));
    const auto = el("label", { class: "row" }, [el("input", { type: "checkbox", checked: !!o.timer,
      on: { change: (e) => {
        clearInterval(o.timer); o.timer = null;
        if (e.target.checked) o.timer = setInterval(state.overlay.flip, o.ms);
      } } }), "auto",
      el("input", { type: "number", min: 100, step: 100, value: o.ms, size: 4,
        on: { change: (e) => { o.ms = Number(e.target.value) || 500;
          if (o.timer) { clearInterval(o.timer); o.timer = setInterval(state.overlay.flip, o.ms); } } } }), "ms"]);

    root.append(el("div", { class: "row" }, [
      "A", select(options, o.a, (v) => { o.a = Number(v); render(); }),
      "B", select(options, o.b, (v) => { o.b = Number(v); render(); }),
      ...modes,
      el("button", { type: "button", text: "blink (B)", on: { click: state.overlay.flip } }),
      auto,
      el("input", { type: "range", min: 0, max: 1, step: 0.05, value: o.alpha, title: "fade",
        on: { input: (e) => { o.alpha = Number(e.target.value); show(); } } }),
    ]));
    root.append(el("div", { class: "pane overlay-pane" }, [label, viewport]));
    applyZoom(imgA, state.zoom);
    applyZoom(imgB, state.zoom);
    live.views.push({ img: imgA, note: noteA, pane: state.panes[o.a] },
                    { img: imgB, note: noteA, pane: state.panes[o.b] });
    show();
  }

  // --- render ----------------------------------------------------------------

  const root = document.getElementById("viewer");

  function guards() {
    const box = document.getElementById("guards");
    box.innerHTML = "";
    const digests = new Set(state.runs.map((r) => (indexes[r] || {}).views_digest));
    const states = new Set(state.runs.map((r) => (indexes[r] || {}).state).filter(Boolean));
    if (digests.size > 1) box.append(el("div", { class: "warn",
      text: "these runs were pictured with different post_views.yaml - their pictures are not comparable and deltas are refused" }));
    if (states.size > 1) box.append(el("div", { class: "warn",
      text: `different driving states (${[...states].join(", ")}): different u∞, compare coefficients with care` }));
  }

  function strip() {
    const box = document.getElementById("strip");
    box.innerHTML = "";
    for (const run of state.runs) {
      const index = indexes[run] || {};
      box.append(el("span", { class: "chip", style: `border-color:${colourOf(run)}` }, [
        el("strong", { text: run }), run === state.ref ? el("span", { class: "badge", text: "REF" }) : null,
        el("span", { class: "muted", text: ` ${index.state || ""} · views ${index.views_digest || "–"}` }),
      ]));
    }
  }

  function render() {
    clearInterval(state.overlay.timer);
    state.overlay.timer = null;
    root.innerHTML = "";
    live.views = [];
    live.positions = [];
    root.append(toolbar());
    if (state.layout === "overlay" && state.panes.length >= 2) overlay(root); else sideBySide(root);
    strip();
    update();
    document.dispatchEvent(new CustomEvent("simdev:ref", { detail: state.ref }));
  }

  // A new position or a Δ toggle: swap pictures and labels, keep the DOM.
  function update() {
    for (const p of live.positions) {
      p.slider.value = p.target.pos;
      p.label.textContent = (choices(p.target.kind, p.target.field)[p.target.pos] || {}).label || "–";
    }
    for (const v of live.views) load(v.img, v.note, urlOf(v.pane));
    preload();
    document.dispatchEvent(new Event("simdev:moved"));
  }

  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea")) return;
    const g = state.global;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      const n = choices(g.kind, g.field).length;
      const step = (event.shiftKey ? 5 : 1) * (event.key === "ArrowLeft" ? -1 : 1);
      g.pos = Math.max(0, Math.min(n - 1, g.pos + step));
      event.preventDefault();
      update();
    } else if ((event.key === "b" || event.key === "B") && state.overlay.flip && state.layout === "overlay") {
      state.overlay.flip();
    }
  });

  Promise.all(state.runs.map((run) => fetch(`/api/runs/${encodeURIComponent(run)}/index`)
    .then((r) => r.json()).then((index) => { indexes[run] = index; })))
    .then(() => { guards(); render(); document.dispatchEvent(new Event("simdev:ready")); });
})();
```

Append to `style.css`:

```css
.toolbar { display: grid; gap: 0.4rem; margin-bottom: 0.6rem; }
.toolbar input[type=range] { width: 22rem; }
.panes { display: grid; gap: 0.6rem; }
.panes.n2 { grid-template-columns: 1fr 1fr; }
.panes.n3, .panes.n4 { grid-template-columns: 1fr 1fr; }
@media (min-width: 1500px) { .panes.n3 { grid-template-columns: 1fr 1fr 1fr; } }
.pane-head { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; border-left: 4px solid; padding: 0.2rem 0.5rem; background: #fff; }
.viewport { position: relative; overflow: hidden; aspect-ratio: 4 / 3; background: rgb(212, 212, 212); cursor: grab; touch-action: none; }
.viewport img { position: absolute; inset: 0; width: 100%; height: 100%; transform-origin: 0 0; user-select: none; }
.pane-note { position: absolute; left: 0.6rem; top: 0.4rem; background: rgba(255, 255, 255, 0.85); padding: 0.1rem 0.4rem; font-size: 12px; }
.pane-note:empty { display: none; }
.handle { position: absolute; top: 0; bottom: 0; width: 4px; margin-left: -2px; background: #fff; box-shadow: 0 0 0 1px #000; cursor: ew-resize; z-index: 2; }
.overlay-label { font-weight: 600; margin: 0.2rem 0; }
.chip { border: 1px solid; border-left-width: 4px; border-radius: 4px; padding: 0.2rem 0.5rem; background: #fff; }
```

- [ ] **Step 3: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_ui_compare_routes.py -v`
Expected: all pass.

- [ ] **Step 4: Manual check against real runs**

Start the UI on a free port against the real runs (does not touch the systemd service):

```bash
.venv/bin/simdev ui --runs-root ~/runs --port 8766 --db ~/runs/ui-delta-spike-2026-10-05/check.db
```

The separate, empty `--db` means this instance's worker has nothing to start; the real queue (and any solve) is never touched. Stop it with Ctrl+C afterwards.

Link the two spike runs so they are visible under the runs root: `ln -s ~/runs/meshstudy-2026-10-04/sweep_tess1mm/c02_combo12 ~/runs/c02_combo12` (remove the link after the check). Open `http://<host>:8766/compare?runs=av001-tc10-cornering-lowspeed-car-001,c02_combo12` and check:

1. Both panes show `cp` on x-planes; the slider and ←/→ move both; Shift+→ jumps five planes.
2. Untick sync on pane 2: it gets its own controls and stays put while the slider moves pane 1.
3. Wheel-zoom on pane 1 zooms pane 2 identically (synced); double-click resets.
4. Tick Δ on pane 2: "computing delta…", then a diverging picture within a few seconds; ←/→ to the next plane computes the next one; going back is instant (cached).
5. Switch to overlay, mode blink: `B` and the button flip A/B; auto blink at 500 ms works; swipe shows a draggable divider; fade blends.
6. Surface view, field cp, Δ ticked: first picture in about 20 s, other views faster.
7. "make REF" on pane 2 moves the REF badge and turns the Δ box off on the new REF pane.
8. With a run from a different driving state added: the warning about states appears.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/ui/static/compare.js pipeline/simdev/ui/static/style.css tests/test_ui_compare_routes.py
git commit -m "feat(ui): compare viewer with sync, zoom, blink, swipe, fade and deltas

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Charts (compare.js, part 2)

**Files:**
- Modify: `pipeline/simdev/ui/static/compare.js` (append a second IIFE)
- Test: manual check list (Step 3)

**Interfaces:**
- Consumes: `window.SimdevCompare` (Task 10), events `simdev:ready`, `simdev:ref` and `simdev:moved`, `GET /api/runs/{name}/forces` (existing: `{"total": {"iteration", "Cd", "Cl"}, "components": {patch: {...}}}`), `GET /api/runs/{name}/summary`, `GET /api/runs/{name}/cplines/{station}`.

- [ ] **Step 1: Implement**

Append to `compare.js`:

```javascript
// --- part 2: charts ----------------------------------------------------------
(function () {
  if (!document.getElementById("compare-data")) return;
  const forces = {}, summaries = {};
  let C = null;

  const width = () => Math.max(320, Math.min(document.querySelector("main").clientWidth - 40, 1100));

  function partSeries(run, part, coefficient) {
    const f = forces[run];
    if (!f) return null;
    if (!part) return { it: f.total.iteration, v: f.total[coefficient] };
    const groups = (summaries[run] || {}).groups_map || {};
    const members = groups[part] || [part];
    const tables = members.map((m) => f.components[m]).filter(Boolean);
    if (tables.length !== members.length) return null;
    const sums = new Map();
    for (const t of tables) t.iteration.forEach((it, i) => sums.set(it, (sums.get(it) || 0) + t[coefficient][i]));
    const it = [...sums.keys()].sort((a, b) => a - b);
    return { it, v: it.map((x) => sums.get(x)) };
  }

  function windowMean(run, s) {
    const [a, b] = (summaries[run] || {}).window || [0, 0];
    const values = s.v.filter((_, i) => s.it[i] >= a && s.it[i] <= b);
    return values.length ? values.reduce((x, y) => x + y, 0) / values.length : null;
  }

  function drawForces() {
    const part = document.getElementById("force-part").value;
    const relative = document.getElementById("force-relative").checked;
    for (const coefficient of ["Cl", "Cd"]) {
      const box = document.getElementById(`plot-${coefficient}`);
      box.innerHTML = "";
      const runs = C.state.runs.filter((r) => partSeries(r, part, coefficient));
      if (!runs.length) { box.textContent = "no history"; continue; }
      const tables = runs.map((r) => {
        const s = partSeries(r, part, coefficient);
        const mean = windowMean(r, s);
        const v = relative && mean ? s.v.map((x) => (100 * (x - mean)) / Math.abs(mean)) : s.v;
        return [s.it, v];
      });
      const joined = uPlot.join(tables);
      const bands = {
        hooks: { drawClear: [(u) => {
          const ctx = u.ctx;
          ctx.save();
          runs.forEach((r) => {
            const [a, b] = (summaries[r] || {}).window || [0, 0];
            ctx.fillStyle = C.colourOf(r) + "18";
            const x0 = u.valToPos(a, "x", true), x1 = u.valToPos(b, "x", true);
            ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
          });
          if (relative) for (const [lim, alpha] of [[1, "22"], [0.5, "33"]]) {
            const y0 = u.valToPos(lim, "y", true), y1 = u.valToPos(-lim, "y", true);
            ctx.fillStyle = "#1f8a4c" + alpha;
            ctx.fillRect(u.bbox.left, y0, u.bbox.width, y1 - y0);
          }
          ctx.restore();
        }] },
      };
      new uPlot({
        width: width(), height: 260, plugins: [bands],
        scales: { x: { time: false } },
        axes: [{}, { label: relative ? `${coefficient} − mean [%]` : coefficient }],
        series: [{ label: "iteration" }, ...runs.map((r) => ({ label: r, stroke: C.colourOf(r), width: 1.2 }))],
      }, joined, box);
    }
  }

  function drawBars() {
    const box = document.getElementById("bars");
    box.innerHTML = "";
    const ref = summaries[C.state.ref];
    if (!ref) { box.textContent = "no results for REF"; return; }
    const parts = [
      ...Object.keys(ref.groups_map || {}).map((g) => ({ label: g, key: g, group: true })),
      ...Object.keys(ref.patches || {}).sort().map((p) => ({ label: p, key: p, group: false })),
    ];
    const valueOf = (s, part, c) => part.group
      ? (s.values || {})[`${c.toLowerCase()}_${part.key}`]
      : ((s.patches || {})[part.key] || {})[c];
    const noiseOf = (s, part, c) => part.group
      ? (s.noise || {})[`${c.toLowerCase()}_${part.key}`]
      : ((s.patches || {})[part.key] || {})[`${c}_noise`];
    for (const run of C.state.runs.filter((r) => r !== C.state.ref && summaries[r])) {
      const s = summaries[run];
      const rows = [];
      let largest = 1e-9;
      for (const part of parts) for (const c of ["Cl", "Cd"]) {
        const a = valueOf(s, part, c), b = valueOf(ref, part, c);
        if (a == null || b == null) continue;
        const na = noiseOf(s, part, c), nb = noiseOf(ref, part, c);
        const noise = na != null && nb != null ? Math.hypot(na, nb) : null;
        rows.push({ label: `${part.label} ${c}`, d: a - b, noise });
        largest = Math.max(largest, Math.abs(a - b), noise || 0);
      }
      const table = document.createElement("table");
      table.className = "bars";
      table.innerHTML = `<caption style="color:${C.colourOf(run)}">${run} − ${C.state.ref}</caption>`;
      for (const row of rows) {
        const tr = table.insertRow();
        tr.insertCell().textContent = row.label;
        const cell = tr.insertCell();
        const bar = document.createElement("div");
        bar.className = "bar";
        const pct = (50 * Math.abs(row.d)) / largest;
        bar.innerHTML = `<span class="fill" style="${row.d < 0 ? "right:50%" : "left:50%"};width:${pct}%;background:${C.colourOf(run)}"></span>`
          + (row.noise != null ? `<span class="whisker" style="left:${50 - (50 * row.noise) / largest}%;width:${(100 * row.noise) / largest}%"></span>` : "");
        cell.append(bar);
        const num = tr.insertCell();
        num.className = "num";
        num.textContent = `${row.d >= 0 ? "+" : ""}${row.d.toFixed(4)}${row.noise != null ? ` ± ${row.noise.toFixed(4)}` : ""}`;
      }
      box.append(table);
    }
  }

  async function drawCp() {
    const station = document.getElementById("cp-station").value;
    const patches = [...document.querySelectorAll(".cp-patch:checked")].map((b) => b.value);
    const data = {};
    await Promise.all(C.state.runs.map(async (run) => {
      const r = await fetch(`/api/runs/${encodeURIComponent(run)}/cplines/${encodeURIComponent(station)}`);
      if (r.ok) data[run] = await r.json();
    }));
    const points = (key) => Object.entries(data).flatMap(([run, d]) => patches.flatMap((p) => {
      const s = (d.patches || {})[p];
      return s ? s.x.map((x, i) => ({ x, y: s[key][i], run })) : [];
    }));
    scatter(document.getElementById("cp-plot"), points("cp"), true, "cp");
    scatter(document.getElementById("cp-outline"), points("z"), false, "z [m]");
  }

  function scatter(canvas, pts, invert, label) {
    // cp axis inverted (suction up), as viz/cplines.py draws it.
    const ctx = canvas.getContext("2d");
    canvas.width = width();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!pts.length) { ctx.fillText("no cp lines for this station", 20, 20); return; }
    const pad = { l: 50, r: 10, t: 10, b: 24 };
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const W = canvas.width - pad.l - pad.r, H = canvas.height - pad.t - pad.b;
    const X = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * W;
    const Y = (y) => pad.t + (invert ? (y - y0) / (y1 - y0 || 1) : 1 - (y - y0) / (y1 - y0 || 1)) * H;
    ctx.strokeStyle = "#999";
    ctx.strokeRect(pad.l, pad.t, W, H);
    ctx.fillStyle = "#333";
    ctx.fillText(label, 4, pad.t + 10);
    ctx.fillText(y0.toFixed(2), 4, invert ? pad.t + 22 : pad.t + H);
    ctx.fillText(y1.toFixed(2), 4, invert ? pad.t + H : pad.t + 22);
    ctx.fillText(`x ${x0.toFixed(3)} … ${x1.toFixed(3)} m`, pad.l, canvas.height - 6);
    for (const p of pts) {
      ctx.fillStyle = C.colourOf(p.run);
      ctx.fillRect(X(p.x) - 1, Y(p.y) - 1, 2, 2);
    }
  }

  function fillSelectors() {
    const parts = new Set();
    for (const run of C.state.runs) {
      Object.keys((summaries[run] || {}).groups_map || {}).forEach((g) => parts.add(g));
      Object.keys((forces[run] || {}).components || {}).forEach((p) => parts.add(p));
    }
    const select = document.getElementById("force-part");
    for (const p of [...parts].sort()) select.append(new Option(p, p));
    const stations = (C.indexes[C.state.ref] || {}).stations || [];
    const station = document.getElementById("cp-station");
    for (const s of stations) station.append(new Option(s.name, s.name));
  }

  document.addEventListener("simdev:ready", async () => {
    C = window.SimdevCompare;
    await Promise.all(C.state.runs.map(async (run) => {
      const [f, s] = await Promise.all([
        fetch(`/api/runs/${encodeURIComponent(run)}/forces`).then((r) => r.json()),
        fetch(`/api/runs/${encodeURIComponent(run)}/summary`).then((r) => (r.ok ? r.json() : null)),
      ]);
      forces[run] = f;
      if (s) summaries[run] = s;
    }));
    fillSelectors();
    drawForces();
    drawBars();
    drawCp();
    document.getElementById("force-part").addEventListener("change", drawForces);
    document.getElementById("force-relative").addEventListener("change", drawForces);
    document.getElementById("cp-station").addEventListener("change", drawCp);
    document.querySelectorAll(".cp-patch").forEach((b) => b.addEventListener("change", drawCp));
    document.addEventListener("simdev:ref", drawBars);
    // The cp station follows the viewer when it is on a matching y-plane.
    document.addEventListener("simdev:moved", () => {
      const g = C.state.global;
      if (g.kind !== "y") return;
      const planes = (((C.indexes[C.state.ref] || {}).planes || {}).y || {})[g.field] || [];
      const plane = planes[g.pos];
      const select = document.getElementById("cp-station");
      if (plane && [...select.options].some((o) => o.value === plane.name) && select.value !== plane.name) {
        select.value = plane.name;
        drawCp();
      }
    });
  });
})();
```

Append to `style.css`:

```css
table.bars { width: auto; margin: 0.6rem 0; }
table.bars caption { text-align: left; font-weight: 600; }
table.bars td { padding: 0.15rem 0.5rem; }
.bar { position: relative; width: 22rem; height: 12px; background: linear-gradient(to right, transparent calc(50% - 1px), #999 calc(50% - 1px), #999 calc(50% + 1px), transparent calc(50% + 1px)); }
.bar .fill { position: absolute; top: 2px; height: 8px; }
.bar .whisker { position: absolute; top: 5px; height: 2px; background: #000; opacity: 0.5; }
```

- [ ] **Step 2: Run the UI test suite**

Run: `.venv/bin/python -m pytest tests/ -q -k "ui"`
Expected: all pass.

- [ ] **Step 3: Manual check (same setup as Task 10, Step 4)**

1. Cl and Cd charts show one line per run in the pane colours, windows shaded.
2. "relative to window mean (%)" draws the ±0.5 % and ±1 % bands; c02's Cl swing is visibly outside ±1 % (it is about 2 %).
3. Selecting `body` or a patch switches both charts.
4. Δ bars list groups and patches with noise whiskers; "make REF" on another pane redraws them against the new REF.
5. cp lines: points in run colours, cp axis with suction up; the outline below; moving the viewer to a y-plane that is a station switches the station.

- [ ] **Step 4: Commit**

```bash
git add pipeline/simdev/ui/static/compare.js pipeline/simdev/ui/static/style.css
git commit -m "feat(ui): compare charts - force histories with convergence bands, delta bars, cp lines

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Documentation and final verification

**Files:**
- Modify: `docs/handbook.md` (new subsection in §6 after "The report, the pictures, and what they cost"; one line in §5 module map)
- Modify: `docs/ui-setup.md` (pages list)

- [ ] **Step 1: Write the handbook section**

Add after the "The report, the pictures, and what they cost" subsection:

```markdown
### The results table and the compare page

**`/results` is the Excel sheet, kept by the runs themselves.** One row per
run directory with a `result.json`, newest first. The change note comes from
the New Run form ("What changed in the geometry?") and is copied into
`<run>/ui/note.json` when the job starts; it and the "compare with" reference
are edited inline and saved to the same file - also for runs started from the
shell, because it is metadata and never touches results. References are run
names, so nothing breaks when the list changes (the old sheet referenced row
numbers, hence its `#REF!` cells).

**Every delta carries its noise.** Noise of a coefficient is half the spread
of the rolling mean of half the window length, inside the averaging window -
how far the reported mean moves depending on where the run stopped
(`ui/summary.py`). The noise of a delta is √(nA² + nB²); a delta below it is
greyed instead of coloured green or red. Green is "better": Cl more negative,
Cd lower, −Cl/Cd higher.

**`/compare?runs=a,b,c` shows pictures side by side.** Sync, zoom and pan
are shared because every picture of a view has the same frame. Blink, swipe
and fade put two panes in one viewport.

**Deltas are computed from the sampled fields, never from the PNGs.** On the
banded colour map a small change turns every shifted band edge into a
one-band ring. `viz/delta.py` probes both runs' slice `.vtp` on a grid laid
out exactly like the camera frame (tolerance 1e-4 m: OpenFOAM leaves cut
points up to ~50 µm off the plane), or interpolates the reference's surface
pressure onto the pane's surface within 0.5 mm. Black means "fluid or surface
in one run only" - geometry that moved. Measured on the 22 M-cell car: a
plane delta takes about a second, the first surface delta of a pair about
20 s. Results are cached in `<run>/ui/delta/<reference>/` and dropped when
either run's spec hash or views digest changes; one delta runs at a time.
```

In §5 module map add: `ui/notes.py, ui/summary.py, ui/results.py, ui/imageindex.py, ui/delta.py, viz/delta.py - results table, compare page, field deltas`.

In `docs/ui-setup.md`, in the list of pages, add **Results** (history table, notes, TSV) and **Compare** (pictures, deltas, charts).

- [ ] **Step 2: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: everything passes (the two VTK helper tests run on this machine).

- [ ] **Step 3: Real-data check of a plane and a surface delta**

```bash
.venv/bin/python - <<'EOF'
from pathlib import Path
from simdev.ui.delta import delta_png
runs = Path.home() / "runs"
ref = runs / "av001-tc10-cornering-lowspeed-car-001"
pane = runs / "meshstudy-2026-10-04/sweep_tess1mm/c02_combo12"
print(delta_png(pane, ref, "y_+0.000", "cpt", None))
print(delta_png(pane, ref, "surface_iso", "cp", None))
EOF
```

Expected: two PNG paths under `c02_combo12/ui/delta/av001-tc10-cornering-lowspeed-car-001/`. Open them and check that they match the spike pictures in `~/runs/ui-delta-spike-2026-10-05/out/y_+0.000_cpt.png` and `out_surface/dcp_iso.png` (same structures over the rear deck and wing). Delete the `ui/delta` folder afterwards if not wanted.

- [ ] **Step 4: Commit**

```bash
git add docs/handbook.md docs/ui-setup.md
git commit -m "docs: results table, compare page and field deltas

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
