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
