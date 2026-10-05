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
