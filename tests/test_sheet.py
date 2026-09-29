from __future__ import annotations

from pathlib import Path

from simdev.viz.sheet import write_contact_sheet

PLAN = {
    "slices": [
        {"name": "x_-0.300", "axis": "x", "offset": -0.300,
         "images": [{"field": "cp", "out": "results/images/slices/x/cp/cp_x_-0.300.png"}]},
        {"name": "x_+0.000", "axis": "x", "offset": 0.0,
         "images": [{"field": "cp", "out": "results/images/slices/x/cp/cp_x_+0.000.png"}]},
        {"name": "x_+0.020", "axis": "x", "offset": 0.02,
         "images": [{"field": "cp", "out": "results/images/slices/x/cp/cp_x_+0.020.png"}]},
    ],
    "surfaces": [
        {"name": "front",
         "images": [{"field": "yplus", "out": "results/images/surface/yplus/front.png"}]},
    ],
}
RECORD = {
    "run": "car-01", "spec_hash": "abc123def456", "views_digest": "deadbeef1234",
    "datum": [0.0036, 0.013, 0.0], "images_written": 4,
    "sample_seconds": 12.3, "render_seconds": 45.6, "reasons": [],
}


def test_the_sheet_lists_every_image(tmp_path: Path) -> None:
    out = write_contact_sheet(tmp_path, PLAN, RECORD)
    text = out.read_text(encoding="utf-8")
    assert text.count("<img") == 4
    assert "cp_x_+0.020.png" in text


def test_the_sheet_carries_the_provenance(tmp_path: Path) -> None:
    """It is the index someone opens six months later."""
    text = write_contact_sheet(tmp_path, PLAN, RECORD).read_text(encoding="utf-8")
    assert "deadbeef1234" in text
    assert "car-01" in text


def test_the_sheet_shows_the_warnings(tmp_path: Path) -> None:
    record = dict(RECORD, reasons=["the x slice range does not cover the geometry"])
    text = write_contact_sheet(tmp_path, PLAN, record).read_text(encoding="utf-8")
    assert "does not cover" in text


def test_slices_are_sorted_by_numeric_offset_not_name(tmp_path: Path) -> None:
    """RULING PLAN-DEFECT-2: Sort by numeric offset, not by string name.

    String sorting of names like 'x_-0.300' vs 'x_+0.000' is broken because
    '+' (ASCII 43) sorts before '-' (ASCII 45), putting all positives before
    all negatives, and within negatives sorting larger magnitudes last.
    Result: sheet starts mid-car, walks to nose, jumps behind datum, walks backward.

    This test verifies the correct geometric order: -0.300 < 0.000 < 0.020.
    """
    text = write_contact_sheet(tmp_path, PLAN, RECORD).read_text(encoding="utf-8")

    # Find the positions of the three images in the HTML.
    pos_neg300 = text.find("cp_x_-0.300.png")
    pos_zero = text.find("cp_x_+0.000.png")
    pos_pos020 = text.find("cp_x_+0.020.png")

    # All should exist.
    assert pos_neg300 >= 0, "x_-0.300 image not found"
    assert pos_zero >= 0, "x_+0.000 image not found"
    assert pos_pos020 >= 0, "x_+0.020 image not found"

    # And in the correct numeric order (smallest to largest offset).
    assert pos_neg300 < pos_zero, (
        "x_-0.300 should appear before x_+0.000 in the sheet "
        "(sorted by numeric offset -0.300 < +0.000)"
    )
    assert pos_zero < pos_pos020, (
        "x_+0.000 should appear before x_+0.020 in the sheet "
        "(sorted by numeric offset +0.000 < +0.020)"
    )
