from __future__ import annotations

import math
from pathlib import Path

import pytest

from simdev.config.schema import AhmedParams
from simdev.geometry.ahmed import build_body, frontal_area, write_ahmed_stl


def test_bounding_box_matches_parameters() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    lo, hi = mesh.bounds
    assert hi[0] - lo[0] == pytest.approx(p.length, abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(p.width, abs=1e-6)
    assert lo[2] == pytest.approx(p.ground_clearance, abs=1e-6)
    assert hi[2] == pytest.approx(p.ground_clearance + p.height, abs=1e-6)


def test_body_is_watertight() -> None:
    assert build_body(AhmedParams()).is_watertight


def test_body_volume_is_positive_and_below_the_bounding_box() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    box_volume = p.length * p.width * p.height
    assert 0.0 < mesh.volume < box_volume


def test_nose_is_inset_by_the_fillet_radius_at_the_tip() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    tip = mesh.vertices[mesh.vertices[:, 0] < 1e-9]
    assert len(tip) > 0
    # At x = 0 the section is inset by the full nose radius on every side.
    assert tip[:, 1].max() == pytest.approx(p.width / 2 - p.nose_radius, abs=1e-6)


def test_slant_reduces_roof_height_by_the_right_amount() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    theta = math.radians(p.slant_angle_deg)
    tail = mesh.vertices[mesh.vertices[:, 0] > p.length - 1e-9]
    expected = p.ground_clearance + p.height - p.slant_length * math.sin(theta)
    assert tail[:, 2].max() == pytest.approx(expected, abs=1e-6)


def test_frontal_area_is_width_times_height() -> None:
    p = AhmedParams()
    assert frontal_area(p) == pytest.approx(0.112032, abs=1e-6)


def test_write_produces_body_and_stilts(tmp_path: Path) -> None:
    paths = write_ahmed_stl(AhmedParams(include_stilts=True), tmp_path)
    assert set(paths) == {"body", "stilts"}
    assert paths["body"].exists() and paths["body"].stat().st_size > 0
    assert paths["stilts"].exists()


def test_write_omits_stilts_when_disabled(tmp_path: Path) -> None:
    paths = write_ahmed_stl(AhmedParams(include_stilts=False), tmp_path)
    assert set(paths) == {"body"}


def test_slant_angle_is_configurable() -> None:
    steep = build_body(AhmedParams(slant_angle_deg=35.0))
    shallow = build_body(AhmedParams(slant_angle_deg=25.0))
    # A steeper slant removes more material.
    assert steep.volume < shallow.volume
