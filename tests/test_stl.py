from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from simdev.config.schema import AhmedParams
from simdev.geometry.ahmed import build_body, write_ahmed_stl
from simdev.geometry.stl import (
    check_geometry,
    projected_frontal_area,
    read_stl_info,
)


def test_read_stl_info_round_trips(tmp_path: Path) -> None:
    p = AhmedParams()
    paths = write_ahmed_stl(p, tmp_path)
    info = read_stl_info(paths["body"])
    assert info.n_triangles > 0
    assert info.is_watertight is True
    assert info.extent[0] == pytest.approx(p.length, abs=1e-6)


def test_projected_frontal_area_of_a_unit_box() -> None:
    box = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    # Projected along x, the box presents 3 x 4.
    assert projected_frontal_area(box, axis=0) == pytest.approx(12.0, rel=1e-9)


def test_projected_frontal_area_along_other_axes() -> None:
    box = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    assert projected_frontal_area(box, axis=1) == pytest.approx(8.0, rel=1e-9)
    assert projected_frontal_area(box, axis=2) == pytest.approx(6.0, rel=1e-9)


def test_ahmed_projected_area_is_close_to_width_times_height() -> None:
    p = AhmedParams()
    area = projected_frontal_area(build_body(p), axis=0)
    # Slightly under width*height because the nose fillet insets the section.
    assert 0.9 * p.width * p.height < area <= p.width * p.height + 1e-9


def test_check_geometry_flags_non_watertight(tmp_path: Path) -> None:
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box.faces = box.faces[:-2]  # punch a hole
    path = tmp_path / "leaky.stl"
    box.export(path)
    warnings = check_geometry(read_stl_info(path))
    assert any("watertight" in w for w in warnings)


def test_check_geometry_flags_millimetre_units(tmp_path: Path) -> None:
    box = trimesh.creation.box(extents=[1044.0, 389.0, 288.0])
    path = tmp_path / "mm.stl"
    box.export(path)
    warnings = check_geometry(read_stl_info(path), expected_length=1.044)
    assert any("unit" in w.lower() or "mm" in w for w in warnings)


def test_check_geometry_is_quiet_on_a_good_mesh(tmp_path: Path) -> None:
    paths = write_ahmed_stl(AhmedParams(), tmp_path)
    assert check_geometry(read_stl_info(paths["body"]), expected_length=1.044) == []
