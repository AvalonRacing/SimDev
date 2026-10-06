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


def test_derived_fields_rotating_frame_signs_and_local_head() -> None:
    # omega=2 about (1, 0); point at (3, 1): dx=2, dy=1.
    # U_rel = (Ux + w*dy, Uy - w*dx, Uz) = (4+2, 1-4, 2) = (6, -3, 2); |U_rel|^2 = 49
    # q = 0.5*10^2 = 50; local freestream |w*r|^2 = 4*5 = 20
    frame = {"mode": "cornering", "u_inf": 10.0, "omega": 2.0, "origin": [1.0, 0.0, 0.0]}
    out = D.derived_fields(np.array([25.0]), np.array([[4.0, 1.0, 2.0]]),
                           np.array([True]), np.array([[3.0, 1.0, 0.0]]), frame)
    assert np.isclose(out["U"][0], 7.0)
    assert np.isclose(out["cp"][0], 0.5)
    assert np.isclose(out["cpt"][0], (25 + 0.5 * 49 - 0.5 * 20) / 50 + 1.0)


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
        f"_, stats = d.surface_delta_field({str(tmp_path / 'pane.vtp')!r}, {str(tmp_path / 'ref.vtp')!r}, 50.0, 50.0)\n"
        "print(json.dumps(stats))\n"
    )
    done = subprocess.run([PVPY, "-c", code], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1])["moved_pct"] == 100.0
