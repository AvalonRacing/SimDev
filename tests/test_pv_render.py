from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RENDERER = (
    Path(__file__).resolve().parents[1]
    / "pipeline" / "simdev" / "viz" / "pv_render.py"
)


def test_the_renderer_imports_nothing_from_simdev() -> None:
    """It runs under /usr/bin/python3, which has paraview and no venv.

    A single `from simdev...` here turns every picture into an ImportError on
    the machine that can actually render.
    """
    text = RENDERER.read_text(encoding="utf-8")
    assert "import simdev" not in text
    assert "from simdev" not in text


def _paraview_python() -> str | None:
    for candidate in ("/usr/bin/python3", sys.executable):
        if shutil.which(candidate) is None and not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", "import paraview.simple"],
            capture_output=True, timeout=180,
        )
        if probe.returncode == 0:
            return candidate
    return None


@pytest.mark.paraview
def test_it_renders_a_plane_to_a_png(tmp_path: Path) -> None:
    interpreter = _paraview_python()
    if interpreter is None:
        pytest.skip("no python with paraview.simple")

    # A one-cell polydata square carrying pMean and UMean, written by hand so
    # the test needs no OpenFOAM run.
    sample = tmp_path / "x_+0.000.vtp"
    sample.write_text(VTP, encoding="utf-8")

    out = tmp_path / "cp.png"
    plan = {
        "views_digest": "deadbeef1234",
        "resolution": [320, 240],
        "streamlines": "off",
        "datum": [0.0, 0.0, 0.0],
        "stamp": {"run": "test", "spec_hash": "abc12345", "window": "1-1", "mean": True},
        "frame": {"mode": "straight", "omega": 0.0, "origin": [0, 0, 0], "u_inf": 10.0},
        "fields": {"cp": {"limits": [-1.0, 1.0], "colormap": "spectrum"}},
        "slices": [{
            "name": "x_+0.000",
            "axis": "x",
            "offset": 0.0,
            "sample": str(sample),
            "camera": {
                "focal": [0.0, 0.0, 0.0], "position": [-2.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0], "parallel_scale": 1.0,
            },
            "images": [{"field": "cp", "out": str(out)}],
        }],
        "surfaces": [],
    }
    plan_path = tmp_path / "render_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [interpreter, str(RENDERER), str(plan_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 1000
    assert json.loads(result.stdout.strip().splitlines()[-1])["written"] == 1


VTP = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="4" NumberOfPolys="1">
      <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">
        0 -1 -1  0 1 -1  0 1 1  0 -1 1
      </DataArray></Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
      </Polys>
      <PointData>
        <DataArray type="Float32" Name="pMean" format="ascii">-40 -10 10 40</DataArray>
        <DataArray type="Float32" Name="UMean" NumberOfComponents="3" format="ascii">
          10 0 0  9 1 0  8 2 0  7 3 0
        </DataArray>
      </PointData>
    </Piece>
  </PolyData>
</VTKFile>
"""
