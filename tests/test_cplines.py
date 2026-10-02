from __future__ import annotations

from pathlib import Path

from simdev.viz.cplines import draw_cp_lines, read_station


def _spec(tmp_path: Path) -> dict:
    return {
        "patches": ["Body", "Wing"],
        "cp_limits": [-3.0, 1.0],
        "x_limits": [-0.25, 0.25],
        "z_limits": [0.0, 0.14],
        "stations": [
            {"name": "y_+0.000", "offset": 0.0,
             "csv": str(tmp_path / "cp_lines" / "y_+0.000.csv"),
             "out": str(tmp_path / "images" / "cp_line_y" / "cp_line_y_01_+0.000.png")},
            {"name": "y_+0.020", "offset": 0.02,
             "csv": str(tmp_path / "cp_lines" / "y_+0.020.csv"),
             "out": str(tmp_path / "images" / "cp_line_y" / "cp_line_y_02_+0.020.png")},
        ],
    }


def test_each_station_with_points_becomes_a_plot(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    csv = Path(spec["stations"][0]["csv"])
    csv.parent.mkdir(parents=True)
    csv.write_text(
        "patch,x,z,pMean\n"
        "Body,0.2,0.05,100.0\nBody,0.0,0.11,-150.0\nWing,-0.19,0.11,-200.0\n",
        encoding="utf-8",
    )
    # The second station has no CSV - the cut found nothing - and is skipped.
    assert draw_cp_lines(spec, u_inf=15.0) == 1
    assert Path(spec["stations"][0]["out"]).stat().st_size > 0
    assert not Path(spec["stations"][1]["out"]).exists()


def test_points_are_grouped_by_part(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("patch,x,z,pMean\nBody,0.1,0,1\nWing,-0.2,0,2\nBody,0.0,0,3\n")
    points = read_station(path)
    assert points["Body"] == ([0.1, 0.0], [0.0, 0.0], [1.0, 3.0])
    assert points["Wing"] == ([-0.2], [0.0], [2.0])


def test_no_cp_lines_block_draws_nothing() -> None:
    assert draw_cp_lines(None, u_inf=15.0) == 0
