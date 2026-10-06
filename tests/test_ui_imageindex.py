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


def test_plan_entry_with_missing_keys_is_skipped(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    (run / "results/images/cp_x").mkdir(parents=True)
    (run / "results/images/cp_x/test.png").write_bytes(b"png")
    plan = {
        "views_digest": "abc",
        "frame": {"u_inf": 6.0},
        "slices": [
            {"name": "x_+0.000", "axis": "x", "offset": 0.0,
             "images": [{"field": "cp", "out": f"{OLD}/results/images/cp_x/test.png"}]},
            {"axis": "x", "offset": 0.0,
             "images": [{"field": "cp", "out": f"{OLD}/results/images/cp_x/test.png"}]},  # missing name
            {"name": "missing_axis", "offset": 0.0,
             "images": [{"field": "cp", "out": f"{OLD}/results/images/cp_x/test.png"}]},  # missing axis
        ],
        "surfaces": [
            {"name": "top", "images": [{"field": "cp", "out": f"{OLD}/results/images/cp_x/test.png"}]},
            {"images": [{"field": "cp", "out": f"{OLD}/results/images/cp_x/test.png"}]},  # missing name
        ],
    }
    (run / "results/render_plan.json").write_text(json.dumps(plan))
    (run / "results/result.json").write_text(json.dumps({"window_start": 0, "window_end": 10}))
    index = build_index(run)
    assert "x" in index["planes"]
    assert "cp" in index["planes"]["x"]
    assert len(index["planes"]["x"]["cp"]) == 1
    assert "top" in index["surfaces"].get("cp", {})
    assert len(index["surfaces"].get("cp", {})) == 1


def test_unanchored_existing_absolute_path_is_skipped(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    (run / "results/images").mkdir(parents=True)
    (run / "results/images/test.png").write_bytes(b"png")
    # Create a file at an absolute path without anchors
    fake_path = tmp_path / "fake.png"
    fake_path.write_bytes(b"png")
    plan = {
        "views_digest": "abc",
        "frame": {"u_inf": 6.0},
        "slices": [
            {"name": "x_+0.000", "axis": "x", "offset": 0.0,
             "images": [{"field": "cp", "out": str(fake_path)}]},
        ],
    }
    (run / "results/render_plan.json").write_text(json.dumps(plan))
    (run / "results/result.json").write_text(json.dumps({"window_start": 0, "window_end": 10}))
    index = build_index(run)
    assert index["planes"] == {}


def test_escaping_path_outside_run_is_skipped(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    (run / "results/images").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    (other / "bad.png").write_bytes(b"png")
    # Path that escapes: results/../../other/bad.png
    escaping_path = f"{OLD}/results/../../other/bad.png"
    plan = {
        "views_digest": "abc",
        "frame": {"u_inf": 6.0},
        "slices": [
            {"name": "x_+0.000", "axis": "x", "offset": 0.0,
             "images": [{"field": "cp", "out": escaping_path}]},
        ],
    }
    (run / "results/render_plan.json").write_text(json.dumps(plan))
    (run / "results/result.json").write_text(json.dumps({"window_start": 0, "window_end": 10}))
    index = build_index(run)
    assert index["planes"] == {}


def test_cp_station_with_escaping_csv_path_returns_empty_patches(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    (run / "results").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    (other / "bad.csv").write_text("patch,x,z,pMean\nBody,0.1,0.05,18\n")
    escaping_path = f"{OLD}/results/../../other/bad.csv"
    plan = {
        "views_digest": "abc",
        "frame": {"u_inf": 6.0},
        "cp_lines": {"stations": [{"name": "y_+0.000", "offset": 0.0, "csv": escaping_path}]},
    }
    (run / "results/render_plan.json").write_text(json.dumps(plan))
    data = cp_station(run, "y_+0.000")
    assert data["u_inf"] == 6.0
    assert data["patches"] == {}
