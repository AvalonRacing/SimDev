from __future__ import annotations

import json
import math
import types
from pathlib import Path
from unittest.mock import patch

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


def test_corrupt_result_json_returns_none(tmp_path: Path) -> None:
    run = tmp_path / "r_corrupt"
    (run / "results").mkdir(parents=True)
    (run / "results" / "result.json").write_text("{ invalid json")
    assert load_summary(run) is None


def test_group_noise_with_hand_computable_rolling_noise(tmp_path: Path) -> None:
    # Use a distinct tmp path to avoid lru_cache interference
    run = tmp_path / "r_group1"
    run.mkdir(parents=True)
    (run / "results").mkdir(parents=True)
    (run / "results" / "result.json").write_text(json.dumps({
        "cd_mean": 0.5, "cl_mean": -1.0, "window_start": 3, "window_end": 6,
        "verdict": "converged",
    }))
    # Body: Cd=[1, 2, 3, 4, 5, 6], window [3,6] -> [3, 4, 5, 6]
    write_coeffs(run / "postProcessing/forceCoeffs_Body/0/coefficient.dat",
                 [(t, float(t), -0.5) for t in range(1, 7)])
    # Wing: Cd=[0, 0, 0, 0, 0, 0], window [3,6] -> [0, 0, 0, 0]
    write_coeffs(run / "postProcessing/forceCoeffs_Wing/0/coefficient.dat",
                 [(t, 0.0, -0.5) for t in range(1, 7)])

    # Mock load_spec to return a group "chassis" = ("Body", "Wing")
    def mock_load_spec(run_dir: Path):
        return types.SimpleNamespace(
            post=types.SimpleNamespace(groups={"chassis": ("Body", "Wing")})
        )

    with patch("simdev.ui.summary.load_spec", side_effect=mock_load_spec):
        summary = load_summary(run)

    assert summary.groups_map == {"chassis": ["Body", "Wing"]}
    # Sum in window: Body [3,4,5,6] + Wing [0,0,0,0] = [3,4,5,6]
    # rolling_noise([3,4,5,6]) with half-window 2 -> means [3.5, 4.5, 5.5] -> spread 2 -> noise 1
    assert summary.noise["Cd_chassis"] == 1.0


def test_group_with_missing_patch_data_has_nan_noise(tmp_path: Path) -> None:
    # Use a distinct tmp path to avoid lru_cache interference
    run = tmp_path / "r_group2"
    run.mkdir(parents=True)
    (run / "results").mkdir(parents=True)
    (run / "results" / "result.json").write_text(json.dumps({
        "cd_mean": 0.5, "cl_mean": -1.0, "window_start": 1, "window_end": 3,
        "verdict": "converged",
    }))
    # Only Body has forceCoeffs data; Wing data is missing
    write_coeffs(run / "postProcessing/forceCoeffs_Body/0/coefficient.dat",
                 [(t, float(t), -0.5) for t in range(1, 4)])

    def mock_load_spec(run_dir: Path):
        return types.SimpleNamespace(
            post=types.SimpleNamespace(groups={"chassis": ("Body", "Wing")})
        )

    with patch("simdev.ui.summary.load_spec", side_effect=mock_load_spec):
        summary = load_summary(run)

    # Wing patch has no data, so group noise should be NaN
    assert math.isnan(summary.noise["Cd_chassis"])
    assert math.isnan(summary.noise["Cl_chassis"])


def test_group_with_misaligned_patch_times(tmp_path: Path) -> None:
    # Use a distinct tmp path to avoid lru_cache interference
    run = tmp_path / "r_group3"
    run.mkdir(parents=True)
    (run / "results").mkdir(parents=True)
    (run / "results" / "result.json").write_text(json.dumps({
        "cd_mean": 0.5, "cl_mean": -1.0, "window_start": 2, "window_end": 4,
        "verdict": "converged",
    }))
    # Body: complete time series
    write_coeffs(run / "postProcessing/forceCoeffs_Body/0/coefficient.dat",
                 [(t, 1.0, -0.5) for t in range(1, 6)])
    # Wing: missing Time=3 (inside window)
    write_coeffs(run / "postProcessing/forceCoeffs_Wing/0/coefficient.dat",
                 [(1, 1.0, -0.5), (2, 1.0, -0.5), (4, 1.0, -0.5), (5, 1.0, -0.5)])

    def mock_load_spec(run_dir: Path):
        return types.SimpleNamespace(
            post=types.SimpleNamespace(groups={"chassis": ("Body", "Wing")})
        )

    with patch("simdev.ui.summary.load_spec", side_effect=mock_load_spec):
        summary = load_summary(run)

    # Groups_map should be set
    assert summary.groups_map == {"chassis": ["Body", "Wing"]}
    # The sum only includes common times: [2, 4] in window (no time 3 from Wing)
    # Sum = [2.0, 2.0] -> rolling_noise with length 1 -> spread 0 -> noise 0
    assert summary.noise["Cd_chassis"] == 0.0
