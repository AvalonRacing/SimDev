from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.migration.split_driving_states import split
from simdev.cad.library import DESIGN_PARTS, STATE_PARTS, Library
from simdev.config.resolve import LegacyDrivingStatesError, resolve
from simdev.config.schema import FlowDirection, Mode

# What split() writes for the shipped testcase state, verbatim.
TESTCASE_STATE = {
    "flow": {"u_inf": 15.0},
    "physics": {"mode": "cornering", "corner_radius": 4.0, "corner_direction": "left"},
    "domain": {"kind": "annulus"},
    "ground": {"motion": "static"},
}


def fake_repo(tmp_path: Path) -> Path:
    folder = tmp_path / "repo" / "CAD" / "Testcase"
    folder.mkdir(parents=True)
    for part in (*STATE_PARTS, *DESIGN_PARTS):
        (folder / f"{part}.step").write_bytes(b"ISO-10303-21;\n" + part.encode())
    return tmp_path / "repo"


def test_split_creates_states_and_a_baseline_design(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    config = {
        "driving_state": "testcase",
        "driving_states": {
            "testcase": {"geometry": {"source_dir": "CAD/Testcase"}, **TESTCASE_STATE},
            "straight": {"geometry": {"source_dir": "CAD/Testcase"},
                         "physics": {"mode": "straight"}},
        },
    }

    done = split(config, repo, repo / "CAD")

    lib = Library(repo / "CAD")
    assert sorted(done) == ["straight", "testcase"]
    assert lib.state("testcase").params == TESTCASE_STATE
    assert sorted(lib.pairs()) == [("baseline", "straight"), ("baseline", "testcase")]


def test_split_is_safe_to_run_twice(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    config = {"driving_states": {"testcase": {"geometry": {"source_dir": "CAD/Testcase"}}}}
    split(config, repo, repo / "CAD")
    assert split(config, repo, repo / "CAD") == []


def test_inline_driving_states_are_refused_with_the_migration_hint() -> None:
    case = yaml.safe_load(Path("cases/car/config.yaml").read_text(encoding="utf-8"))
    case["driving_states"] = {"x": {}}
    with pytest.raises(LegacyDrivingStatesError, match="split_driving_states"):
        resolve(case, profile="car_dev")


def test_the_car_case_no_longer_carries_states() -> None:
    case = yaml.safe_load(Path("cases/car/config.yaml").read_text(encoding="utf-8"))
    assert "driving_states" not in case
    assert "driving_state" not in case


def test_the_migrated_testcase_resolves_to_the_requested_corner() -> None:
    """The same assertions the inline-state version of this test made."""
    case = yaml.safe_load(Path("cases/car/config.yaml").read_text(encoding="utf-8"))
    spec = resolve(case, profile="car_dev", state=TESTCASE_STATE)

    assert spec.flow.u_inf == 15.0
    assert spec.physics.corner_radius == 4.0
    assert spec.physics.mode is Mode.CORNERING
    assert spec.domain.kind == "annulus"
    assert spec.flow.direction is FlowDirection.MINUS_X
    assert spec.geometry.symmetric is False
    assert spec.omega_signed == pytest.approx(15.0 / 4.0)
