from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simdev.config.validate import ValidationError
from simdev.run.status import read_status
from simdev.stages.prepare import prepare

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044, "c_of_r": [0.5, 0.0, 0.0]},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {"include_stilts": False},
        "patches": [
            {"name": "body", "role": "body"},
            {"name": "ground", "role": "ground"},
            {"name": "symmetry", "role": "symmetry"},
            {"name": "inlet", "role": "inlet"},
            {"name": "outlet", "role": "outlet"},
            {"name": "farfield", "role": "farfield"},
        ],
    },
}


@pytest.fixture()
def case_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    return path


def test_prepare_writes_a_complete_case(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")

    for relative in (
        "system/blockMeshDict",
        "system/snappyHexMeshDict",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/decomposeParDict",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "0/U",
        "0/p",
        "0/k",
        "0/omega",
        "0/nut",
        "caseSpec.json",
    ):
        assert (run_dir / relative).exists(), relative


def test_prepare_writes_the_geometry(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    assert (run_dir / "constant" / "triSurface" / "body.stl").exists()


def test_prepare_records_ok_status(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = prepare(case_file, run_dir, profile="dev")
    status = read_status(run_dir, "prepare")
    assert status is not None
    assert status.state == "ok"
    assert status.input_hash == result.spec.spec_hash()


def test_prepare_sizes_the_domain_from_the_geometry(
    case_file: Path, tmp_path: Path
) -> None:
    result = prepare(case_file, tmp_path / "run", profile="dev")
    # 5 body lengths upstream of the nose at x = 0.
    assert result.domain.x_min == pytest.approx(-5.0 * 1.044, rel=1e-6)


def test_prepare_uses_projected_area_for_blockage(
    case_file: Path, tmp_path: Path
) -> None:
    result = prepare(case_file, tmp_path / "run", profile="dev")
    # Half model, so half the projected area, and below width*height.
    assert 0.0 < result.frontal_area < 0.112032 / 2 + 1e-6


def test_prepare_rejects_an_invalid_case(tmp_path: Path) -> None:
    bad = dict(CASE)
    bad["geometry"] = dict(CASE["geometry"])
    bad["geometry"]["patches"] = [
        p for p in CASE["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValidationError):
        prepare(path, tmp_path / "run", profile="dev")


def test_prepare_is_idempotent_and_skips_on_unchanged_input(
    case_file: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    marker = run_dir / "system" / "controlDict"
    marker.write_text("TOUCHED", encoding="utf-8")

    prepare(case_file, run_dir, profile="dev")
    assert marker.read_text(encoding="utf-8") == "TOUCHED"


def test_force_rerenders(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    marker = run_dir / "system" / "controlDict"
    marker.write_text("TOUCHED", encoding="utf-8")

    prepare(case_file, run_dir, profile="dev", force=True)
    assert marker.read_text(encoding="utf-8") != "TOUCHED"


def test_profile_change_invalidates_the_skip(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    marker = run_dir / "system" / "controlDict"
    marker.write_text("TOUCHED", encoding="utf-8")

    # A different profile is a different spec hash, so it must re-render.
    # The production profile's 0.469 mm surface cells are a wall-resolved
    # density, so it only validates against a wall-resolved treatment - the
    # high_y_plus stack this case carries by default does not fit in them.
    prepare(case_file, run_dir, profile="production", wall_treatment="low_y_plus")
    assert marker.read_text(encoding="utf-8") != "TOUCHED"
