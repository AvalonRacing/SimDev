from __future__ import annotations

from pathlib import Path

import pytest

from simdev.cad.checks import default_library, state_params_check
from simdev.cad.library import LibraryError

gmsh = pytest.importorskip("gmsh", reason="the STEP check needs gmsh")

from simdev.cad.stepcheck import check_step_files  # noqa: E402

CAR_CASE = Path("cases/car/config.yaml")


def write_box_step(path: Path) -> Path:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def test_a_real_step_file_passes(tmp_path: Path) -> None:
    good = write_box_step(tmp_path / "Body.step")
    assert check_step_files([good]) == {}


def test_a_file_without_the_step_header_fails_without_gmsh(tmp_path: Path) -> None:
    bad = tmp_path / "Body.step"
    bad.write_text("solid ascii stl\n")
    assert "not a STEP file" in check_step_files([bad])[bad]


def test_a_corrupt_step_body_fails(tmp_path: Path) -> None:
    good = write_box_step(tmp_path / "Body.step")
    broken = tmp_path / "Wing.step"
    broken.write_bytes(b"ISO-10303-21;\nHEADER;\nthis is not step\n")
    errors = check_step_files([good, broken])
    assert good not in errors
    assert broken in errors


def test_state_params_that_resolve_pass() -> None:
    check = state_params_check(CAR_CASE)
    check({"flow": {"u_inf": 15.0}, "physics": {"mode": "cornering", "corner_radius": 4.0},
           "domain": {"kind": "annulus"}, "ground": {"motion": "static"}})


def test_state_params_that_do_not_resolve_are_refused() -> None:
    check = state_params_check(CAR_CASE)
    with pytest.raises(LibraryError, match="do not resolve"):
        check({"physics": {"mode": "sideways"}})


def test_default_library_points_at_the_repo_cad_folder() -> None:
    assert default_library().root.name == "CAD"
