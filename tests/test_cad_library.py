"""The CAD library: states, designs, slots and assembly - files only."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simdev.cad.library import (
    DESIGN_PARTS,
    STATE_PARTS,
    Library,
    LibraryError,
    match_parts,
)

CORNER = {
    "flow": {"u_inf": 15.0},
    "physics": {"mode": "cornering", "corner_radius": 4.0, "corner_direction": "left"},
    "domain": {"kind": "annulus"},
    "ground": {"motion": "static"},
}


def make_files(tmp_path: Path, names: list[str], tag: str = "a") -> dict[str, Path]:
    """Fake uploads. The library does not parse STEP; its step_check does."""
    folder = tmp_path / f"upload-{tag}"
    folder.mkdir(exist_ok=True)
    out = {}
    for name in names:
        path = folder / name
        path.write_bytes(b"ISO-10303-21;\n" + f"{name} {tag}".encode())
        out[name] = path
    return out


def state_files(tmp_path: Path, tag: str = "a") -> dict[str, Path]:
    return make_files(tmp_path, [f"{p}.step" for p in STATE_PARTS], tag)


def design_files(tmp_path: Path, tag: str = "a") -> dict[str, Path]:
    return make_files(tmp_path, [f"{p}.step" for p in DESIGN_PARTS], tag)


@pytest.fixture
def lib(tmp_path: Path) -> Library:
    return Library(tmp_path / "CAD")


@pytest.fixture
def stocked(lib: Library, tmp_path: Path) -> Library:
    lib.create_state("corner", "4 m left", CORNER, state_files(tmp_path))
    lib.create_state("straight", "", {"physics": {"mode": "straight"}}, state_files(tmp_path, "s"))
    lib.create_design("v01")
    lib.add_slot("v01", "corner", design_files(tmp_path))
    return lib


# --- matching uploads to parts ---------------------------------------------


def test_case_and_extension_variants_are_accepted(tmp_path: Path) -> None:
    files = make_files(tmp_path, ["BODY.STP", "wing.step"])
    assert set(match_parts(files, DESIGN_PARTS, require_all=True)) == {"Body", "Wing"}


def test_two_files_for_one_part_are_rejected(tmp_path: Path) -> None:
    files = make_files(tmp_path, ["Body.step", "body.STEP", "Wing.step"])
    with pytest.raises(LibraryError) as error:
        match_parts(files, DESIGN_PARTS, require_all=True)
    assert "second file for part Body" in str(error.value)
    assert "Body.step" in str(error.value)
    assert "body.STEP" in str(error.value)


def test_unexpected_and_missing_parts_are_named(tmp_path: Path) -> None:
    files = make_files(tmp_path, ["Body.step", "Spoiler.step"])
    with pytest.raises(LibraryError) as error:
        match_parts(files, DESIGN_PARTS, require_all=True)
    assert "Spoiler.step" in str(error.value)
    assert "missing parts: Wing" in str(error.value)


def test_a_non_step_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="not a .step"):
        match_parts(make_files(tmp_path, ["Body.stl"]), DESIGN_PARTS, require_all=False)


# --- states -----------------------------------------------------------------


def test_create_state_writes_parts_and_parameters(lib: Library, tmp_path: Path) -> None:
    state = lib.create_state("corner", "4 m left", CORNER, state_files(tmp_path))
    folder = lib.root / "states" / "corner"
    assert sorted(p.name for p in folder.glob("*.step")) == sorted(f"{p}.step" for p in STATE_PARTS)
    assert state.description == "4 m left"
    assert state.params == CORNER
    assert lib.state("corner") == state


def test_parts_are_stored_under_their_canonical_name(lib: Library, tmp_path: Path) -> None:
    files = make_files(tmp_path, [f"{p.upper()}.STP" for p in STATE_PARTS])
    lib.create_state("corner", "", CORNER, files)
    assert (lib.root / "states" / "corner" / "Tire_FL.step").is_file()


def test_a_failed_state_upload_leaves_nothing_behind(lib: Library, tmp_path: Path) -> None:
    files = state_files(tmp_path)
    files.pop("Chassis.step")
    with pytest.raises(LibraryError, match="missing parts: Chassis"):
        lib.create_state("corner", "", CORNER, files)
    assert lib.states() == []
    assert not list(lib.root.glob(".tmp-*"))


@pytest.mark.parametrize("name", ["Corner", "corner r4", "../x", "-x", ""])
def test_bad_names_are_rejected(lib: Library, tmp_path: Path, name: str) -> None:
    with pytest.raises(LibraryError, match="lowercase"):
        lib.create_state(name, "", CORNER, state_files(tmp_path))


def test_step_check_failures_name_the_part(tmp_path: Path) -> None:
    def broken_wing(paths: list[Path]) -> dict[Path, str]:
        return {p: "gmsh could not read it" for p in paths if p.name == "Wing.step"}

    lib = Library(tmp_path / "CAD", step_check=broken_wing)
    lib.create_state("corner", "", CORNER, state_files(tmp_path))
    lib.create_design("v01")
    with pytest.raises(LibraryError, match="Wing: gmsh could not read it"):
        lib.add_slot("v01", "corner", design_files(tmp_path))


def test_params_check_failures_are_reported(tmp_path: Path) -> None:
    def refuse(params: dict) -> None:
        raise LibraryError("state parameters do not resolve: bad mode")

    lib = Library(tmp_path / "CAD", params_check=refuse)
    with pytest.raises(LibraryError, match="bad mode"):
        lib.create_state("corner", "", CORNER, state_files(tmp_path))


def test_update_state_rewrites_parameters(stocked: Library) -> None:
    stocked.update_state("corner", "8 m", {**CORNER, "flow": {"u_inf": 20.0}})
    assert stocked.state("corner").params["flow"]["u_inf"] == 20.0
    assert stocked.state("corner").description == "8 m"


def test_replace_state_parts_replaces_only_those_given(stocked: Library, tmp_path: Path) -> None:
    folder = stocked.root / "states" / "corner"
    before = (folder / "Chassis.step").read_bytes()
    stocked.replace_state_parts("corner", make_files(tmp_path, ["SUS_FL.step"], "new"))
    assert b"new" in (folder / "SUS_FL.step").read_bytes()
    assert (folder / "Chassis.step").read_bytes() == before


def test_delete_state_is_blocked_by_its_slots(stocked: Library) -> None:
    with pytest.raises(LibraryError, match="v01/corner"):
        stocked.delete_state("corner")
    stocked.delete_state("straight")
    assert [s.name for s in stocked.states()] == ["corner"]


def test_rename_state_carries_its_slots(stocked: Library) -> None:
    stocked.rename_state("corner", "corner-r4")
    assert stocked.pairs() == [("v01", "corner-r4")]


# --- designs and slots ------------------------------------------------------


def test_pairs_list_only_slots(stocked: Library) -> None:
    assert stocked.pairs() == [("v01", "corner")]
    assert stocked.designs() == {"v01": ["corner"]}


def test_a_slot_needs_an_existing_state(stocked: Library, tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="no driving state 'wet'"):
        stocked.add_slot("v01", "wet", design_files(tmp_path))


def test_move_slot_to_another_state(stocked: Library) -> None:
    stocked.move_slot("v01", "corner", "straight")
    assert stocked.pairs() == [("v01", "straight")]


def test_moving_onto_an_existing_slot_needs_overwrite(stocked: Library, tmp_path: Path) -> None:
    stocked.add_slot("v01", "straight", design_files(tmp_path, "s"))
    with pytest.raises(LibraryError, match="overwrite"):
        stocked.move_slot("v01", "corner", "straight")
    stocked.move_slot("v01", "corner", "straight", overwrite=True)
    body = stocked.root / "designs" / "v01" / "straight" / "Body.step"
    assert b"Body.step a" in body.read_bytes()
    assert stocked.pairs() == [("v01", "straight")]


def test_replace_slot_parts(stocked: Library, tmp_path: Path) -> None:
    stocked.replace_slot_parts("v01", "corner", make_files(tmp_path, ["Wing.step"], "w2"))
    wing = stocked.root / "designs" / "v01" / "corner" / "Wing.step"
    assert b"w2" in wing.read_bytes()


def test_delete_slot_and_design(stocked: Library) -> None:
    stocked.delete_slot("v01", "corner")
    assert stocked.pairs() == []
    stocked.delete_design("v01")
    assert stocked.designs() == {}


def test_rename_design(stocked: Library) -> None:
    stocked.rename_design("v01", "v02")
    assert stocked.pairs() == [("v02", "corner")]


# --- assembly ---------------------------------------------------------------


def test_assemble_copies_fifteen_parts_and_digests_them(stocked: Library, tmp_path: Path) -> None:
    target = tmp_path / "run" / "cad"
    digests = stocked.assemble("v01", "corner", target)

    assert sorted(digests) == sorted((*STATE_PARTS, *DESIGN_PARTS))
    assert sorted(p.name for p in target.glob("*.step")) == sorted(f"{p}.step" for p in digests)
    assert (target / ".simdev-cache").is_symlink()


def test_an_assembled_run_is_independent_of_later_library_edits(stocked: Library, tmp_path: Path) -> None:
    target = tmp_path / "run" / "cad"
    stocked.assemble("v01", "corner", target)
    before = (target / "Body.step").read_bytes()

    stocked.replace_slot_parts("v01", "corner", make_files(tmp_path, ["Body.step"], "b2"))
    stocked.delete_slot("v01", "corner")

    assert (target / "Body.step").read_bytes() == before


def test_assemble_without_a_slot_names_the_pair(stocked: Library, tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="no Body and Wing for state 'straight'"):
        stocked.assemble("v01", "straight", tmp_path / "run" / "cad")


def test_state_file_is_plain_yaml(stocked: Library) -> None:
    raw = yaml.safe_load((stocked.root / "states" / "corner" / "state.yaml").read_text())
    assert raw == {"description": "4 m left", **CORNER}


# --- path traversal protection ------------------------------------------


def test_delete_design_rejects_parent_directory_traversal(lib: Library) -> None:
    """Ensure delete_design('..') cannot reach the CAD root."""
    lib.create_design("v01")
    before = sorted(lib.designs_dir.iterdir())
    with pytest.raises(LibraryError, match="no design"):
        lib.delete_design("..")
    assert sorted(lib.designs_dir.iterdir()) == before


def test_delete_design_rejects_empty_name(lib: Library) -> None:
    """Ensure delete_design('') cannot reach designs/."""
    lib.create_design("v01")
    before = sorted(lib.designs_dir.iterdir())
    with pytest.raises(LibraryError, match="no design"):
        lib.delete_design("")
    assert sorted(lib.designs_dir.iterdir()) == before


def test_delete_slot_rejects_invalid_state_name(stocked: Library) -> None:
    """Ensure delete_slot('v01', '..') cannot escape the state directory."""
    before = list(stocked.root.glob("**/*"))
    with pytest.raises(LibraryError, match="no Body and Wing"):
        stocked.delete_slot("v01", "..")
    after = list(stocked.root.glob("**/*"))
    assert before == after


def test_state_rejects_path_traversal(lib: Library, tmp_path: Path) -> None:
    """Ensure state('..') cannot reach outside states/."""
    lib.create_state("corner", "", CORNER, state_files(tmp_path))
    with pytest.raises(LibraryError, match="no driving state"):
        lib.state("../states/corner")
