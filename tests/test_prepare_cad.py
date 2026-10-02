from __future__ import annotations

import json
from pathlib import Path

import pytest

import simdev.cli as cli
from simdev.cad.library import DESIGN_PARTS, STATE_PARTS, Library, LibraryError
from simdev.stages.prepare import cad_layers

CORNER = {"flow": {"u_inf": 15.0}, "physics": {"mode": "cornering", "corner_radius": 4.0}}


def stock(root: Path, tmp_path: Path) -> Library:
    upload = tmp_path / "upload"
    upload.mkdir()
    def files(parts):
        out = {}
        for part in parts:
            path = upload / f"{part}.step"
            path.write_bytes(b"ISO-10303-21;\n" + part.encode())
            out[path.name] = path
        return out
    lib = Library(root)
    lib.create_state("corner", "", CORNER, files(STATE_PARTS))
    lib.create_design("v01")
    lib.add_slot("v01", "corner", files(DESIGN_PARTS))
    return lib


def test_no_design_means_no_layers(tmp_path: Path) -> None:
    assert cad_layers(tmp_path / "run", None, None, None) == (None, {})


def test_design_and_state_go_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="together"):
        cad_layers(tmp_path / "run", "v01", None, tmp_path / "CAD")


def test_layers_assemble_the_run_geometry(tmp_path: Path) -> None:
    stock(tmp_path / "CAD", tmp_path)
    run_dir = tmp_path / "run"

    params, overrides = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")

    assert params == CORNER
    assert overrides["driving_state"] == "corner"
    assert overrides["geometry"]["source_dir"] == str(run_dir / "cad")
    assert overrides["cad"]["design"] == "v01"
    assert len(overrides["cad"]["parts"]) == 15
    assert (run_dir / "cad" / "Body.step").is_file()


def test_an_unknown_pair_is_a_library_error(tmp_path: Path) -> None:
    stock(tmp_path / "CAD", tmp_path)
    with pytest.raises(LibraryError):
        cad_layers(tmp_path / "run", "v02", "corner", tmp_path / "CAD")


def test_the_cli_passes_design_and_state_to_prepare(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake_prepare(case, run_dir, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli, "prepare", fake_prepare)
    code = cli.main([
        "prepare", "cases/car/config.yaml", "--run-dir", str(tmp_path / "run"),
        "--design", "v01", "--state", "corner", "--cad-root", str(tmp_path / "CAD"),
    ])
    assert code == 0
    assert seen["design"] == "v01"
    assert seen["state"] == "corner"
    assert seen["cad_root"] == tmp_path / "CAD"


def _prepared(run_dir: Path, overrides: dict) -> None:
    """Stand in for the caseSpec.json prepare writes once the run is built."""
    spec = {"cad": overrides["cad"], "geometry": overrides["geometry"]}
    (run_dir / "caseSpec.json").write_text(json.dumps({"spec": spec, "hash": "h"}))


def _edit_library(lib: Library, tmp_path: Path) -> None:
    new_body = tmp_path / "new" / "Body.step"
    new_body.parent.mkdir()
    new_body.write_bytes(b"ISO-10303-21;\nBody, edited")
    lib.replace_slot_parts("v01", "corner", {"Body.step": new_body})
    lib.update_state("corner", "", {**CORNER, "flow": {"u_inf": 25.0}})


def test_a_prepared_run_keeps_its_cad_and_params(tmp_path: Path) -> None:
    lib = stock(tmp_path / "CAD", tmp_path)
    run_dir = tmp_path / "run"
    params, overrides = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")
    _prepared(run_dir, overrides)
    _edit_library(lib, tmp_path)

    again, again_overrides = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")
    assert again == params == CORNER
    assert again_overrides["cad"]["parts"] == overrides["cad"]["parts"]
    assert (run_dir / "cad" / "Body.step").read_bytes().endswith(b"Body")


def test_force_reassembles_from_the_library_as_it_is_now(tmp_path: Path) -> None:
    lib = stock(tmp_path / "CAD", tmp_path)
    run_dir = tmp_path / "run"
    _, overrides = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")
    _prepared(run_dir, overrides)
    _edit_library(lib, tmp_path)

    params, forced = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD", force=True)
    assert params["flow"]["u_inf"] == 25.0
    assert forced["cad"]["parts"]["Body"] != overrides["cad"]["parts"]["Body"]
    assert forced["cad"]["parts"]["Wing"] == overrides["cad"]["parts"]["Wing"]


def test_a_prepared_run_survives_its_slot_being_deleted(tmp_path: Path) -> None:
    lib = stock(tmp_path / "CAD", tmp_path)
    run_dir = tmp_path / "run"
    params, overrides = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")
    _prepared(run_dir, overrides)
    lib.delete_slot("v01", "corner")

    again, again_overrides = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")
    assert again == params
    assert again_overrides == overrides


def test_a_tampered_copy_is_reassembled(tmp_path: Path) -> None:
    stock(tmp_path / "CAD", tmp_path)
    run_dir = tmp_path / "run"
    _, overrides = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")
    _prepared(run_dir, overrides)
    (run_dir / "cad" / "Wing.step").write_bytes(b"changed by hand")

    _, again = cad_layers(run_dir, "v01", "corner", tmp_path / "CAD")
    assert again["cad"]["parts"] == overrides["cad"]["parts"]
    assert (run_dir / "cad" / "Wing.step").read_bytes().endswith(b"Wing")
